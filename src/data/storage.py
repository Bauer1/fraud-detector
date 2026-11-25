"""Operational storage layer: SQLAlchemy DB for decisions, tri-state feedback, audit trail, idempotency, and monitoring."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    asc,
    create_engine,
    desc,
    inspect,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

from src.config import load_config, resolve_path
from src.data.schemas import ensure_utc_datetime
from src.features.online_store import SCHEMA_VERSION

Base = declarative_base()


def _as_utc(dt: datetime | None) -> datetime | None:
    return ensure_utc_datetime(dt)


class ScoredTransactionORM(Base):
    __tablename__ = "scored_transactions"

    tx_id = Column(String(128), primary_key=True, index=True)
    step = Column(Integer, nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=True)
    tx_type = Column(String(32), nullable=False)
    amount = Column(Float, nullable=False)
    name_orig = Column(String(128), nullable=False, index=True)
    name_dest = Column(String(128), nullable=False, index=True)
    device_id = Column(String(128), nullable=True)
    channel = Column(String(32), nullable=True)
    ip_country = Column(String(16), nullable=True)
    raw_payload_json = Column(Text, nullable=False, default="{}")

    # Scoring & governance metadata
    action = Column(String(16), nullable=False, index=True)
    effective_action = Column(String(16), nullable=False, default="ALLOW")
    decision_mode = Column(String(16), nullable=False, default="ACTIVE")
    technical_status = Column(String(32), nullable=False, default="OK")
    fallback_reason = Column(Text, nullable=True)
    model_version = Column(String(128), nullable=False, default="unversioned", index=True)
    feature_schema_version = Column(String(64), nullable=False, default=SCHEMA_VERSION)

    risk_score = Column(Float, nullable=False, index=True)
    ml_score = Column(Float, nullable=False)
    rule_score = Column(Float, nullable=False)
    anomaly_score = Column(Float, nullable=False)
    triggered_rules_json = Column(Text, nullable=False, default="[]")
    top_factors_json = Column(Text, nullable=False, default="[]")
    features_json = Column(Text, nullable=False, default="{}")
    explanation_summary = Column(Text, nullable=False, default="")
    latency_ms = Column(Float, nullable=False, default=0.0)

    # Tri-state feedback (`fraud` / `not_fraud` / `unknown`) + provenance
    label_status = Column(String(16), nullable=True, index=True)
    analyst_label = Column(Boolean, nullable=True, index=True)
    label_source = Column(String(64), nullable=True)
    analyst_id = Column(String(64), nullable=True)
    analyst_notes = Column(Text, nullable=True)
    labeled_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class FeedbackAuditORM(Base):
    """Immutable audit log of all analyst / chargeback / investigation label changes."""

    __tablename__ = "feedback_audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tx_id = Column(String(128), nullable=False, index=True)
    previous_label_status = Column(String(16), nullable=True)
    new_label_status = Column(String(16), nullable=False)
    label_source = Column(String(64), nullable=False, default="analyst_review")
    analyst_id = Column(String(64), nullable=False)
    notes = Column(Text, nullable=False, default="")
    labeled_at = Column(DateTime(timezone=True), nullable=False)
    recorded_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class StorageManager:
    """Manages operational persistence, label audit history, and live telemetry queries."""

    def __init__(self, database_url: str | None = None):
        cfg = load_config()
        url = database_url or cfg["paths"]["database_url"]
        if url.startswith("sqlite:///") and not url.startswith("sqlite:///:memory:"):
            rel_path = url.replace("sqlite:///", "", 1)
            abs_path = resolve_path(rel_path)
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite:///{abs_path.as_posix()}"

        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        engine_kwargs: dict[str, Any] = {"connect_args": connect_args, "future": True}
        if url == "sqlite:///:memory:":
            engine_kwargs["poolclass"] = StaticPool
        self.engine = create_engine(url, **engine_kwargs)
        Base.metadata.create_all(self.engine)
        self._ensure_backward_compatible_columns()
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)

    def close(self) -> None:
        self.engine.dispose()

    def _ensure_backward_compatible_columns(self) -> None:
        """Safely migrate legacy SQLite tables missing newly introduced governance/audit columns."""
        inspector = inspect(self.engine)
        if not inspector.has_table("scored_transactions"):
            return
        existing_cols = {col["name"] for col in inspector.get_columns("scored_transactions")}
        migrations = [
            ("raw_payload_json", "TEXT NOT NULL DEFAULT '{}'"),
            ("effective_action", " VARCHAR(16) NOT NULL DEFAULT 'ALLOW'"),
            ("decision_mode", "VARCHAR(16) NOT NULL DEFAULT 'ACTIVE'"),
            ("technical_status", "VARCHAR(32) NOT NULL DEFAULT 'OK'"),
            ("fallback_reason", "TEXT"),
            ("model_version", "VARCHAR(128) NOT NULL DEFAULT 'unversioned'"),
            ("feature_schema_version", f"VARCHAR(64) NOT NULL DEFAULT '{SCHEMA_VERSION}'"),
            ("label_status", "VARCHAR(16)"),
            ("label_source", "VARCHAR(64)"),
        ]
        with self.engine.begin() as conn:
            for col_name, col_ddl in migrations:
                if col_name not in existing_cols:
                    conn.execute(text(f"ALTER TABLE scored_transactions ADD COLUMN {col_name} {col_ddl}"))

    def save_scored_transaction(
        self,
        tx: dict[str, Any],
        decision: dict[str, Any],
        features: dict[str, float],
    ) -> None:
        with self.SessionLocal() as session:
            ts = tx.get("timestamp")
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            ts = _as_utc(ts)

            # Store exact input snapshot preserving None vs explicit values for lossless idempotency & audit
            raw_payload = {
                "oldbalanceOrg": tx.get("oldbalanceOrg"),
                "newbalanceOrig": tx.get("newbalanceOrig"),
                "oldbalanceDest": tx.get("oldbalanceDest"),
                "newbalanceDest": tx.get("newbalanceDest"),
                "device_id": tx.get("device_id"),
                "channel": tx.get("channel"),
                "ip_country": tx.get("ip_country"),
            }

            act = str(decision["action"])
            eff_act = str(decision.get("effective_action", act))
            rec = ScoredTransactionORM(
                tx_id=str(tx["tx_id"]),
                step=int(tx.get("step") or 0),
                timestamp=ts,
                tx_type=str(tx["type"]),
                amount=float(tx["amount"]),
                name_orig=str(tx["nameOrig"]),
                name_dest=str(tx["nameDest"]),
                device_id=str(tx["device_id"]) if tx.get("device_id") is not None else None,
                channel=str(tx["channel"]) if tx.get("channel") is not None else None,
                ip_country=str(tx["ip_country"]).upper() if tx.get("ip_country") is not None else None,
                raw_payload_json=json.dumps(raw_payload, ensure_ascii=False),
                action=act,
                effective_action=eff_act,
                decision_mode=str(decision.get("decision_mode", "ACTIVE")),
                technical_status=str(decision.get("technical_status", "OK")),
                fallback_reason=decision.get("fallback_reason"),
                model_version=str(decision.get("model_version", "unversioned")),
                feature_schema_version=str(decision.get("feature_schema_version", SCHEMA_VERSION)),
                risk_score=float(decision["risk_score"]),
                ml_score=float(decision["ml_score"]),
                rule_score=float(decision["rule_score"]),
                anomaly_score=float(decision["anomaly_score"]),
                triggered_rules_json=json.dumps(decision.get("triggered_rules", []), ensure_ascii=False),
                top_factors_json=json.dumps(decision.get("top_factors", []), ensure_ascii=False),
                features_json=json.dumps(
                    {k: float(v) for k, v in features.items() if isinstance(v, (int, float))},
                    ensure_ascii=False,
                ),
                explanation_summary=str(decision.get("explanation_summary", "")),
                latency_ms=float(decision.get("latency_ms", 0.0)),
            )
            session.merge(rec)
            session.commit()

    def record_feedback(
        self,
        tx_id: str,
        is_fraud: bool | None = None,
        analyst_id: str = "analyst_1",
        notes: str = "",
        *,
        label_status: str | None = None,
        label_source: str = "analyst_review",
        labeled_at: datetime | None = None,
    ) -> tuple[datetime, int] | None:
        """Record tri-state label (`fraud` / `not_fraud` / `unknown`) and append an immutable audit log entry."""
        if label_status is None:
            if is_fraud is True:
                resolved_status = "fraud"
            elif is_fraud is False:
                resolved_status = "not_fraud"
            else:
                resolved_status = "unknown"
        else:
            resolved_status = label_status.strip().lower()
            if resolved_status not in ("fraud", "not_fraud", "unknown"):
                raise ValueError(f"Unsupported label_status='{label_status}'")

        resolved_bool: bool | None = (
            True if resolved_status == "fraud" else (False if resolved_status == "not_fraud" else None)
        )

        with self.SessionLocal() as session:
            rec = session.get(ScoredTransactionORM, tx_id)
            if rec is None:
                return None

            prev_status = rec.label_status
            if prev_status is None and rec.analyst_label is not None:
                prev_status = "fraud" if rec.analyst_label else "not_fraud"

            now = _as_utc(labeled_at) or datetime.now(timezone.utc)
            rec.label_status = resolved_status
            rec.analyst_label = resolved_bool
            rec.label_source = label_source
            rec.analyst_id = analyst_id
            rec.analyst_notes = notes
            rec.labeled_at = now

            audit_row = FeedbackAuditORM(
                tx_id=tx_id,
                previous_label_status=prev_status,
                new_label_status=resolved_status,
                label_source=label_source,
                analyst_id=analyst_id,
                notes=notes,
                labeled_at=now,
            )
            session.add(audit_row)
            session.commit()
            session.refresh(audit_row)
            return now, int(audit_row.id)

    def get_feedback_audit_trail(self, tx_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Return chronological history of label changes for auditability."""
        with self.SessionLocal() as session:
            q = session.query(FeedbackAuditORM)
            if tx_id:
                q = q.filter(FeedbackAuditORM.tx_id == tx_id)
            rows = q.order_by(desc(FeedbackAuditORM.id)).limit(limit).all()
            return [
                {
                    "audit_id": r.id,
                    "tx_id": r.tx_id,
                    "previous_label": r.previous_label_status,
                    "new_label": r.new_label_status,
                    "label_source": r.label_source,
                    "analyst_id": r.analyst_id,
                    "notes": r.notes,
                    "labeled_at": _as_utc(r.labeled_at).isoformat() if r.labeled_at else None,
                    "recorded_at": _as_utc(r.recorded_at).isoformat() if r.recorded_at else None,
                }
                for r in rows
            ]

    def get_alerts(
        self,
        actions: list[str] | None = None,
        only_unlabeled: bool = False,
        label_status: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        with self.SessionLocal() as session:
            q = session.query(ScoredTransactionORM)
            if actions:
                q = q.filter(ScoredTransactionORM.action.in_(actions))
            if only_unlabeled:
                q = q.filter(
                    (ScoredTransactionORM.label_status.is_(None))
                    | (ScoredTransactionORM.label_status == "unknown")
                )
            elif label_status:
                q = q.filter(ScoredTransactionORM.label_status == label_status)
            rows = q.order_by(desc(ScoredTransactionORM.risk_score), desc(ScoredTransactionORM.created_at)).limit(limit).all()
            return [self._row_to_dict(r) for r in rows]

    def get_transaction(self, tx_id: str) -> dict[str, Any] | None:
        with self.SessionLocal() as session:
            rec = session.get(ScoredTransactionORM, tx_id)
            return self._row_to_dict(rec) if rec else None

    def get_operational_history(self, as_of: datetime | None = None) -> list[dict[str, Any]]:
        """Return scored operational transactions ordered chronologically up to `as_of` for clean state warm-up."""
        cutoff = _as_utc(as_of)
        with self.SessionLocal() as session:
            rows = (
                session.query(ScoredTransactionORM)
                .order_by(asc(ScoredTransactionORM.timestamp), asc(ScoredTransactionORM.step), asc(ScoredTransactionORM.tx_id))
                .all()
            )
            events: list[dict[str, Any]] = []
            for r in rows:
                ts_utc = _as_utc(r.timestamp)
                if cutoff is not None and ts_utc is not None and ts_utc > cutoff:
                    continue
                raw_snap = json.loads(r.raw_payload_json or "{}")
                events.append({
                    "tx_id": r.tx_id,
                    "step": r.step,
                    "timestamp": ts_utc,
                    "type": r.tx_type,
                    "amount": r.amount,
                    "nameOrig": r.name_orig,
                    "oldbalanceOrg": raw_snap.get("oldbalanceOrg"),
                    "newbalanceOrig": raw_snap.get("newbalanceOrig"),
                    "nameDest": r.name_dest,
                    "oldbalanceDest": raw_snap.get("oldbalanceDest"),
                    "newbalanceDest": raw_snap.get("newbalanceDest"),
                    "device_id": raw_snap.get("device_id", r.device_id),
                    "channel": raw_snap.get("channel", r.channel),
                    "ip_country": raw_snap.get("ip_country", r.ip_country),
                })
            return events

    def get_recent_features_df(self, limit: int = 5000) -> pd.DataFrame:
        """Return DataFrame of recent scored transactions' features, scores, and label maturity metadata."""
        with self.SessionLocal() as session:
            rows = (
                session.query(ScoredTransactionORM)
                .order_by(desc(ScoredTransactionORM.created_at))
                .limit(limit)
                .all()
            )
            if not rows:
                return pd.DataFrame()
            records = []
            for r in rows:
                feat = json.loads(r.features_json or "{}")
                feat["tx_id"] = r.tx_id
                feat["risk_score"] = r.risk_score
                feat["ml_score"] = r.ml_score
                feat["action"] = r.action
                feat["effective_action"] = r.effective_action
                feat["decision_mode"] = r.decision_mode
                feat["technical_status"] = r.technical_status
                feat["model_version"] = r.model_version
                feat["amount"] = r.amount
                feat["analyst_label"] = r.analyst_label
                feat["label_status"] = r.label_status or (
                    "fraud" if r.analyst_label is True else ("not_fraud" if r.analyst_label is False else None)
                )
                feat["label_source"] = r.label_source
                feat["event_timestamp"] = _as_utc(r.timestamp) or _as_utc(r.created_at)
                feat["created_at"] = _as_utc(r.created_at)
                feat["latency_ms"] = r.latency_ms
                records.append(feat)
            return pd.DataFrame(records)

    def count_unknown_feedback(self) -> int:
        with self.SessionLocal() as session:
            return int(
                session.query(ScoredTransactionORM)
                .filter(ScoredTransactionORM.label_status == "unknown")
                .count()
            )

    def get_labeled_feedback_df(self) -> pd.DataFrame:
        """Return ONLY definitively labeled (`fraud` / `not_fraud`) cases for candidate retraining.

        Cases marked `unknown` or unlabeled are strictly excluded so immature transactions
        are never miscast as negative examples.
        """
        with self.SessionLocal() as session:
            rows = (
                session.query(ScoredTransactionORM)
                .filter(ScoredTransactionORM.analyst_label.isnot(None))
                .all()
            )
            if not rows:
                return pd.DataFrame()
            records = []
            for r in rows:
                if r.label_status == "unknown":
                    continue
                feat = json.loads(r.features_json or "{}")
                ts_utc = _as_utc(r.timestamp)
                feat["tx_id"] = r.tx_id
                feat["step"] = r.step
                feat["timestamp"] = ts_utc.isoformat() if ts_utc else None
                feat["amount"] = r.amount
                feat["label_source"] = r.label_source or "analyst_review"
                feat["isFraud"] = int(bool(r.analyst_label))
                records.append(feat)
            return pd.DataFrame(records)

    def get_summary_stats(
        self,
        maturity_hours: float = 72.0,
        max_daily_reviews: int = 150,
    ) -> dict[str, Any]:
        with self.SessionLocal() as session:
            rows = session.query(ScoredTransactionORM).all()
            total = len(rows)
            if total == 0:
                return {
                    "total_scored": 0,
                    "action_counts": {"ALLOW": 0, "OTP": 0, "REVIEW": 0, "BLOCK": 0},
                    "alert_rate": 0.0,
                    "blocked_amount": 0.0,
                    "avg_latency_ms": 0.0,
                    "p50_latency_ms": 0.0,
                    "p95_latency_ms": 0.0,
                    "p99_latency_ms": 0.0,
                    "degraded_fallback_count": 0,
                    "shadow_mode_count": 0,
                    "labeled_count": 0,
                    "confirmed_fraud_count": 0,
                    "confirmed_not_fraud_count": 0,
                    "unknown_label_count": 0,
                    "immature_unlabeled_count": 0,
                    "mature_unlabeled_count": 0,
                    "reviews_last_24h": 0,
                    "daily_review_capacity": max_daily_reviews,
                    "review_capacity_utilization": 0.0,
                    "label_coverage_by_action": {"ALLOW": 0.0, "OTP": 0.0, "REVIEW": 0.0, "BLOCK": 0.0},
                }

            now = datetime.now(timezone.utc)
            maturity_cutoff = now - timedelta(hours=float(maturity_hours))
            day_cutoff = now - timedelta(hours=24.0)

            counts = {"ALLOW": 0, "OTP": 0, "REVIEW": 0, "BLOCK": 0}
            labeled_by_action = {"ALLOW": 0, "OTP": 0, "REVIEW": 0, "BLOCK": 0}
            blocked_amount = 0.0
            latencies: list[float] = []
            labeled = 0
            confirmed_fraud = 0
            confirmed_not_fraud = 0
            unknown_cnt = 0
            immature_unlabeled = 0
            mature_unlabeled = 0
            degraded_cnt = 0
            shadow_cnt = 0
            reviews_24h = 0

            for r in rows:
                act = r.action
                counts[act] = counts.get(act, 0) + 1
                if act == "BLOCK":
                    blocked_amount += r.amount
                latencies.append(float(r.latency_ms or 0.0))

                if r.technical_status == "DEGRADED_FALLBACK":
                    degraded_cnt += 1
                if r.decision_mode == "SHADOW":
                    shadow_cnt += 1

                created_utc = _as_utc(r.created_at) or now
                if act == "REVIEW" and created_utc >= day_cutoff:
                    reviews_24h += 1

                l_status = r.label_status
                if l_status is None and r.analyst_label is not None:
                    l_status = "fraud" if r.analyst_label else "not_fraud"

                if l_status == "fraud":
                    labeled += 1
                    confirmed_fraud += 1
                    labeled_by_action[act] = labeled_by_action.get(act, 0) + 1
                elif l_status == "not_fraud":
                    labeled += 1
                    confirmed_not_fraud += 1
                    labeled_by_action[act] = labeled_by_action.get(act, 0) + 1
                elif l_status == "unknown":
                    unknown_cnt += 1
                else:
                    ev_utc = _as_utc(r.timestamp) or created_utc
                    if ev_utc >= maturity_cutoff:
                        immature_unlabeled += 1
                    else:
                        mature_unlabeled += 1

            lat_arr = np.asarray(latencies, dtype=float)
            alerts = counts["OTP"] + counts["REVIEW"] + counts["BLOCK"]
            coverage_by_action = {
                k: round(labeled_by_action.get(k, 0) / counts[k], 4) if counts.get(k, 0) > 0 else 0.0
                for k in ("ALLOW", "OTP", "REVIEW", "BLOCK")
            }
            return {
                "total_scored": total,
                "action_counts": counts,
                "alert_rate": round(alerts / total, 4),
                "blocked_amount": round(blocked_amount, 2),
                "avg_latency_ms": round(float(lat_arr.mean()), 2),
                "p50_latency_ms": round(float(np.percentile(lat_arr, 50)), 2),
                "p95_latency_ms": round(float(np.percentile(lat_arr, 95)), 2),
                "p99_latency_ms": round(float(np.percentile(lat_arr, 99)), 2),
                "degraded_fallback_count": degraded_cnt,
                "shadow_mode_count": shadow_cnt,
                "labeled_count": labeled,
                "confirmed_fraud_count": confirmed_fraud,
                "confirmed_not_fraud_count": confirmed_not_fraud,
                "unknown_label_count": unknown_cnt,
                "immature_unlabeled_count": immature_unlabeled,
                "mature_unlabeled_count": mature_unlabeled,
                "reviews_last_24h": reviews_24h,
                "daily_review_capacity": max_daily_reviews,
                "review_capacity_utilization": round(reviews_24h / max(max_daily_reviews, 1), 4),
                "label_coverage_by_action": coverage_by_action,
            }

    @staticmethod
    def _row_to_dict(r: ScoredTransactionORM) -> dict[str, Any]:
        ts_utc = _as_utc(r.timestamp)
        created_utc = _as_utc(r.created_at)
        labeled_utc = _as_utc(r.labeled_at)
        raw_snap = json.loads(r.raw_payload_json or "{}")
        l_status = r.label_status
        if l_status is None and r.analyst_label is not None:
            l_status = "fraud" if r.analyst_label else "not_fraud"

        return {
            "tx_id": r.tx_id,
            "step": r.step,
            "timestamp": ts_utc.isoformat() if ts_utc else None,
            "type": r.tx_type,
            "amount": r.amount,
            "nameOrig": r.name_orig,
            "oldbalanceOrg": raw_snap.get("oldbalanceOrg"),
            "newbalanceOrig": raw_snap.get("newbalanceOrig"),
            "nameDest": r.name_dest,
            "oldbalanceDest": raw_snap.get("oldbalanceDest"),
            "newbalanceDest": raw_snap.get("newbalanceDest"),
            "device_id": raw_snap.get("device_id", r.device_id),
            "channel": raw_snap.get("channel", r.channel),
            "ip_country": raw_snap.get("ip_country", r.ip_country),
            "action": r.action,
            "effective_action": r.effective_action or r.action,
            "decision_mode": r.decision_mode or "ACTIVE",
            "technical_status": r.technical_status or "OK",
            "fallback_reason": r.fallback_reason,
            "model_version": r.model_version or "unversioned",
            "feature_schema_version": r.feature_schema_version or SCHEMA_VERSION,
            "risk_score": r.risk_score,
            "ml_score": r.ml_score,
            "rule_score": r.rule_score,
            "anomaly_score": r.anomaly_score,
            "triggered_rules": json.loads(r.triggered_rules_json or "[]"),
            "top_factors": json.loads(r.top_factors_json or "[]"),
            "features": json.loads(r.features_json or "{}"),
            "explanation_summary": r.explanation_summary,
            "latency_ms": r.latency_ms,
            "label_status": l_status,
            "analyst_label": r.analyst_label,
            "label_source": r.label_source,
            "analyst_id": r.analyst_id,
            "analyst_notes": r.analyst_notes,
            "labeled_at": labeled_utc.isoformat() if labeled_utc else None,
            "created_at": created_utc.isoformat() if created_utc else None,
        }
