"""Operational storage layer: SQLAlchemy DB for alerts, feedback, and live stream metrics."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    desc,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from src.config import load_config, resolve_path

Base = declarative_base()


class ScoredTransactionORM(Base):
    __tablename__ = "scored_transactions"

    tx_id = Column(String(64), primary_key=True, index=True)
    step = Column(Integer, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=True)
    tx_type = Column(String(32), nullable=False)
    amount = Column(Float, nullable=False)
    name_orig = Column(String(64), nullable=False, index=True)
    name_dest = Column(String(64), nullable=False, index=True)
    device_id = Column(String(64), nullable=False)
    channel = Column(String(32), nullable=False)
    ip_country = Column(String(16), nullable=False, default="RU")

    # Scoring outputs
    action = Column(String(16), nullable=False, index=True)
    risk_score = Column(Float, nullable=False, index=True)
    ml_score = Column(Float, nullable=False)
    rule_score = Column(Float, nullable=False)
    anomaly_score = Column(Float, nullable=False)
    triggered_rules_json = Column(Text, nullable=False, default="[]")
    top_factors_json = Column(Text, nullable=False, default="[]")
    features_json = Column(Text, nullable=False, default="{}")
    explanation_summary = Column(Text, nullable=False, default="")
    latency_ms = Column(Float, nullable=False, default=0.0)

    # Feedback loop (analyst labeling)
    analyst_label = Column(Boolean, nullable=True, index=True)
    analyst_id = Column(String(64), nullable=True)
    analyst_notes = Column(Text, nullable=True)
    labeled_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class StorageManager:
    """Manages SQLite/PostgreSQL operational persistence for the Fraud Agent."""

    def __init__(self, database_url: str | None = None):
        cfg = load_config()
        url = database_url or cfg["paths"]["database_url"]
        if url.startswith("sqlite:///") and not url.startswith("sqlite:///:memory:"):
            rel_path = url.replace("sqlite:///", "", 1)
            abs_path = resolve_path(rel_path)
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite:///{abs_path.as_posix()}"

        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self.engine = create_engine(url, connect_args=connect_args, future=True)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)

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
            rec = ScoredTransactionORM(
                tx_id=tx["tx_id"],
                step=int(tx["step"]),
                timestamp=ts,
                tx_type=str(tx["type"]),
                amount=float(tx["amount"]),
                name_orig=str(tx["nameOrig"]),
                name_dest=str(tx["nameDest"]),
                device_id=str(tx.get("device_id", "DEV_UNKNOWN")),
                channel=str(tx.get("channel", "MOBILE")),
                ip_country=str(tx.get("ip_country", "RU")),
                action=str(decision["action"]),
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
        is_fraud: bool,
        analyst_id: str = "analyst_1",
        notes: str = "",
    ) -> datetime | None:
        with self.SessionLocal() as session:
            rec = session.get(ScoredTransactionORM, tx_id)
            if rec is None:
                return None
            now = datetime.now(timezone.utc)
            rec.analyst_label = bool(is_fraud)
            rec.analyst_id = analyst_id
            rec.analyst_notes = notes
            rec.labeled_at = now
            session.commit()
            return now

    def get_alerts(
        self,
        actions: list[str] | None = None,
        only_unlabeled: bool = False,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        with self.SessionLocal() as session:
            q = session.query(ScoredTransactionORM)
            if actions:
                q = q.filter(ScoredTransactionORM.action.in_(actions))
            if only_unlabeled:
                q = q.filter(ScoredTransactionORM.analyst_label.is_(None))
            rows = q.order_by(desc(ScoredTransactionORM.risk_score), desc(ScoredTransactionORM.created_at)).limit(limit).all()
            return [self._row_to_dict(r) for r in rows]

    def get_transaction(self, tx_id: str) -> dict[str, Any] | None:
        with self.SessionLocal() as session:
            rec = session.get(ScoredTransactionORM, tx_id)
            return self._row_to_dict(rec) if rec else None

    def get_recent_features_df(self, limit: int = 5000) -> pd.DataFrame:
        """Return DataFrame of recent scored transactions' features and scores for drift monitoring."""
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
                feat["risk_score"] = r.risk_score
                feat["ml_score"] = r.ml_score
                feat["action"] = r.action
                feat["amount"] = r.amount
                feat["analyst_label"] = r.analyst_label
                records.append(feat)
            return pd.DataFrame(records)

    def get_labeled_feedback_df(self) -> pd.DataFrame:
        """Return all analyst-labeled cases with features for retraining."""
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
                feat = json.loads(r.features_json or "{}")
                feat["tx_id"] = r.tx_id
                feat["step"] = r.step
                feat["amount"] = r.amount
                feat["isFraud"] = int(bool(r.analyst_label))
                records.append(feat)
            return pd.DataFrame(records)

    def get_summary_stats(self) -> dict[str, Any]:
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
                    "labeled_count": 0,
                    "confirmed_fraud_count": 0,
                }
            counts = {"ALLOW": 0, "OTP": 0, "REVIEW": 0, "BLOCK": 0}
            blocked_amount = 0.0
            latencies = []
            labeled = 0
            confirmed_fraud = 0
            for r in rows:
                counts[r.action] = counts.get(r.action, 0) + 1
                if r.action == "BLOCK":
                    blocked_amount += r.amount
                latencies.append(r.latency_ms)
                if r.analyst_label is not None:
                    labeled += 1
                    if r.analyst_label:
                        confirmed_fraud += 1
            alerts = counts["OTP"] + counts["REVIEW"] + counts["BLOCK"]
            return {
                "total_scored": total,
                "action_counts": counts,
                "alert_rate": alerts / total,
                "blocked_amount": round(blocked_amount, 2),
                "avg_latency_ms": round(sum(latencies) / total, 2),
                "labeled_count": labeled,
                "confirmed_fraud_count": confirmed_fraud,
            }

    @staticmethod
    def _row_to_dict(r: ScoredTransactionORM) -> dict[str, Any]:
        return {
            "tx_id": r.tx_id,
            "step": r.step,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            "type": r.tx_type,
            "amount": r.amount,
            "nameOrig": r.name_orig,
            "nameDest": r.name_dest,
            "device_id": r.device_id,
            "channel": r.channel,
            "ip_country": r.ip_country,
            "action": r.action,
            "risk_score": r.risk_score,
            "ml_score": r.ml_score,
            "rule_score": r.rule_score,
            "anomaly_score": r.anomaly_score,
            "triggered_rules": json.loads(r.triggered_rules_json or "[]"),
            "top_factors": json.loads(r.top_factors_json or "[]"),
            "features": json.loads(r.features_json or "{}"),
            "explanation_summary": r.explanation_summary,
            "latency_ms": r.latency_ms,
            "analyst_label": r.analyst_label,
            "analyst_id": r.analyst_id,
            "analyst_notes": r.analyst_notes,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
