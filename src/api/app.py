"""FastAPI Scoring & Operations Service for the AI Fraud Detection Agent."""
from __future__ import annotations

import json
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from src.agent.decision_agent import FraudDecisionAgent
from src.agent.explainer import FraudExplainer
from src.config import load_config, resolve_path
from src.data.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    ScoreResponse,
    TransactionRequest,
)
from src.data.storage import StorageManager
from src.features.online_store import (
    ALL_FEATURE_NAMES,
    DuplicateTransactionConflictError,
    FeatureStoreError,
    OnlineFeatureStore,
    StaleEventError,
    canonical_tx_fingerprint,
)
from src.monitoring.drift import compute_drift_report, compute_feedback_quality
from src.rules.engine import RuleEngine
from src.training.pipeline import run_training_pipeline
from src.training.retrain import retrain_with_feedback

VALID_INITIAL_STATE_MODES = frozenset({"empty", "operational_db", "pre_test_warmup"})


class ServiceState:
    """Thread-safe container for scoring models, clean online feature state, and persistence."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.cfg: dict[str, Any] = load_config()
        self.storage: StorageManager = StorageManager(self.cfg["paths"]["database_url"])
        self.feature_store: OnlineFeatureStore | None = None
        self.agent: FraudDecisionAgent | None = None
        self.metadata: dict[str, Any] = {}
        self.reference_df: pd.DataFrame = pd.DataFrame()

    def _build_initial_feature_store(
        self,
        mode: str,
        as_of: datetime | None = None,
    ) -> OnlineFeatureStore:
        if mode not in VALID_INITIAL_STATE_MODES:
            raise ValueError(f"Unsupported initial_state_mode='{mode}'. Expected one of {sorted(VALID_INITIAL_STATE_MODES)}")

        scoring_cfg = self.cfg.get("scoring", {})
        mule_th = int(self.cfg["rules"]["mule_in_degree_threshold"])
        sender_ooo = scoring_cfg.get("max_sender_out_of_order_seconds", 300.0)
        global_stale = scoring_cfg.get("max_global_staleness_seconds", 86400.0)
        idem_size = int(scoring_cfg.get("max_idempotency_cache_size", 50000))
        startup_ts = as_of or datetime.now(timezone.utc)

        if mode == "empty":
            return OnlineFeatureStore(
                mule_in_degree_threshold=mule_th,
                max_sender_out_of_order_sec=sender_ooo,
                max_global_staleness_sec=global_stale,
                max_idempotency_entries=idem_size,
                state_origin="empty",
                as_of_timestamp=startup_ts.isoformat(),
            )

        if mode == "operational_db":
            op_events = self.storage.get_operational_history(as_of=startup_ts)
            return OnlineFeatureStore.from_history(
                op_events,
                as_of_timestamp=startup_ts,
                mule_in_degree_threshold=mule_th,
                max_sender_out_of_order_sec=sender_ooo,
                max_global_staleness_sec=global_stale,
                max_idempotency_entries=idem_size,
                state_origin="operational_db",
            )

        # mode == "pre_test_warmup"
        artifacts_dir = resolve_path(self.cfg["paths"]["artifacts_dir"])
        warmup_path = artifacts_dir / "initial_feature_store_pre_test.joblib"
        if not warmup_path.exists():
            raise FileNotFoundError(
                f"Pre-test warm-up store not found at {warmup_path}. Run `python -m src.training.pipeline` first."
            )
        store: OnlineFeatureStore = joblib.load(warmup_path)
        if getattr(store, "state_origin", "") != "pre_test_warmup":
            raise RuntimeError(
                f"Refusing to load feature store with untrusted state_origin='{getattr(store, 'state_origin', None)}'"
            )
        store.max_sender_out_of_order_sec = sender_ooo
        store.max_global_staleness_sec = global_stale
        store.max_idempotency_entries = idem_size
        return store

    def reload_model_artifacts(self) -> None:
        """Atomically reload model, thresholds, metadata, and reference distribution without resetting live feature state."""
        with self._lock:
            artifacts_dir = resolve_path(self.cfg["paths"]["artifacts_dir"])
            gbdt_path = artifacts_dir / "gbdt_model.joblib"
            if not gbdt_path.exists():
                auto_train = bool(self.cfg.get("scoring", {}).get("auto_train_on_startup", False))
                if auto_train:
                    run_training_pipeline(cfg=self.cfg)
                else:
                    raise FileNotFoundError(
                        f"Model artifact not found at {gbdt_path}. Run `python -m src.training.pipeline` before starting API."
                    )

            gbdt_model = joblib.load(gbdt_path)
            anomaly_detector = joblib.load(artifacts_dir / "anomaly_detector.joblib")
            background_median = joblib.load(artifacts_dir / "background_median.joblib")

            meta_path = artifacts_dir / "metadata.json"
            metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

            ref_path = artifacts_dir / "reference_distribution.parquet"
            reference_df = pd.read_parquet(ref_path) if ref_path.exists() else pd.DataFrame()

            thresholds = metadata.get("optimized_thresholds", self.cfg["agent"]["thresholds"])
            rule_engine = RuleEngine(rules_cfg=self.cfg["rules"])
            explainer = FraudExplainer(
                ml_model=gbdt_model,
                feature_names=ALL_FEATURE_NAMES,
                background_median=background_median,
                llm_cfg=self.cfg.get("llm", {}),
            )
            new_agent = FraudDecisionAgent(
                rule_engine=rule_engine,
                ml_model=gbdt_model,
                anomaly_detector=anomaly_detector,
                explainer=explainer,
                weights=self.cfg["agent"]["weights"],
                thresholds=thresholds,
                feature_names=ALL_FEATURE_NAMES,
            )

            self.agent = new_agent
            self.metadata = metadata
            self.reference_df = reference_df

    def load_artifacts(
        self,
        *,
        initial_state_mode: str | None = None,
        as_of: datetime | None = None,
    ) -> None:
        """Load model artifacts and initialize a clean, leak-free OnlineFeatureStore."""
        with self._lock:
            self.reload_model_artifacts()
            mode = initial_state_mode or self.cfg.get("scoring", {}).get("initial_state_mode", "empty")
            self.feature_store = self._build_initial_feature_store(mode=mode, as_of=as_of)

    def score(self, tx: TransactionRequest) -> ScoreResponse:
        """Thread-safe, idempotent, event-order-checked transaction scoring with atomic state commit."""
        with self._lock:
            if self.agent is None or self.feature_store is None:
                try:
                    self.load_artifacts()
                except FileNotFoundError as exc:
                    raise HTTPException(status_code=503, detail=str(exc)) from exc

            assert self.agent is not None
            assert self.feature_store is not None

            tx_dict = tx.model_dump()
            tx_dict["type"] = tx.type.value
            tx_dict["channel"] = tx.channel.value

            # 1. Persistent idempotency & conflict check by tx_id
            existing_rec = self.storage.get_transaction(tx.tx_id)
            if existing_rec is not None:
                existing_fp = canonical_tx_fingerprint(existing_rec)
                incoming_fp = canonical_tx_fingerprint(tx_dict)
                if existing_fp != incoming_fp:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Conflicting transaction payload for already-scored tx_id='{tx.tx_id}'",
                    )
                return ScoreResponse(
                    tx_id=existing_rec["tx_id"],
                    action=existing_rec["action"],
                    risk_score=existing_rec["risk_score"],
                    ml_score=existing_rec["ml_score"],
                    rule_score=existing_rec["rule_score"],
                    anomaly_score=existing_rec["anomaly_score"],
                    triggered_rules=existing_rec["triggered_rules"],
                    top_factors=existing_rec["top_factors"],
                    explanation_summary=existing_rec["explanation_summary"],
                    latency_ms=existing_rec["latency_ms"],
                    is_duplicate=True,
                )

            # 2. Compute features without mutating store yet (validates schema, order, and in-memory tx_id cache)
            try:
                in_mem_cached = self.feature_store.get_cached_features(tx_dict)
                if in_mem_cached is not None:
                    # Only reachable if custom caller populated store directly outside storage
                    decision = self.agent.evaluate_transaction(tx_dict, in_mem_cached, explain=True)
                    return ScoreResponse(**decision, is_duplicate=True)

                features = self.feature_store.compute_features(
                    tx_dict,
                    update_state=False,
                    strict_fields=True,
                )
            except DuplicateTransactionConflictError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except (StaleEventError, FeatureStoreError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

            # 3. Run multi-layer fraud evaluation
            decision = self.agent.evaluate_transaction(tx_dict, features, explain=True)

            # 4. Persist to operational DB, then commit OnlineFeatureStore state atomically
            self.storage.save_scored_transaction(tx_dict, decision, features)
            self.feature_store.commit_event(tx_dict, features)
            return ScoreResponse(**decision, is_duplicate=False)


state = ServiceState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.load_artifacts()
    yield


app = FastAPI(
    title="AI Fraud Detection Agent API",
    description="Многоуровневый ИИ-агент обнаружения мошеннических переводов (Правила + GBDT + Isolation Forest + SHAP).",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    with state._lock:
        fs = state.feature_store
        return {
            "status": "ok",
            "model_loaded": state.agent is not None,
            "model_backend": state.metadata.get("model_backend", "unknown"),
            "feature_store": {
                "initialized": fs is not None,
                "state_origin": fs.state_origin if fs else None,
                "as_of_timestamp": fs.as_of_timestamp if fs else None,
                "ingested_events": fs.ingested_events if fs else 0,
            },
            "thresholds": {
                "otp": state.agent.t_otp,
                "review": state.agent.t_review,
                "block": state.agent.t_block,
            } if state.agent else {},
        }


@app.post("/score", response_model=ScoreResponse)
def score_transaction(tx: TransactionRequest) -> ScoreResponse:
    return state.score(tx)


@app.get("/alerts")
def list_alerts(
    action: str | None = Query(default=None, description="Comma-separated actions, e.g. BLOCK,REVIEW,OTP"),
    only_unlabeled: bool = Query(default=False),
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[dict[str, Any]]:
    actions = [a.strip().upper() for a in action.split(",") if a.strip()] if action else None
    return state.storage.get_alerts(actions=actions, only_unlabeled=only_unlabeled, limit=limit)


@app.get("/alerts/{tx_id}")
def get_alert_detail(tx_id: str) -> dict[str, Any]:
    rec = state.storage.get_transaction(tx_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Transaction {tx_id} not found")
    return rec


@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(fb: FeedbackRequest) -> FeedbackResponse:
    labeled_at = state.storage.record_feedback(
        tx_id=fb.tx_id,
        is_fraud=fb.is_fraud,
        analyst_id=fb.analyst_id,
        notes=fb.notes,
    )
    if labeled_at is None:
        raise HTTPException(status_code=404, detail=f"Transaction {fb.tx_id} not found")
    return FeedbackResponse(
        status="recorded",
        tx_id=fb.tx_id,
        is_fraud=fb.is_fraud,
        labeled_at=labeled_at,
    )


@app.get("/metrics/summary")
def get_metrics_summary() -> dict[str, Any]:
    summary = state.storage.get_summary_stats()
    recent_df = state.storage.get_recent_features_df(limit=5000)
    feedback_q = compute_feedback_quality(recent_df)
    return {
        "operational": summary,
        "feedback_quality": feedback_q,
        "thresholds": state.metadata.get("optimized_thresholds", {}),
    }


@app.get("/monitoring/drift")
def get_drift_report() -> dict[str, Any]:
    recent_df = state.storage.get_recent_features_df(limit=5000)
    monitored_cols = state.metadata.get("monitored_columns", [])
    if recent_df.empty or len(recent_df) < 20 or state.reference_df.empty:
        return {
            "source": "train_vs_test_offline_baseline",
            "report": state.metadata.get("train_vs_test_drift", {}),
        }
    report = compute_drift_report(
        reference_df=state.reference_df,
        current_df=recent_df,
        columns=monitored_cols,
        warning_threshold=state.cfg["monitoring"]["psi_warning_threshold"],
        critical_threshold=state.cfg["monitoring"]["psi_critical_threshold"],
        bins=state.cfg["monitoring"]["psi_bins"],
    )
    return {"source": "live_stream_vs_training_reference", "report": report}


@app.get("/ablations")
def get_ablations() -> dict[str, Any]:
    return {
        "model_backend": state.metadata.get("model_backend", "unknown"),
        "dataset_summary": state.metadata.get("dataset_summary", {}),
        "optimized_thresholds": state.metadata.get("optimized_thresholds", {}),
        "ablations": state.metadata.get("ablations", []),
    }


@app.post("/retrain")
def trigger_retrain() -> dict[str, Any]:
    res = retrain_with_feedback(cfg=state.cfg, storage=state.storage)
    state.reload_model_artifacts()
    return res
