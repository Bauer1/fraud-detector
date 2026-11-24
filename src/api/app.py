"""FastAPI Scoring & Operations Service for the AI Fraud Detection Agent."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
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
from src.features.online_store import ALL_FEATURE_NAMES, OnlineFeatureStore
from src.monitoring.drift import compute_drift_report, compute_feedback_quality
from src.rules.engine import RuleEngine
from src.training.pipeline import run_training_pipeline
from src.training.retrain import retrain_with_feedback


class ServiceState:
    def __init__(self) -> None:
        self.cfg: dict[str, Any] = load_config()
        self.storage: StorageManager = StorageManager(self.cfg["paths"]["database_url"])
        self.feature_store: OnlineFeatureStore | None = None
        self.agent: FraudDecisionAgent | None = None
        self.metadata: dict[str, Any] = {}
        self.reference_df: pd.DataFrame = pd.DataFrame()

    def load_artifacts(self) -> None:
        artifacts_dir = resolve_path(self.cfg["paths"]["artifacts_dir"])
        gbdt_path = artifacts_dir / "gbdt_model.joblib"
        if not gbdt_path.exists():
            # Automatically bootstrap artifacts on first startup if missing
            run_training_pipeline(cfg=self.cfg)

        gbdt_model = joblib.load(artifacts_dir / "gbdt_model.joblib")
        anomaly_detector = joblib.load(artifacts_dir / "anomaly_detector.joblib")
        background_median = joblib.load(artifacts_dir / "background_median.joblib")
        self.feature_store = joblib.load(artifacts_dir / "online_feature_store.joblib")

        meta_path = artifacts_dir / "metadata.json"
        self.metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

        ref_path = artifacts_dir / "reference_distribution.parquet"
        self.reference_df = pd.read_parquet(ref_path) if ref_path.exists() else pd.DataFrame()

        thresholds = self.metadata.get("optimized_thresholds", self.cfg["agent"]["thresholds"])
        rule_engine = RuleEngine(rules_cfg=self.cfg["rules"])
        explainer = FraudExplainer(
            ml_model=gbdt_model,
            feature_names=ALL_FEATURE_NAMES,
            background_median=background_median,
            llm_cfg=self.cfg.get("llm", {}),
        )
        self.agent = FraudDecisionAgent(
            rule_engine=rule_engine,
            ml_model=gbdt_model,
            anomaly_detector=anomaly_detector,
            explainer=explainer,
            weights=self.cfg["agent"]["weights"],
            thresholds=thresholds,
            feature_names=ALL_FEATURE_NAMES,
        )


state = ServiceState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.load_artifacts()
    yield


app = FastAPI(
    title="AI Fraud Detection Agent API",
    description="Многоуровневый ИИ-агент обнаружения мошеннических переводов (Правила + GBDT + Isolation Forest + SHAP).",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": state.agent is not None,
        "model_backend": state.metadata.get("model_backend", "unknown"),
        "thresholds": {
            "otp": state.agent.t_otp,
            "review": state.agent.t_review,
            "block": state.agent.t_block,
        } if state.agent else {},
    }


@app.post("/score", response_model=ScoreResponse)
def score_transaction(tx: TransactionRequest) -> ScoreResponse:
    if state.agent is None or state.feature_store is None:
        state.load_artifacts()

    tx_dict = tx.model_dump()
    tx_dict["type"] = tx.type.value
    tx_dict["channel"] = tx.channel.value

    features = state.feature_store.compute_features(tx_dict, update_state=True)
    decision = state.agent.evaluate_transaction(tx_dict, features, explain=True)
    state.storage.save_scored_transaction(tx_dict, decision, features)
    return ScoreResponse(**decision)


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
    result = retrain_with_feedback(cfg=state.cfg, storage=state.storage)
    state.load_artifacts()
    return result
