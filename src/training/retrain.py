"""Feedback-loop retraining pipeline: incorporates analyst-labeled cases into model updates."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.agent.decision_agent import FraudDecisionAgent
from src.agent.explainer import FraudExplainer
from src.agent.thresholds import optimize_thresholds_by_cost
from src.config import load_config, resolve_path
from src.data.storage import StorageManager
from src.evaluation.metrics import evaluate_predictions
from src.features.engineering import time_based_split
from src.features.online_store import ALL_FEATURE_NAMES
from src.models.anomaly import FraudAnomalyDetector
from src.models.supervised import FraudGBDTModel
from src.rules.engine import RuleEngine


def retrain_with_feedback(
    cfg: dict[str, Any] | None = None,
    storage: StorageManager | None = None,
    feedback_weight: int = 3,
) -> dict[str, Any]:
    """Retrain GBDT + Isolation Forest incorporating analyst feedback from operational DB.

    Feedback cases are upweighted (default 3x) so newly confirmed fraud patterns (concept drift)
    are learned rapidly without losing historical recall.
    """
    cfg = cfg or load_config()
    storage = storage or StorageManager(cfg["paths"]["database_url"])
    artifacts_dir = resolve_path(cfg["paths"]["artifacts_dir"])
    processed_path = resolve_path(cfg["paths"]["processed_data"])

    if not processed_path.exists():
        raise FileNotFoundError("Processed features parquet not found. Run initial training first.")

    feat_df = pd.read_parquet(processed_path)
    train_df, val_df, test_df = time_based_split(
        feat_df,
        train_frac=cfg["split"]["train_frac"],
        val_frac=cfg["split"]["val_frac"],
    )

    feedback_df = storage.get_labeled_feedback_df()
    n_feedback = len(feedback_df)

    # Combine historical training data with upweighted analyst feedback
    if n_feedback > 0:
        for col in ALL_FEATURE_NAMES:
            if col not in feedback_df.columns:
                feedback_df[col] = 0.0
        fb_clean = feedback_df[ALL_FEATURE_NAMES + ["isFraud", "amount"]].copy()
        fb_repeated = pd.concat([fb_clean] * max(1, feedback_weight), ignore_index=True)
        X_train = pd.concat([train_df[ALL_FEATURE_NAMES], fb_repeated[ALL_FEATURE_NAMES]], ignore_index=True)
        y_train = np.concatenate([train_df["isFraud"].values, fb_repeated["isFraud"].values])
    else:
        X_train = train_df[ALL_FEATURE_NAMES]
        y_train = train_df["isFraud"].values

    # Retrain models
    new_gbdt = FraudGBDTModel(params=cfg["model"]["lgbm_params"])
    new_gbdt.fit(X_train, y_train)

    new_anom = FraudAnomalyDetector(params=cfg["model"]["isolation_forest"])
    new_anom.fit(X_train.loc[y_train == 0])

    rule_engine = RuleEngine(rules_cfg=cfg["rules"])
    bg_median = np.median(X_train.values, axis=0)
    explainer = FraudExplainer(new_gbdt, ALL_FEATURE_NAMES, bg_median, cfg.get("llm", {}))
    agent = FraudDecisionAgent(
        rule_engine=rule_engine,
        ml_model=new_gbdt,
        anomaly_detector=new_anom,
        explainer=explainer,
        weights=cfg["agent"]["weights"],
        thresholds=cfg["agent"]["thresholds"],
        feature_names=ALL_FEATURE_NAMES,
    )

    # Re-optimize thresholds on validation set
    val_ml = new_gbdt.predict_score(val_df[ALL_FEATURE_NAMES])
    val_rule = np.array(rule_engine.score_batch(val_df), dtype=float)
    val_anom = new_anom.predict_anomaly_score(val_df[ALL_FEATURE_NAMES])
    val_risk = np.asarray(agent.combine_scores(val_ml, val_rule, val_anom), dtype=float)

    best_thresholds, _ = optimize_thresholds_by_cost(
        y_true=val_df["isFraud"].values,
        scores=val_risk,
        amounts=val_df["amount"].values,
        costs_cfg=cfg["agent"]["costs"],
    )

    # Evaluate on held-out test set
    test_ml = new_gbdt.predict_score(test_df[ALL_FEATURE_NAMES])
    test_rule = np.array(rule_engine.score_batch(test_df), dtype=float)
    test_anom = new_anom.predict_anomaly_score(test_df[ALL_FEATURE_NAMES])
    test_risk = np.asarray(agent.combine_scores(test_ml, test_rule, test_anom), dtype=float)

    test_metrics = evaluate_predictions(
        y_true=test_df["isFraud"].values,
        y_score=test_risk,
        amounts=test_df["amount"].values,
        thresholds=best_thresholds,
        costs_cfg=cfg["agent"]["costs"],
    )

    # Persist updated artifacts
    joblib.dump(new_gbdt, artifacts_dir / "gbdt_model.joblib")
    joblib.dump(new_anom, artifacts_dir / "anomaly_detector.joblib")
    joblib.dump(bg_median, artifacts_dir / "background_median.joblib")

    meta_path = artifacts_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta["last_retrained_at"] = datetime.now(timezone.utc).isoformat()
    meta["feedback_samples_used"] = int(n_feedback)
    meta["optimized_thresholds"] = best_thresholds
    meta["post_retrain_test_metrics"] = test_metrics
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "status": "retrained",
        "feedback_samples_used": int(n_feedback),
        "optimized_thresholds": best_thresholds,
        "test_metrics": test_metrics,
        "retrained_at": meta["last_retrained_at"],
    }
