"""End-to-end training pipeline: simulation -> leak-free features -> time split -> models -> thresholds -> ablations."""
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
from src.data.simulator import load_or_generate_dataset
from src.evaluation.ablation import run_ablation_study
from src.evaluation.metrics import evaluate_predictions
from src.features.engineering import (
    build_features_chronological,
    build_initial_feature_store,
    sort_transactions_deterministically,
    time_based_split,
)
from src.features.online_store import (
    ALL_FEATURE_NAMES,
    FEATURE_CONTRACT,
    normalize_event_datetime,
)
from src.models.anomaly import FraudAnomalyDetector
from src.models.supervised import FraudGBDTModel
from src.monitoring.drift import compute_drift_report
from src.rules.engine import RuleEngine


def run_training_pipeline(
    cfg: dict[str, Any] | None = None,
    force_regenerate_data: bool = False,
) -> dict[str, Any]:
    """Execute full reproducible training, evaluation, ablation, and clean artifact export pipeline."""
    cfg = cfg or load_config()
    artifacts_dir = resolve_path(cfg["paths"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    processed_path = resolve_path(cfg["paths"]["processed_data"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load or simulate dataset
    raw_df = load_or_generate_dataset(cfg=cfg, force_regenerate=force_regenerate_data)
    raw_sorted = sort_transactions_deterministically(raw_df, deduplicate_retries=True)

    # 2. Strictly chronological leak-free offline feature engineering (never exported as live API store!)
    feat_df = build_features_chronological(
        raw_sorted,
        mule_in_degree_threshold=cfg["rules"]["mule_in_degree_threshold"],
        return_store=False,
    )
    feat_df.to_parquet(processed_path, index=False)

    # 3. Strict time-based split (Train / Val / Test)
    train_df, val_df, test_df = time_based_split(
        feat_df,
        train_frac=cfg["split"]["train_frac"],
        val_frac=cfg["split"]["val_frac"],
    )

    # Build optional point-in-time historical warm-up store ONLY from pre-test data (train + val)
    val_max_dt = normalize_event_datetime(val_df.iloc[-1].to_dict())
    test_min_dt = normalize_event_datetime(test_df.iloc[0].to_dict())
    pre_test_count = len(train_df) + len(val_df)
    pre_test_raw_df = raw_sorted.iloc[:pre_test_count].copy()

    scoring_cfg = cfg.get("scoring", {})
    pre_test_store = build_initial_feature_store(
        pre_test_raw_df,
        as_of_timestamp=val_max_dt,
        mule_in_degree_threshold=cfg["rules"]["mule_in_degree_threshold"],
        max_sender_out_of_order_sec=scoring_cfg.get("max_sender_out_of_order_seconds", 300.0),
        max_global_staleness_sec=scoring_cfg.get("max_global_staleness_seconds", 86400.0),
        state_origin="pre_test_warmup",
    )

    # Remove legacy full-dataset feature store artifact if present so API can never accidentally load it
    legacy_store_path = artifacts_dir / "online_feature_store.joblib"
    if legacy_store_path.exists():
        legacy_store_path.unlink()

    X_train = train_df[ALL_FEATURE_NAMES]
    y_train = train_df["isFraud"].values
    X_val = val_df[ALL_FEATURE_NAMES]
    y_val = val_df["isFraud"].values
    X_test = test_df[ALL_FEATURE_NAMES]
    y_test = test_df["isFraud"].values

    # 4. Train Supervised Main Model (LightGBM or HistGBDT fallback)
    gbdt_model = FraudGBDTModel(params=cfg["model"]["lgbm_params"])
    gbdt_model.fit(X_train, y_train)

    # 5. Train Anomaly Detector (Isolation Forest on legitimate training transactions)
    normal_mask = y_train == 0
    anomaly_detector = FraudAnomalyDetector(params=cfg["model"]["isolation_forest"])
    anomaly_detector.fit(X_train.loc[normal_mask])

    # 6. Rule Engine & Composite Decision Agent
    rule_engine = RuleEngine(rules_cfg=cfg["rules"])
    background_median = np.median(X_train.values, axis=0)
    explainer = FraudExplainer(
        ml_model=gbdt_model,
        feature_names=ALL_FEATURE_NAMES,
        background_median=background_median,
        llm_cfg=cfg.get("llm", {}),
    )
    agent = FraudDecisionAgent(
        rule_engine=rule_engine,
        ml_model=gbdt_model,
        anomaly_detector=anomaly_detector,
        explainer=explainer,
        weights=cfg["agent"]["weights"],
        thresholds=cfg["agent"]["thresholds"],
        feature_names=ALL_FEATURE_NAMES,
    )

    # 7. Optimize decision thresholds on validation set by expected business cost
    val_ml = gbdt_model.predict_score(X_val)
    val_rule = np.array(rule_engine.score_batch(val_df), dtype=float)
    val_anom = anomaly_detector.predict_anomaly_score(X_val)
    val_risk = np.asarray(agent.combine_scores(val_ml, val_rule, val_anom), dtype=float)

    best_thresholds, val_cost_metrics = optimize_thresholds_by_cost(
        y_true=y_val,
        scores=val_risk,
        amounts=val_df["amount"].values,
        costs_cfg=cfg["agent"]["costs"],
    )
    agent.t_otp = best_thresholds["otp"]
    agent.t_review = best_thresholds["review"]
    agent.t_block = best_thresholds["block"]

    # 8. Run Ablation Study & baseline evaluation on held-out future Test Set
    ablations = run_ablation_study(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        cfg=cfg,
        full_gbdt=gbdt_model,
        anomaly_detector=anomaly_detector,
        rule_engine=rule_engine,
        decision_agent=agent,
    )

    # 9. Compute Train-vs-Test Drift Report & Baseline Test Metrics
    test_ml = gbdt_model.predict_score(X_test)
    test_rule = np.array(rule_engine.score_batch(test_df), dtype=float)
    test_anom = anomaly_detector.predict_anomaly_score(X_test)
    test_risk = np.asarray(agent.combine_scores(test_ml, test_rule, test_anom), dtype=float)

    baseline_test_metrics = evaluate_predictions(
        y_true=y_test,
        y_score=test_risk,
        amounts=test_df["amount"].values,
        thresholds=best_thresholds,
        costs_cfg=cfg["agent"]["costs"],
    )

    ref_monitor_df = train_df[ALL_FEATURE_NAMES].copy()
    ref_monitor_df["ml_score"] = gbdt_model.predict_score(X_train)
    ref_monitor_df["risk_score"] = np.asarray(
        agent.combine_scores(
            ref_monitor_df["ml_score"].values,
            np.array(rule_engine.score_batch(train_df), dtype=float),
            anomaly_detector.predict_anomaly_score(X_train),
        ),
        dtype=float,
    )

    cur_monitor_df = test_df[ALL_FEATURE_NAMES].copy()
    cur_monitor_df["ml_score"] = test_ml
    cur_monitor_df["risk_score"] = test_risk

    monitored_cols = [
        "amount",
        "balance_drain_ratio",
        "orig_tx_count_1h",
        "orig_tx_count_24h",
        "amount_zscore",
        "is_new_recipient",
        "is_new_device",
        "dest_in_degree",
        "device_shared_accounts",
        "channel_API",
        "ml_score",
        "risk_score",
    ]
    drift_report = compute_drift_report(
        reference_df=ref_monitor_df,
        current_df=cur_monitor_df,
        columns=monitored_cols,
        warning_threshold=cfg["monitoring"]["psi_warning_threshold"],
        critical_threshold=cfg["monitoring"]["psi_critical_threshold"],
        bins=cfg["monitoring"]["psi_bins"],
    )

    # 10. Save production model artifacts, pre-test warm-up snapshot, and holdout manifest
    trained_at = datetime.now(timezone.utc).isoformat()
    joblib.dump(gbdt_model, artifacts_dir / "gbdt_model.joblib")
    joblib.dump(anomaly_detector, artifacts_dir / "anomaly_detector.joblib")
    joblib.dump(background_median, artifacts_dir / "background_median.joblib")
    joblib.dump(pre_test_store, artifacts_dir / "initial_feature_store_pre_test.joblib")
    ref_monitor_df[monitored_cols].sample(min(5000, len(ref_monitor_df)), random_state=42).to_parquet(
        artifacts_dir / "reference_distribution.parquet", index=False
    )

    train_max_dt = normalize_event_datetime(train_df.iloc[-1].to_dict())
    val_min_dt = normalize_event_datetime(val_df.iloc[0].to_dict())
    test_max_dt = normalize_event_datetime(test_df.iloc[-1].to_dict())

    holdout_manifest = {
        "created_at": trained_at,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_max_timestamp": train_max_dt.isoformat(),
        "val_min_timestamp": val_min_dt.isoformat(),
        "val_max_timestamp": val_max_dt.isoformat(),
        "test_min_timestamp": test_min_dt.isoformat(),
        "test_max_timestamp": test_max_dt.isoformat(),
        "val_tx_ids": val_df["tx_id"].astype(str).tolist(),
        "test_tx_ids": test_df["tx_id"].astype(str).tolist(),
    }
    with open(artifacts_dir / "holdout_manifest.json", "w", encoding="utf-8") as f:
        json.dump(holdout_manifest, f, indent=2, ensure_ascii=False)

    metadata = {
        "trained_at": trained_at,
        "model_backend": gbdt_model.backend_name,
        "feature_names": ALL_FEATURE_NAMES,
        "feature_contract": FEATURE_CONTRACT,
        "monitored_columns": monitored_cols,
        "initial_state_snapshot": {
            "artifact": "initial_feature_store_pre_test.joblib",
            "state_origin": pre_test_store.state_origin,
            "as_of_timestamp": pre_test_store.as_of_timestamp,
            "ingested_events": pre_test_store.ingested_events,
            "contains_test_events": False,
        },
        "split_boundaries": {
            "train_max_timestamp": train_max_dt.isoformat(),
            "val_min_timestamp": val_min_dt.isoformat(),
            "val_max_timestamp": val_max_dt.isoformat(),
            "test_min_timestamp": test_min_dt.isoformat(),
            "test_max_timestamp": test_max_dt.isoformat(),
        },
        "dataset_summary": {
            "total_rows": int(len(feat_df)),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_fraud_rate": round(float(y_train.mean()), 5),
            "val_fraud_rate": round(float(y_val.mean()), 5),
            "test_fraud_rate": round(float(y_test.mean()), 5),
            "scenario_counts": raw_sorted["fraud_scenario"].value_counts().to_dict(),
        },
        "optimized_thresholds": best_thresholds,
        "validation_cost_metrics": val_cost_metrics,
        "baseline_test_metrics": baseline_test_metrics,
        "ablations": ablations,
        "train_vs_test_drift": drift_report,
    }
    with open(artifacts_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata


if __name__ == "__main__":
    meta = run_training_pipeline()
    print(f"Training complete ({meta['model_backend']}). Optimized thresholds: {meta['optimized_thresholds']}")
