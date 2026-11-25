"""End-to-end offline training, calibration, capacity-aware threshold optimization, verification, and registry registration."""
from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.agent.decision_agent import FraudDecisionAgent
from src.agent.explainer import FraudExplainer
from src.agent.thresholds import estimate_span_days, optimize_thresholds_by_cost
from src.config import load_config, resolve_path
from src.data.paysim_loader import (
    load_paysim_csv_dataset,
    profile_enriched_simulation,
    profile_paysim_csv,
)
from src.data.simulator import load_or_generate_dataset
from src.evaluation.ablation import run_ablation_study
from src.evaluation.metrics import (
    compute_fp_fn_error_analysis,
    compute_segmented_metrics,
    compute_temporal_slice_stability,
    evaluate_predictions,
    measure_latency_percentiles,
)
from src.features.engineering import (
    build_features_chronological,
    build_initial_feature_store,
    sort_transactions_deterministically,
    time_based_split,
)
from src.features.online_store import (
    ALL_FEATURE_NAMES,
    FEATURE_AVAILABILITY_REGISTRY,
    FEATURE_CONTRACT,
    SCHEMA_VERSION,
    normalize_event_datetime,
    resolve_model_feature_names,
)
from src.models.anomaly import FraudAnomalyDetector
from src.models.registry import ModelRegistry
from src.models.supervised import FraudGBDTModel
from src.monitoring.drift import compute_drift_report
from src.rules.engine import RuleEngine


def run_training_pipeline(
    cfg: dict[str, Any] | None = None,
    *,
    dataset_source: str | None = None,
    force_regenerate_data: bool = False,
    promote: bool | None = None,
    approved_by: str = "offline_pipeline_bootstrap",
) -> dict[str, Any]:
    """Execute reproducible training, temporal evaluation, ablation, and versioned registry registration.

    Supports both strictly separated data sources:
    - `enriched_simulation` ("обогащённая симуляция проекта")
    - `paysim_csv` ("исходный PaySim CSV")
    """
    cfg = cfg or load_config()
    ds_source = dataset_source or cfg.get("datasets", {}).get("active_training_source", "enriched_simulation")
    if ds_source not in ("enriched_simulation", "paysim_csv"):
        raise ValueError(f"Unsupported dataset_source='{ds_source}'. Expected 'enriched_simulation' or 'paysim_csv'.")

    reports_dir = resolve_path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = resolve_path(cfg["paths"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Profile data source & build Data Manifest before training
    load_audit: dict[str, Any] = {}
    if ds_source == "paysim_csv":
        ps_cfg = cfg.get("datasets", {}).get("paysim_csv", {})
        data_manifest = profile_paysim_csv(force_recompute=force_regenerate_data)
        raw_df, load_audit = load_paysim_csv_dataset(
            sampling_mode=ps_cfg.get("sampling_mode", "chronological_stride"),
            max_rows=int(ps_cfg.get("max_rows", 120_000)),
        )
        processed_path = resolve_path(cfg["paths"].get("paysim_processed_data", "data/processed/paysim_features.parquet"))
    else:
        raw_df = load_or_generate_dataset(cfg=cfg, force_regenerate=force_regenerate_data)
        data_manifest = profile_enriched_simulation(raw_df)
        load_audit = {
            "dataset_id": "enriched_simulation",
            "sampling_mode": "full",
            "loaded_rows": int(len(raw_df)),
        }
        processed_path = resolve_path(cfg["paths"]["processed_data"])

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    raw_sorted = sort_transactions_deterministically(raw_df, deduplicate_retries=True)

    # 2. Strictly chronological point-in-time feature engineering + observability capture
    feat_df, batch_store_obs = build_features_chronological(
        raw_sorted,
        mule_in_degree_threshold=cfg["rules"]["mule_in_degree_threshold"],
        return_store=True,
    )
    store_observability = batch_store_obs.get_observability_stats()
    del batch_store_obs
    feat_df.to_parquet(processed_path, index=False)

    # Resolve strictly allowed pre-decision model features for this data source
    fc_cfg = cfg.get("feature_contract", {})
    model_feature_names = resolve_model_feature_names(ds_source, fc_cfg)

    # 3. Strict time-based split (Train / Val / Test)
    train_df, val_df, test_df = time_based_split(
        feat_df,
        train_frac=cfg["split"]["train_frac"],
        val_frac=cfg["split"]["val_frac"],
    )

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

    legacy_store_path = artifacts_dir / "online_feature_store.joblib"
    if legacy_store_path.exists():
        legacy_store_path.unlink()

    X_train = train_df[model_feature_names]
    y_train = train_df["isFraud"].values
    X_val = val_df[model_feature_names]
    y_val = val_df["isFraud"].values
    X_test = test_df[model_feature_names]
    y_test = test_df["isFraud"].values

    # 4. Train Supervised Main Model (GBDT) + Chronological Validation Calibration
    gbdt_model = FraudGBDTModel(params=cfg["model"]["lgbm_params"])
    gbdt_model.fit(X_train, y_train)

    cal_method = cfg["model"].get("calibration_method", "sigmoid")
    half_val = len(val_df) // 2
    if half_val >= 50 and int(y_val[:half_val].sum()) >= 8 and int(y_val[half_val:].sum()) >= 8:
        cal_diag = gbdt_model.calibrate(X_val.iloc[:half_val], y_val[:half_val], method=cal_method)
        thresh_val_df = val_df.iloc[half_val:].copy()
    else:
        cal_diag = gbdt_model.calibrate(X_val, y_val, method=cal_method)
        thresh_val_df = val_df

    # 5. Train Auxiliary Anomaly Detector (Isolation Forest on legitimate training transactions)
    normal_mask = y_train == 0
    anomaly_detector = FraudAnomalyDetector(params=cfg["model"]["isolation_forest"])
    anomaly_detector.fit(X_train.loc[normal_mask])

    # 6. Prepare Registry Version ID & Decision Service
    registry = ModelRegistry()
    version_id = registry.generate_version_id(dataset_source=ds_source, prefix="model")

    rule_engine = RuleEngine(rules_cfg=cfg["rules"])
    background_median = np.median(X_train.values, axis=0)
    explainer = FraudExplainer(
        ml_model=gbdt_model,
        feature_names=model_feature_names,
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
        feature_names=model_feature_names,
        model_version=version_id,
        feature_schema_version=SCHEMA_VERSION,
        anomaly_max_standalone_score=float(cfg["agent"].get("anomaly_max_standalone_score", 0.14)),
        fallback_policy=scoring_cfg.get("fallback_policy", "rules_only"),
    )

    # 7. Optimize decision thresholds on validation set under cost matrix AND analyst REVIEW capacity limits
    tv_ml = gbdt_model.predict_score(thresh_val_df[model_feature_names])
    tv_rule = np.array(rule_engine.score_batch(thresh_val_df), dtype=float)
    tv_anom = anomaly_detector.predict_anomaly_score(thresh_val_df[model_feature_names])
    tv_risk = np.asarray(agent.combine_scores(tv_ml, tv_rule, tv_anom), dtype=float)
    val_span_days = estimate_span_days(
        thresh_val_df["timestamp"]
        if "timestamp" in thresh_val_df.columns and thresh_val_df["timestamp"].notna().any()
        else thresh_val_df["step"]
    )

    best_thresholds, val_cost_metrics = optimize_thresholds_by_cost(
        y_true=thresh_val_df["isFraud"].values,
        scores=tv_risk,
        amounts=thresh_val_df["amount"].values,
        costs_cfg=cfg["agent"]["costs"],
        capacity_cfg=cfg["agent"].get("capacity", {}),
        span_days=val_span_days,
    )
    agent.t_otp = best_thresholds["otp"]
    agent.t_review = best_thresholds["review"]
    agent.t_block = best_thresholds["block"]

    # 8. Single-shot Evaluation, Ablation, Segments, Temporal Stability, FP/FN Analysis & Latency on Test Holdout
    ablations = run_ablation_study(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        cfg=cfg,
        full_gbdt=gbdt_model,
        anomaly_detector=anomaly_detector,
        rule_engine=rule_engine,
        decision_agent=agent,
        dataset_source=ds_source,
    )

    test_ml = gbdt_model.predict_score(X_test)
    test_rule = np.array(rule_engine.score_batch(test_df), dtype=float)
    test_anom = anomaly_detector.predict_anomaly_score(X_test)
    test_risk = np.asarray(agent.combine_scores(test_ml, test_rule, test_anom), dtype=float)
    test_span_days = estimate_span_days(
        test_df["timestamp"]
        if "timestamp" in test_df.columns and test_df["timestamp"].notna().any()
        else test_df["step"]
    )

    baseline_test_metrics = evaluate_predictions(
        y_true=y_test,
        y_score=test_risk,
        amounts=test_df["amount"].values,
        thresholds=best_thresholds,
        costs_cfg=cfg["agent"]["costs"],
        capacity_cfg=cfg["agent"].get("capacity", {}),
        span_days=test_span_days,
    )

    temporal_stability = compute_temporal_slice_stability(
        df=test_df,
        y_score=test_risk,
        thresholds=best_thresholds,
        costs_cfg=cfg["agent"]["costs"],
        capacity_cfg=cfg["agent"].get("capacity", {}),
        n_slices=4,
        n_bootstrap=100,
        random_state=cfg["project"]["random_seed"],
    )
    segmented_metrics = compute_segmented_metrics(test_df, test_risk, best_thresholds)
    error_analysis = compute_fp_fn_error_analysis(test_df, test_risk, best_thresholds)
    latency_benchmark = measure_latency_percentiles(agent, test_df, ALL_FEATURE_NAMES, max_samples=150)

    # 9. Compute Train-vs-Test Drift Report on active model features + scores
    ref_monitor_df = train_df[model_feature_names].copy()
    ref_monitor_df["ml_score"] = gbdt_model.predict_score(X_train)
    ref_monitor_df["risk_score"] = np.asarray(
        agent.combine_scores(
            ref_monitor_df["ml_score"].values,
            np.array(rule_engine.score_batch(train_df), dtype=float),
            anomaly_detector.predict_anomaly_score(X_train),
        ),
        dtype=float,
    )

    cur_monitor_df = test_df[model_feature_names].copy()
    cur_monitor_df["ml_score"] = test_ml
    cur_monitor_df["risk_score"] = test_risk

    candidate_monitored = [
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
    ]
    monitored_cols = [c for c in candidate_monitored if c in model_feature_names] + ["ml_score", "risk_score"]

    drift_report = compute_drift_report(
        reference_df=ref_monitor_df,
        current_df=cur_monitor_df,
        columns=monitored_cols,
        warning_threshold=cfg["monitoring"]["psi_warning_threshold"],
        critical_threshold=cfg["monitoring"]["psi_critical_threshold"],
        bins=cfg["monitoring"]["psi_bins"],
    )

    trained_at = datetime.now(timezone.utc).isoformat()
    train_max_dt = normalize_event_datetime(train_df.iloc[-1].to_dict())
    val_min_dt = normalize_event_datetime(val_df.iloc[0].to_dict())
    test_max_dt = normalize_event_datetime(test_df.iloc[-1].to_dict())

    holdout_manifest = {
        "created_at": trained_at,
        "dataset_source": ds_source,
        "model_version": version_id,
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

    excluded_features = [f for f in ALL_FEATURE_NAMES if f not in model_feature_names]
    metadata: dict[str, Any] = {
        "model_version": version_id,
        "trained_at": trained_at,
        "dataset_source": ds_source,
        "dataset_title_ru": data_manifest.get("dataset_title_ru", ds_source),
        "pilot_proposal_notice": cfg["project"].get("pilot_proposal_notice", "PROPOSED_PILOT_DEFAULTS"),
        "data_manifest_ref": {
            "dataset_id": data_manifest.get("dataset_id"),
            "file_path": data_manifest.get("file_path"),
            "file_sha256": data_manifest.get("file_sha256"),
            "source_total_rows": data_manifest.get("row_count"),
            "load_audit": load_audit,
        },
        "model_backend": gbdt_model.backend_name,
        "feature_schema_version": SCHEMA_VERSION,
        "feature_names": model_feature_names,
        "excluded_features": excluded_features,
        "feature_availability_map": {f: FEATURE_AVAILABILITY_REGISTRY[f] for f in model_feature_names},
        "feature_contract": FEATURE_CONTRACT,
        "feature_store_observability": store_observability,
        "calibration_diagnostics": cal_diag,
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
            "dataset_source": ds_source,
            "total_rows": int(len(feat_df)),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_fraud_rate": round(float(y_train.mean()), 6),
            "val_fraud_rate": round(float(y_val.mean()), 6),
            "test_fraud_rate": round(float(y_test.mean()), 6),
            "scenario_counts": (
                raw_sorted["fraud_scenario"].value_counts().to_dict()
                if "fraud_scenario" in raw_sorted.columns
                else {}
            ),
        },
        "optimized_thresholds": best_thresholds,
        "validation_cost_metrics": val_cost_metrics,
        "baseline_test_metrics": baseline_test_metrics,
        "temporal_slice_stability": temporal_stability,
        "segmented_test_metrics": segmented_metrics,
        "error_analysis": error_analysis,
        "latency_benchmark_ms": latency_benchmark,
        "ablations": ablations,
        "train_vs_test_drift": drift_report,
    }

    # 10. Stage artifacts in a temporary directory and register immutable candidate in ModelRegistry
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)
        joblib.dump(gbdt_model, tmp_dir / "gbdt_model.joblib")
        joblib.dump(anomaly_detector, tmp_dir / "anomaly_detector.joblib")
        joblib.dump(background_median, tmp_dir / "background_median.joblib")
        joblib.dump(pre_test_store, tmp_dir / "initial_feature_store_pre_test.joblib")
        ref_monitor_df[monitored_cols].sample(min(5000, len(ref_monitor_df)), random_state=42).to_parquet(
            tmp_dir / "reference_distribution.parquet", index=False
        )
        (tmp_dir / "holdout_manifest.json").write_text(
            json.dumps(holdout_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        reg_entry = registry.register_candidate(
            version_id=version_id,
            source_files={
                "gbdt_model.joblib": tmp_dir / "gbdt_model.joblib",
                "anomaly_detector.joblib": tmp_dir / "anomaly_detector.joblib",
                "background_median.joblib": tmp_dir / "background_median.joblib",
                "initial_feature_store_pre_test.joblib": tmp_dir / "initial_feature_store_pre_test.joblib",
                "reference_distribution.parquet": tmp_dir / "reference_distribution.parquet",
                "holdout_manifest.json": tmp_dir / "holdout_manifest.json",
            },
            metadata=metadata,
            config_snapshot=cfg,
            min_pr_auc_ratio=float(cfg.get("governance", {}).get("min_pr_auc_guardrail_ratio", 0.95)),
        )

    # Determine whether to promote:
    # - Initial bootstrap when no active model exists yet for the configured active source, OR explicit `promote=True`.
    idx_state = registry.list_registry()
    configured_active_ds = cfg.get("datasets", {}).get("active_training_source", "enriched_simulation")
    no_global_active = idx_state.get("active_version") is None
    no_ds_active = ds_source not in idx_state.get("active_by_dataset", {})

    if promote is True or (promote is None and (no_global_active or no_ds_active)):
        sync_global = (ds_source == configured_active_ds) or no_global_active
        registry.promote_version(
            version_id,
            approved_by=approved_by,
            reason=f"Initial/explicit offline release for {ds_source}",
            override_guardrail=False,
            sync_global_active=sync_global,
        )

    final_meta = registry.get_version_metadata(version_id) or metadata
    benchmark_out = reports_dir / f"benchmark_{ds_source}.json"
    benchmark_out.write_text(json.dumps(final_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return final_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and verify anti-fraud models on separated data sources")
    parser.add_argument(
        "--dataset",
        choices=["enriched_simulation", "paysim_csv", "both"],
        default="enriched_simulation",
        help="Target dataset source ('both' runs separate pipelines for PaySim CSV and Enriched Simulation)",
    )
    parser.add_argument("--force-regenerate", action="store_true", help="Regenerate synthetic dataset / re-profile CSV")
    parser.add_argument("--promote", action="store_true", help="Approve and promote trained version after guardrails")
    args = parser.parse_args()

    sources = ["paysim_csv", "enriched_simulation"] if args.dataset == "both" else [args.dataset]
    cfg = load_config()
    active_ds = cfg.get("datasets", {}).get("active_training_source", "enriched_simulation")

    for src in sources:
        should_promote = True if (args.promote and src == active_ds) else (False if not args.promote else None)
        # On initial bootstrap (`active_version is None`), allow automatic bootstrap promotion
        reg = ModelRegistry().list_registry()
        if src not in reg.get("active_by_dataset", {}):
            should_promote = None
        meta = run_training_pipeline(
            cfg=cfg,
            dataset_source=src,
            force_regenerate_data=args.force_regenerate,
            promote=should_promote,
        )
        test_m = meta["baseline_test_metrics"]
        print(
            f"[{src}] version={meta['model_version']} | backend={meta['model_backend']} | "
            f"features={len(meta['feature_names'])} | PR-AUC={test_m['pr_auc']:.4f} | "
            f"thresholds={meta['optimized_thresholds']} | daily_reviews={test_m['daily_review_count']}"
        )


if __name__ == "__main__":
    main()
