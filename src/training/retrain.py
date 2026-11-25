"""Feedback-loop candidate training pipeline with holdout protection, label maturity, and release governance.

Per Technical Specification Sections 1, 3.7, 4, and 8:
- Retraining builds and evaluates an immutable CANDIDATE version in `ModelRegistry` (`models/registry/<version>/`).
- Previous model artifacts in `models/artifacts/` and prior registry versions are NEVER overwritten.
- `/retrain` does NOT automatically promote/publish the candidate (`governance.auto_promote_on_retrain = false`).
- `unknown` and immature unverified cases are never treated as negative (`not_fraud`) examples.
"""
from __future__ import annotations

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
from src.data.storage import StorageManager
from src.evaluation.metrics import evaluate_predictions
from src.features.engineering import time_based_split
from src.features.online_store import (
    SCHEMA_VERSION,
    normalize_event_datetime,
    resolve_model_feature_names,
)
from src.models.anomaly import FraudAnomalyDetector
from src.models.registry import ModelRegistry
from src.models.supervised import FraudGBDTModel
from src.rules.engine import RuleEngine


def _load_holdout_protection_sets(
    artifacts_dir: Path,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[set[str], set[str], datetime, datetime, datetime, datetime]:
    """Load locked val/test IDs and temporal boundaries from manifest + current split."""
    val_ids = set(val_df["tx_id"].astype(str).tolist())
    test_ids = set(test_df["tx_id"].astype(str).tolist())

    val_min_dt = normalize_event_datetime(val_df.iloc[0].to_dict())
    val_max_dt = normalize_event_datetime(val_df.iloc[-1].to_dict())
    test_min_dt = normalize_event_datetime(test_df.iloc[0].to_dict())
    test_max_dt = normalize_event_datetime(test_df.iloc[-1].to_dict())

    manifest_path = artifacts_dir / "holdout_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        val_ids.update(str(x) for x in manifest.get("val_tx_ids", []))
        test_ids.update(str(x) for x in manifest.get("test_tx_ids", []))
        if manifest.get("val_min_timestamp"):
            val_min_dt = min(val_min_dt, normalize_event_datetime({"timestamp": manifest["val_min_timestamp"]}))
        if manifest.get("test_max_timestamp"):
            test_max_dt = max(test_max_dt, normalize_event_datetime({"timestamp": manifest["test_max_timestamp"]}))

    return val_ids, test_ids, val_min_dt, val_max_dt, test_min_dt, test_max_dt


def retrain_with_feedback(
    cfg: dict[str, Any] | None = None,
    storage: StorageManager | None = None,
    feedback_weight: int = 3,
    allow_holdout_feedback: bool | None = None,
    *,
    promote: bool = False,
    approved_by: str | None = None,
    registry: ModelRegistry | None = None,
) -> dict[str, Any]:
    """Train a candidate GBDT + Isolation Forest incorporating definitive analyst feedback without auto-publishing."""
    cfg = cfg or load_config()
    owns_storage = storage is None
    storage = storage or StorageManager(cfg["paths"]["database_url"])
    try:
        artifacts_dir = resolve_path(cfg["paths"]["artifacts_dir"])
        ds_source = cfg.get("datasets", {}).get("active_training_source", "enriched_simulation")
        if ds_source == "paysim_csv":
            processed_path = resolve_path(cfg["paths"].get("paysim_processed_data", "data/processed/paysim_features.parquet"))
        else:
            processed_path = resolve_path(cfg["paths"]["processed_data"])

        if not processed_path.exists():
            raise FileNotFoundError(f"Processed features parquet not found at {processed_path}. Run initial training first.")

        if allow_holdout_feedback is None:
            exclude_cfg = bool(cfg.get("retraining", {}).get("exclude_holdout_feedback", True))
            allow_holdout_feedback = not exclude_cfg

        model_feature_names = resolve_model_feature_names(ds_source, cfg.get("feature_contract", {}))

        feat_df = pd.read_parquet(processed_path)
        train_df, val_df, test_df = time_based_split(
            feat_df,
            train_frac=cfg["split"]["train_frac"],
            val_frac=cfg["split"]["val_frac"],
        )

        val_ids, test_ids, val_min_dt, val_max_dt, test_min_dt, test_max_dt = _load_holdout_protection_sets(
            artifacts_dir=artifacts_dir,
            val_df=val_df,
            test_df=test_df,
        )
        protected_ids = val_ids | test_ids

        # Fetch only definitive (`fraud` / `not_fraud`) feedback; `unknown` cases are strictly excluded!
        feedback_df = storage.get_labeled_feedback_df()
        n_unknown_excluded = storage.count_unknown_feedback() if hasattr(storage, "count_unknown_feedback") else 0
        n_feedback_total = int(len(feedback_df))
        n_excluded_holdout = 0
        max_used_feedback_dt: datetime | None = None
        used_feedback_ids: set[str] = set()

        eligible_rows: list[dict[str, Any]] = []
        if n_feedback_total > 0:
            for rec in feedback_df.to_dict(orient="records"):
                tx_id = str(rec.get("tx_id", ""))
                rec_dt = normalize_event_datetime(rec)
                in_holdout_ids = tx_id in protected_ids
                in_holdout_window = val_min_dt <= rec_dt <= test_max_dt

                if not allow_holdout_feedback and (in_holdout_ids or in_holdout_window):
                    n_excluded_holdout += 1
                    continue

                eligible_rows.append(rec)
                if tx_id:
                    used_feedback_ids.add(tx_id)
                if max_used_feedback_dt is None or rec_dt > max_used_feedback_dt:
                    max_used_feedback_dt = rec_dt

        n_feedback_used = len(eligible_rows)

        if n_feedback_used > 0:
            fb_eligible_df = pd.DataFrame(eligible_rows)
            for col in model_feature_names:
                if col not in fb_eligible_df.columns:
                    fb_eligible_df[col] = 0.0
            fb_clean = fb_eligible_df[model_feature_names + ["isFraud"]].copy()
            fb_repeated = pd.concat([fb_clean] * max(1, int(feedback_weight)), ignore_index=True)
            X_train = pd.concat([train_df[model_feature_names], fb_repeated[model_feature_names]], ignore_index=True)
            y_train = np.concatenate([train_df["isFraud"].values, fb_repeated["isFraud"].values])
        else:
            X_train = train_df[model_feature_names]
            y_train = train_df["isFraud"].values

        clean_val_df = val_df.loc[~val_df["tx_id"].astype(str).isin(used_feedback_ids)].copy()
        if len(clean_val_df) < 10:
            clean_val_df = val_df

        reg = registry or ModelRegistry(
            registry_dir=cfg["paths"].get("registry_dir", "models/registry"),
            active_artifacts_dir=cfg["paths"]["artifacts_dir"],
        )
        candidate_version = reg.generate_version_id(dataset_source=ds_source, prefix="retrain")

        new_gbdt = FraudGBDTModel(params=cfg["model"]["lgbm_params"])
        new_gbdt.fit(X_train, y_train)
        cal_diag = new_gbdt.calibrate(
            clean_val_df[model_feature_names],
            clean_val_df["isFraud"].values,
            method=cfg["model"].get("calibration_method", "sigmoid"),
        )

        new_anom = FraudAnomalyDetector(params=cfg["model"]["isolation_forest"])
        new_anom.fit(X_train.loc[y_train == 0])

        rule_engine = RuleEngine(rules_cfg=cfg["rules"])
        bg_median = np.median(X_train.values, axis=0)
        explainer = FraudExplainer(new_gbdt, model_feature_names, bg_median, cfg.get("llm", {}))
        agent = FraudDecisionAgent(
            rule_engine=rule_engine,
            ml_model=new_gbdt,
            anomaly_detector=new_anom,
            explainer=explainer,
            weights=cfg["agent"]["weights"],
            thresholds=cfg["agent"]["thresholds"],
            feature_names=model_feature_names,
            model_version=candidate_version,
            feature_schema_version=SCHEMA_VERSION,
            anomaly_max_standalone_score=float(cfg["agent"].get("anomaly_max_standalone_score", 0.14)),
            fallback_policy=cfg.get("scoring", {}).get("fallback_policy", "rules_only"),
        )

        val_ml = new_gbdt.predict_score(clean_val_df[model_feature_names])
        val_rule = np.array(rule_engine.score_batch(clean_val_df), dtype=float)
        val_anom = new_anom.predict_anomaly_score(clean_val_df[model_feature_names])
        val_risk = np.asarray(agent.combine_scores(val_ml, val_rule, val_anom), dtype=float)
        val_days = estimate_span_days(
            clean_val_df["timestamp"]
            if "timestamp" in clean_val_df.columns and clean_val_df["timestamp"].notna().any()
            else clean_val_df["step"]
        )

        best_thresholds, val_cost_metrics = optimize_thresholds_by_cost(
            y_true=clean_val_df["isFraud"].values,
            scores=val_risk,
            amounts=clean_val_df["amount"].values,
            costs_cfg=cfg["agent"]["costs"],
            capacity_cfg=cfg["agent"].get("capacity", {}),
            span_days=val_days,
        )
        agent.t_otp = best_thresholds["otp"]
        agent.t_review = best_thresholds["review"]
        agent.t_block = best_thresholds["block"]

        test_metrics: dict[str, Any] | None = None
        future_slice_metrics: dict[str, Any] | None = None
        test_days = estimate_span_days(
            test_df["timestamp"]
            if "timestamp" in test_df.columns and test_df["timestamp"].notna().any()
            else test_df["step"]
        )

        used_any_holdout = bool(
            allow_holdout_feedback
            and n_feedback_used > 0
            and (
                bool(used_feedback_ids & protected_ids)
                or (max_used_feedback_dt is not None and max_used_feedback_dt >= test_min_dt)
            )
        )

        if used_any_holdout:
            original_test_invalidated = True
            is_independent_test = False
            assert max_used_feedback_dt is not None
            test_dts = [normalize_event_datetime(r) for r in test_df.to_dict(orient="records")]
            future_mask = [
                (dt > max_used_feedback_dt) and (str(tid) not in used_feedback_ids)
                for dt, tid in zip(test_dts, test_df["tx_id"])
            ]
            future_df = test_df.loc[future_mask].copy()
            if len(future_df) >= 20 and future_df["isFraud"].nunique() >= 2:
                evaluation_scope = "post_feedback_future_slice_only"
                f_ml = new_gbdt.predict_score(future_df[model_feature_names])
                f_rule = np.array(rule_engine.score_batch(future_df), dtype=float)
                f_anom = new_anom.predict_anomaly_score(future_df[model_feature_names])
                f_risk = np.asarray(agent.combine_scores(f_ml, f_rule, f_anom), dtype=float)
                future_slice_metrics = evaluate_predictions(
                    y_true=future_df["isFraud"].values,
                    y_score=f_risk,
                    amounts=future_df["amount"].values,
                    thresholds=best_thresholds,
                    costs_cfg=cfg["agent"]["costs"],
                    capacity_cfg=cfg["agent"].get("capacity", {}),
                    span_days=max(test_days * (len(future_df) / max(len(test_df), 1)), 1.0 / 24.0),
                )
                future_slice_metrics["future_slice_rows"] = int(len(future_df))
                future_slice_metrics["feedback_cutoff_utc"] = max_used_feedback_dt.isoformat()
            else:
                evaluation_scope = "no_independent_future_holdout"
        else:
            original_test_invalidated = False
            test_ml = new_gbdt.predict_score(test_df[model_feature_names])
            test_rule = np.array(rule_engine.score_batch(test_df), dtype=float)
            test_anom = new_anom.predict_anomaly_score(test_df[model_feature_names])
            test_risk = np.asarray(agent.combine_scores(test_ml, test_rule, test_anom), dtype=float)

            test_metrics = evaluate_predictions(
                y_true=test_df["isFraud"].values,
                y_score=test_risk,
                amounts=test_df["amount"].values,
                thresholds=best_thresholds,
                costs_cfg=cfg["agent"]["costs"],
                capacity_cfg=cfg["agent"].get("capacity", {}),
                span_days=test_days,
            )
            if max_used_feedback_dt is not None and max_used_feedback_dt > test_max_dt:
                evaluation_scope = "historical_regression_holdout"
                is_independent_test = False
            else:
                evaluation_scope = "independent_locked_test"
                is_independent_test = True

        retrained_at = datetime.now(timezone.utc).isoformat()
        base_meta_path = artifacts_dir / "metadata.json"
        base_meta = json.loads(base_meta_path.read_text(encoding="utf-8")) if base_meta_path.exists() else {}

        cand_meta = dict(base_meta)
        cand_meta["model_version"] = candidate_version
        cand_meta["dataset_source"] = ds_source
        cand_meta["feature_names"] = model_feature_names
        cand_meta["feature_schema_version"] = SCHEMA_VERSION
        cand_meta["calibration_diagnostics"] = cal_diag
        cand_meta["last_retrained_at"] = retrained_at
        cand_meta["optimized_thresholds"] = best_thresholds
        cand_meta["validation_cost_metrics"] = val_cost_metrics
        cand_meta["retrain_audit"] = {
            "retrained_at": retrained_at,
            "feedback_samples_total": n_feedback_total,
            "feedback_samples_used": n_feedback_used,
            "feedback_excluded_holdout": n_excluded_holdout,
            "feedback_unknown_excluded": n_unknown_excluded,
            "evaluation_scope": evaluation_scope,
            "is_independent_test": is_independent_test,
            "original_test_invalidated": original_test_invalidated,
        }
        if test_metrics is not None:
            cand_meta["post_retrain_test_metrics"] = test_metrics
        else:
            cand_meta.pop("post_retrain_test_metrics", None)
            cand_meta.pop("baseline_test_metrics", None)
        if future_slice_metrics is not None:
            cand_meta["post_retrain_future_slice_metrics"] = future_slice_metrics

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)
            joblib.dump(new_gbdt, tmp_dir / "gbdt_model.joblib")
            joblib.dump(new_anom, tmp_dir / "anomaly_detector.joblib")
            joblib.dump(bg_median, tmp_dir / "background_median.joblib")

            copy_files: dict[str, Path] = {
                "gbdt_model.joblib": tmp_dir / "gbdt_model.joblib",
                "anomaly_detector.joblib": tmp_dir / "anomaly_detector.joblib",
                "background_median.joblib": tmp_dir / "background_median.joblib",
            }
            for aux_name in ("holdout_manifest.json", "initial_feature_store_pre_test.joblib", "reference_distribution.parquet"):
                aux_p = artifacts_dir / aux_name
                if aux_p.exists():
                    copy_files[aux_name] = aux_p

            reg_entry = reg.register_candidate(
                version_id=candidate_version,
                source_files=copy_files,
                metadata=cand_meta,
                config_snapshot=cfg,
                min_pr_auc_ratio=float(cfg.get("governance", {}).get("min_pr_auc_guardrail_ratio", 0.95)),
            )

        auto_promote_cfg = bool(cfg.get("governance", {}).get("auto_promote_on_retrain", False))
        did_promote = False
        if promote or auto_promote_cfg:
            reg.promote_version(
                candidate_version,
                approved_by=approved_by or "explicit_retrain_approver",
                reason="Explicitly promoted during retrain workflow",
            )
            did_promote = True

        idx_now = reg.list_registry()
        return {
            "status": "promoted" if did_promote else "candidate_registered",
            "candidate_version": candidate_version,
            "active_version": idx_now.get("active_version"),
            "auto_promoted": did_promote,
            "comparison_vs_champion": reg_entry["comparison_vs_champion"],
            "feedback_samples_total": n_feedback_total,
            "feedback_samples_used": n_feedback_used,
            "feedback_excluded_holdout": n_excluded_holdout,
            "feedback_unknown_excluded": n_unknown_excluded,
            "evaluation_scope": evaluation_scope,
            "is_independent_test": is_independent_test,
            "original_test_invalidated": original_test_invalidated,
            "optimized_thresholds": best_thresholds,
            "test_metrics": test_metrics,
            "future_slice_metrics": future_slice_metrics,
            "retrained_at": retrained_at,
        }
    finally:
        if owns_storage:
            storage.close()
