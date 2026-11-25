"""Feedback-loop retraining pipeline with strict holdout leakage protection."""
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
from src.features.online_store import ALL_FEATURE_NAMES, normalize_event_datetime
from src.models.anomaly import FraudAnomalyDetector
from src.models.supervised import FraudGBDTModel
from src.rules.engine import RuleEngine


def _load_holdout_protection_sets(
    artifacts_dir,
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
) -> dict[str, Any]:
    """Retrain GBDT + Isolation Forest incorporating analyst feedback without contaminating evaluation.

    Guarantees:
    1. Default (`allow_holdout_feedback=False`, controlled by `retraining.exclude_holdout_feedback`):
       Any labeled transaction belonging to `val_df` or `test_df` (by `tx_id` or benchmark window
       `[val_min_timestamp, test_max_timestamp]`) is strictly excluded from `X_train` so the held-out
       test benchmark remains genuinely untouched.
    2. Even if `allow_holdout_feedback=True` is explicitly requested for rolling online adaptation:
       - `val_df` cases used in training are removed from validation threshold tuning;
       - The original `test_df` is marked `original_test_invalidated=True`, `test_metrics=None`
         (never reporting contaminated test metrics as an independent benchmark);
       - Optional metrics are computed ONLY on strictly future unseen events (`timestamp > max_feedback_ts`
         and disjoint `tx_id`) under `future_slice_metrics`.
    """
    cfg = cfg or load_config()
    storage = storage or StorageManager(cfg["paths"]["database_url"])
    artifacts_dir = resolve_path(cfg["paths"]["artifacts_dir"])
    processed_path = resolve_path(cfg["paths"]["processed_data"])

    if not processed_path.exists():
        raise FileNotFoundError("Processed features parquet not found. Run initial training first.")

    if allow_holdout_feedback is None:
        exclude_cfg = bool(cfg.get("retraining", {}).get("exclude_holdout_feedback", True))
        allow_holdout_feedback = not exclude_cfg

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

    feedback_df = storage.get_labeled_feedback_df()
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

    # Combine historical training data with eligible, non-leaking analyst feedback
    if n_feedback_used > 0:
        fb_eligible_df = pd.DataFrame(eligible_rows)
        for col in ALL_FEATURE_NAMES:
            if col not in fb_eligible_df.columns:
                fb_eligible_df[col] = 0.0
        fb_clean = fb_eligible_df[ALL_FEATURE_NAMES + ["isFraud"]].copy()
        fb_repeated = pd.concat([fb_clean] * max(1, int(feedback_weight)), ignore_index=True)
        X_train = pd.concat([train_df[ALL_FEATURE_NAMES], fb_repeated[ALL_FEATURE_NAMES]], ignore_index=True)
        y_train = np.concatenate([train_df["isFraud"].values, fb_repeated["isFraud"].values])
    else:
        X_train = train_df[ALL_FEATURE_NAMES]
        y_train = train_df["isFraud"].values

    # Ensure validation set is strictly disjoint from any used feedback
    clean_val_df = val_df.loc[~val_df["tx_id"].astype(str).isin(used_feedback_ids)].copy()
    if len(clean_val_df) < 10:
        clean_val_df = val_df

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

    # Re-optimize thresholds on uncontaminated validation set
    val_ml = new_gbdt.predict_score(clean_val_df[ALL_FEATURE_NAMES])
    val_rule = np.array(rule_engine.score_batch(clean_val_df), dtype=float)
    val_anom = new_anom.predict_anomaly_score(clean_val_df[ALL_FEATURE_NAMES])
    val_risk = np.asarray(agent.combine_scores(val_ml, val_rule, val_anom), dtype=float)

    best_thresholds, _ = optimize_thresholds_by_cost(
        y_true=clean_val_df["isFraud"].values,
        scores=val_risk,
        amounts=clean_val_df["amount"].values,
        costs_cfg=cfg["agent"]["costs"],
    )

    # Honest post-retrain evaluation (never calling a contaminated test set an independent evaluation)
    test_metrics: dict[str, Any] | None = None
    future_slice_metrics: dict[str, Any] | None = None

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
        # Evaluate ONLY on strictly future transactions after max_used_feedback_dt and disjoint by tx_id
        assert max_used_feedback_dt is not None
        test_dts = [normalize_event_datetime(r) for r in test_df.to_dict(orient="records")]
        future_mask = [
            (dt > max_used_feedback_dt) and (str(tid) not in used_feedback_ids)
            for dt, tid in zip(test_dts, test_df["tx_id"])
        ]
        future_df = test_df.loc[future_mask].copy()
        if len(future_df) >= 20 and future_df["isFraud"].nunique() >= 2:
            evaluation_scope = "post_feedback_future_slice_only"
            f_ml = new_gbdt.predict_score(future_df[ALL_FEATURE_NAMES])
            f_rule = np.array(rule_engine.score_batch(future_df), dtype=float)
            f_anom = new_anom.predict_anomaly_score(future_df[ALL_FEATURE_NAMES])
            f_risk = np.asarray(agent.combine_scores(f_ml, f_rule, f_anom), dtype=float)
            future_slice_metrics = evaluate_predictions(
                y_true=future_df["isFraud"].values,
                y_score=f_risk,
                amounts=future_df["amount"].values,
                thresholds=best_thresholds,
                costs_cfg=cfg["agent"]["costs"],
            )
            future_slice_metrics["future_slice_rows"] = int(len(future_df))
            future_slice_metrics["feedback_cutoff_utc"] = max_used_feedback_dt.isoformat()
        else:
            evaluation_scope = "no_independent_future_holdout"
    else:
        original_test_invalidated = False
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
        if max_used_feedback_dt is not None and max_used_feedback_dt > test_max_dt:
            # Feedback came from later production traffic outside the locked benchmark window
            evaluation_scope = "historical_regression_holdout"
            is_independent_test = False
        else:
            evaluation_scope = "independent_locked_test"
            is_independent_test = True

    # Persist updated model artifacts
    joblib.dump(new_gbdt, artifacts_dir / "gbdt_model.joblib")
    joblib.dump(new_anom, artifacts_dir / "anomaly_detector.joblib")
    joblib.dump(bg_median, artifacts_dir / "background_median.joblib")

    retrained_at = datetime.now(timezone.utc).isoformat()
    meta_path = artifacts_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta["last_retrained_at"] = retrained_at
    meta["optimized_thresholds"] = best_thresholds
    meta["retrain_audit"] = {
        "retrained_at": retrained_at,
        "feedback_samples_total": n_feedback_total,
        "feedback_samples_used": n_feedback_used,
        "feedback_excluded_holdout": n_excluded_holdout,
        "evaluation_scope": evaluation_scope,
        "is_independent_test": is_independent_test,
        "original_test_invalidated": original_test_invalidated,
    }
    if test_metrics is not None:
        meta["post_retrain_test_metrics"] = test_metrics
    else:
        meta.pop("post_retrain_test_metrics", None)
    if future_slice_metrics is not None:
        meta["post_retrain_future_slice_metrics"] = future_slice_metrics
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "status": "retrained",
        "feedback_samples_total": n_feedback_total,
        "feedback_samples_used": n_feedback_used,
        "feedback_excluded_holdout": n_excluded_holdout,
        "evaluation_scope": evaluation_scope,
        "is_independent_test": is_independent_test,
        "original_test_invalidated": original_test_invalidated,
        "optimized_thresholds": best_thresholds,
        "test_metrics": test_metrics,
        "future_slice_metrics": future_slice_metrics,
        "retrained_at": retrained_at,
    }
