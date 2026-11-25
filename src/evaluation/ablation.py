"""Ablation study comparing Rules, Baseline LR, Pre-Decision GBDT groups, Auxiliary Anomaly, Full Service, and Leakage Audit."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.agent.decision_agent import FraudDecisionAgent
from src.agent.thresholds import estimate_span_days, optimize_thresholds_by_cost
from src.evaluation.metrics import evaluate_predictions
from src.features.online_store import (
    ALL_FEATURE_NAMES,
    BEHAVIORAL_FEATURE_NAMES,
    PAYSIM_BEHAVIORAL_FEATURE_NAMES,
    resolve_model_feature_names,
    resolve_tx_only_feature_names,
)
from src.models.anomaly import FraudAnomalyDetector
from src.models.supervised import FraudGBDTModel, build_baseline_model
from src.rules.engine import RuleEngine


def run_ablation_study(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: dict[str, Any],
    full_gbdt: FraudGBDTModel,
    anomaly_detector: FraudAnomalyDetector,
    rule_engine: RuleEngine,
    decision_agent: FraudDecisionAgent,
    *,
    dataset_source: str = "enriched_simulation",
) -> list[dict[str, Any]]:
    """Run controlled ablations on the exact same chronological train/val/test split.

    Strictly separates pre-decision deployable variants from diagnostic-only rows
    (`isFlaggedFraud` PaySim benchmark and `[DO NOT DEPLOY] Post-Tx Balance Leakage Audit`).
    """
    y_train = train_df["isFraud"].values
    y_val = val_df["isFraud"].values
    y_test = test_df["isFraud"].values
    amt_val = val_df["amount"].values
    amt_test = test_df["amount"].values
    costs_cfg = cfg["agent"]["costs"]
    capacity_cfg = cfg["agent"].get("capacity", {})
    seed = cfg["project"]["random_seed"]
    cal_method = cfg["model"].get("calibration_method", "sigmoid")

    val_days = estimate_span_days(
        val_df["timestamp"] if "timestamp" in val_df.columns and val_df["timestamp"].notna().any() else val_df["step"]
    )
    test_days = estimate_span_days(
        test_df["timestamp"] if "timestamp" in test_df.columns and test_df["timestamp"].notna().any() else test_df["step"]
    )

    fc_cfg = cfg.get("feature_contract", {})
    clean_all_feats = resolve_model_feature_names(dataset_source, fc_cfg)
    clean_tx_feats = resolve_tx_only_feature_names(dataset_source, fc_cfg)
    beh_feats = PAYSIM_BEHAVIORAL_FEATURE_NAMES if dataset_source == "paysim_csv" else BEHAVIORAL_FEATURE_NAMES
    tx_plus_beh_feats = [f for f in (clean_tx_feats + beh_feats) if f in clean_all_feats]

    results: list[dict[str, Any]] = []

    def _eval_variant(
        name: str,
        val_scores: np.ndarray,
        test_scores: np.ndarray,
        fixed_thresholds: dict[str, float] | None = None,
        deployable: bool = True,
    ) -> None:
        if fixed_thresholds is None:
            th, _ = optimize_thresholds_by_cost(
                y_val, val_scores, amt_val, costs_cfg, capacity_cfg=capacity_cfg, span_days=val_days
            )
        else:
            th = fixed_thresholds
        m = evaluate_predictions(
            y_test, test_scores, amt_test, th, costs_cfg, capacity_cfg=capacity_cfg, span_days=test_days
        )
        m["variant"] = name
        m["thresholds"] = th
        m["deployable_pre_decision"] = deployable
        results.append(m)

    # 0. Original PaySim native rule benchmark (`isFlaggedFraud`, if present in CSV)
    if dataset_source == "paysim_csv" and "isFlaggedFraud" in test_df.columns:
        val_flagged = val_df["isFlaggedFraud"].astype(float).values
        test_flagged = test_df["isFlaggedFraud"].astype(float).values
        _eval_variant(
            "0. PaySim Native Rule (isFlaggedFraud Benchmark)",
            val_flagged,
            test_flagged,
            fixed_thresholds={"otp": 0.25, "review": 0.50, "block": 0.85},
            deployable=False,
        )

    # 1. Rules Only
    val_rule = np.array(rule_engine.score_batch(val_df), dtype=float)
    test_rule = np.array(rule_engine.score_batch(test_df), dtype=float)
    _eval_variant("1. Rules Only", val_rule, test_rule)

    # 2. Baseline Logistic Regression (Pre-Decision Transaction features only)
    lr_tx = build_baseline_model(random_state=seed)
    lr_tx.fit(train_df[clean_tx_feats].values, y_train)
    _eval_variant(
        "2. Baseline LogReg (Pre-Decision Tx Only)",
        lr_tx.predict_proba(val_df[clean_tx_feats].values)[:, 1],
        lr_tx.predict_proba(test_df[clean_tx_feats].values)[:, 1],
    )

    # 3. Baseline Logistic Regression (All Pre-Decision features)
    lr_all = build_baseline_model(random_state=seed)
    lr_all.fit(train_df[clean_all_feats].values, y_train)
    _eval_variant(
        "3. Baseline LogReg (All Pre-Decision Features)",
        lr_all.predict_proba(val_df[clean_all_feats].values)[:, 1],
        lr_all.predict_proba(test_df[clean_all_feats].values)[:, 1],
    )

    # 4. GBDT (Pre-Decision Transaction features only)
    gbdt_tx = FraudGBDTModel(params=cfg["model"]["lgbm_params"])
    gbdt_tx.fit(train_df[clean_tx_feats], y_train)
    gbdt_tx.calibrate(val_df[clean_tx_feats], y_val, method=cal_method)
    _eval_variant(
        f"4. GBDT ({gbdt_tx.backend_name}, Pre-Decision Tx Only)",
        gbdt_tx.predict_score(val_df[clean_tx_feats]),
        gbdt_tx.predict_score(test_df[clean_tx_feats]),
    )

    # 5. GBDT (Pre-Decision Tx + Behavioral, No Graph)
    gbdt_beh = FraudGBDTModel(params=cfg["model"]["lgbm_params"])
    gbdt_beh.fit(train_df[tx_plus_beh_feats], y_train)
    gbdt_beh.calibrate(val_df[tx_plus_beh_feats], y_val, method=cal_method)
    _eval_variant(
        f"5. GBDT ({gbdt_beh.backend_name}, Tx + Behavioral, No Graph)",
        gbdt_beh.predict_score(val_df[tx_plus_beh_feats]),
        gbdt_beh.predict_score(test_df[tx_plus_beh_feats]),
    )

    # 6. GBDT (All Pre-Decision Features: Tx + Behavioral + Graph, Calibrated)
    val_ml = full_gbdt.predict_score(val_df[clean_all_feats])
    test_ml = full_gbdt.predict_score(test_df[clean_all_feats])
    _eval_variant(
        f"6. GBDT ({full_gbdt.backend_name}, All Pre-Decision Features)",
        val_ml,
        test_ml,
    )

    # 7. Anomaly Detector Only (Auxiliary Diagnostic — never deployed standalone)
    val_anom = anomaly_detector.predict_anomaly_score(val_df[clean_all_feats])
    test_anom = anomaly_detector.predict_anomaly_score(test_df[clean_all_feats])
    _eval_variant("7. Isolation Forest Only (Auxiliary Diagnostic)", val_anom, test_anom, deployable=False)

    # 8. Full Decision Service (Rules + Calibrated GBDT + Auxiliary IsoForest)
    val_agent = decision_agent.combine_scores(val_ml, val_rule, val_anom)
    test_agent = decision_agent.combine_scores(test_ml, test_rule, test_anom)
    _eval_variant(
        "8. Full Decision Service (Rules + GBDT + Aux IsoForest)",
        np.asarray(val_agent, dtype=float),
        np.asarray(test_agent, dtype=float),
        fixed_thresholds={
            "otp": decision_agent.t_otp,
            "review": decision_agent.t_review,
            "block": decision_agent.t_block,
        },
        deployable=True,
    )

    # 9. Explicit Post-Tx Balance Leakage Audit (quantifies why newbalanceOrig/Dest are excluded)
    leak_audit_feats = (
        resolve_model_feature_names(
            dataset_source,
            {"allow_post_tx_balances": True, "allow_dest_balances": True, "allow_sender_pre_balance": True},
        )
    )
    if set(leak_audit_feats) != set(clean_all_feats) and set(leak_audit_feats).issubset(train_df.columns):
        gbdt_leak = FraudGBDTModel(params=cfg["model"]["lgbm_params"])
        gbdt_leak.fit(train_df[leak_audit_feats], y_train)
        gbdt_leak.calibrate(val_df[leak_audit_feats], y_val, method=cal_method)
        _eval_variant(
            "9. [LEAKAGE AUDIT - EXCLUDED] GBDT + Post-Tx Balances",
            gbdt_leak.predict_score(val_df[leak_audit_feats]),
            gbdt_leak.predict_score(test_df[leak_audit_feats]),
            deployable=False,
        )

    return results
