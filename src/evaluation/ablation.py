"""Ablation study comparing Rules, Baseline LR, Tx-only GBDT, Full GBDT, Isolation Forest, and Full Agent."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.agent.decision_agent import FraudDecisionAgent
from src.agent.thresholds import optimize_thresholds_by_cost
from src.evaluation.metrics import evaluate_predictions
from src.features.online_store import ALL_FEATURE_NAMES, TX_FEATURE_NAMES
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
) -> list[dict[str, Any]]:
    """Run controlled ablations on the exact same chronological train/val/test split."""
    y_train = train_df["isFraud"].values
    y_val = val_df["isFraud"].values
    y_test = test_df["isFraud"].values
    amt_val = val_df["amount"].values
    amt_test = test_df["amount"].values
    costs_cfg = cfg["agent"]["costs"]
    seed = cfg["project"]["random_seed"]

    results: list[dict[str, Any]] = []

    def _eval_variant(name: str, val_scores: np.ndarray, test_scores: np.ndarray, fixed_thresholds: dict[str, float] | None = None) -> None:
        if fixed_thresholds is None:
            th, _ = optimize_thresholds_by_cost(y_val, val_scores, amt_val, costs_cfg)
        else:
            th = fixed_thresholds
        m = evaluate_predictions(y_test, test_scores, amt_test, th, costs_cfg)
        m["variant"] = name
        m["thresholds"] = th
        results.append(m)

    # 1. Rules Only
    val_rule = np.array(rule_engine.score_batch(val_df), dtype=float)
    test_rule = np.array(rule_engine.score_batch(test_df), dtype=float)
    _eval_variant("1. Rules Only", val_rule, test_rule)

    # 2. Baseline Logistic Regression (Transaction features only)
    lr_tx = build_baseline_model(random_state=seed)
    lr_tx.fit(train_df[TX_FEATURE_NAMES].values, y_train)
    _eval_variant(
        "2. Baseline LogReg (Tx Only)",
        lr_tx.predict_proba(val_df[TX_FEATURE_NAMES].values)[:, 1],
        lr_tx.predict_proba(test_df[TX_FEATURE_NAMES].values)[:, 1],
    )

    # 3. Baseline Logistic Regression (All features: Tx + Behavioral + Graph)
    lr_all = build_baseline_model(random_state=seed)
    lr_all.fit(train_df[ALL_FEATURE_NAMES].values, y_train)
    _eval_variant(
        "3. Baseline LogReg (All Features)",
        lr_all.predict_proba(val_df[ALL_FEATURE_NAMES].values)[:, 1],
        lr_all.predict_proba(test_df[ALL_FEATURE_NAMES].values)[:, 1],
    )

    # 4. GBDT (Transaction features only — ablation without behavioral/graph)
    gbdt_tx = FraudGBDTModel(params=cfg["model"]["lgbm_params"])
    gbdt_tx.fit(train_df[TX_FEATURE_NAMES], y_train)
    _eval_variant(
        f"4. GBDT ({gbdt_tx.backend_name}, Tx Only)",
        gbdt_tx.predict_score(val_df[TX_FEATURE_NAMES]),
        gbdt_tx.predict_score(test_df[TX_FEATURE_NAMES]),
    )

    # 5. GBDT (Full features: Tx + Behavioral + Graph)
    val_ml = full_gbdt.predict_score(val_df[ALL_FEATURE_NAMES])
    test_ml = full_gbdt.predict_score(test_df[ALL_FEATURE_NAMES])
    _eval_variant(
        f"5. GBDT ({full_gbdt.backend_name}, Full Features)",
        val_ml,
        test_ml,
    )

    # 6. Anomaly Detector Only (Isolation Forest)
    val_anom = anomaly_detector.predict_anomaly_score(val_df[ALL_FEATURE_NAMES])
    test_anom = anomaly_detector.predict_anomaly_score(test_df[ALL_FEATURE_NAMES])
    _eval_variant("6. Isolation Forest Only", val_anom, test_anom)

    # 7. Full AI Decision Agent (Rules + GBDT + Isolation Forest)
    val_agent = decision_agent.combine_scores(val_ml, val_rule, val_anom)
    test_agent = decision_agent.combine_scores(test_ml, test_rule, test_anom)
    _eval_variant(
        "7. Full AI Agent (Rules + GBDT + IsoForest)",
        np.asarray(val_agent, dtype=float),
        np.asarray(test_agent, dtype=float),
        fixed_thresholds={
            "otp": decision_agent.t_otp,
            "review": decision_agent.t_review,
            "block": decision_agent.t_block,
        },
    )

    return results
