"""Anti-fraud evaluation metrics: PR-AUC, ROC-AUC, Recall@FPR, Precision@K, and Business Cost."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from src.agent.thresholds import compute_business_cost_metrics


def recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01) -> float:
    """Compute maximum recall achievable at FPR <= target_fpr."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    if len(np.unique(y)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y, s)
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return 0.0
    return float(tpr[valid[-1]])


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 100) -> float:
    """Compute precision among the top-K highest-scored alerts."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    k = min(k, len(y))
    if k <= 0:
        return 0.0
    top_idx = np.argsort(s)[::-1][:k]
    return float(y[top_idx].mean())


def evaluate_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amounts: np.ndarray,
    thresholds: dict[str, float],
    costs_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Compute complete suite of ranking and business-cost metrics."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)

    pr_auc = float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else 0.0
    roc_auc = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else 0.0

    cost_m = compute_business_cost_metrics(
        y_true=y,
        scores=s,
        amounts=amounts,
        t_otp=thresholds["otp"],
        t_review=thresholds["review"],
        t_block=thresholds["block"],
        costs_cfg=costs_cfg,
    )

    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "recall_at_1pct_fpr": round(recall_at_fpr(y, s, target_fpr=0.01), 4),
        "recall_at_0_5pct_fpr": round(recall_at_fpr(y, s, target_fpr=0.005), 4),
        "precision_at_50": round(precision_at_k(y, s, k=50), 4),
        "precision_at_100": round(precision_at_k(y, s, k=100), 4),
        "precision_at_200": round(precision_at_k(y, s, k=200), 4),
        **cost_m,
    }
