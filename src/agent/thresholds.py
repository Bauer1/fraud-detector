"""Business-cost threshold optimizer for multi-tier actions (ALLOW / OTP / REVIEW / BLOCK)."""
from __future__ import annotations

from typing import Any

import numpy as np


def compute_business_cost_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    t_otp: float,
    t_review: float,
    t_block: float,
    costs_cfg: dict[str, Any],
) -> dict[str, float]:
    """Compute financial impact of a 4-action policy (ALLOW, OTP, REVIEW, BLOCK).

    Missed fraud costs the transfer amount; false alarms cost OTP/review/customer friction.
    """
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    amt = np.asarray(amounts, dtype=float)

    otp_cost = float(costs_cfg.get("otp_cost", 2.0))
    review_cost = float(costs_cfg.get("review_cost", 25.0))
    false_block_cost = float(costs_cfg.get("false_block_cost", 150.0))

    otp_catch = float(costs_cfg.get("otp_fraud_catch_rate", 0.70))
    review_catch = float(costs_cfg.get("review_fraud_catch_rate", 0.92))
    block_catch = float(costs_cfg.get("block_fraud_catch_rate", 1.00))

    is_block = s >= t_block
    is_review = (s >= t_review) & (s < t_block)
    is_otp = (s >= t_otp) & (s < t_review)
    is_allow = s < t_otp

    is_fraud = y == 1
    is_legit = y == 0

    total_fraud_amount = float(amt[is_fraud].sum())

    # Prevented fraud amount
    fraud_prevented = float(
        amt[is_fraud & is_block].sum() * block_catch
        + amt[is_fraud & is_review].sum() * review_catch
        + amt[is_fraud & is_otp].sum() * otp_catch
    )
    missed_fraud = total_fraud_amount - fraud_prevented

    # Operational & friction costs on all intervened transactions (and extra churn cost on legit blocks)
    fp_block_cost = float((is_legit & is_block).sum()) * false_block_cost
    op_review_cost = float(is_review.sum()) * review_cost
    op_otp_cost = float(is_otp.sum()) * otp_cost
    total_intervention_cost = fp_block_cost + op_review_cost + op_otp_cost

    # Baseline cost (if we had no anti-fraud system at all) = total_fraud_amount
    total_system_cost = missed_fraud + total_intervention_cost
    net_savings = fraud_prevented - total_intervention_cost
    savings_rate = (net_savings / total_fraud_amount) if total_fraud_amount > 0 else 0.0

    return {
        "total_fraud_amount": round(total_fraud_amount, 2),
        "fraud_prevented_amount": round(fraud_prevented, 2),
        "missed_fraud_amount": round(missed_fraud, 2),
        "false_positive_cost": round(total_intervention_cost, 2),
        "total_system_cost": round(total_system_cost, 2),
        "net_savings": round(net_savings, 2),
        "savings_rate": round(savings_rate, 4),
        "n_allow": int(is_allow.sum()),
        "n_otp": int(is_otp.sum()),
        "n_review": int(is_review.sum()),
        "n_block": int(is_block.sum()),
    }


def optimize_thresholds_by_cost(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    costs_cfg: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    """Grid-search (t_otp, t_review, t_block) on validation set to maximize net business savings."""
    best_savings = -float("inf")
    best_thresholds = {"otp": 0.30, "review": 0.55, "block": 0.80}
    best_metrics: dict[str, float] = {}

    otp_grid = np.linspace(0.15, 0.50, 8)
    review_grid = np.linspace(0.35, 0.75, 9)
    block_grid = np.linspace(0.55, 0.95, 9)

    for t_otp in otp_grid:
        for t_rev in review_grid:
            if t_rev <= t_otp + 0.05:
                continue
            for t_blk in block_grid:
                if t_blk <= t_rev + 0.05:
                    continue
                m = compute_business_cost_metrics(
                    y_true=y_true,
                    scores=scores,
                    amounts=amounts,
                    t_otp=float(t_otp),
                    t_review=float(t_rev),
                    t_block=float(t_blk),
                    costs_cfg=costs_cfg,
                )
                if m["net_savings"] > best_savings:
                    best_savings = m["net_savings"]
                    best_thresholds = {
                        "otp": round(float(t_otp), 3),
                        "review": round(float(t_rev), 3),
                        "block": round(float(t_blk), 3),
                    }
                    best_metrics = m

    return best_thresholds, best_metrics
