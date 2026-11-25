"""Business-cost & analyst-capacity threshold optimizer for multi-tier actions (ALLOW / OTP / REVIEW / BLOCK)."""
from __future__ import annotations

from typing import Any

import numpy as np


def estimate_span_days(steps_or_timestamps: Any, default_days: float = 1.0) -> float:
    """Estimate calendar span in days for daily operational load & FP/day metrics."""
    if steps_or_timestamps is None:
        return max(default_days, 1.0 / 24.0)
    arr = list(steps_or_timestamps)
    if not arr:
        return max(default_days, 1.0 / 24.0)
    first, last = arr[0], arr[-1]
    if hasattr(first, "timestamp") and hasattr(last, "timestamp"):
        span_h = abs(float(last.timestamp()) - float(first.timestamp())) / 3600.0
        return max(span_h / 24.0, 1.0 / 24.0)
    try:
        vals = np.asarray(arr, dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            span_h = float(vals.max() - vals.min()) + 1.0
            return max(span_h / 24.0, 1.0 / 24.0)
    except Exception:
        pass
    return max(default_days, 1.0 / 24.0)


def compute_business_cost_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    t_otp: float,
    t_review: float,
    t_block: float,
    costs_cfg: dict[str, Any],
    capacity_cfg: dict[str, Any] | None = None,
    span_days: float = 1.0,
) -> dict[str, Any]:
    """Compute financial & operational impact of a 4-action policy (ALLOW, OTP, REVIEW, BLOCK).

    Note: Cost weights and catch rates are proposed pilot parameters requiring bank sign-off.
    """
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    amt = np.asarray(amounts, dtype=float)
    n_total = max(len(y), 1)
    days = max(float(span_days), 1.0 / 24.0)

    otp_cost = float(costs_cfg.get("otp_cost", 2.0))
    review_cost = float(costs_cfg.get("review_cost", 25.0))
    false_block_cost = float(costs_cfg.get("false_block_cost", 150.0))

    otp_catch = float(costs_cfg.get("otp_fraud_catch_rate", 0.70))
    review_catch = float(costs_cfg.get("review_fraud_catch_rate", 0.92))
    block_catch = float(costs_cfg.get("block_fraud_catch_rate", 1.00))

    cap = capacity_cfg or {}
    max_review_rate = float(cap.get("max_review_rate", 1.0))
    max_daily_reviews = float(cap.get("max_daily_reviews", 1e9))

    is_block = s >= t_block
    is_review = (s >= t_review) & (s < t_block)
    is_otp = (s >= t_otp) & (s < t_review)
    is_allow = s < t_otp

    is_fraud = y == 1
    is_legit = y == 0
    n_fraud = int(is_fraud.sum())

    total_fraud_amount = float(amt[is_fraud].sum())

    fraud_prevented = float(
        amt[is_fraud & is_block].sum() * block_catch
        + amt[is_fraud & is_review].sum() * review_catch
        + amt[is_fraud & is_otp].sum() * otp_catch
    )
    missed_fraud = total_fraud_amount - fraud_prevented

    weighted_caught_cases = (
        float((is_fraud & is_block).sum()) * block_catch
        + float((is_fraud & is_review).sum()) * review_catch
        + float((is_fraud & is_otp).sum()) * otp_catch
    )
    fraud_case_capture_rate = (weighted_caught_cases / n_fraud) if n_fraud > 0 else 0.0
    fraud_amount_capture_rate = (fraud_prevented / total_fraud_amount) if total_fraud_amount > 0 else 0.0

    fp_block_count = int((is_legit & is_block).sum())
    fp_review_count = int((is_legit & is_review).sum())
    fp_alert_count = fp_block_count + fp_review_count

    fp_block_cost = float(fp_block_count) * false_block_cost
    op_review_cost = float(is_review.sum()) * review_cost
    op_otp_cost = float(is_otp.sum()) * otp_cost
    total_intervention_cost = fp_block_cost + op_review_cost + op_otp_cost

    total_system_cost = missed_fraud + total_intervention_cost
    expected_cost_per_tx = total_system_cost / n_total
    net_savings = fraud_prevented - total_intervention_cost
    savings_rate = (net_savings / total_fraud_amount) if total_fraud_amount > 0 else 0.0

    n_review_int = int(is_review.sum())
    review_rate = n_review_int / n_total
    daily_review_count = n_review_int / days
    capacity_within_limit = bool(
        (review_rate <= max_review_rate + 1e-9) and (daily_review_count <= max_daily_reviews + 1e-9)
    )

    return {
        "total_fraud_amount": round(total_fraud_amount, 2),
        "fraud_prevented_amount": round(fraud_prevented, 2),
        "missed_fraud_amount": round(missed_fraud, 2),
        "fraud_amount_capture_rate": round(fraud_amount_capture_rate, 4),
        "fraud_case_capture_rate": round(fraud_case_capture_rate, 4),
        "false_positive_cost": round(total_intervention_cost, 2),
        "total_system_cost": round(total_system_cost, 2),
        "expected_cost_per_tx": round(expected_cost_per_tx, 4),
        "baseline_no_protection_cost": round(total_fraud_amount, 2),
        "net_savings": round(net_savings, 2),
        "savings_rate": round(savings_rate, 4),
        "fp_block_count": fp_block_count,
        "fp_review_count": fp_review_count,
        "fp_per_1000_tx": round(1000.0 * fp_alert_count / n_total, 3),
        "fp_per_day": round(fp_alert_count / days, 2),
        "review_rate": round(review_rate, 5),
        "daily_review_count": round(daily_review_count, 2),
        "capacity_within_limit": capacity_within_limit,
        "n_allow": int(is_allow.sum()),
        "n_otp": int(is_otp.sum()),
        "n_review": n_review_int,
        "n_block": int(is_block.sum()),
    }


def optimize_thresholds_by_cost(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    costs_cfg: dict[str, Any],
    capacity_cfg: dict[str, Any] | None = None,
    span_days: float = 1.0,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Select `(t_otp, t_review, t_block)` on validation set minimizing expected cost under analyst capacity limits."""
    best_feasible_savings = -float("inf")
    best_feasible_thresholds: dict[str, float] | None = None
    best_feasible_metrics: dict[str, Any] | None = None

    best_unconstrained_savings = -float("inf")
    best_unconstrained_thresholds = {"otp": 0.30, "review": 0.55, "block": 0.80}
    best_unconstrained_metrics: dict[str, Any] = {}

    otp_grid = [0.03, 0.06, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    review_grid = [0.10, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    block_grid = [0.25, 0.35, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

    for t_otp in otp_grid:
        for t_rev in review_grid:
            if t_rev <= t_otp + 0.03:
                continue
            for t_blk in block_grid:
                if t_blk <= t_rev + 0.04:
                    continue
                m = compute_business_cost_metrics(
                    y_true=y_true,
                    scores=scores,
                    amounts=amounts,
                    t_otp=float(t_otp),
                    t_review=float(t_rev),
                    t_block=float(t_blk),
                    costs_cfg=costs_cfg,
                    capacity_cfg=capacity_cfg,
                    span_days=span_days,
                )
                cand_th = {
                    "otp": round(float(t_otp), 3),
                    "review": round(float(t_rev), 3),
                    "block": round(float(t_blk), 3),
                }
                if m["net_savings"] > best_unconstrained_savings:
                    best_unconstrained_savings = m["net_savings"]
                    best_unconstrained_thresholds = cand_th
                    best_unconstrained_metrics = m

                if m["capacity_within_limit"] and m["net_savings"] > best_feasible_savings:
                    best_feasible_savings = m["net_savings"]
                    best_feasible_thresholds = cand_th
                    best_feasible_metrics = m

    if best_feasible_thresholds is not None and best_feasible_metrics is not None:
        return best_feasible_thresholds, best_feasible_metrics
    return best_unconstrained_thresholds, best_unconstrained_metrics
