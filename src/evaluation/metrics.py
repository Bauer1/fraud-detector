"""Anti-fraud evaluation suite: ranking, business cost, capacity, temporal stability, segments, FP/FN analysis, and latency."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from src.agent.thresholds import compute_business_cost_metrics, estimate_span_days


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
    capacity_cfg: dict[str, Any] | None = None,
    span_days: float = 1.0,
) -> dict[str, Any]:
    """Compute complete suite of ranking, operational load, and business-cost metrics."""
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
        capacity_cfg=capacity_cfg,
        span_days=span_days,
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


def compute_temporal_slice_stability(
    df: pd.DataFrame,
    y_score: np.ndarray,
    thresholds: dict[str, float],
    costs_cfg: dict[str, Any],
    capacity_cfg: dict[str, Any] | None = None,
    n_slices: int = 4,
    n_bootstrap: int = 100,
    random_state: int = 42,
) -> dict[str, Any]:
    """Evaluate ranking & cost dispersion across chronological slices + 95% bootstrap CI on PR-AUC."""
    y = df["isFraud"].astype(int).to_numpy()
    s = np.asarray(y_score, dtype=float)
    amt = df["amount"].astype(float).to_numpy()
    n = len(df)

    # 1. Bootstrap 95% CI on test PR-AUC
    rng = np.random.default_rng(random_state)
    boot_prs: list[float] = []
    if n >= 30 and len(np.unique(y)) > 1:
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            if len(np.unique(y[idx])) > 1:
                boot_prs.append(float(average_precision_score(y[idx], s[idx])))

    ci_low = round(float(np.percentile(boot_prs, 2.5)), 4) if boot_prs else None
    ci_high = round(float(np.percentile(boot_prs, 97.5)), 4) if boot_prs else None

    # 2. Multi-slice chronological dispersion
    slices_out: list[dict[str, Any]] = []
    edges = np.linspace(0, n, n_slices + 1, dtype=int)
    for i in range(n_slices):
        lo, hi = int(edges[i]), int(edges[i + 1])
        if hi - lo < 10:
            continue
        sub = df.iloc[lo:hi]
        sub_y = y[lo:hi]
        sub_s = s[lo:hi]
        sub_amt = amt[lo:hi]
        span_d = estimate_span_days(sub["timestamp"] if "timestamp" in sub.columns and sub["timestamp"].notna().any() else sub["step"])
        m = evaluate_predictions(sub_y, sub_s, sub_amt, thresholds, costs_cfg, capacity_cfg, span_days=span_d)
        slices_out.append({
            "slice_index": i + 1,
            "rows": int(hi - lo),
            "fraud_count": int(sub_y.sum()),
            "fraud_rate": round(float(sub_y.mean()), 5),
            "min_step": int(sub["step"].min()) if "step" in sub.columns else None,
            "max_step": int(sub["step"].max()) if "step" in sub.columns else None,
            "pr_auc": m["pr_auc"] if int(sub_y.sum()) > 0 else None,
            "recall_at_1pct_fpr": m["recall_at_1pct_fpr"] if int(sub_y.sum()) > 0 else None,
            "expected_cost_per_tx": m["expected_cost_per_tx"],
            "review_rate": m["review_rate"],
        })

    valid_prs = [sl["pr_auc"] for sl in slices_out if sl["pr_auc"] is not None]
    return {
        "pr_auc_bootstrap_ci95": {"low": ci_low, "high": ci_high, "n_bootstrap": n_bootstrap},
        "slice_pr_auc_mean": round(float(np.mean(valid_prs)), 4) if valid_prs else None,
        "slice_pr_auc_std": round(float(np.std(valid_prs)), 4) if valid_prs else None,
        "slice_pr_auc_min": round(float(np.min(valid_prs)), 4) if valid_prs else None,
        "slice_pr_auc_max": round(float(np.max(valid_prs)), 4) if valid_prs else None,
        "slices": slices_out,
    }


def compute_segmented_metrics(
    df: pd.DataFrame,
    y_score: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, list[dict[str, Any]]]:
    """Compute ranking & alert metrics segmented by operation type, amount tier, time of day, and channel."""
    work = df.copy()
    work["__score"] = np.asarray(y_score, dtype=float)
    work["__y"] = work["isFraud"].astype(int)
    work["__alert"] = (work["__score"] >= thresholds["review"]).astype(int)

    work["amount_bucket"] = pd.cut(
        work["amount"],
        bins=[-1.0, 1_000.0, 10_000.0, 50_000.0, float("inf")],
        labels=["<1k", "1k-10k", "10k-50k", ">=50k"],
    ).astype(str)
    if "is_night" in work.columns:
        work["time_segment"] = np.where(work["is_night"] >= 0.5, "night_00_05_utc", "day_06_23_utc")
    else:
        work["time_segment"] = "all_hours"

    dimensions = ["type", "amount_bucket", "time_segment"]
    if "channel" in work.columns and work["channel"].notna().any():
        dimensions.append("channel")

    segmented: dict[str, list[dict[str, Any]]] = {}
    for dim in dimensions:
        rows_list: list[dict[str, Any]] = []
        for seg_val, grp in work.groupby(dim, dropna=False):
            yv = grp["__y"].to_numpy()
            sv = grp["__score"].to_numpy()
            n_g = len(grp)
            n_f = int(yv.sum())
            pr = round(float(average_precision_score(yv, sv)), 4) if len(np.unique(yv)) > 1 else None
            rec1 = round(recall_at_fpr(yv, sv, 0.01), 4) if len(np.unique(yv)) > 1 else None
            high_tier_recall = (
                round(float(((yv == 1) & (sv >= thresholds["otp"])).sum() / n_f), 4) if n_f > 0 else None
            )
            rows_list.append({
                "segment": str(seg_val),
                "rows": int(n_g),
                "fraud_count": n_f,
                "fraud_rate": round(n_f / max(n_g, 1), 5),
                "pr_auc": pr,
                "recall_at_1pct_fpr": rec1,
                "intervention_recall": high_tier_recall,
                "review_or_block_rate": round(float(grp["__alert"].mean()), 5),
            })
        segmented[dim] = rows_list
    return segmented


def compute_fp_fn_error_analysis(
    df: pd.DataFrame,
    y_score: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Structured error analysis of False Positives (`REVIEW/BLOCK` on legit) and False Negatives (`ALLOW` on fraud)."""
    work = df.copy()
    s = np.asarray(y_score, dtype=float)
    y = work["isFraud"].astype(int).to_numpy()

    fp_mask = (y == 0) & (s >= thresholds["review"])
    fn_mask = (y == 1) & (s < thresholds["otp"])

    fp_df = work.loc[fp_mask]
    fn_df = work.loc[fn_mask]

    def _summarize(sub: pd.DataFrame) -> dict[str, Any]:
        if sub.empty:
            return {"count": 0, "total_amount": 0.0, "median_amount": 0.0, "by_type": {}, "cold_start_share": 0.0}
        cold_share = (
            float((sub["orig_past_tx_count"] == 0).mean()) if "orig_past_tx_count" in sub.columns else 0.0
        )
        summary: dict[str, Any] = {
            "count": int(len(sub)),
            "total_amount": round(float(sub["amount"].sum()), 2),
            "median_amount": round(float(sub["amount"].median()), 2),
            "by_type": {str(k): int(v) for k, v in sub["type"].value_counts().items()} if "type" in sub.columns else {},
            "cold_start_share": round(cold_share, 4),
        }
        if "fraud_scenario" in sub.columns:
            summary["by_scenario"] = {str(k): int(v) for k, v in sub["fraud_scenario"].value_counts().items()}
        return summary

    return {
        "false_positives_review_or_block": _summarize(fp_df),
        "false_negatives_allowed": _summarize(fn_df),
    }


def measure_latency_percentiles(
    agent: Any,
    sample_df: pd.DataFrame,
    all_feature_cols: list[str],
    max_samples: int = 150,
) -> dict[str, float]:
    """Measure single-event end-to-end decision & SHAP latency percentiles (p50 / p95 / p99)."""
    if sample_df.empty:
        return {"samples": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    sub = sample_df.iloc[: min(max_samples, len(sample_df))]
    latencies: list[float] = []
    for row in sub.to_dict(orient="records"):
        feats = {c: float(row.get(c, 0.0)) for c in all_feature_cols if c in row}
        t0 = time.perf_counter()
        agent.evaluate_transaction(row, feats, explain=True)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(latencies, dtype=float)
    return {
        "samples": int(len(arr)),
        "mean_ms": round(float(arr.mean()), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
    }
