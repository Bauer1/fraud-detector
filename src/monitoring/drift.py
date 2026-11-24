"""Population Stability Index (PSI) drift detection and live feedback performance tracking."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Compute Population Stability Index (PSI) between reference (`expected`) and live (`actual`) arrays.

    Interpretation:
      - PSI < 0.10: No significant drift
      - 0.10 <= PSI < 0.25: Moderate drift (warning)
      - PSI >= 0.25: Significant drift (retrain / investigate)
    """
    exp = np.asarray(expected, dtype=float)
    act = np.asarray(actual, dtype=float)
    exp = exp[np.isfinite(exp)]
    act = act[np.isfinite(act)]
    if len(exp) < 10 or len(act) < 10:
        return 0.0

    # Quantile-based bin edges from reference distribution
    quantiles = np.linspace(0.0, 100.0, bins + 1)
    edges = np.unique(np.percentile(exp, quantiles))
    if len(edges) < 3:
        # Fallback for binary or low-cardinality features
        uniq = np.unique(np.concatenate([exp, act]))
        if len(uniq) <= 1:
            return 0.0
        exp_counts = np.array([(exp == u).sum() for u in uniq], dtype=float)
        act_counts = np.array([(act == u).sum() for u in uniq], dtype=float)
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf
        exp_counts, _ = np.histogram(exp, bins=edges)
        act_counts, _ = np.histogram(act, bins=edges)

    eps = 1e-6
    exp_pct = np.clip(exp_counts / max(exp_counts.sum(), 1.0), eps, 1.0)
    act_pct = np.clip(act_counts / max(act_counts.sum(), 1.0), eps, 1.0)
    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return round(max(psi, 0.0), 4)


def compute_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    columns: list[str],
    warning_threshold: float = 0.10,
    critical_threshold: float = 0.25,
    bins: int = 10,
) -> dict[str, Any]:
    """Compute PSI across monitored features and scores."""
    feature_psi: list[dict[str, Any]] = []
    for col in columns:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        psi_val = compute_psi(reference_df[col].values, current_df[col].values, bins=bins)
        if psi_val >= critical_threshold:
            status = "CRITICAL"
        elif psi_val >= warning_threshold:
            status = "WARNING"
        else:
            status = "OK"
        feature_psi.append({
            "feature": col,
            "psi": psi_val,
            "status": status,
            "ref_mean": round(float(np.nanmean(reference_df[col].values)), 4),
            "cur_mean": round(float(np.nanmean(current_df[col].values)), 4),
        })

    feature_psi.sort(key=lambda x: x["psi"], reverse=True)
    n_crit = sum(1 for x in feature_psi if x["status"] == "CRITICAL")
    n_warn = sum(1 for x in feature_psi if x["status"] == "WARNING")
    overall = "CRITICAL" if n_crit > 0 else ("WARNING" if n_warn > 0 else "OK")

    return {
        "overall_status": overall,
        "n_reference": int(len(reference_df)),
        "n_current": int(len(current_df)),
        "n_critical": n_crit,
        "n_warning": n_warn,
        "features": feature_psi,
    }


def compute_feedback_quality(labeled_df: pd.DataFrame) -> dict[str, Any]:
    """Compute precision/recall of alerts on analyst-labeled cases."""
    if labeled_df.empty or "analyst_label" not in labeled_df.columns:
        return {"n_labeled": 0, "precision": None, "recall": None, "confirmed_fraud_rate": None}

    valid = labeled_df[labeled_df["analyst_label"].notna()].copy()
    if valid.empty:
        return {"n_labeled": 0, "precision": None, "recall": None, "confirmed_fraud_rate": None}

    y_true = valid["analyst_label"].astype(int).values
    y_pred = valid["action"].isin(["OTP", "REVIEW", "BLOCK"]).astype(int).values

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0

    return {
        "n_labeled": int(len(valid)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "confirmed_fraud_rate": round(float(y_true.mean()), 4),
    }
