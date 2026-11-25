"""PSI drift detection, mature-label quality, selection-bias tracking, and operational alerting."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Compute Population Stability Index (PSI) between reference (`expected`) and live (`actual`) arrays.

    Note: Per Technical Specification Section 5 & 8, PSI is an investigation signal, never the sole
    basis for automatic retraining or model publication.
    """
    exp = np.asarray(expected, dtype=float)
    act = np.asarray(actual, dtype=float)
    exp = exp[np.isfinite(exp)]
    act = act[np.isfinite(act)]
    if len(exp) < 10 or len(act) < 10:
        return 0.0

    quantiles = np.linspace(0.0, 100.0, bins + 1)
    edges = np.unique(np.percentile(exp, quantiles))
    if len(edges) < 3:
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

    action_distribution: dict[str, float] = {}
    if "action" in current_df.columns and not current_df.empty:
        vc = current_df["action"].value_counts(normalize=True)
        action_distribution = {str(k): round(float(v), 4) for k, v in vc.items()}

    return {
        "overall_status": overall,
        "psi_policy_note": "PSI indicates input/score shift for analyst & DS investigation; does not auto-publish retrained models.",
        "n_reference": int(len(reference_df)),
        "n_current": int(len(current_df)),
        "n_critical": n_crit,
        "n_warning": n_warn,
        "current_action_shares": action_distribution,
        "features": feature_psi,
    }


def compute_feedback_quality(
    labeled_df: pd.DataFrame,
    costs_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute precision/recall, observed decision cost, and selection-bias coverage on definitive labels.

    Transactions marked `unknown` or unlabeled are NEVER treated as negative (`not_fraud`).
    """
    if labeled_df.empty or "analyst_label" not in labeled_df.columns:
        return {
            "n_labeled": 0,
            "n_unknown": 0,
            "precision": None,
            "recall": None,
            "confirmed_fraud_rate": None,
            "selection_bias_note": "No labeled observations yet.",
        }

    n_unknown = (
        int((labeled_df["label_status"] == "unknown").sum())
        if "label_status" in labeled_df.columns
        else 0
    )

    valid = labeled_df[labeled_df["analyst_label"].notna()].copy()
    if "label_status" in valid.columns:
        valid = valid[valid["label_status"] != "unknown"]

    if valid.empty:
        return {
            "n_labeled": 0,
            "n_unknown": n_unknown,
            "precision": None,
            "recall": None,
            "confirmed_fraud_rate": None,
            "selection_bias_note": "Only `unknown` or unlabeled observations present; awaiting label maturity.",
        }

    y_true = valid["analyst_label"].astype(int).values
    y_pred = valid["action"].isin(["OTP", "REVIEW", "BLOCK"]).astype(int).values

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0

    # Selection bias diagnostics: compare label coverage on intervened vs ALLOW traffic
    allow_total = int((labeled_df["action"] == "ALLOW").sum()) if "action" in labeled_df.columns else 0
    allow_labeled = int((valid["action"] == "ALLOW").sum()) if "action" in valid.columns else 0
    intervened_total = int(labeled_df["action"].isin(["OTP", "REVIEW", "BLOCK"]).sum()) if "action" in labeled_df.columns else 0
    intervened_labeled = int(valid["action"].isin(["OTP", "REVIEW", "BLOCK"]).sum()) if "action" in valid.columns else 0

    observed_cost = None
    if costs_cfg and "amount" in valid.columns:
        otp_c = float(costs_cfg.get("otp_cost", 2.0))
        rev_c = float(costs_cfg.get("review_cost", 25.0))
        blk_fp_c = float(costs_cfg.get("false_block_cost", 150.0))
        fn_amt = float(valid.loc[(y_true == 1) & (valid["action"] == "ALLOW"), "amount"].sum())
        fp_blk = float(((y_true == 0) & (valid["action"] == "BLOCK")).sum()) * blk_fp_c
        rev_op = float((valid["action"] == "REVIEW").sum()) * rev_c
        otp_op = float((valid["action"] == "OTP").sum()) * otp_c
        observed_cost = round(fn_amt + fp_blk + rev_op + otp_op, 2)

    return {
        "n_labeled": int(len(valid)),
        "n_unknown": n_unknown,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "confirmed_fraud_rate": round(float(y_true.mean()), 4),
        "observed_labeled_cost": observed_cost,
        "selection_bias": {
            "intervened_label_coverage": round(intervened_labeled / max(intervened_total, 1), 4),
            "allow_label_coverage": round(allow_labeled / max(allow_total, 1), 4),
            "warning": (
                "Recall on labeled alerts overestimates true stream recall when ALLOW coverage is low; "
                "rely on matured claims/chargebacks and control sampling."
                if allow_labeled < max(5, int(0.02 * allow_total))
                else "Adequate ALLOW label sample."
            ),
        },
    }


def build_operational_alerts(
    summary_stats: dict[str, Any],
    drift_report: dict[str, Any] | None,
    monitoring_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Generate structured monitoring alerts for latency SLA, analyst queue capacity, fallbacks, and PSI drift."""
    m_cfg = monitoring_cfg or {}
    sla_p95 = float(m_cfg.get("latency_sla_p95_ms", 150.0))
    alerts: list[dict[str, Any]] = []

    if summary_stats.get("total_scored", 0) > 0:
        p95 = float(summary_stats.get("p95_latency_ms", 0.0))
        if p95 > sla_p95:
            alerts.append({
                "code": "LATENCY_P95_ABOVE_PROPOSED_SLA",
                "severity": "WARNING",
                "message": f"p95 latency ({p95:.1f} ms) exceeds proposed pilot SLA ({sla_p95:.1f} ms).",
                "runbook_action": "Inspect SHAP / DB I/O latency; verify timeout fallback budget.",
            })

        cap_util = float(summary_stats.get("review_capacity_utilization", 0.0))
        if cap_util > 1.0:
            alerts.append({
                "code": "ANALYST_REVIEW_CAPACITY_EXCEEDED",
                "severity": "CRITICAL",
                "message": (
                    f"24h REVIEW volume ({summary_stats.get('reviews_last_24h')}) exceeds daily limit "
                    f"({summary_stats.get('daily_review_capacity')})."
                ),
                "runbook_action": "Review triggered rules or recalibrate `review` threshold on validation.",
            })
        elif cap_util >= 0.80:
            alerts.append({
                "code": "ANALYST_REVIEW_CAPACITY_HIGH",
                "severity": "WARNING",
                "message": f"24h REVIEW queue at {cap_util:.0%} of daily analyst capacity.",
                "runbook_action": "Monitor queue backlog and analyst SLA.",
            })

        fb_cnt = int(summary_stats.get("degraded_fallback_count", 0))
        if fb_cnt > 0:
            alerts.append({
                "code": "FAILSAFE_FALLBACK_INVOKED",
                "severity": "CRITICAL",
                "message": f"{fb_cnt} transactions were scored via degraded fail-safe fallback policy.",
                "runbook_action": "Check service logs and `fallback_reason` in `scored_transactions`; consider rollback if model defect.",
            })

    if drift_report and drift_report.get("overall_status") in ("WARNING", "CRITICAL"):
        alerts.append({
            "code": f"PSI_DRIFT_{drift_report['overall_status']}",
            "severity": drift_report["overall_status"],
            "message": (
                f"PSI drift status is {drift_report['overall_status']} "
                f"(critical={drift_report.get('n_critical', 0)}, warning={drift_report.get('n_warning', 0)})."
            ),
            "runbook_action": "Investigate shifted features and matured labels before training/promoting a candidate.",
        })

    return alerts
