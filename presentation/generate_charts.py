"""Generate high-resolution charts from real project data and artifacts for the presentation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.agent.explainer import FraudExplainer
from src.features.engineering import time_based_split
from src.features.online_store import ALL_FEATURE_NAMES

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "presentation" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

# Clean light theme matching the light presentation & PDF user guide
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#f8fafc",
    "axes.edgecolor": "#cbd5e1",
    "axes.labelcolor": "#1e293b",
    "text.color": "#0f172a",
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "grid.color": "#cbd5e1",
    "grid.alpha": 0.65,
    "font.sans-serif": ["DejaVu Sans", "Arial"],
})


def make_charts() -> None:
    meta = json.loads((ROOT / "models" / "artifacts" / "metadata.json").read_text(encoding="utf-8"))
    raw_df = pd.read_parquet(ROOT / "data" / "raw" / "transactions.parquet")

    # 1. Daily fraud rate + train/val/test split boundaries
    raw_df["day"] = raw_df["step"] // 24
    daily = raw_df.groupby("day")["isFraud"].mean() * 100.0
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=180)
    ax.plot(daily.index, daily.values, color="#0284c7", linewidth=2.4, marker="o", markersize=4.5)
    ax.fill_between(daily.index, daily.values, color="#0284c7", alpha=0.12)
    ax.axvline(18, color="#d97706", linestyle="--", linewidth=1.6, label="Val (20%)")
    ax.axvline(24, color="#dc2626", linestyle="--", linewidth=1.6, label="Test (20%, дрейф)")
    ax.set_title("Динамика доли мошенничества по дням (разбиение по времени)", fontsize=11.5, fontweight="bold", pad=10)
    ax.set_xlabel("День симуляции")
    ax.set_ylabel("Доля фрода (%)")
    ax.grid(True)
    ax.legend(facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSETS / "fraud_by_day.png", dpi=180)
    plt.close(fig)

    # 2. Ablation study: PR-AUC and Net Savings
    ab_df = pd.DataFrame(meta["ablations"])
    short_names = [
        "1. Только правила",
        "2. LogReg (Tx Only)",
        "3. LogReg (Все прич.)",
        "4. GBDT (Tx Only)",
        "5. GBDT (Все прич.)",
        "6. IsoForest Only",
        "7. Full AI Agent",
    ]
    ab_df["short"] = short_names[: len(ab_df)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=180)
    colors_pr = ["#94a3b8", "#94a3b8", "#0284c7", "#0284c7", "#16a34a", "#d97706", "#7c3aed"]
    y_pos = np.arange(len(ab_df))

    bars1 = ax1.barh(y_pos, ab_df["pr_auc"], color=colors_pr, height=0.62)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(ab_df["short"], fontsize=9.5)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1.15)
    ax1.set_title("PR-AUC (Test Set)", fontsize=11.5, fontweight="bold")
    ax1.grid(True, axis="x")
    for b, val in zip(bars1, ab_df["pr_auc"]):
        ax1.text(val + 0.015, b.get_y() + b.get_height() / 2, f"{val:.4f}", va="center", fontsize=8.8, color="#0f172a", fontweight="bold")

    bars2 = ax2.barh(y_pos, ab_df["net_savings"] / 1e6, color=colors_pr, height=0.62)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([""] * len(ab_df))
    ax2.invert_yaxis()
    ax2.set_xlim(0, 3.4)
    ax2.set_title("Чистая экономия (млн)", fontsize=11.5, fontweight="bold")
    ax2.grid(True, axis="x")
    for b, val in zip(bars2, ab_df["net_savings"] / 1e6):
        ax2.text(val + 0.05, b.get_y() + b.get_height() / 2, f"{val:.2f}M", va="center", fontsize=8.8, color="#0f172a", fontweight="bold")

    fig.tight_layout()
    fig.savefig(ASSETS / "ablation_comparison.png", dpi=180)
    plt.close(fig)

    # 3. PSI Drift Chart
    psi_df = pd.DataFrame(meta["train_vs_test_drift"]["features"]).head(8)
    fig, ax = plt.subplots(figsize=(7.6, 3.9), dpi=180)
    bar_colors = [
        "#dc2626" if s == "CRITICAL" else ("#d97706" if s == "WARNING" else "#16a34a")
        for s in psi_df["status"]
    ]
    bars = ax.barh(psi_df["feature"], psi_df["psi"], color=bar_colors, height=0.6)
    ax.invert_yaxis()
    ax.axvline(0.10, color="#d97706", linestyle="--", linewidth=1.4, label="Warning (0.10)")
    ax.axvline(0.25, color="#dc2626", linestyle="--", linewidth=1.4, label="Critical (0.25)")
    ax.set_title("Мониторинг дрейфа данных (PSI: Train vs Test)", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Population Stability Index (PSI)")
    ax.set_xlim(0, max(psi_df["psi"].max() * 1.18, 0.35))
    ax.grid(True, axis="x")
    ax.legend(facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9)
    for b, val in zip(bars, psi_df["psi"]):
        ax.text(val + 0.03, b.get_y() + b.get_height() / 2, f"{val:.3f}", va="center", fontsize=8.8, color="#0f172a")
    fig.tight_layout()
    fig.savefig(ASSETS / "psi_drift.png", dpi=180)
    plt.close(fig)

    # 4. SHAP Local Explanation Example on a real test fraud case
    feat_df = pd.read_parquet(ROOT / "data" / "processed" / "features.parquet")
    _, _, test_df = time_based_split(feat_df)
    gbdt = joblib.load(ROOT / "models" / "artifacts" / "gbdt_model.joblib")
    bg = joblib.load(ROOT / "models" / "artifacts" / "background_median.joblib")
    explainer = FraudExplainer(gbdt, ALL_FEATURE_NAMES, bg)

    # Pick a clear high-risk fraud case
    fraud_rows = test_df[test_df["isFraud"] == 1]
    sample = fraud_rows.iloc[0]
    factors = explainer.explain_single(sample[ALL_FEATURE_NAMES].to_dict(), top_k=6)
    f_df = pd.DataFrame(factors).sort_values("shap_value")

    fig, ax = plt.subplots(figsize=(7.0, 3.8), dpi=180)
    f_colors = ["#dc2626" if v >= 0 else "#16a34a" for v in f_df["shap_value"]]
    ax.barh(f_df["feature"], f_df["shap_value"], color=f_colors, height=0.58)
    ax.axvline(0, color="#64748b", linewidth=1.0)
    ax.set_title(f"Пример локального SHAP-объяснения ({sample['tx_id']})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Вклад признака в риск мошенничества (SHAP)")
    ax.grid(True, axis="x")
    fig.tight_layout()
    fig.savefig(ASSETS / "shap_case_example.png", dpi=180)
    plt.close(fig)

    print("Generated 4 presentation charts in", ASSETS)


if __name__ == "__main__":
    make_charts()
