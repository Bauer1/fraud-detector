"""Leak-free chronological batch feature engineering pipeline."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.features.online_store import ALL_FEATURE_NAMES, OnlineFeatureStore


def build_features_chronological(
    df: pd.DataFrame,
    mule_in_degree_threshold: int = 5,
    return_store: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, OnlineFeatureStore]:
    """Compute all transactional, behavioral, deviation, and graph features strictly chronologically.

    Input DataFrame is sorted by timestamp (or step) first to guarantee zero future leakage.
    """
    sort_col = "timestamp" if "timestamp" in df.columns else "step"
    df_sorted = df.sort_values(sort_col, kind="mergesort").reset_index(drop=True)

    store = OnlineFeatureStore(mule_in_degree_threshold=mule_in_degree_threshold)
    feature_rows: list[dict[str, float]] = []

    records = df_sorted.to_dict(orient="records")
    for tx in records:
        feats = store.compute_features(tx, update_state=True)
        feature_rows.append(feats)

    feat_df = pd.DataFrame(feature_rows, columns=ALL_FEATURE_NAMES)

    # Preserve metadata and target columns alongside features
    meta_cols = [c for c in ["tx_id", "step", "timestamp", "type", "nameOrig", "nameDest", "device_id", "channel", "ip_country", "isFraud", "fraud_scenario"] if c in df_sorted.columns]
    result = pd.concat([df_sorted[meta_cols].reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)

    if return_store:
        return result, store
    return result


def time_based_split(
    df: pd.DataFrame,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame chronologically into train / validation / test sets without shuffling."""
    sort_col = "timestamp" if "timestamp" in df.columns else "step"
    df_sorted = df.sort_values(sort_col, kind="mergesort").reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return (
        df_sorted.iloc[:train_end].copy(),
        df_sorted.iloc[train_end:val_end].copy(),
        df_sorted.iloc[val_end:].copy(),
    )
