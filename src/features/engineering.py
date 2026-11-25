"""Leak-free chronological batch feature engineering and deterministic time splitting."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from src.features.online_store import (
    ALL_FEATURE_NAMES,
    DuplicateTransactionConflictError,
    OnlineFeatureStore,
    canonical_tx_fingerprint,
    normalize_event_datetime,
)


def sort_transactions_deterministically(df: pd.DataFrame, *, deduplicate_retries: bool = True) -> pd.DataFrame:
    """Sort transactions chronologically by UTC timestamp (or step) with deterministic `tx_id` tie-breaking.

    If `deduplicate_retries=True` and `tx_id` is present:
    - Exact retry rows (identical `tx_id` and canonical transaction fields) keep only the first occurrence.
    - Conflicting rows sharing `tx_id` with different canonical transaction payloads raise
      `DuplicateTransactionConflictError`.
    """
    if df.empty:
        return df.copy()

    records = df.to_dict(orient="records")
    has_full_tx_fields = {"type", "amount", "nameOrig", "nameDest"}.issubset(df.columns)

    sort_ts: list[float] = []
    tie_ids: list[str] = []
    for idx, rec in enumerate(records):
        if "timestamp" in df.columns or "step" in df.columns:
            ts_val = normalize_event_datetime(rec).timestamp()
        else:
            ts_val = float(idx)
        sort_ts.append(ts_val)
        tx_id_val = rec.get("tx_id")
        tie_ids.append(str(tx_id_val) if tx_id_val is not None else f"__row_{idx:09d}")

    out = df.copy()
    out["__sort_ts_utc"] = sort_ts
    out["__sort_tie_id"] = tie_ids
    out = out.sort_values(["__sort_ts_utc", "__sort_tie_id"], kind="mergesort").reset_index(drop=True)

    if deduplicate_retries and "tx_id" in out.columns and has_full_tx_fields:
        seen_fp: dict[str, tuple[Any, ...]] = {}
        keep_mask: list[bool] = []
        for rec in out.to_dict(orient="records"):
            tx_id = str(rec["tx_id"])
            fp = canonical_tx_fingerprint(rec)
            if tx_id in seen_fp:
                if seen_fp[tx_id] != fp:
                    raise DuplicateTransactionConflictError(
                        f"Conflicting duplicate tx_id='{tx_id}' found in batch dataset"
                    )
                keep_mask.append(False)
            else:
                seen_fp[tx_id] = fp
                keep_mask.append(True)
        out = out.loc[keep_mask].reset_index(drop=True)

    return out.drop(columns=["__sort_ts_utc", "__sort_tie_id"])


def build_features_chronological(
    df: pd.DataFrame,
    mule_in_degree_threshold: int = 5,
    return_store: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, OnlineFeatureStore]:
    """Compute all transactional, behavioral, deviation, and graph features strictly chronologically.

    Input DataFrame is sorted deterministically by (UTC timestamp, tx_id) first to guarantee
    zero future leakage and reproducible tie-breaking for simultaneous events.
    """
    df_sorted = sort_transactions_deterministically(df, deduplicate_retries=True)

    store = OnlineFeatureStore(
        mule_in_degree_threshold=mule_in_degree_threshold,
        max_sender_out_of_order_sec=0.0,
        max_global_staleness_sec=None,
        state_origin="batch_offline_full_dataset",
    )
    feature_rows: list[dict[str, float]] = []

    records = df_sorted.to_dict(orient="records")
    for tx in records:
        feats = store.compute_features(tx, update_state=True)
        feature_rows.append(feats)

    feat_df = pd.DataFrame(feature_rows, columns=ALL_FEATURE_NAMES)

    meta_cols = [
        c
        for c in [
            "tx_id",
            "step",
            "timestamp",
            "type",
            "nameOrig",
            "nameDest",
            "device_id",
            "channel",
            "ip_country",
            "isFraud",
            "fraud_scenario",
        ]
        if c in df_sorted.columns
    ]
    result = pd.concat([df_sorted[meta_cols].reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)

    if return_store:
        return result, store
    return result


def build_initial_feature_store(
    history_df: pd.DataFrame,
    *,
    as_of_timestamp: datetime | str,
    mule_in_degree_threshold: int = 5,
    max_sender_out_of_order_sec: float | None = 300.0,
    max_global_staleness_sec: float | None = 86400.0,
    state_origin: str = "pre_test_warmup",
) -> OnlineFeatureStore:
    """Build a point-in-time OnlineFeatureStore strictly from historical events <= `as_of_timestamp`."""
    return OnlineFeatureStore.from_history(
        history_df,
        as_of_timestamp=as_of_timestamp,
        mule_in_degree_threshold=mule_in_degree_threshold,
        max_sender_out_of_order_sec=max_sender_out_of_order_sec,
        max_global_staleness_sec=max_global_staleness_sec,
        state_origin=state_origin,
    )


def time_based_split(
    df: pd.DataFrame,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame chronologically into train / validation / test sets without shuffling."""
    df_sorted = sort_transactions_deterministically(df, deduplicate_retries=False)
    n = len(df_sorted)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return (
        df_sorted.iloc[:train_end].copy(),
        df_sorted.iloc[train_end:val_end].copy(),
        df_sorted.iloc[val_end:].copy(),
    )
