"""Dedicated profiler, manifest builder, and loader for Original PaySim CSV vs Enriched Project Simulation.

Maintains strict separation between the two data sources required by the Technical Specification:
1. `paysim_csv` ("исходный PaySim CSV"): `data/paysim dataset.csv`
   - Raw columns: `step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud`.
   - Does NOT contain native `tx_id`, wall-clock `timestamp`, `device_id`, `channel`, or `ip_country`.
   - Assigns deterministic observation ID `PAYSIM_{source_row_idx:08d}` and documents equal-step ordering rule:
     `(step ASC, source_row_idx ASC)`.
   - Keeps `isFlaggedFraud` strictly as an offline benchmark column, never as a model input.
2. `enriched_simulation` ("обогащённая симуляция проекта"): `src/data/simulator.py`
   - Synthetic multi-scenario dataset with explicit `tx_id`, UTC `timestamp`, `device_id`, `channel`, `ip_country`.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path

PAYSIM_EXPECTED_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]


def compute_file_sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Stream-compute SHA-256 checksum without loading the entire file into RAM."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_bytes)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def profile_paysim_csv(
    csv_path: str | Path | None = None,
    manifest_out_path: str | Path | None = None,
    chunksize: int = 1_000_000,
    force_recompute: bool = False,
) -> dict[str, Any]:
    """Profile the complete `data/paysim dataset.csv` file in chunks and write a reproducible Data Manifest."""
    cfg = load_config()
    path = resolve_path(csv_path or cfg["paths"].get("paysim_csv", "data/paysim dataset.csv"))
    if not path.exists():
        raise FileNotFoundError(f"PaySim CSV not found at {path}")

    manifests_dir = resolve_path(cfg["paths"].get("manifests_dir", "models/manifests"))
    manifests_dir.mkdir(parents=True, exist_ok=True)
    out_path = resolve_path(manifest_out_path) if manifest_out_path else (manifests_dir / "paysim_csv_manifest.json")

    file_size = path.stat().st_size
    sha256 = compute_file_sha256(path)

    if out_path.exists() and not force_recompute:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if existing.get("file_sha256") == sha256 and existing.get("file_size_bytes") == file_size:
            return existing

    total_rows = 0
    fraud_count = 0
    flagged_count = 0
    flagged_and_fraud = 0
    null_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    fraud_by_type: dict[str, int] = {}
    min_step: int | None = None
    max_step: int | None = None
    non_monotonic_step_transitions = 0
    prev_last_step: int | None = None

    amt_min = float("inf")
    amt_max = 0.0
    amt_sum = 0.0
    negative_amounts = 0
    zero_amounts = 0
    exact_chunk_duplicates = 0
    post_orig_balance_changed = 0
    post_dest_balance_changed = 0
    schema_cols: list[str] = []
    dtypes_map: dict[str, str] = {}

    for chunk in pd.read_csv(path, chunksize=chunksize):
        if not schema_cols:
            schema_cols = list(chunk.columns)
            dtypes_map = {col: str(dtype) for col, dtype in chunk.dtypes.items()}
            missing_expected = [c for c in PAYSIM_EXPECTED_COLUMNS if c not in schema_cols]
            if missing_expected:
                raise ValueError(f"PaySim CSV missing expected columns: {missing_expected}")
            null_counts = {col: 0 for col in schema_cols}

        total_rows += len(chunk)
        for col in schema_cols:
            null_counts[col] += int(chunk[col].isna().sum())

        c_fraud = chunk["isFraud"].astype(int)
        c_flagged = chunk["isFlaggedFraud"].astype(int)
        fraud_count += int(c_fraud.sum())
        flagged_count += int(c_flagged.sum())
        flagged_and_fraud += int(((c_fraud == 1) & (c_flagged == 1)).sum())

        for t_name, cnt in chunk["type"].value_counts().items():
            type_counts[str(t_name)] = type_counts.get(str(t_name), 0) + int(cnt)
        for t_name, cnt in chunk.loc[c_fraud == 1, "type"].value_counts().items():
            fraud_by_type[str(t_name)] = fraud_by_type.get(str(t_name), 0) + int(cnt)

        s_min = int(chunk["step"].min())
        s_max = int(chunk["step"].max())
        min_step = s_min if min_step is None else min(min_step, s_min)
        max_step = s_max if max_step is None else max(max_step, s_max)

        step_arr = chunk["step"].to_numpy()
        if prev_last_step is not None and len(step_arr) > 0 and int(step_arr[0]) < prev_last_step:
            non_monotonic_step_transitions += 1
        if len(step_arr) > 1:
            non_monotonic_step_transitions += int((np.diff(step_arr) < 0).sum())
        if len(step_arr) > 0:
            prev_last_step = int(step_arr[-1])

        amt = chunk["amount"].to_numpy(dtype=float)
        amt_min = min(amt_min, float(np.nanmin(amt)))
        amt_max = max(amt_max, float(np.nanmax(amt)))
        amt_sum += float(np.nansum(amt))
        negative_amounts += int((amt < 0).sum())
        zero_amounts += int((amt == 0).sum())

        exact_chunk_duplicates += int(chunk.duplicated().sum())
        post_orig_balance_changed += int((chunk["newbalanceOrig"] != chunk["oldbalanceOrg"]).sum())
        post_dest_balance_changed += int((chunk["newbalanceDest"] != chunk["oldbalanceDest"]).sum())

    manifest: dict[str, Any] = {
        "dataset_id": "paysim_csv",
        "dataset_title_ru": "Исходный PaySim CSV",
        "source_origin": "Kaggle PaySim1 synthetic mobile money dataset (local file data/paysim dataset.csv)",
        "profiled_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_path": "data/paysim dataset.csv",
        "file_size_bytes": int(file_size),
        "file_sha256": sha256,
        "schema": {
            "columns": schema_cols,
            "dtypes": dtypes_map,
            "absent_telemetry_columns": ["tx_id", "timestamp", "device_id", "channel", "ip_country"],
        },
        "row_count": int(total_rows),
        "class_balance": {
            "fraud_count": int(fraud_count),
            "legitimate_count": int(total_rows - fraud_count),
            "fraud_rate": round(fraud_count / max(total_rows, 1), 6),
            "isFlaggedFraud_count": int(flagged_count),
            "isFlaggedFraud_true_positives": int(flagged_and_fraud),
        },
        "type_distribution": type_counts,
        "fraud_by_type": fraud_by_type,
        "temporal_coverage": {
            "unit": "step (1 step = 1 simulated hour; no absolute calendar timestamp in CSV)",
            "min_step": int(min_step or 0),
            "max_step": int(max_step or 0),
            "span_hours": int((max_step or 0) - (min_step or 0) + 1),
            "approx_days": round(((max_step or 0) - (min_step or 0) + 1) / 24.0, 2),
            "non_monotonic_step_transitions": int(non_monotonic_step_transitions),
            "equal_step_ordering_rule": "Deterministic tie-break by (step ASC, source_row_idx ASC) via generated ID PAYSIM_{row_idx:08d}",
        },
        "data_quality": {
            "null_counts": null_counts,
            "within_chunk_exact_duplicates": int(exact_chunk_duplicates),
            "negative_amounts": int(negative_amounts),
            "zero_amounts": int(zero_amounts),
            "amount_min": round(amt_min if amt_min != float("inf") else 0.0, 2),
            "amount_max": round(amt_max, 2),
            "amount_mean": round(amt_sum / max(total_rows, 1), 2),
        },
        "feature_availability_and_leakage_notes": [
            "CSV labels `isFraud` are synthetic (PaySim simulator) and do not prove real bank transferability.",
            "`isFlaggedFraud` is PaySim's built-in rule output and is strictly forbidden as a model input.",
            "`newbalanceOrig` and `newbalanceDest` reflect post-settlement state (`post_orig_balance_changed="
            f"{post_orig_balance_changed}`, `post_dest_balance_changed={post_dest_balance_changed}`) and are excluded by default under the pre-decision contract.",
            "Raw `nameOrig` and `nameDest` are pseudonymous entity keys used strictly for point-in-time aggregates, never fed to ML as raw IDs.",
            "Zero-amount rows (`zero_amounts` in PaySim) are filtered/logged during validation because transfer amount must be > 0.",
        ],
    }
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def profile_enriched_simulation(
    df: pd.DataFrame,
    parquet_path: str | Path | None = None,
    manifest_out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a reproducible Data Manifest for the project's Enriched Synthetic Simulation dataset."""
    cfg = load_config()
    manifests_dir = resolve_path(cfg["paths"].get("manifests_dir", "models/manifests"))
    manifests_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        resolve_path(manifest_out_path)
        if manifest_out_path
        else (manifests_dir / "enriched_simulation_manifest.json")
    )

    p_path = resolve_path(parquet_path or cfg["paths"]["raw_data"])
    sha256 = compute_file_sha256(p_path) if p_path.exists() else None
    file_size = int(p_path.stat().st_size) if p_path.exists() else None

    total_rows = len(df)
    fraud_count = int(df["isFraud"].sum()) if "isFraud" in df.columns else 0
    dup_tx_id = int(df["tx_id"].duplicated().sum()) if "tx_id" in df.columns else 0

    manifest: dict[str, Any] = {
        "dataset_id": "enriched_simulation",
        "dataset_title_ru": "Обогащённая симуляция проекта",
        "source_origin": "Project synthetic generator (`src/data/simulator.py`) with 5 injected fraud scenarios and concept drift",
        "profiled_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_path": str(cfg["paths"]["raw_data"]),
        "file_size_bytes": file_size,
        "file_sha256": sha256,
        "schema": {
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "synthetic_enriched_columns": ["tx_id", "timestamp", "device_id", "channel", "ip_country", "fraud_scenario"],
        },
        "row_count": int(total_rows),
        "class_balance": {
            "fraud_count": fraud_count,
            "legitimate_count": int(total_rows - fraud_count),
            "fraud_rate": round(fraud_count / max(total_rows, 1), 6),
            "scenario_counts": df["fraud_scenario"].value_counts().to_dict() if "fraud_scenario" in df.columns else {},
        },
        "type_distribution": df["type"].value_counts().to_dict() if "type" in df.columns else {},
        "temporal_coverage": {
            "min_step": int(df["step"].min()) if "step" in df.columns and not df.empty else 0,
            "max_step": int(df["step"].max()) if "step" in df.columns and not df.empty else 0,
            "min_timestamp_utc": str(df["timestamp"].min()) if "timestamp" in df.columns and not df.empty else None,
            "max_timestamp_utc": str(df["timestamp"].max()) if "timestamp" in df.columns and not df.empty else None,
            "ordering_rule": "Deterministic sort by (UTC timestamp ASC, tx_id ASC)",
        },
        "data_quality": {
            "null_counts": {col: int(df[col].isna().sum()) for col in df.columns},
            "duplicate_tx_ids": dup_tx_id,
            "amount_min": round(float(df["amount"].min()), 2) if not df.empty else 0.0,
            "amount_max": round(float(df["amount"].max()), 2) if not df.empty else 0.0,
            "amount_mean": round(float(df["amount"].mean()), 2) if not df.empty else 0.0,
        },
        "feature_availability_and_leakage_notes": [
            "Synthetic dataset created for end-to-end engineering, ablation, and drift smoke-testing.",
            "Metrics on this simulation are prototype benchmark results and must NEVER be presented as production bank performance.",
            "Post-transaction balances (`newbalanceOrig`, `newbalanceDest`) are excluded in `pre_decision` mode.",
        ],
    }
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def load_paysim_csv_dataset(
    csv_path: str | Path | None = None,
    *,
    sampling_mode: str = "chronological_stride",
    max_rows: int = 120_000,
    chunksize: int = 500_000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and validate observations from Original PaySim CSV without fabricating absent telemetry.

    Parameters
    ----------
    sampling_mode:
        - `"chronological_stride"`: deterministic uniform row stride across the full 6.36M-row CSV
          so all 743 hourly steps (31 days) and natural ~0.13% class balance are represented.
        - `"chronological_prefix"`: first `max_rows` valid rows from step 1 onward.
        - `"full"`: load all valid rows from the CSV.
    """
    cfg = load_config()
    path = resolve_path(csv_path or cfg["paths"].get("paysim_csv", "data/paysim dataset.csv"))
    if not path.exists():
        raise FileNotFoundError(f"PaySim CSV not found at {path}")

    manifest = profile_paysim_csv(csv_path=path, force_recompute=False)
    total_csv_rows = int(manifest["row_count"])

    if sampling_mode == "full" or max_rows <= 0 or max_rows >= total_csv_rows:
        stride = 1
    elif sampling_mode == "chronological_stride":
        stride = max(1, total_csv_rows // max_rows)
    elif sampling_mode == "chronological_prefix":
        stride = 1
    else:
        raise ValueError(f"Unsupported PaySim sampling_mode='{sampling_mode}'")

    chunks_collected: list[pd.DataFrame] = []
    global_row_offset = 0
    dropped_non_positive_amount = 0
    collected_rows = 0

    for chunk in pd.read_csv(path, chunksize=chunksize):
        n_chunk = len(chunk)
        chunk = chunk.copy()
        chunk["source_row_idx"] = np.arange(global_row_offset, global_row_offset + n_chunk, dtype=np.int64)
        global_row_offset += n_chunk

        # Documented validation rule: transfer attempt must have strictly positive finite amount
        invalid_amt = chunk["amount"].isna() | (chunk["amount"] <= 0.0)
        dropped_non_positive_amount += int(invalid_amt.sum())
        chunk = chunk.loc[~invalid_amt]

        if sampling_mode == "chronological_stride" and stride > 1:
            chunk = chunk.loc[(chunk["source_row_idx"] % stride) == 0]
        elif sampling_mode == "chronological_prefix":
            rem = max_rows - collected_rows
            if rem <= 0:
                break
            chunk = chunk.iloc[:rem]

        if not chunk.empty:
            chunk["tx_id"] = [f"PAYSIM_{idx:08d}" for idx in chunk["source_row_idx"]]
            # Do NOT fabricate timestamp, device_id, channel, or ip_country — keep None/absent!
            chunks_collected.append(chunk)
            collected_rows += len(chunk)

        if sampling_mode == "chronological_prefix" and collected_rows >= max_rows:
            break

    df = pd.concat(chunks_collected, ignore_index=True) if chunks_collected else pd.DataFrame()
    if not df.empty:
        # Explicit deterministic ordering rule for equal `step`: (step ASC, source_row_idx ASC)
        df = df.sort_values(["step", "source_row_idx"], kind="mergesort").reset_index(drop=True)
        if sampling_mode == "chronological_stride" and max_rows > 0 and len(df) > max_rows:
            df = df.iloc[:max_rows].copy()

    load_audit = {
        "dataset_id": "paysim_csv",
        "sampling_mode": sampling_mode,
        "stride": stride,
        "source_total_rows": total_csv_rows,
        "loaded_rows": int(len(df)),
        "dropped_non_positive_amount_rows": int(dropped_non_positive_amount),
        "equal_step_tie_breaker": "(step ASC, source_row_idx ASC)",
        "min_step": int(df["step"].min()) if not df.empty else 0,
        "max_step": int(df["step"].max()) if not df.empty else 0,
        "fraud_count": int(df["isFraud"].sum()) if not df.empty else 0,
        "fraud_rate": round(float(df["isFraud"].mean()), 6) if not df.empty else 0.0,
        "isFlaggedFraud_count": int(df["isFlaggedFraud"].sum()) if not df.empty else 0,
    }
    return df, load_audit
