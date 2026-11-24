"""PaySim-compatible realistic transaction simulator with injected fraud scenarios.

Generates chronological mobile/bank transfer data with:
1. Normal customer behavior (consistent amount distributions, preferred devices, regular recipients, daytime activity).
2. Five realistic fraud scenarios:
   - account_takeover: new device, night hours, large transfer draining balance.
   - velocity_burst: rapid sequence of transfers within 1 hour.
   - mule_chain: multiple compromised senders funnelling funds to shared mule/dropper recipients & shared devices.
   - anomalous_amount: sudden high-z-score transfer to a brand-new recipient.
   - novel_micro_smurfing (injected mainly in the late/test period): structured medium-large transfers via API/WEB
     to test concept drift and Isolation Forest anomaly detection.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path


def generate_transactions(
    n_customers: int = 600,
    n_merchants: int = 250,
    n_transactions: int = 30000,
    n_days: int = 30,
    fraud_rate: float = 0.015,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate a chronological PaySim-compatible DataFrame with rich behavioral & graph patterns."""
    rng = np.random.default_rng(random_seed)

    # Customer profiles
    customers = [f"C{100000 + i}" for i in range(n_customers)]
    merchants = [f"M{200000 + i}" for i in range(n_merchants)]
    mules = [f"C_MULE_{i:03d}" for i in range(18)]

    cust_mean_log_amt = rng.uniform(5.2, 8.2, size=n_customers)  # ~180 to ~3600 typical mean
    cust_std_log_amt = rng.uniform(0.35, 0.75, size=n_customers)
    cust_balance = np.exp(cust_mean_log_amt) * rng.uniform(5.0, 25.0, size=n_customers)
    cust_primary_device = [f"DEV_{i:05d}" for i in range(n_customers)]
    cust_preferred_channel = rng.choice(["MOBILE", "WEB", "ATM"], size=n_customers, p=[0.72, 0.22, 0.06])

    # Pre-assign regular recipients (3 to 8 known recipients per customer)
    all_normal_dests = customers + merchants
    cust_known_dests: list[list[str]] = []
    for i in range(n_customers):
        k = int(rng.integers(3, 9))
        choices = rng.choice(all_normal_dests, size=k, replace=False).tolist()
        choices = [d for d in choices if d != customers[i]]
        cust_known_dests.append(choices if choices else [merchants[0]])

    # Track recipient balances
    dest_balances: dict[str, float] = {d: float(rng.uniform(500.0, 50000.0)) for d in all_normal_dests + mules}
    # Add blacklisted entities occasionally
    blacklisted_dest = "M_BLACKLIST_999"
    dest_balances[blacklisted_dest] = 0.0

    start_time = datetime(2026, 1, 1, 0, 0, 0)
    total_hours = n_days * 24

    n_fraud_target = max(50, int(n_transactions * fraud_rate))
    n_normal = n_transactions - n_fraud_target

    # Generate normal timestamps (strong daytime preference, lower at night 0-5h)
    hour_probs = np.array([
        0.012, 0.008, 0.006, 0.006, 0.008, 0.015,
        0.030, 0.045, 0.060, 0.068, 0.072, 0.074,
        0.072, 0.068, 0.066, 0.065, 0.064, 0.062,
        0.056, 0.048, 0.038, 0.028, 0.018, 0.011,
    ])
    hour_probs = hour_probs / hour_probs.sum()

    normal_days = rng.integers(0, n_days, size=n_normal)
    normal_hours = rng.choice(24, size=n_normal, p=hour_probs)
    normal_minutes = rng.integers(0, 60, size=n_normal)
    normal_seconds = rng.integers(0, 60, size=n_normal)
    normal_steps = normal_days * 24 + normal_hours
    normal_ts_offsets = normal_steps * 3600 + normal_minutes * 60 + normal_seconds

    records: list[dict[str, Any]] = []

    # Build normal transaction drafts (sorted later before balance replay)
    tx_types_normal = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
    tx_type_probs = [0.35, 0.28, 0.20, 0.12, 0.05]

    normal_cust_idx = rng.integers(0, n_customers, size=n_normal)
    normal_types = rng.choice(tx_types_normal, size=n_normal, p=tx_type_probs)

    for idx in range(n_normal):
        c_idx = int(normal_cust_idx[idx])
        tx_type = str(normal_types[idx])
        amt = float(np.round(np.exp(rng.normal(cust_mean_log_amt[c_idx], cust_std_log_amt[c_idx])), 2))
        amt = float(np.clip(amt, 5.0, 120000.0))

        # Inject ~4% realistic "hard negatives" (legitimate large transfers: rent, travel, savings move)
        is_hard_neg = rng.random() < 0.04
        force_drain_legit = False
        if is_hard_neg and tx_type in ("TRANSFER", "CASH_OUT"):
            amt = float(np.round(amt * rng.uniform(3.5, 11.0), 2))
            amt = float(np.clip(amt, 5000.0, 220000.0))
            force_drain_legit = rng.random() < 0.45

        # 82% known recipient, 18% new recipient (hard negatives more often go to new recipients)
        new_rec_prob = 0.45 if is_hard_neg else 0.16
        if rng.random() >= new_rec_prob:
            dest = str(rng.choice(cust_known_dests[c_idx]))
        else:
            dest = str(rng.choice(all_normal_dests))
            if dest == customers[c_idx]:
                dest = merchants[0]

        # 92% primary device, 8% secondary/new device
        device = cust_primary_device[c_idx] if rng.random() < 0.92 else f"DEV_ALT_{c_idx}_{rng.integers(1, 4)}"
        channel = str(cust_preferred_channel[c_idx]) if rng.random() < 0.90 else str(rng.choice(["MOBILE", "WEB", "ATM", "API"], p=[0.4, 0.35, 0.15, 0.10]))
        ip_country = "RU" if rng.random() < 0.96 else str(rng.choice(["KZ", "BY", "TR", "AE", "DE"]))

        records.append({
            "offset_sec": int(normal_ts_offsets[idx]),
            "step": int(normal_steps[idx]),
            "type": tx_type,
            "amount": amt,
            "cust_idx": c_idx,
            "nameOrig": customers[c_idx],
            "nameDest": dest,
            "device_id": device,
            "channel": channel,
            "ip_country": ip_country,
            "isFraud": 0,
            "fraud_scenario": "legitimate",
            "force_drain": force_drain_legit,
        })

    # Inject fraud scenarios (with realistic overlap & stealthy partial-drain cases)
    fraud_count = 0
    shared_fraud_devices = [f"DEV_FRAUD_RING_{i:02d}" for i in range(6)]

    # 1. Account Takeover (~30% of fraud)
    n_ato = int(n_fraud_target * 0.30)
    for _ in range(n_ato):
        c_idx = int(rng.integers(0, n_customers))
        day = int(rng.integers(2, n_days))
        hour = int(rng.choice([0, 1, 2, 3, 4, 5, 11, 15, 22, 23], p=[0.14, 0.14, 0.14, 0.12, 0.10, 0.08, 0.06, 0.06, 0.08, 0.08]))
        offset = (day * 24 + hour) * 3600 + int(rng.integers(0, 3600))
        amt = float(np.round(np.exp(cust_mean_log_amt[c_idx]) * rng.uniform(3.0, 14.0), 2))
        amt = float(np.clip(amt, 4000.0, 260000.0))
        dest = str(rng.choice(mules)) if rng.random() < 0.45 else f"C_EXT_{rng.integers(1000, 9999)}"
        records.append({
            "offset_sec": offset,
            "step": day * 24 + hour,
            "type": str(rng.choice(["TRANSFER", "CASH_OUT"])),
            "amount": amt,
            "cust_idx": c_idx,
            "nameOrig": customers[c_idx],
            "nameDest": dest,
            "device_id": f"DEV_ATO_{rng.integers(10000, 99999)}" if rng.random() < 0.80 else cust_primary_device[c_idx],
            "channel": str(rng.choice(["WEB", "MOBILE"])),
            "ip_country": str(rng.choice(["NL", "DE", "TR", "NG", "RU"], p=[0.20, 0.15, 0.15, 0.15, 0.35])),
            "isFraud": 1,
            "fraud_scenario": "account_takeover",
            "force_drain": rng.random() < 0.60,
        })
        fraud_count += 1

    # 2. Rapid Velocity Burst (~22% of fraud, bursts of 3-6 tx within 45 min)
    n_vel_target = int(n_fraud_target * 0.22)
    vel_added = 0
    while vel_added < n_vel_target:
        c_idx = int(rng.integers(0, n_customers))
        day = int(rng.integers(2, n_days))
        hour = int(rng.integers(0, 24))
        base_offset = (day * 24 + hour) * 3600 + int(rng.integers(0, 600))
        burst_len = int(rng.integers(3, 7))
        burst_dev = f"DEV_VEL_{rng.integers(1000, 9999)}" if rng.random() < 0.65 else cust_primary_device[c_idx]
        for b in range(burst_len):
            if vel_added >= n_vel_target:
                break
            off = base_offset + b * int(rng.integers(60, 480))
            amt = float(np.round(np.exp(cust_mean_log_amt[c_idx]) * rng.uniform(1.6, 5.0), 2))
            amt = float(np.clip(amt, 1800.0, 85000.0))
            records.append({
                "offset_sec": off,
                "step": off // 3600,
                "type": str(rng.choice(["TRANSFER", "CASH_OUT"])),
                "amount": amt,
                "cust_idx": c_idx,
                "nameOrig": customers[c_idx],
                "nameDest": str(rng.choice(mules)) if rng.random() < 0.4 else f"C_EXT_{rng.integers(1000, 9999)}",
                "device_id": burst_dev,
                "channel": "MOBILE",
                "ip_country": str(rng.choice(["RU", "TR", "CY"], p=[0.65, 0.20, 0.15])),
                "isFraud": 1,
                "fraud_scenario": "velocity_burst",
                "force_drain": b == burst_len - 1 and rng.random() < 0.5,
            })
            vel_added += 1
            fraud_count += 1

    # 3. Mule / Dropper Chain (~23% of fraud: many senders -> shared mule & shared ring device)
    n_mule = int(n_fraud_target * 0.23)
    for _ in range(n_mule):
        c_idx = int(rng.integers(0, n_customers))
        day = int(rng.integers(3, n_days))
        hour = int(rng.integers(6, 23))
        offset = (day * 24 + hour) * 3600 + int(rng.integers(0, 3600))
        mule_dest = str(rng.choice(mules[:8]))
        ring_dev = str(rng.choice(shared_fraud_devices)) if rng.random() < 0.70 else cust_primary_device[c_idx]
        amt = float(np.round(np.exp(cust_mean_log_amt[c_idx]) * rng.uniform(1.8, 7.5), 2))
        amt = float(np.clip(amt, 2500.0, 150000.0))
        records.append({
            "offset_sec": offset,
            "step": day * 24 + hour,
            "type": "TRANSFER",
            "amount": amt,
            "cust_idx": c_idx,
            "nameOrig": customers[c_idx],
            "nameDest": mule_dest,
            "device_id": ring_dev,
            "channel": str(rng.choice(["MOBILE", "WEB"])),
            "ip_country": "RU",
            "isFraud": 1,
            "fraud_scenario": "mule_chain",
            "force_drain": rng.random() < 0.35,
        })
        fraud_count += 1

    # 4. Anomalous Amount to New Recipient (~15% of fraud)
    n_anom = int(n_fraud_target * 0.15)
    for _ in range(n_anom):
        c_idx = int(rng.integers(0, n_customers))
        day = int(rng.integers(2, n_days))
        hour = int(rng.integers(7, 22))
        offset = (day * 24 + hour) * 3600 + int(rng.integers(0, 3600))
        amt = float(np.round(np.exp(cust_mean_log_amt[c_idx]) * rng.uniform(5.0, 18.0), 2))
        amt = float(np.clip(amt, 10000.0, 290000.0))
        dest = blacklisted_dest if rng.random() < 0.10 else f"C_NEW_{rng.integers(10000, 99999)}"
        records.append({
            "offset_sec": offset,
            "step": day * 24 + hour,
            "type": str(rng.choice(["TRANSFER", "CASH_OUT"])),
            "amount": amt,
            "cust_idx": c_idx,
            "nameOrig": customers[c_idx],
            "nameDest": dest,
            "device_id": cust_primary_device[c_idx] if rng.random() < 0.55 else f"DEV_NEW_{rng.integers(1000, 9999)}",
            "channel": "WEB",
            "ip_country": str(rng.choice(["RU", "AE", "HK"], p=[0.6, 0.2, 0.2])),
            "isFraud": 1,
            "fraud_scenario": "anomalous_amount",
            "force_drain": rng.random() < 0.45,
        })
        fraud_count += 1

    # 5. Novel Late-Period Pattern (Concept Drift / Unseen Scheme, ~10% of fraud, concentrated in last 20% of days)
    n_novel = max(10, n_fraud_target - fraud_count)
    late_start_day = int(n_days * 0.80)
    for _ in range(n_novel):
        c_idx = int(rng.integers(0, n_customers))
        day = int(rng.integers(late_start_day, n_days))
        hour = int(rng.integers(1, 7))
        offset = (day * 24 + hour) * 3600 + int(rng.integers(0, 3600))
        amt = float(np.round(rng.uniform(18000.0, 88000.0), 2))
        records.append({
            "offset_sec": offset,
            "step": day * 24 + hour,
            "type": "CASH_OUT",
            "amount": amt,
            "cust_idx": c_idx,
            "nameOrig": customers[c_idx],
            "nameDest": str(rng.choice(mules[8:])),
            "device_id": "DEV_BLACK_001" if rng.random() < 0.15 else f"DEV_API_{rng.integers(100, 999)}",
            "channel": "API",
            "ip_country": str(rng.choice(["SG", "CY", "NL"])),
            "isFraud": 1,
            "fraud_scenario": "novel_api_cashout",
            "force_drain": rng.random() < 0.55,
        })

    # Sort strictly chronologically before computing realistic account balances
    df = pd.DataFrame(records)
    df = df.sort_values("offset_sec", kind="mergesort").reset_index(drop=True)

    # Replay balances chronologically so oldbalanceOrg / newbalanceOrig are realistic
    old_bal_orig = np.zeros(len(df), dtype=np.float64)
    new_bal_orig = np.zeros(len(df), dtype=np.float64)
    old_bal_dest = np.zeros(len(df), dtype=np.float64)
    new_bal_dest = np.zeros(len(df), dtype=np.float64)

    current_bal = cust_balance.copy()

    for i, row in enumerate(df.itertuples(index=False)):
        c_idx = row.cust_idx
        amt = row.amount
        tx_type = row.type

        # Top up legitimate customer balance periodically if running low
        if row.isFraud == 0 and tx_type != "CASH_IN" and current_bal[c_idx] < amt * 1.15:
            current_bal[c_idx] += amt * rng.uniform(1.5, 4.0)

        if row.force_drain:
            # Fraudster drains nearly the entire balance (PaySim hallmark)
            old_b = max(current_bal[c_idx], amt * rng.uniform(1.0, 1.04))
            actual_amt = amt
            new_b = max(0.0, old_b - actual_amt)
            if new_b / (old_b + 1e-6) > 0.05:
                # Adjust old_b so drain ratio is ~96-100%
                old_b = actual_amt * rng.uniform(1.0, 1.03)
                new_b = max(0.0, old_b - actual_amt)
            current_bal[c_idx] = new_b
        elif tx_type == "CASH_IN":
            old_b = current_bal[c_idx]
            new_b = old_b + amt
            current_bal[c_idx] = new_b
        else:
            old_b = current_bal[c_idx]
            new_b = max(0.0, old_b - amt)
            current_bal[c_idx] = new_b

        old_bal_orig[i] = round(old_b, 2)
        new_bal_orig[i] = round(new_b, 2)

        d_name = row.nameDest
        d_old = dest_balances.get(d_name, float(rng.uniform(0.0, 15000.0)))
        # In PaySim, merchant/cashout destinations sometimes don't reflect balance immediately
        d_new = d_old if tx_type == "CASH_OUT" else d_old + amt
        dest_balances[d_name] = d_new
        old_bal_dest[i] = round(d_old, 2)
        new_bal_dest[i] = round(d_new, 2)

    df["oldbalanceOrg"] = old_bal_orig
    df["newbalanceOrig"] = new_bal_orig
    df["oldbalanceDest"] = old_bal_dest
    df["newbalanceDest"] = new_bal_dest
    df["tx_id"] = [f"TX_{i + 1:07d}" for i in range(len(df))]
    df["timestamp"] = [start_time + timedelta(seconds=int(s)) for s in df["offset_sec"]]

    # Select final column order (PaySim core + extended telemetry)
    cols = [
        "tx_id",
        "step",
        "timestamp",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
        "device_id",
        "channel",
        "ip_country",
        "isFraud",
        "fraud_scenario",
    ]
    return df[cols]


def load_or_generate_dataset(
    cfg: dict[str, Any] | None = None,
    force_regenerate: bool = False,
) -> pd.DataFrame:
    """Load raw Parquet dataset or generate it via the simulator."""
    cfg = cfg or load_config()
    raw_path = resolve_path(cfg["paths"]["raw_data"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_path.exists() and not force_regenerate:
        return pd.read_parquet(raw_path)

    sim_cfg = cfg["simulation"]
    df = generate_transactions(
        n_customers=sim_cfg["n_customers"],
        n_merchants=sim_cfg["n_merchants"],
        n_transactions=sim_cfg["n_transactions"],
        n_days=sim_cfg["n_days"],
        fraud_rate=sim_cfg["fraud_rate"],
        random_seed=cfg["project"]["random_seed"],
    )
    df.to_parquet(raw_path, index=False)
    return df


if __name__ == "__main__":
    df_out = load_or_generate_dataset(force_regenerate=True)
    print(f"Generated {len(df_out):,} transactions | Fraud rate: {df_out['isFraud'].mean():.3%}")
    print(df_out["fraud_scenario"].value_counts())
