"""Chronological pre-decision transaction stream replayer -> FastAPI /score endpoint."""
from __future__ import annotations

import argparse
import os
import time
from typing import Any

import httpx
import pandas as pd

from src.config import load_config, resolve_path
from src.data.paysim_loader import load_paysim_csv_dataset
from src.features.engineering import sort_transactions_deterministically


def replay_stream(
    api_url: str = "http://127.0.0.1:8000",
    n_events: int = 300,
    delay_ms: int = 10,
    simulate_feedback: bool = True,
    in_process: bool = False,
    dataset_source: str = "enriched_simulation",
    shadow_mode: bool = False,
) -> dict[str, Any]:
    """Stream held-out test-period transactions chronologically using strictly pre-decision fields."""
    cfg = load_config()
    if dataset_source == "paysim_csv":
        ps_cfg = cfg.get("datasets", {}).get("paysim_csv", {})
        df_raw, _ = load_paysim_csv_dataset(
            sampling_mode=ps_cfg.get("sampling_mode", "chronological_stride"),
            max_rows=int(ps_cfg.get("max_rows", 120_000)),
        )
        df = sort_transactions_deterministically(df_raw, deduplicate_retries=True)
    else:
        raw_path = resolve_path(cfg["paths"]["raw_data"])
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data not found at {raw_path}. Run training pipeline first.")
        df = sort_transactions_deterministically(pd.read_parquet(raw_path), deduplicate_retries=True)

    test_start = int(len(df) * (cfg["split"]["train_frac"] + cfg["split"]["val_frac"]))
    test_df = df.iloc[test_start : test_start + n_events].copy()

    action_counts = {"ALLOW": 0, "OTP": 0, "REVIEW": 0, "BLOCK": 0}
    feedback_sent = 0
    errors = 0

    def _build_pre_decision_payload(row: dict[str, Any]) -> dict[str, Any]:
        ts = row.get("timestamp")
        ts_str = None
        if ts is not None and not pd.isna(ts):
            ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        payload: dict[str, Any] = {
            "tx_id": str(row["tx_id"]),
            "step": int(row["step"]),
            "timestamp": ts_str,
            "type": str(row["type"]),
            "amount": float(row["amount"]),
            "nameOrig": str(row["nameOrig"]),
            "oldbalanceOrg": float(row["oldbalanceOrg"]) if pd.notna(row.get("oldbalanceOrg")) else None,
            "nameDest": str(row["nameDest"]),
        }
        if pd.notna(row.get("device_id")):
            payload["device_id"] = str(row["device_id"])
        if pd.notna(row.get("channel")):
            payload["channel"] = str(row["channel"])
        if pd.notna(row.get("ip_country")):
            payload["ip_country"] = str(row["ip_country"])
        return payload

    if in_process:
        from src.api.app import score_transaction, state, submit_feedback
        from src.data.schemas import FeedbackRequest, LabelStatus, TransactionRequest

        if state.agent is None:
            state.load_artifacts()

        for idx, row in enumerate(test_df.to_dict(orient="records")):
            tx_req = TransactionRequest(**_build_pre_decision_payload(row))
            res = score_transaction(tx_req, shadow=shadow_mode)
            act = res.action.value
            action_counts[act] = action_counts.get(act, 0) + 1
            if simulate_feedback and act in ("REVIEW", "BLOCK") and feedback_sent < 30:
                # Simulate realistic mix of matured fraud/not_fraud and immature `unknown` cases
                if idx % 6 == 0:
                    lbl = LabelStatus.UNKNOWN
                else:
                    lbl = LabelStatus.FRAUD if bool(row["isFraud"]) else LabelStatus.NOT_FRAUD
                submit_feedback(
                    FeedbackRequest(
                        tx_id=str(row["tx_id"]),
                        label=lbl,
                        label_source="simulation",
                        analyst_id="sim_analyst_1",
                        notes=f"Simulated post-review label (source={dataset_source})",
                    )
                )
                feedback_sent += 1
        return {
            "dataset_source": dataset_source,
            "replayed_events": len(test_df),
            "action_counts": action_counts,
            "feedback_sent": feedback_sent,
            "errors": errors,
        }

    headers: dict[str, str] = {}
    api_key = os.getenv("FRAUD_API_KEY", "").strip()
    if api_key:
        headers["X-API-Key"] = api_key

    with httpx.Client(base_url=api_url, headers=headers, timeout=15.0) as client:
        for idx, row in enumerate(test_df.to_dict(orient="records")):
            payload = _build_pre_decision_payload(row)
            resp = client.post("/score", params={"shadow": shadow_mode}, json=payload)
            if resp.status_code != 200:
                errors += 1
                continue
            res = resp.json()
            act = res["action"]
            action_counts[act] = action_counts.get(act, 0) + 1

            if simulate_feedback and act in ("REVIEW", "BLOCK") and feedback_sent < 30:
                lbl_str = "unknown" if (idx % 6 == 0) else ("fraud" if bool(row["isFraud"]) else "not_fraud")
                fb_payload = {
                    "tx_id": str(row["tx_id"]),
                    "label": lbl_str,
                    "label_source": "simulation",
                    "analyst_id": "sim_analyst_1",
                    "notes": f"Simulated post-review label (source={dataset_source})",
                }
                fb_resp = client.post("/feedback", json=fb_payload)
                if fb_resp.status_code == 200:
                    feedback_sent += 1

            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)

    return {
        "dataset_source": dataset_source,
        "replayed_events": len(test_df),
        "action_counts": action_counts,
        "feedback_sent": feedback_sent,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay chronological pre-decision transactions to Fraud Service API")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="FastAPI base URL")
    parser.add_argument("--n-events", type=int, default=300, help="Number of test transactions to replay")
    parser.add_argument("--delay-ms", type=int, default=10, help="Delay between requests in milliseconds")
    parser.add_argument("--dataset", choices=["enriched_simulation", "paysim_csv"], default="enriched_simulation")
    parser.add_argument("--shadow", action="store_true", help="Replay in shadow pilot mode (effective_action=ALLOW)")
    parser.add_argument("--no-feedback", action="store_true", help="Disable simulated analyst feedback")
    parser.add_argument("--in-process", action="store_true", help="Run in-process without HTTP server")
    args = parser.parse_args()

    stats = replay_stream(
        api_url=args.api_url,
        n_events=args.n_events,
        delay_ms=args.delay_ms,
        simulate_feedback=not args.no_feedback,
        in_process=args.in_process,
        dataset_source=args.dataset,
        shadow_mode=args.shadow,
    )
    print("Stream replay complete:", stats)


if __name__ == "__main__":
    main()
