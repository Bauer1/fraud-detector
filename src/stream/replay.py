"""Chronological transaction stream replayer -> FastAPI /score endpoint.

Replays held-out test transactions in real time (or accelerated speed) and optionally
simulates delayed analyst feedback to populate the operational DB and dashboard.
"""
from __future__ import annotations

import argparse
import time
from typing import Any

import httpx
import pandas as pd

from src.config import load_config, resolve_path
from src.features.engineering import sort_transactions_deterministically


def replay_stream(
    api_url: str = "http://127.0.0.1:8000",
    n_events: int = 300,
    delay_ms: int = 10,
    simulate_feedback: bool = True,
    in_process: bool = False,
) -> dict[str, Any]:
    """Stream test-period transactions chronologically to the FastAPI scoring service."""
    cfg = load_config()
    raw_path = resolve_path(cfg["paths"]["raw_data"])
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found at {raw_path}. Run training pipeline first.")

    df = sort_transactions_deterministically(pd.read_parquet(raw_path), deduplicate_retries=True)
    test_start = int(len(df) * (cfg["split"]["train_frac"] + cfg["split"]["val_frac"]))
    test_df = df.iloc[test_start : test_start + n_events].copy()

    action_counts = {"ALLOW": 0, "OTP": 0, "REVIEW": 0, "BLOCK": 0}
    feedback_sent = 0
    errors = 0

    if in_process:
        from src.api.app import score_transaction, state, submit_feedback
        from src.data.schemas import FeedbackRequest, TransactionRequest

        if state.agent is None:
            state.load_artifacts()

        for row in test_df.to_dict(orient="records"):
            tx_req = TransactionRequest(
                tx_id=row["tx_id"],
                step=int(row["step"]),
                timestamp=row["timestamp"].isoformat() if hasattr(row["timestamp"], "isoformat") else str(row["timestamp"]),
                type=row["type"],
                amount=float(row["amount"]),
                nameOrig=row["nameOrig"],
                oldbalanceOrg=float(row["oldbalanceOrg"]),
                newbalanceOrig=float(row["newbalanceOrig"]),
                nameDest=row["nameDest"],
                oldbalanceDest=float(row["oldbalanceDest"]),
                newbalanceDest=float(row["newbalanceDest"]),
                device_id=row["device_id"],
                channel=row["channel"],
                ip_country=row["ip_country"],
            )
            res = score_transaction(tx_req)
            act = res.action.value
            action_counts[act] = action_counts.get(act, 0) + 1
            if simulate_feedback and act in ("REVIEW", "BLOCK") and feedback_sent < 30:
                submit_feedback(
                    FeedbackRequest(
                        tx_id=row["tx_id"],
                        is_fraud=bool(row["isFraud"]),
                        analyst_id="sim_analyst_1",
                        notes=f"Simulated ground-truth label (scenario={row.get('fraud_scenario', 'unknown')})",
                    )
                )
                feedback_sent += 1
        return {
            "replayed_events": len(test_df),
            "action_counts": action_counts,
            "feedback_sent": feedback_sent,
            "errors": errors,
        }

    with httpx.Client(base_url=api_url, timeout=15.0) as client:
        for row in test_df.to_dict(orient="records"):
            payload = {
                "tx_id": row["tx_id"],
                "step": int(row["step"]),
                "timestamp": row["timestamp"].isoformat() if hasattr(row["timestamp"], "isoformat") else str(row["timestamp"]),
                "type": row["type"],
                "amount": float(row["amount"]),
                "nameOrig": row["nameOrig"],
                "oldbalanceOrg": float(row["oldbalanceOrg"]),
                "newbalanceOrig": float(row["newbalanceOrig"]),
                "nameDest": row["nameDest"],
                "oldbalanceDest": float(row["oldbalanceDest"]),
                "newbalanceDest": float(row["newbalanceDest"]),
                "device_id": row["device_id"],
                "channel": row["channel"],
                "ip_country": row["ip_country"],
            }
            resp = client.post("/score", json=payload)
            if resp.status_code != 200:
                errors += 1
                continue
            res = resp.json()
            act = res["action"]
            action_counts[act] = action_counts.get(act, 0) + 1

            # Simulate delayed analyst review on a portion of REVIEW/BLOCK alerts
            if simulate_feedback and act in ("REVIEW", "BLOCK") and feedback_sent < 30:
                fb_payload = {
                    "tx_id": row["tx_id"],
                    "is_fraud": bool(row["isFraud"]),
                    "analyst_id": "sim_analyst_1",
                    "notes": f"Simulated ground-truth label (scenario={row.get('fraud_scenario', 'unknown')})",
                }
                fb_resp = client.post("/feedback", json=fb_payload)
                if fb_resp.status_code == 200:
                    feedback_sent += 1

            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)

    return {
        "replayed_events": len(test_df),
        "action_counts": action_counts,
        "feedback_sent": feedback_sent,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay chronological transactions to Fraud Agent API")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="FastAPI base URL")
    parser.add_argument("--n-events", type=int, default=300, help="Number of test transactions to replay")
    parser.add_argument("--delay-ms", type=int, default=10, help="Delay between requests in milliseconds")
    parser.add_argument("--no-feedback", action="store_true", help="Disable simulated analyst feedback")
    parser.add_argument("--in-process", action="store_true", help="Run against in-process FastAPI TestClient (no server needed)")
    args = parser.parse_args()

    stats = replay_stream(
        api_url=args.api_url,
        n_events=args.n_events,
        delay_ms=args.delay_ms,
        simulate_feedback=not args.no_feedback,
        in_process=args.in_process,
    )
    print("Stream replay complete:", stats)


if __name__ == "__main__":
    main()
