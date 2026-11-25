"""Stage 1 correctness tests: feature contract, event ordering, concurrency, clean startup, and retrain holdout safety."""
from __future__ import annotations

import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import joblib
import pandas as pd
from fastapi import HTTPException
from pydantic import ValidationError

from src.api.app import health, score_transaction, state
from src.config import load_config, resolve_path
from src.data.schemas import TransactionRequest
from src.data.storage import StorageManager
from src.features.engineering import build_features_chronological, time_based_split
from src.features.online_store import (
    COLD_START_TIME_SINCE_LAST_HOURS,
    DuplicateTransactionConflictError,
    OnlineFeatureStore,
    StaleEventError,
    UnknownFeatureFieldError,
    normalize_event_datetime,
)
from src.training.retrain import retrain_with_feedback


def _make_tx(
    tx_id: str,
    ts: datetime | str | None,
    *,
    tx_type: str = "TRANSFER",
    amount: float = 1000.0,
    orig: str = "C1",
    dest: str = "C2",
    old_orig: float = 10000.0,
    new_orig: float | None = None,
    old_dest: float = 0.0,
    new_dest: float | None = None,
    device_id: str = "DEV_1",
    channel: str = "MOBILE",
    ip_country: str = "RU",
    step: int = 10,
) -> dict:
    if new_orig is None:
        new_orig = old_orig + amount if tx_type == "CASH_IN" else max(0.0, old_orig - amount)
    if new_dest is None:
        new_dest = old_dest if tx_type in ("CASH_IN", "CASH_OUT") else old_dest + amount
    return {
        "tx_id": tx_id,
        "step": step,
        "timestamp": ts,
        "type": tx_type,
        "amount": amount,
        "nameOrig": orig,
        "oldbalanceOrg": old_orig,
        "newbalanceOrig": new_orig,
        "nameDest": dest,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
        "device_id": device_id,
        "channel": channel,
        "ip_country": ip_country,
    }


class TestFeatureContract(unittest.TestCase):
    def test_cash_in_does_not_drain_or_pollute_outgoing_velocity(self) -> None:
        store = OnlineFeatureStore()
        t0 = datetime(2026, 1, 5, 12, 0, 0, tzinfo=timezone.utc)

        f_in = store.compute_features(
            _make_tx("TX_CI_1", t0, tx_type="CASH_IN", amount=50000.0, orig="C1", dest="M200001", old_orig=1000.0, new_orig=51000.0)
        )
        self.assertEqual(f_in["balance_drain_ratio"], 0.0)
        self.assertEqual(f_in["orig_balance_zero_after"], 0.0)
        self.assertEqual(f_in["orig_balance_error"], 0.0)
        self.assertEqual(f_in["dest_balance_error"], 0.0)
        self.assertEqual(f_in["type_CASH_IN"], 1.0)

        # Next outgoing transfer sees 0 past outgoing velocity, but knows device DEV_1 and last activity timestamp!
        f_out = store.compute_features(
            _make_tx("TX_TR_1", t0 + timedelta(minutes=30), tx_type="TRANSFER", amount=2000.0, orig="C1", dest="C99", device_id="DEV_1")
        )
        self.assertEqual(f_out["orig_tx_count_1h"], 0.0)
        self.assertEqual(f_out["orig_tx_sum_1h"], 0.0)
        self.assertEqual(f_out["orig_past_tx_count"], 0.0)
        self.assertEqual(f_out["is_new_device"], 0.0)
        self.assertAlmostEqual(f_out["time_since_last_tx_hours"], 0.5, places=5)

    def test_payment_and_merchant_not_flagged_as_mule(self) -> None:
        store = OnlineFeatureStore(mule_in_degree_threshold=3)
        t0 = datetime(2026, 1, 5, 14, 0, 0, tzinfo=timezone.utc)

        for i in range(6):
            f = store.compute_features(
                _make_tx(
                    f"TX_PAY_{i}",
                    t0 + timedelta(minutes=i),
                    tx_type="PAYMENT",
                    amount=500.0,
                    orig=f"C_BUYER_{i}",
                    dest="M200010",
                    old_dest=0.0,
                    new_dest=0.0,
                )
            )
            self.assertEqual(f["dest_balance_error"], 0.0)
            self.assertEqual(f["dest_in_degree"], 0.0)
            self.assertEqual(f["is_mule_candidate"], 0.0)

    def test_timezone_normalization_naive_and_offset_match_utc(self) -> None:
        s1 = OnlineFeatureStore()
        s2 = OnlineFeatureStore()
        s3 = OnlineFeatureStore()

        f_naive = s1.compute_features(_make_tx("T1", datetime(2026, 1, 2, 3, 15, 0)))
        f_utc = s2.compute_features(_make_tx("T1", "2026-01-02T03:15:00Z"))
        f_plus5 = s3.compute_features(_make_tx("T1", "2026-01-02T08:15:00+05:00"))

        self.assertEqual(f_naive["hour_of_day"], 3.0)
        self.assertEqual(f_naive["is_night"], 1.0)
        self.assertEqual(f_naive, f_utc)
        self.assertEqual(f_utc, f_plus5)

    def test_cold_start_and_unknown_fields_contract(self) -> None:
        store = OnlineFeatureStore()
        f = store.compute_features(_make_tx("T_COLD", "2026-01-01T12:00:00Z", tx_type="CUSTOM_TYPE", channel="POS"))
        self.assertEqual(f["orig_past_tx_count"], 0.0)
        self.assertEqual(f["amount_zscore"], 0.0)
        self.assertEqual(f["amount_to_mean_ratio"], 1.0)
        self.assertEqual(f["is_new_recipient"], 0.0)
        self.assertEqual(f["is_new_device"], 0.0)
        self.assertEqual(f["time_since_last_tx_hours"], COLD_START_TIME_SINCE_LAST_HOURS)
        self.assertEqual(sum(f[f"type_{t}"] for t in ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]), 0.0)
        self.assertEqual(sum(f[f"channel_{c}"] for c in ["MOBILE", "WEB", "API", "ATM"]), 0.0)

        # Strict mode in OnlineFeatureStore rejects unknown fields
        bad_tx = _make_tx("T_BAD_FIELD", "2026-01-01T12:05:00Z")
        bad_tx["oldBalanceOrig_typo"] = 999.0
        with self.assertRaises(UnknownFeatureFieldError):
            store.compute_features(bad_tx, strict_fields=True)

        # Pydantic TransactionRequest rejects extra fields
        with self.assertRaises(ValidationError):
            TransactionRequest(**bad_tx)


class TestEventOrderingAndBoundaries(unittest.TestCase):
    def test_exact_window_boundaries_1h_24h_7d(self) -> None:
        store = OnlineFeatureStore()
        t0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        store.compute_features(_make_tx("B0", t0, amount=100.0, orig="C_WIN"))
        # Exactly +1h: B0 is still inside [t - 1h, t]
        f_at_1h = store.compute_features(_make_tx("B1", t0 + timedelta(hours=1), amount=200.0, orig="C_WIN"))
        self.assertEqual(f_at_1h["orig_tx_count_1h"], 1.0)
        self.assertEqual(f_at_1h["orig_tx_sum_1h"], 100.0)

        # +1h + 1s from B0: B0 drops out of 1h window, B1 remains in 1h, both remain in 24h
        f_after_1h = store.compute_features(_make_tx("B2", t0 + timedelta(hours=1, seconds=1), amount=300.0, orig="C_WIN"))
        self.assertEqual(f_after_1h["orig_tx_count_1h"], 1.0)
        self.assertEqual(f_after_1h["orig_tx_sum_1h"], 200.0)
        self.assertEqual(f_after_1h["orig_tx_count_24h"], 2.0)

        # +7d + 1s from B0: B0 evicted from 7d window, B1 & B2 remain in 7d
        f_after_7d = store.compute_features(_make_tx("B3", t0 + timedelta(days=7, seconds=1), amount=400.0, orig="C_WIN"))
        self.assertEqual(f_after_7d["orig_tx_count_7d"], 2.0)
        self.assertEqual(f_after_7d["orig_tx_sum_7d"], 500.0)

    def test_stale_event_rejected_and_minor_jitter_preserves_sorted_window(self) -> None:
        store = OnlineFeatureStore(max_sender_out_of_order_sec=300.0, max_global_staleness_sec=86400.0)
        t0 = datetime(2026, 1, 10, 12, 0, 0, tzinfo=timezone.utc)

        store.compute_features(_make_tx("E1", t0, amount=1000.0, orig="C_ORD"))

        # 2 minutes earlier (within 300s tolerance): allowed, must NOT see future E1 in window!
        f_jitter = store.compute_features(_make_tx("E0_JITTER", t0 - timedelta(minutes=2), amount=500.0, orig="C_ORD"))
        self.assertEqual(f_jitter["orig_tx_count_1h"], 0.0)

        # Next event at t0 + 1m must see BOTH E0_JITTER and E1 (sorted deque preserved!)
        f_next = store.compute_features(_make_tx("E2", t0 + timedelta(minutes=1), amount=700.0, orig="C_ORD"))
        self.assertEqual(f_next["orig_tx_count_1h"], 2.0)
        self.assertEqual(f_next["orig_tx_sum_1h"], 1500.0)

        # 10 minutes earlier than watermark (> 300s tolerance): strictly rejected!
        with self.assertRaises(StaleEventError):
            store.compute_features(_make_tx("E_TOO_OLD", t0 - timedelta(minutes=10), amount=250.0, orig="C_ORD"))

        # Another sender lagging global watermark by > 24h: strictly rejected!
        with self.assertRaises(StaleEventError):
            store.compute_features(_make_tx("E_GLOBAL_OLD", t0 - timedelta(hours=25), amount=250.0, orig="C_OTHER"))

    def test_duplicate_tx_id_idempotency_and_conflict(self) -> None:
        store = OnlineFeatureStore()
        t0 = datetime(2026, 1, 10, 15, 0, 0, tzinfo=timezone.utc)
        tx = _make_tx("DUP_1", t0, amount=1500.0, orig="C_DUP")

        f1 = store.compute_features(tx)
        f2 = store.compute_features(tx)
        self.assertEqual(f1, f2)
        self.assertEqual(store.ingested_events, 1)

        # Subsequent event sees only 1 prior transaction, not 2!
        f_after = store.compute_features(_make_tx("DUP_2", t0 + timedelta(minutes=5), amount=500.0, orig="C_DUP"))
        self.assertEqual(f_after["orig_tx_count_1h"], 1.0)
        self.assertEqual(f_after["orig_tx_sum_1h"], 1500.0)

        # Conflicting reuse of DUP_1 with different amount raises conflict error
        with self.assertRaises(DuplicateTransactionConflictError):
            store.compute_features(_make_tx("DUP_1", t0, amount=9999.0, orig="C_DUP"))

    def test_same_timestamp_deterministic_tie_breaking_online_and_batch(self) -> None:
        same_ts = datetime(2026, 1, 12, 9, 0, 0, tzinfo=timezone.utc)
        rows = [
            {**_make_tx("TX_SAME_B", same_ts, amount=2000.0, orig="C_TIE", dest="C_D2"), "isFraud": 0},
            {**_make_tx("TX_SAME_A", same_ts, amount=1000.0, orig="C_TIE", dest="C_D1"), "isFraud": 0},
        ]
        f_order_1 = build_features_chronological(pd.DataFrame(rows))
        f_order_2 = build_features_chronological(pd.DataFrame(list(reversed(rows))))

        pd.testing.assert_frame_equal(f_order_1, f_order_2)
        self.assertEqual(list(f_order_1["tx_id"]), ["TX_SAME_A", "TX_SAME_B"])
        self.assertEqual(f_order_1.loc[0, "orig_tx_count_1h"], 0.0)
        self.assertEqual(f_order_1.loc[1, "orig_tx_count_1h"], 1.0)
        self.assertEqual(f_order_1.loc[1, "time_since_last_tx_hours"], 0.0)


class TestInitialStateAndConcurrency(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_storage = state.storage
        state.storage = StorageManager("sqlite:///:memory:")
        state.load_artifacts(initial_state_mode="empty")

    def tearDown(self) -> None:
        state.storage.close()
        state.storage = self._orig_storage
        state.load_artifacts(initial_state_mode="empty")

    def test_clean_startup_has_no_full_dataset_leakage_and_warmup_excludes_test(self) -> None:
        cfg = load_config()
        artifacts_dir = resolve_path(cfg["paths"]["artifacts_dir"])

        # Legacy full-dataset store must NOT exist
        self.assertFalse((artifacts_dir / "online_feature_store.joblib").exists())

        h = health()
        self.assertEqual(h["feature_store"]["state_origin"], "empty")
        self.assertEqual(h["feature_store"]["ingested_events"], 0)

        # Pre-test warm-up store must contain 0 test_tx_ids
        manifest = json.loads((artifacts_dir / "holdout_manifest.json").read_text(encoding="utf-8"))
        test_ids = set(manifest["test_tx_ids"])
        val_max_ts = normalize_event_datetime({"timestamp": manifest["val_max_timestamp"]}).timestamp()

        warmup_store: OnlineFeatureStore = joblib.load(artifacts_dir / "initial_feature_store_pre_test.joblib")
        self.assertEqual(warmup_store.state_origin, "pre_test_warmup")
        self.assertEqual(len(test_ids & set(warmup_store._processed_tx.keys())), 0)
        self.assertLessEqual(warmup_store._global_max_ts, val_max_ts + 1e-6)

    def test_concurrent_scoring_idempotency_and_failure_atomicity(self) -> None:
        base_ts = datetime(2026, 2, 1, 10, 0, 0, tzinfo=timezone.utc)
        n_tx = 32

        requests = [
            TransactionRequest(**_make_tx(f"TX_CONC_{i:03d}", base_ts + timedelta(seconds=i), amount=500.0 + i, orig="C_PARALLEL"))
            for i in range(n_tx)
        ]

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(score_transaction, requests))

        self.assertEqual(len(results), n_tx)
        self.assertTrue(all(not r.is_duplicate for r in results))
        self.assertEqual(state.feature_store.ingested_events, n_tx)

        # Parallel retries of an already-scored tx_id must all return is_duplicate=True without mutating state
        with ThreadPoolExecutor(max_workers=8) as pool:
            dup_results = list(pool.map(score_transaction, [requests[0]] * 10))
        self.assertTrue(all(r.is_duplicate for r in dup_results))
        self.assertEqual(state.feature_store.ingested_events, n_tx)

        # Conflicting tx_id -> HTTP 409, state untouched
        conflict_req = TransactionRequest(**_make_tx("TX_CONC_000", base_ts, amount=99999.0, orig="C_PARALLEL"))
        with self.assertRaises(HTTPException) as ctx_conflict:
            score_transaction(conflict_req)
        self.assertEqual(ctx_conflict.exception.status_code, 409)
        self.assertEqual(state.feature_store.ingested_events, n_tx)

        # Stale transaction -> HTTP 422, state untouched
        stale_req = TransactionRequest(**_make_tx("TX_STALE_999", base_ts - timedelta(hours=2), amount=600.0, orig="C_PARALLEL"))
        with self.assertRaises(HTTPException) as ctx_stale:
            score_transaction(stale_req)
        self.assertEqual(ctx_stale.exception.status_code, 422)
        self.assertEqual(state.feature_store.ingested_events, n_tx)


class TestRetrainHoldoutProtection(unittest.TestCase):
    def test_labeled_test_cases_cannot_contaminate_independent_test_evaluation(self) -> None:
        cfg = load_config()
        processed_path = resolve_path(cfg["paths"]["processed_data"])
        feat_df = pd.read_parquet(processed_path)
        _, _, test_df = time_based_split(feat_df, cfg["split"]["train_frac"], cfg["split"]["val_frac"])

        temp_storage = StorageManager("sqlite:///:memory:")
        sample_test_rows = test_df.iloc[:5]
        for row in sample_test_rows.to_dict(orient="records"):
            tx_payload = _make_tx(
                str(row["tx_id"]),
                row["timestamp"],
                tx_type=str(row["type"]),
                amount=float(row["amount"]),
                orig=str(row["nameOrig"]),
                dest=str(row["nameDest"]),
                step=int(row["step"]),
            )
            fake_decision = {
                "action": "REVIEW",
                "risk_score": 0.7,
                "ml_score": 0.7,
                "rule_score": 0.3,
                "anomaly_score": 0.4,
                "triggered_rules": [],
                "top_factors": [],
                "explanation_summary": "test",
                "latency_ms": 1.0,
            }
            temp_storage.save_scored_transaction(tx_payload, fake_decision, {})
            temp_storage.record_feedback(str(row["tx_id"]), bool(row["isFraud"]), analyst_id="test_analyst")

        import tempfile
        from pathlib import Path
        from src.models.registry import ModelRegistry

        artifacts_dir = resolve_path(cfg["paths"]["artifacts_dir"])
        backup_files = {
            name: (artifacts_dir / name).read_bytes()
            for name in ["gbdt_model.joblib", "anomaly_detector.joblib", "background_median.joblib", "metadata.json"]
            if (artifacts_dir / name).exists()
        }
        with tempfile.TemporaryDirectory() as tmp_reg_dir:
            temp_registry = ModelRegistry(
                registry_dir=Path(tmp_reg_dir) / "registry",
                active_artifacts_dir=Path(tmp_reg_dir) / "active",
            )
            try:
                # 1. Default strict protection: blocks all 5 test cases from entering X_train
                res_strict = retrain_with_feedback(cfg=cfg, storage=temp_storage, registry=temp_registry)
                self.assertEqual(res_strict["feedback_samples_total"], 5)
                self.assertEqual(res_strict["feedback_excluded_holdout"], 5)
                self.assertEqual(res_strict["feedback_samples_used"], 0)
                self.assertTrue(res_strict["is_independent_test"])
                self.assertFalse(res_strict["original_test_invalidated"])
                self.assertIsNotNone(res_strict["test_metrics"])

                # 2. Explicit adaptation mode: refuses to report contaminated test_metrics as independent test
                res_adapt = retrain_with_feedback(
                    cfg=cfg, storage=temp_storage, allow_holdout_feedback=True, registry=temp_registry
                )
                self.assertEqual(res_adapt["feedback_samples_used"], 5)
                self.assertFalse(res_adapt["is_independent_test"])
                self.assertTrue(res_adapt["original_test_invalidated"])
                self.assertIsNone(res_adapt["test_metrics"])
                self.assertEqual(res_adapt["evaluation_scope"], "post_feedback_future_slice_only")
                self.assertIsNotNone(res_adapt["future_slice_metrics"])
            finally:
                temp_storage.close()
                for name, raw_bytes in backup_files.items():
                    (artifacts_dir / name).write_bytes(raw_bytes)


if __name__ == "__main__":
    unittest.main()
