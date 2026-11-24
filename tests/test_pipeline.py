"""Unit and integration tests for the AI Fraud Detection Agent.

Compatible with both `python -m unittest` and `pytest`.
Covers:
1. Strictly leak-free chronological feature engineering
2. Rule Engine hard blocks & heuristics
3. Business-cost metrics & threshold optimization
4. PSI drift detection
5. FastAPI end-to-end scoring, alerts, feedback, and monitoring endpoints
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.agent.thresholds import compute_business_cost_metrics, optimize_thresholds_by_cost
from src.api.app import (
    get_ablations,
    get_alert_detail,
    get_drift_report,
    get_metrics_summary,
    health,
    list_alerts,
    score_transaction,
    state,
    submit_feedback,
)
from src.data.schemas import FeedbackRequest, TransactionRequest
from src.evaluation.metrics import precision_at_k, recall_at_fpr
from src.features.engineering import build_features_chronological, time_based_split
from src.monitoring.drift import compute_psi
from src.rules.engine import RuleEngine


class TestFeatureLeakage(unittest.TestCase):
    def test_rolling_features_use_only_past_transactions(self) -> None:
        base = datetime(2026, 1, 1, 10, 0, 0)
        df = pd.DataFrame([
            {
                "tx_id": "T1",
                "step": 10,
                "timestamp": base,
                "type": "TRANSFER",
                "amount": 1000.0,
                "nameOrig": "C1",
                "oldbalanceOrg": 5000.0,
                "newbalanceOrig": 4000.0,
                "nameDest": "M1",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 1000.0,
                "device_id": "D1",
                "channel": "MOBILE",
                "ip_country": "RU",
                "isFraud": 0,
            },
            {
                "tx_id": "T2",
                "step": 10,
                "timestamp": base + timedelta(minutes=15),
                "type": "TRANSFER",
                "amount": 2000.0,
                "nameOrig": "C1",
                "oldbalanceOrg": 4000.0,
                "newbalanceOrig": 2000.0,
                "nameDest": "M2",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 2000.0,
                "device_id": "D2",
                "channel": "WEB",
                "ip_country": "RU",
                "isFraud": 1,
            },
            {
                "tx_id": "T3",
                "step": 10,
                "timestamp": base + timedelta(minutes=30),
                "type": "TRANSFER",
                "amount": 500.0,
                "nameOrig": "C1",
                "oldbalanceOrg": 2000.0,
                "newbalanceOrig": 1500.0,
                "nameDest": "M1",
                "oldbalanceDest": 1000.0,
                "newbalanceDest": 1500.0,
                "device_id": "D1",
                "channel": "MOBILE",
                "ip_country": "RU",
                "isFraud": 0,
            },
        ])
        feats = build_features_chronological(df)
        # First tx must see 0 past transactions (no future leakage from T2/T3!)
        self.assertEqual(feats.loc[0, "orig_tx_count_1h"], 0.0)
        self.assertEqual(feats.loc[0, "orig_tx_sum_1h"], 0.0)
        self.assertEqual(feats.loc[0, "orig_past_tx_count"], 0.0)

        # Second tx sees only T1
        self.assertEqual(feats.loc[1, "orig_tx_count_1h"], 1.0)
        self.assertEqual(feats.loc[1, "orig_tx_sum_1h"], 1000.0)
        self.assertEqual(feats.loc[1, "is_new_recipient"], 1.0)
        self.assertEqual(feats.loc[1, "is_new_device"], 1.0)

        # Third tx sees T1 + T2, and M1/D1 are now known
        self.assertEqual(feats.loc[2, "orig_tx_count_1h"], 2.0)
        self.assertEqual(feats.loc[2, "orig_tx_sum_1h"], 3000.0)
        self.assertEqual(feats.loc[2, "is_new_recipient"], 0.0)
        self.assertEqual(feats.loc[2, "is_new_device"], 0.0)

    def test_time_based_split_preserves_order(self) -> None:
        df = pd.DataFrame({"step": [5, 1, 4, 2, 3, 6, 7, 8, 9, 10]})
        tr, va, te = time_based_split(df, train_frac=0.6, val_frac=0.2)
        self.assertLessEqual(tr["step"].max(), va["step"].min())
        self.assertLessEqual(va["step"].max(), te["step"].min())


class TestRuleEngine(unittest.TestCase):
    def test_blacklisted_account_triggers_hard_block(self) -> None:
        engine = RuleEngine()
        ev = engine.evaluate(
            {"nameOrig": "C_BLACKLIST_001", "nameDest": "M1", "device_id": "D1", "amount": 500.0, "type": "TRANSFER"},
            {"amount": 500.0, "balance_drain_ratio": 0.1, "hour_of_day": 14},
        )
        self.assertTrue(ev.hard_block)
        self.assertEqual(ev.rule_score, 1.0)
        self.assertTrue(any(h["rule_name"] == "BLACKLISTED_ACCOUNT" for h in ev.hits))

    def test_normal_small_transfer_triggers_no_rules(self) -> None:
        engine = RuleEngine()
        ev = engine.evaluate(
            {"nameOrig": "C100", "nameDest": "M100", "device_id": "D100", "amount": 350.0, "type": "PAYMENT"},
            {
                "amount": 350.0,
                "balance_drain_ratio": 0.05,
                "hour_of_day": 14,
                "orig_tx_count_1h": 0.0,
                "orig_tx_sum_1h": 0.0,
                "is_new_device": 0.0,
                "is_new_recipient": 0.0,
                "device_shared_accounts": 0.0,
                "dest_in_degree": 1.0,
                "amount_zscore": 0.2,
            },
        )
        self.assertFalse(ev.hard_block)
        self.assertEqual(ev.rule_score, 0.0)
        self.assertEqual(len(ev.hits), 0)


class TestMetricsAndDrift(unittest.TestCase):
    def test_ranking_and_cost_metrics(self) -> None:
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1])
        s = np.array([0.05, 0.10, 0.12, 0.20, 0.35, 0.70, 0.85, 0.95])
        amt = np.array([100, 200, 150, 300, 400, 10000, 20000, 30000], dtype=float)
        self.assertEqual(precision_at_k(y, s, k=3), 1.0)
        self.assertGreaterEqual(recall_at_fpr(y, s, target_fpr=0.2), 1.0)

        costs = {"otp_cost": 2.0, "review_cost": 25.0, "false_block_cost": 150.0}
        th, m = optimize_thresholds_by_cost(y, s, amt, costs)
        self.assertLess(th["otp"], th["review"])
        self.assertLess(th["review"], th["block"])
        self.assertGreater(m["net_savings"], 50000.0)

    def test_psi_detects_shift(self) -> None:
        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, 1000)
        same = rng.normal(0, 1, 1000)
        shifted = rng.normal(2.5, 1.5, 1000)
        self.assertLess(compute_psi(ref, same), 0.10)
        self.assertGreater(compute_psi(ref, shifted), 0.25)


class TestFastAPIService(unittest.TestCase):
    def test_score_alert_and_feedback_flow(self) -> None:
        state.load_artifacts()

        h = health()
        self.assertEqual(h["status"], "ok")
        self.assertTrue(h["model_loaded"])

        tx = TransactionRequest(
            tx_id="TX_TEST_UNIT_001",
            step=750,
            timestamp=datetime.now(timezone.utc),
            type="TRANSFER",
            amount=260000.0,
            nameOrig="C_BLACKLIST_001",
            oldbalanceOrg=260500.0,
            newbalanceOrig=500.0,
            nameDest="C_MULE_001",
            oldbalanceDest=0.0,
            newbalanceDest=260000.0,
            device_id="DEV_BLACK_001",
            channel="API",
            ip_country="NL",
        )
        res = score_transaction(tx)
        self.assertEqual(res.action.value, "BLOCK")
        self.assertEqual(res.risk_score, 1.0)
        self.assertGreater(len(res.triggered_rules), 0)
        self.assertGreater(len(res.top_factors), 0)

        # Verify alert retrieval
        detail = get_alert_detail("TX_TEST_UNIT_001")
        self.assertEqual(detail["action"], "BLOCK")
        alerts = list_alerts(action="BLOCK", only_unlabeled=False, limit=10)
        self.assertTrue(any(a["tx_id"] == "TX_TEST_UNIT_001" for a in alerts))

        # Submit analyst feedback
        fb = submit_feedback(
            FeedbackRequest(
                tx_id="TX_TEST_UNIT_001",
                is_fraud=True,
                analyst_id="test_analyst",
                notes="Confirmed ATO + blacklist hit",
            )
        )
        self.assertTrue(fb.is_fraud)

        # Check summary, drift & ablations endpoints
        self.assertIn("operational", get_metrics_summary())
        self.assertIn("report", get_drift_report())
        self.assertGreater(len(get_ablations()["ablations"]), 0)


if __name__ == "__main__":
    unittest.main()
