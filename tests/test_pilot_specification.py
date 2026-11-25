"""Comprehensive verification tests for Pilot Technical Specification v1.0 requirements.

Covers:
1. Data Manifests, Original PaySim CSV equal-step tie-breaking, and absent-telemetry safety
2. Pre-decision feature availability contract & explicit rejection of forbidden leakage fields
3. Probability calibration, capacity-constrained threshold selection, auxiliary anomaly cap, shadow mode & fail-safe fallback
4. Tri-state feedback (`fraud` / `not_fraud` / `unknown`), label maturity, selection bias, and immutable label audit log
5. Versioned Model Registry: candidate registration without overwrite, guardrail check, promotion, and instant rollback
6. PII pseudonymization, API-key authentication, and rate limiting
"""
from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException
from starlette.requests import Request

from src.agent.decision_agent import FraudDecisionAgent
from src.agent.thresholds import optimize_thresholds_by_cost
from src.api.app import (
    enforce_gateway_security,
    get_alert_audit_trail,
    get_alert_detail,
    get_data_manifests,
    get_metrics_summary,
    health,
    list_model_registry,
    promote_model_version,
    rollback_active_model,
    score_transaction,
    state,
    submit_feedback,
    trigger_retrain,
)
from src.config import load_config, resolve_path
from src.data.schemas import (
    FeedbackRequest,
    LabelStatus,
    PromoteModelRequest,
    RollbackModelRequest,
    TransactionRequest,
)
from src.data.storage import StorageManager
from src.features.engineering import build_features_chronological
from src.features.online_store import (
    ForbiddenLeakageFieldError,
    OnlineFeatureStore,
    POST_TX_BALANCE_LEAKAGE_FEATURES,
    resolve_model_feature_names,
)
from src.models.registry import ModelGovernanceError, ModelRegistry
from src.models.supervised import FraudGBDTModel
from src.rules.engine import RuleEngine
from src.security import SlidingWindowRateLimiter, pseudonymize_id, pseudonymize_transaction_dict


class TestPaySimAndFeatureAvailabilityContract(unittest.TestCase):
    def test_manifests_exist_and_separate_sources(self) -> None:
        manifests = get_data_manifests()
        self.assertIn("paysim_csv", manifests)
        self.assertIn("enriched_simulation", manifests)

        ps = manifests["paysim_csv"]
        self.assertEqual(ps["dataset_id"], "paysim_csv")
        self.assertGreater(ps["row_count"], 6_000_000)
        self.assertTrue(len(ps["file_sha256"]) == 64)
        self.assertIn("device_id", ps["schema"]["absent_telemetry_columns"])
        self.assertIn("channel", ps["schema"]["absent_telemetry_columns"])

    def test_post_tx_balances_and_absent_telemetry_excluded_from_model_features(self) -> None:
        cfg = load_config()
        fc = cfg.get("feature_contract", {})

        paysim_feats = resolve_model_feature_names("paysim_csv", fc)
        sim_feats = resolve_model_feature_names("enriched_simulation", fc)

        for leak_col in POST_TX_BALANCE_LEAKAGE_FEATURES:
            self.assertNotIn(leak_col, paysim_feats)
            self.assertNotIn(leak_col, sim_feats)

        # PaySim CSV must NOT include fabricated telemetry features
        for tel_col in ("is_foreign_ip", "channel_MOBILE", "channel_API", "is_new_device", "device_shared_accounts"):
            self.assertNotIn(tel_col, paysim_feats)
            self.assertIn(tel_col, sim_feats)

    def test_absent_device_never_links_unrelated_accounts_as_shared_ring(self) -> None:
        store = OnlineFeatureStore()
        for i in range(10):
            f = store.compute_features(
                {
                    "tx_id": f"PAYSIM_{i:08d}",
                    "step": 1,
                    "type": "TRANSFER",
                    "amount": 500.0 + i,
                    "nameOrig": f"C_ORIG_{i}",
                    "oldbalanceOrg": 5000.0,
                    "nameDest": f"C_DEST_{i}",
                    "device_id": None,
                    "channel": None,
                    "ip_country": None,
                }
            )
            self.assertEqual(f["device_shared_accounts"], 0.0)
            self.assertEqual(f["is_new_device"], 0.0)
            self.assertEqual(f["is_foreign_ip"], 0.0)

        obs = store.get_observability_stats()
        self.assertEqual(obs["cold_start_events"], 10)
        self.assertEqual(obs["cold_start_rate"], 1.0)
        self.assertEqual(obs["missing_optional_counts"]["device_id"], 10)

    def test_paysim_equal_step_tie_breaker_is_deterministic(self) -> None:
        rows = [
            {
                "tx_id": "PAYSIM_00000010",
                "source_row_idx": 10,
                "step": 5,
                "type": "TRANSFER",
                "amount": 1000.0,
                "nameOrig": "C_SAME",
                "oldbalanceOrg": 5000.0,
                "nameDest": "C_REC_1",
                "isFraud": 0,
            },
            {
                "tx_id": "PAYSIM_00000011",
                "source_row_idx": 11,
                "step": 5,
                "type": "TRANSFER",
                "amount": 2000.0,
                "nameOrig": "C_SAME",
                "oldbalanceOrg": 4000.0,
                "nameDest": "C_REC_2",
                "isFraud": 1,
            },
        ]
        f1 = build_features_chronological(pd.DataFrame(rows))
        f2 = build_features_chronological(pd.DataFrame(list(reversed(rows))))
        pd.testing.assert_frame_equal(f1, f2)
        self.assertEqual(list(f1["tx_id"]), ["PAYSIM_00000010", "PAYSIM_00000011"])

    def test_forbidden_leakage_fields_rejected_in_online_mode(self) -> None:
        store = OnlineFeatureStore()
        leaked_payload = {
            "tx_id": "TX_LEAK_1",
            "step": 1,
            "type": "TRANSFER",
            "amount": 1200.0,
            "nameOrig": "C1",
            "nameDest": "C2",
            "isFlaggedFraud": 1,
        }
        with self.assertRaises(ForbiddenLeakageFieldError):
            store.compute_features(leaked_payload, strict_fields=True)


class TestDecisionPolicyCalibrationAndFailSafe(unittest.TestCase):
    def test_probability_calibration_reduces_balanced_class_bias(self) -> None:
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.normal(0, 1, size=(1000, 4)), columns=["f1", "f2", "f3", "f4"])
        # Rare ~2% positive class
        logits = 1.8 * X["f1"].values - 1.2 * X["f2"].values - 4.2
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.random(1000) < probs).astype(int)
        if y[:500].sum() < 5:
            y[:6] = 1
        if y[500:].sum() < 5:
            y[500:506] = 1

        model = FraudGBDTModel(params={"n_estimators": 60, "class_weight": "balanced", "random_state": 42})
        model.fit(X.iloc[:600], y[:600])
        diag = model.calibrate(X.iloc[600:], y[600:], method="sigmoid")
        self.assertEqual(diag["method"], "sigmoid")
        self.assertLess(diag["brier_after"], diag["brier_before"])

    def test_capacity_constrained_threshold_optimization_respects_review_limit(self) -> None:
        rng = np.random.default_rng(7)
        y = np.zeros(1000, dtype=int)
        y[-20:] = 1
        scores = np.concatenate([rng.uniform(0.0, 0.55, size=980), rng.uniform(0.65, 0.99, size=20)])
        amounts = np.full(1000, 15000.0)

        th, m = optimize_thresholds_by_cost(
            y_true=y,
            scores=scores,
            amounts=amounts,
            costs_cfg={"otp_cost": 2.0, "review_cost": 10.0, "false_block_cost": 150.0},
            capacity_cfg={"max_review_rate": 0.03, "max_daily_reviews": 15},
            span_days=2.0,
        )
        self.assertTrue(m["capacity_within_limit"])
        self.assertLessEqual(m["review_rate"], 0.03 + 1e-6)
        self.assertLessEqual(m["daily_review_count"], 15.0 + 1e-6)
        self.assertLess(th["otp"], th["review"])
        self.assertLess(th["review"], th["block"])

    def test_standalone_anomaly_score_never_triggers_review_or_block(self) -> None:
        agent = FraudDecisionAgent(
            rule_engine=RuleEngine(),
            ml_model=None,
            anomaly_detector=None,
            explainer=None,
            thresholds={"otp": 0.20, "review": 0.45, "block": 0.75},
            anomaly_max_standalone_score=0.14,
        )
        # Extreme anomaly=1.0, but ML=0.0 and Rules=0.0 -> must stay below OTP/REVIEW/BLOCK!
        standalone_risk = agent.combine_scores(ml_score=0.0, rule_score=0.0, anomaly_score=1.0, hard_block=False)
        self.assertLessEqual(standalone_risk, 0.14)
        self.assertEqual(agent.select_action(float(standalone_risk)), "ALLOW")

    def test_failsafe_fallback_on_missing_or_broken_ml_model(self) -> None:
        agent = FraudDecisionAgent(
            rule_engine=RuleEngine(),
            ml_model=None,  # Simulates degraded ML component
            anomaly_detector=None,
            explainer=None,
            model_version="v-fallback-test",
            fallback_policy="rules_only",
        )
        dec_blacklist = agent.evaluate_transaction(
            {"tx_id": "FB_1", "nameOrig": "C_BLACKLIST_001", "nameDest": "C2", "amount": 1000.0, "type": "TRANSFER"},
            {"amount": 1000.0},
        )
        self.assertEqual(dec_blacklist["technical_status"], "DEGRADED_FALLBACK")
        self.assertEqual(dec_blacklist["action"], "BLOCK")
        self.assertEqual(dec_blacklist["model_version"], "v-fallback-test")

        dec_clean = agent.evaluate_transaction(
            {"tx_id": "FB_2", "nameOrig": "C_OK", "nameDest": "M_OK", "amount": 250.0, "type": "PAYMENT"},
            {"amount": 250.0},
        )
        self.assertEqual(dec_clean["technical_status"], "DEGRADED_FALLBACK")
        self.assertEqual(dec_clean["action"], "ALLOW")


class TestGovernanceAuditShadowAndSecurity(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory()
        tmp_root = Path(self._tmp_dir.name)
        self._orig_storage = state.storage
        self._orig_registry = state.registry

        state.storage = StorageManager("sqlite:///:memory:")
        state.registry = ModelRegistry(
            registry_dir=tmp_root / "registry",
            active_artifacts_dir=tmp_root / "active_artifacts",
        )
        # Seed isolated registry with current active artifacts so governance tests never mutate workspace registry
        prod_art = resolve_path(load_config()["paths"]["artifacts_dir"])
        seed_meta = load_config()
        state.registry.register_candidate(
            version_id="model-sim-v001-seed",
            source_files={
                name: prod_art / name
                for name in (
                    "gbdt_model.joblib",
                    "anomaly_detector.joblib",
                    "background_median.joblib",
                    "metadata.json",
                    "holdout_manifest.json",
                    "reference_distribution.parquet",
                )
                if (prod_art / name).exists()
            },
            metadata={"dataset_source": "enriched_simulation", "baseline_test_metrics": {"pr_auc": 0.90}},
            config_snapshot=seed_meta,
        )
        state.registry.promote_version("model-sim-v001-seed", approved_by="unit_test_setup")
        state.load_artifacts(initial_state_mode="empty")

    def tearDown(self) -> None:
        state.storage.close()
        state.storage = self._orig_storage
        state.registry = self._orig_registry
        state.load_artifacts(initial_state_mode="empty")
        self._tmp_dir.cleanup()

    def test_shadow_mode_and_tristate_feedback_audit_trail(self) -> None:
        tx = TransactionRequest(
            tx_id="TX_SHADOW_001",
            step=10,
            timestamp=datetime.now(timezone.utc),
            type="TRANSFER",
            amount=275000.0,
            nameOrig="C_BLACKLIST_001",
            oldbalanceOrg=280000.0,
            nameDest="C_MULE_001",
        )
        res = score_transaction(tx, shadow=True)
        self.assertEqual(res.decision_mode, "SHADOW")
        self.assertEqual(res.action.value, "BLOCK")
        self.assertEqual(res.effective_action.value, "ALLOW")
        self.assertEqual(res.technical_status.value, "OK")
        self.assertTrue(len(res.model_version) > 0)

        # 1. Mark as `unknown` (under investigation / immature)
        fb1 = submit_feedback(
            FeedbackRequest(
                tx_id="TX_SHADOW_001",
                label=LabelStatus.UNKNOWN,
                label_source="analyst_review",
                analyst_id="analyst_a",
                notes="Waiting for customer callback",
            )
        )
        self.assertEqual(fb1.label, LabelStatus.UNKNOWN)
        self.assertIsNone(fb1.is_fraud)

        # `unknown` must NOT appear in definitive retraining dataset!
        self.assertEqual(len(state.storage.get_labeled_feedback_df()), 0)
        self.assertEqual(state.storage.count_unknown_feedback(), 1)

        # 2. Matured chargeback confirms `fraud`
        fb2 = submit_feedback(
            FeedbackRequest(
                tx_id="TX_SHADOW_001",
                label=LabelStatus.FRAUD,
                label_source="chargeback",
                analyst_id="claims_system",
                notes="Confirmed unauthorized transfer",
            )
        )
        self.assertEqual(fb2.label, LabelStatus.FRAUD)
        self.assertTrue(fb2.is_fraud)
        self.assertEqual(len(state.storage.get_labeled_feedback_df()), 1)

        # Verify immutable audit log preserves both transitions!
        audit = get_alert_audit_trail("TX_SHADOW_001")
        self.assertEqual(len(audit), 2)
        self.assertEqual(audit[0]["previous_label"], "unknown")
        self.assertEqual(audit[0]["new_label"], "fraud")
        self.assertEqual(audit[0]["label_source"], "chargeback")
        self.assertIsNone(audit[1]["previous_label"])
        self.assertEqual(audit[1]["new_label"], "unknown")

        detail = get_alert_detail("TX_SHADOW_001")
        self.assertEqual(len(detail["label_audit_history"]), 2)

        summary = get_metrics_summary()
        self.assertIn("p95_latency_ms", summary["operational"])
        self.assertIn("selection_bias", summary["feedback_quality"])

    def test_retrain_registers_candidate_without_auto_promotion_and_supports_rollback(self) -> None:
        before_active = list_model_registry()["active_version"]
        self.assertEqual(before_active, "model-sim-v001-seed")

        # Run retrain against isolated registry -> must NOT auto-promote!
        from src.training.retrain import retrain_with_feedback

        cfg = load_config()
        cfg["paths"]["artifacts_dir"] = str(state.registry.active_artifacts_dir)
        cfg["paths"]["registry_dir"] = str(state.registry.registry_dir)

        res = retrain_with_feedback(cfg=cfg, storage=state.storage, promote=False, registry=state.registry)
        self.assertEqual(res["status"], "candidate_registered")
        self.assertFalse(res["auto_promoted"])
        self.assertEqual(res["active_version"], "model-sim-v001-seed")

        cand_ver = res["candidate_version"]
        reg_after_cand = list_model_registry()
        self.assertEqual(reg_after_cand["active_version"], "model-sim-v001-seed")
        self.assertEqual(reg_after_cand["versions"][cand_ver]["status"], "candidate")

        # Promote candidate explicitly
        prom = promote_model_version(cand_ver, PromoteModelRequest(approved_by="risk_director", reason="Validated"))
        self.assertEqual(prom["active_version"], cand_ver)
        self.assertEqual(list_model_registry()["active_version"], cand_ver)

        # Rollback restores `model-sim-v001-seed` immediately without overwriting candidate directory
        rb = rollback_active_model(RollbackModelRequest(initiated_by="oncall_sre", reason="Canary latency check"))
        self.assertEqual(rb["active_version"], "model-sim-v001-seed")
        self.assertTrue((state.registry.registry_dir / cand_ver / "gbdt_model.joblib").exists())

    def test_pseudonymization_api_key_auth_and_rate_limiter(self) -> None:
        tx_p = pseudonymize_transaction_dict(
            {"nameOrig": "C123456", "nameDest": "M999888", "device_id": "DEV_777"},
            salt="bank-secret-salt",
        )
        self.assertTrue(tx_p["nameOrig"].startswith("C_PSEUDO_"))
        self.assertTrue(tx_p["nameDest"].startswith("M_PSEUDO_"))
        self.assertTrue(tx_p["device_id"].startswith("DEV_PSEUDO_"))
        self.assertEqual(tx_p["nameOrig"], pseudonymize_id("C123456", salt="bank-secret-salt"))

        limiter = SlidingWindowRateLimiter(max_requests_per_minute=2)
        self.assertTrue(limiter.allow("client_1"))
        self.assertTrue(limiter.allow("client_1"))
        self.assertFalse(limiter.allow("client_1"))

        def _build_request(headers: list[tuple[bytes, bytes]]) -> Request:
            return Request(
                {
                    "type": "http",
                    "method": "GET",
                    "path": "/alerts",
                    "headers": headers,
                    "client": ("127.0.0.1", 50000),
                }
            )

        with patch.dict(os.environ, {"FRAUD_API_KEY": "top-secret-service-key"}):
            state.cfg = load_config()
            with self.assertRaises(HTTPException) as ctx_unauth:
                enforce_gateway_security(_build_request([]))
            self.assertEqual(ctx_unauth.exception.status_code, 401)

            # Valid X-API-Key header passes gateway security check
            enforce_gateway_security(_build_request([(b"x-api-key", b"top-secret-service-key")]))
            # Valid Bearer token also passes
            enforce_gateway_security(_build_request([(b"authorization", b"Bearer top-secret-service-key")]))
        state.cfg = load_config()


if __name__ == "__main__":
    unittest.main()
