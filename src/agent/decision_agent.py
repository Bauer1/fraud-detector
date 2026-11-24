"""Level 3: Decision Agent combining Rules, Supervised ML, Anomaly Detection, and Explainability."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.agent.explainer import FraudExplainer
from src.features.online_store import ALL_FEATURE_NAMES
from src.models.anomaly import FraudAnomalyDetector
from src.models.supervised import FraudGBDTModel
from src.rules.engine import RuleEngine


class FraudDecisionAgent:
    """Multi-layered AI Agent that scores transfers, selects an action, and explains its decision."""

    def __init__(
        self,
        rule_engine: RuleEngine,
        ml_model: FraudGBDTModel,
        anomaly_detector: FraudAnomalyDetector,
        explainer: FraudExplainer,
        weights: dict[str, float] | None = None,
        thresholds: dict[str, float] | None = None,
        feature_names: list[str] | None = None,
    ):
        self.rule_engine = rule_engine
        self.ml_model = ml_model
        self.anomaly_detector = anomaly_detector
        self.explainer = explainer
        self.feature_names = feature_names or list(ALL_FEATURE_NAMES)

        w = weights or {"ml_score": 0.65, "rule_score": 0.25, "anomaly_score": 0.10}
        total_w = sum(w.values()) or 1.0
        self.w_ml = float(w.get("ml_score", 0.65)) / total_w
        self.w_rule = float(w.get("rule_score", 0.25)) / total_w
        self.w_anom = float(w.get("anomaly_score", 0.10)) / total_w

        t = thresholds or {"otp": 0.30, "review": 0.55, "block": 0.80}
        self.t_otp = float(t.get("otp", 0.30))
        self.t_review = float(t.get("review", 0.55))
        self.t_block = float(t.get("block", 0.80))

    def combine_scores(
        self,
        ml_score: np.ndarray | float,
        rule_score: np.ndarray | float,
        anomaly_score: np.ndarray | float,
        hard_block: np.ndarray | bool = False,
    ) -> np.ndarray | float:
        """Compute composite risk score = f(rules, ml_score, anomaly_score)."""
        combined = self.w_ml * ml_score + self.w_rule * rule_score + self.w_anom * anomaly_score
        # Boost if anomaly detector flags an extreme novel anomaly
        if isinstance(combined, np.ndarray):
            novel_boost = np.where(np.asarray(anomaly_score) > 0.85, 0.12, 0.0)
            combined = np.clip(combined + novel_boost, 0.0, 1.0)
            if isinstance(hard_block, np.ndarray):
                combined = np.where(hard_block, 1.0, combined)
            elif hard_block:
                combined = np.ones_like(combined)
            return combined
        else:
            if hard_block:
                return 1.0
            if float(anomaly_score) > 0.85:
                combined += 0.12
            return float(min(max(combined, 0.0), 1.0))

    def select_action(self, risk_score: float, hard_block: bool = False) -> str:
        if hard_block or risk_score >= self.t_block:
            return "BLOCK"
        if risk_score >= self.t_review:
            return "REVIEW"
        if risk_score >= self.t_otp:
            return "OTP"
        return "ALLOW"

    def evaluate_transaction(
        self,
        tx: dict[str, Any],
        features: dict[str, float],
        explain: bool = True,
    ) -> dict[str, Any]:
        """Score a single transaction end-to-end and return full decision & explanation payload."""
        t0 = time.perf_counter()

        # Level 1: Rule Engine
        rule_ev = self.rule_engine.evaluate(tx, features)

        # Level 2: ML + Anomaly Detector
        x_vec = np.array([[float(features.get(f, 0.0)) for f in self.feature_names]], dtype=float)
        ml_score = float(self.ml_model.predict_score(x_vec)[0])
        anomaly_score = float(self.anomaly_detector.predict_anomaly_score(x_vec)[0])

        # Level 3: Composite Risk & Action
        risk_score = float(
            self.combine_scores(
                ml_score=ml_score,
                rule_score=rule_ev.rule_score,
                anomaly_score=anomaly_score,
                hard_block=rule_ev.hard_block,
            )
        )
        action = self.select_action(risk_score, hard_block=rule_ev.hard_block)

        # Explainability (always compute for alerts; lightweight or full for ALLOW)
        top_factors = self.explainer.explain_single(features, top_k=5) if explain else []
        summary = self.explainer.generate_case_summary(
            tx=tx,
            action=action,
            risk_score=risk_score,
            ml_score=ml_score,
            rule_score=rule_ev.rule_score,
            anomaly_score=anomaly_score,
            triggered_rules=rule_ev.hits,
            top_factors=top_factors,
        )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "tx_id": str(tx.get("tx_id", "")),
            "action": action,
            "risk_score": round(risk_score, 4),
            "ml_score": round(ml_score, 4),
            "rule_score": round(rule_ev.rule_score, 4),
            "anomaly_score": round(anomaly_score, 4),
            "triggered_rules": rule_ev.hits,
            "top_factors": top_factors,
            "explanation_summary": summary,
            "latency_ms": round(latency_ms, 2),
        }
