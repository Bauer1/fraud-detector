"""Level 3: Decision Agent combining Rules, Supervised ML, Auxiliary Anomaly Signal, Shadow Mode, and Fail-Safe Fallback."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.agent.explainer import FraudExplainer
from src.features.online_store import ENRICHED_PRE_DECISION_FEATURE_NAMES, SCHEMA_VERSION
from src.models.anomaly import FraudAnomalyDetector
from src.models.supervised import FraudGBDTModel
from src.rules.engine import RuleEngine


class FraudDecisionAgent:
    """Multi-layered Decision Service with deterministic rule priority, auxiliary anomaly cap, and fail-safe fallback."""

    def __init__(
        self,
        rule_engine: RuleEngine,
        ml_model: FraudGBDTModel | None,
        anomaly_detector: FraudAnomalyDetector | None,
        explainer: FraudExplainer | None,
        weights: dict[str, float] | None = None,
        thresholds: dict[str, float] | None = None,
        feature_names: list[str] | None = None,
        model_version: str = "unversioned",
        feature_schema_version: str = SCHEMA_VERSION,
        anomaly_max_standalone_score: float = 0.14,
        fallback_policy: str = "rules_only",
    ):
        self.rule_engine = rule_engine
        self.ml_model = ml_model
        self.anomaly_detector = anomaly_detector
        self.explainer = explainer
        self.feature_names = feature_names or list(ENRICHED_PRE_DECISION_FEATURE_NAMES)
        self.model_version = model_version
        self.feature_schema_version = feature_schema_version
        self.anomaly_max_standalone_score = float(anomaly_max_standalone_score)
        self.fallback_policy = fallback_policy

        w = weights or {"ml_score": 0.70, "rule_score": 0.25, "anomaly_score": 0.05}
        total_w = sum(w.values()) or 1.0
        self.w_ml = float(w.get("ml_score", 0.70)) / total_w
        self.w_rule = float(w.get("rule_score", 0.25)) / total_w
        self.w_anom = float(w.get("anomaly_score", 0.05)) / total_w

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
        """Compute composite risk score while ensuring anomaly score NEVER defines fraud on its own."""
        standalone_cap = min(self.anomaly_max_standalone_score, max(0.0, self.t_review - 0.01))

        if isinstance(ml_score, np.ndarray) or isinstance(rule_score, np.ndarray) or isinstance(anomaly_score, np.ndarray):
            ml_arr = np.asarray(ml_score, dtype=float)
            rule_arr = np.asarray(rule_score, dtype=float)
            anom_arr = np.asarray(anomaly_score, dtype=float)

            combined = self.w_ml * ml_arr + self.w_rule * rule_arr + self.w_anom * anom_arr
            # Auxiliary boost ONLY when corroborated by supervised ML or rules
            corroborated = (anom_arr > 0.85) & ((ml_arr >= 0.10) | (rule_arr >= 0.20))
            combined = np.where(corroborated, combined + 0.08, combined)

            # Enforce standalone anomaly cap when neither ML nor rules indicate risk
            uncorroborated_anomaly = (ml_arr < 0.05) & (rule_arr <= 1e-6)
            combined = np.where(uncorroborated_anomaly, np.minimum(combined, standalone_cap), combined)
            combined = np.clip(combined, 0.0, 1.0)

            if isinstance(hard_block, np.ndarray):
                combined = np.where(hard_block, 1.0, combined)
            elif hard_block:
                combined = np.ones_like(combined)
            return combined

        if hard_block:
            return 1.0

        ml_f = float(ml_score)
        rule_f = float(rule_score)
        anom_f = float(anomaly_score)

        combined_f = self.w_ml * ml_f + self.w_rule * rule_f + self.w_anom * anom_f
        if anom_f > 0.85 and (ml_f >= 0.10 or rule_f >= 0.20):
            combined_f += 0.08
        if ml_f < 0.05 and rule_f <= 1e-6:
            combined_f = min(combined_f, standalone_cap)
        return float(min(max(combined_f, 0.0), 1.0))

    def select_action(self, risk_score: float, hard_block: bool = False) -> str:
        if hard_block or risk_score >= self.t_block:
            return "BLOCK"
        if risk_score >= self.t_review:
            return "REVIEW"
        if risk_score >= self.t_otp:
            return "OTP"
        return "ALLOW"

    def evaluate_fallback_only(
        self,
        tx: dict[str, Any],
        features: dict[str, float] | None,
        reason: str,
        shadow_mode: bool = False,
        t0: float | None = None,
    ) -> dict[str, Any]:
        """Deterministic fail-safe decision policy when ML/Feature components fail or time out."""
        start = t0 if t0 is not None else time.perf_counter()
        safe_feats = features or {"amount": float(tx.get("amount", 0.0))}
        rule_ev = self.rule_engine.evaluate(tx, safe_feats)

        if rule_ev.hard_block:
            action = "BLOCK"
            risk_score = 1.0
        elif self.fallback_policy == "hold_for_review":
            action = "REVIEW"
            risk_score = max(rule_ev.rule_score, self.t_review)
        else:
            # Default business fail-safe: "rules_only"
            if rule_ev.rule_score >= 0.55:
                action = "REVIEW"
            elif rule_ev.rule_score >= 0.30:
                action = "OTP"
            else:
                action = "ALLOW"
            risk_score = rule_ev.rule_score

        decision_mode = "SHADOW" if shadow_mode else "ACTIVE"
        effective_action = "ALLOW" if shadow_mode else action
        latency_ms = (time.perf_counter() - start) * 1000.0
        summary = (
            f"[FAIL-SAFE: {self.fallback_policy}] Применена резервная политика из-за деградации компонента "
            f"({reason}). Решение по правилам: {action} (правила={rule_ev.rule_score:.2f})."
        )
        return {
            "tx_id": str(tx.get("tx_id", "")),
            "action": action,
            "effective_action": effective_action,
            "decision_mode": decision_mode,
            "technical_status": "DEGRADED_FALLBACK",
            "fallback_reason": reason,
            "model_version": self.model_version,
            "feature_schema_version": self.feature_schema_version,
            "risk_score": round(float(risk_score), 4),
            "ml_score": 0.0,
            "rule_score": round(float(rule_ev.rule_score), 4),
            "anomaly_score": 0.0,
            "triggered_rules": rule_ev.hits,
            "top_factors": [],
            "explanation_summary": summary,
            "latency_ms": round(latency_ms, 2),
        }

    def evaluate_transaction(
        self,
        tx: dict[str, Any],
        features: dict[str, float],
        explain: bool = True,
        shadow_mode: bool = False,
        timeout_budget_ms: float | None = None,
    ) -> dict[str, Any]:
        """Score a single transaction end-to-end with deterministic rules, ML, auxiliary anomaly, and fail-safe guard."""
        t0 = time.perf_counter()
        try:
            if self.ml_model is None or self.anomaly_detector is None:
                return self.evaluate_fallback_only(
                    tx=tx,
                    features=features,
                    reason="ml_or_anomaly_model_unavailable",
                    shadow_mode=shadow_mode,
                    t0=t0,
                )

            rule_ev = self.rule_engine.evaluate(tx, features)

            x_vec = np.array([[float(features.get(f, 0.0)) for f in self.feature_names]], dtype=float)
            ml_score = float(self.ml_model.predict_score(x_vec)[0])
            anomaly_score = float(self.anomaly_detector.predict_anomaly_score(x_vec)[0])

            risk_score = float(
                self.combine_scores(
                    ml_score=ml_score,
                    rule_score=rule_ev.rule_score,
                    anomaly_score=anomaly_score,
                    hard_block=rule_ev.hard_block,
                )
            )
            action = self.select_action(risk_score, hard_block=rule_ev.hard_block)

            top_factors = (
                self.explainer.explain_single(features, top_k=5)
                if (explain and self.explainer is not None)
                else []
            )
            summary = (
                self.explainer.generate_case_summary(
                    tx=tx,
                    action=action,
                    risk_score=risk_score,
                    ml_score=ml_score,
                    rule_score=rule_ev.rule_score,
                    anomaly_score=anomaly_score,
                    triggered_rules=rule_ev.hits,
                    top_factors=top_factors,
                )
                if self.explainer is not None
                else f"Решение: {action} (риск={risk_score:.2f})"
            )

            latency_ms = (time.perf_counter() - t0) * 1000.0
            technical_status = "OK"
            fallback_reason: str | None = None
            if timeout_budget_ms is not None and latency_ms > timeout_budget_ms:
                return self.evaluate_fallback_only(
                    tx=tx,
                    features=features,
                    reason=f"latency_budget_exceeded_{latency_ms:.1f}ms_gt_{timeout_budget_ms:.1f}ms",
                    shadow_mode=shadow_mode,
                    t0=t0,
                )

            decision_mode = "SHADOW" if shadow_mode else "ACTIVE"
            effective_action = "ALLOW" if shadow_mode else action

            return {
                "tx_id": str(tx.get("tx_id", "")),
                "action": action,
                "effective_action": effective_action,
                "decision_mode": decision_mode,
                "technical_status": technical_status,
                "fallback_reason": fallback_reason,
                "model_version": self.model_version,
                "feature_schema_version": self.feature_schema_version,
                "risk_score": round(risk_score, 4),
                "ml_score": round(ml_score, 4),
                "rule_score": round(rule_ev.rule_score, 4),
                "anomaly_score": round(anomaly_score, 4),
                "triggered_rules": rule_ev.hits,
                "top_factors": top_factors,
                "explanation_summary": summary,
                "latency_ms": round(latency_ms, 2),
            }
        except Exception as exc:
            return self.evaluate_fallback_only(
                tx=tx,
                features=features,
                reason=f"scoring_exception:{type(exc).__name__}",
                shadow_mode=shadow_mode,
                t0=t0,
            )
