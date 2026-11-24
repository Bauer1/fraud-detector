"""SHAP-based local explainability (+ fast local attribution fallback) and optional LLM case summary."""
from __future__ import annotations

import os
from typing import Any

import httpx
import numpy as np

from src.features.online_store import ALL_FEATURE_NAMES, FEATURE_DESCRIPTIONS_RU

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class FraudExplainer:
    """Produces top-K local feature contributions (SHAP) and analyst-friendly case summaries.

    Uses `shap.TreeExplainer` when `shap` and a supported tree model are available,
    otherwise falls back to fast baseline-perturbation marginal attribution (exact 1st-order Shapley
    approximation against the training median baseline) so explanations are always fast and accurate.
    """

    def __init__(
        self,
        ml_model: Any,
        feature_names: list[str] | None = None,
        background_median: np.ndarray | None = None,
        llm_cfg: dict[str, Any] | None = None,
    ):
        self.ml_model = ml_model
        self.feature_names = feature_names or list(ALL_FEATURE_NAMES)
        self.background_median = (
            np.asarray(background_median, dtype=float)
            if background_median is not None
            else np.zeros(len(self.feature_names), dtype=float)
        )
        self.llm_cfg = llm_cfg or {}
        self._tree_explainer = None

        if HAS_SHAP and hasattr(ml_model, "model"):
            try:
                self._tree_explainer = shap.TreeExplainer(ml_model.model)
            except Exception:
                self._tree_explainer = None

    def explain_single(self, features: dict[str, float], top_k: int = 5) -> list[dict[str, Any]]:
        """Return top_k contributing features with their values, SHAP attributions, and descriptions."""
        x_vec = np.array([float(features.get(f, 0.0)) for f in self.feature_names], dtype=float)

        if self._tree_explainer is not None:
            try:
                shap_vals = self._tree_explainer.shap_values(x_vec.reshape(1, -1))
                if isinstance(shap_vals, list):
                    vals = np.asarray(shap_vals[1][0], dtype=float)
                elif shap_vals.ndim == 3:
                    vals = np.asarray(shap_vals[0, :, 1], dtype=float)
                else:
                    vals = np.asarray(shap_vals[0], dtype=float)
            except Exception:
                vals = self._perturbation_shap(x_vec)
        else:
            vals = self._perturbation_shap(x_vec)

        order = np.argsort(np.abs(vals))[::-1][:top_k]
        top_factors: list[dict[str, Any]] = []
        for idx in order:
            fname = self.feature_names[idx]
            fval = float(x_vec[idx])
            sval = float(vals[idx])
            desc = self._format_factor_description(fname, fval, sval)
            top_factors.append({
                "feature": fname,
                "value": round(fval, 4),
                "shap_value": round(sval, 4),
                "description": desc,
            })
        return top_factors

    def _perturbation_shap(self, x_vec: np.ndarray) -> np.ndarray:
        """Fast leave-one-out marginal log-odds / probability attribution against background median."""
        base_prob = float(self.ml_model.predict_score(x_vec.reshape(1, -1))[0])
        n_feat = len(x_vec)
        perturbed = np.tile(x_vec, (n_feat, 1))
        for i in range(n_feat):
            perturbed[i, i] = self.background_median[i]
        pert_probs = self.ml_model.predict_score(perturbed)
        # Positive value => this feature pushed the fraud probability UP compared to baseline
        return base_prob - pert_probs

    @staticmethod
    def _format_factor_description(feature: str, value: float, shap_val: float) -> str:
        base_label = FEATURE_DESCRIPTIONS_RU.get(feature, feature)
        direction = "повышает риск" if shap_val >= 0 else "снижает риск"
        if feature == "amount_to_mean_ratio" and value > 1.5:
            return f"Сумма в {value:.1f}× выше среднего чека клиента ({direction})"
        if feature == "amount_zscore":
            return f"Отклонение суммы от нормы клиента: z-score = {value:.1f} ({direction})"
        if feature == "balance_drain_ratio":
            return f"Списание {value:.0%} от баланса счёта ({direction})"
        if feature == "orig_balance_zero_after" and value >= 0.5:
            return f"Счёт отправителя опустошён до нуля ({direction})"
        if feature == "is_new_recipient" and value >= 0.5:
            return f"Перевод новому получателю ({direction})"
        if feature == "is_new_device" and value >= 0.5:
            return f"Операция с нового устройства ({direction})"
        if feature == "dest_in_degree":
            return f"Получатель принял переводы от {int(value)} разных отправителей ({direction})"
        if feature == "device_shared_accounts" and value > 0:
            return f"Устройство связано ещё с {int(value)} аккаунтами ({direction})"
        return f"{base_label}: {value:.2f} ({direction})"

    def generate_case_summary(
        self,
        tx: dict[str, Any],
        action: str,
        risk_score: float,
        ml_score: float,
        rule_score: float,
        anomaly_score: float,
        triggered_rules: list[dict[str, Any]],
        top_factors: list[dict[str, Any]],
    ) -> str:
        """Generate a concise Russian case summary for the analyst.

        Optionally calls an OpenAI-compatible LLM strictly for explanation (never for scoring),
        falling back cleanly to a structured deterministic summary.
        """
        if self.llm_cfg.get("enabled") and os.getenv("OPENAI_API_KEY"):
            llm_text = self._try_llm_summary(
                tx, action, risk_score, ml_score, rule_score, anomaly_score, triggered_rules, top_factors
            )
            if llm_text:
                return llm_text

        action_ru = {
            "ALLOW": "Пропустить (ALLOW)",
            "OTP": "Запросить подтверждение OTP (OTP)",
            "REVIEW": "Направить на ручную проверку (REVIEW)",
            "BLOCK": "Заблокировать перевод (BLOCK)",
        }.get(action, action)

        parts = [
            f"Решение агента: {action_ru}. Итоговый риск: {risk_score:.2f} "
            f"(ML: {ml_score:.2f}, Правила: {rule_score:.2f}, Аномалия: {anomaly_score:.2f})."
        ]
        if triggered_rules:
            rule_reasons = "; ".join(r["reason"] for r in triggered_rules[:3])
            parts.append(f"Сработавшие правила: {rule_reasons}")

        risk_up = [f["description"] for f in top_factors if f["shap_value"] > 0][:3]
        if risk_up:
            parts.append("Ключевые факторы риска (SHAP): " + "; ".join(risk_up) + ".")
        elif action == "ALLOW":
            parts.append("Поведенческий профиль и параметры перевода соответствуют норме клиента.")

        return " ".join(parts)

    def _try_llm_summary(
        self,
        tx: dict[str, Any],
        action: str,
        risk_score: float,
        ml_score: float,
        rule_score: float,
        anomaly_score: float,
        triggered_rules: list[dict[str, Any]],
        top_factors: list[dict[str, Any]],
    ) -> str | None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        api_base = self.llm_cfg.get("api_base", "https://api.openai.com/v1").rstrip("/")
        model = self.llm_cfg.get("model", "gpt-4o-mini")
        timeout = float(self.llm_cfg.get("timeout_seconds", 10))

        prompt = (
            "Ты ассистент антифрод-аналитика. Напиши краткое резюме кейса (2–3 предложения на русском) "
            "строго на основе переданных сигналов. Не меняй и не оспаривай решение.\n"
            f"Транзакция: {tx.get('type')} на сумму {tx.get('amount')}, канал {tx.get('channel')}, "
            f"отправитель {tx.get('nameOrig')}, получатель {tx.get('nameDest')}.\n"
            f"Решение: {action}, риск={risk_score:.2f} (ML={ml_score:.2f}, правила={rule_score:.2f}, аномалия={anomaly_score:.2f}).\n"
            f"Правила: {[r['reason'] for r in triggered_rules]}\n"
            f"Топ-факторы SHAP: {[f['description'] for f in top_factors[:4]]}"
        )
        try:
            resp = httpx.post(
                f"{api_base}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 220,
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return None
