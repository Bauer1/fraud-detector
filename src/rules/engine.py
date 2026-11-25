"""Level 1: Fast, transparent Rule Engine for hard limits, blacklists, velocity, and graph heuristics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config import load_config


@dataclass
class RuleEvaluation:
    rule_score: float
    hard_block: bool
    hits: list[dict[str, Any]] = field(default_factory=list)


class RuleEngine:
    """Evaluates deterministic anti-fraud rules over transaction metadata and computed features."""

    def __init__(self, rules_cfg: dict[str, Any] | None = None):
        cfg = rules_cfg if rules_cfg is not None else load_config()["rules"]
        self.hard_limit_amount = float(cfg.get("hard_limit_amount", 250000.0))
        self.night_high_amount = float(cfg.get("night_high_amount", 50000.0))
        self.night_hours = set(cfg.get("night_hours", [0, 1, 2, 3, 4, 5]))
        self.max_tx_1h = int(cfg.get("max_tx_1h", 6))
        self.max_amount_1h = float(cfg.get("max_amount_1h", 100000.0))
        self.drain_ratio_threshold = float(cfg.get("drain_ratio_threshold", 0.95))
        self.drain_min_amount = float(cfg.get("drain_min_amount", 15000.0))
        self.mule_in_degree_threshold = int(cfg.get("mule_in_degree_threshold", 5))
        self.shared_device_threshold = int(cfg.get("shared_device_accounts_threshold", 3))
        self.blacklisted_accounts = set(cfg.get("blacklisted_accounts", []))
        self.blacklisted_devices = set(cfg.get("blacklisted_devices", []))

    def evaluate(self, tx: dict[str, Any], features: dict[str, float]) -> RuleEvaluation:
        hits: list[dict[str, Any]] = []
        hard_block = False
        score = 0.0

        orig = str(tx.get("nameOrig") or "")
        dest = str(tx.get("nameDest") or "")
        device = str(tx.get("device_id") or "")
        amount = float(tx.get("amount") or features.get("amount", 0.0))
        tx_type = str(tx.get("type") or "")

        # 1. Blacklists (HARD BLOCK)
        if orig in self.blacklisted_accounts or dest in self.blacklisted_accounts:
            hard_block = True
            score = 1.0
            hits.append({
                "rule_name": "BLACKLISTED_ACCOUNT",
                "severity": "HARD_BLOCK",
                "score_contribution": 1.0,
                "reason": f"Аккаунт ({orig if orig in self.blacklisted_accounts else dest}) находится в чёрном списке.",
            })

        if device in self.blacklisted_devices:
            hard_block = True
            score = 1.0
            hits.append({
                "rule_name": "BLACKLISTED_DEVICE",
                "severity": "HARD_BLOCK",
                "score_contribution": 1.0,
                "reason": f"Устройство {device} находится в чёрном списке.",
            })

        # 2. Hard amount limit
        if amount >= self.hard_limit_amount:
            score += 0.55
            hits.append({
                "rule_name": "HARD_AMOUNT_LIMIT",
                "severity": "HIGH",
                "score_contribution": 0.55,
                "reason": f"Сумма перевода ({amount:,.0f}) превышает жёсткий лимит ({self.hard_limit_amount:,.0f}).",
            })

        # 3. Balance drain on TRANSFER / CASH_OUT
        drain_ratio = float(features.get("balance_drain_ratio", 0.0))
        if tx_type in ("TRANSFER", "CASH_OUT") and drain_ratio >= self.drain_ratio_threshold and amount >= self.drain_min_amount:
            score += 0.40
            hits.append({
                "rule_name": "ACCOUNT_DRAIN",
                "severity": "HIGH",
                "score_contribution": 0.40,
                "reason": f"Опустошение счёта ({drain_ratio:.0%} от баланса) на крупную сумму {amount:,.0f}.",
            })

        # 4. Velocity burst (1h)
        cnt_1h = float(features.get("orig_tx_count_1h", 0.0))
        sum_1h = float(features.get("orig_tx_sum_1h", 0.0))
        if cnt_1h >= self.max_tx_1h or (sum_1h + amount) >= self.max_amount_1h and cnt_1h >= 2:
            score += 0.45
            hits.append({
                "rule_name": "VELOCITY_BURST_1H",
                "severity": "HIGH",
                "score_contribution": 0.45,
                "reason": f"Высокая частота операций за 1 час ({int(cnt_1h)} операций, сумма {sum_1h:,.0f}).",
            })

        # 5. Nighttime high-value transfer with new device/recipient
        hour = int(features.get("hour_of_day", 12))
        is_new_dev = float(features.get("is_new_device", 0.0)) >= 0.5
        is_new_rec = float(features.get("is_new_recipient", 0.0)) >= 0.5
        if hour in self.night_hours and amount >= self.night_high_amount and (is_new_dev or is_new_rec):
            score += 0.35
            hits.append({
                "rule_name": "NIGHT_HIGH_RISK_TRANSFER",
                "severity": "MEDIUM",
                "score_contribution": 0.35,
                "reason": f"Крупный ночной перевод ({amount:,.0f} в {hour:02d}:00) с новым устройством или получателем.",
            })

        # 6. Graph: Shared device ring
        shared_accs = float(features.get("device_shared_accounts", 0.0))
        if shared_accs >= self.shared_device_threshold:
            score += 0.45
            hits.append({
                "rule_name": "SHARED_DEVICE_RING",
                "severity": "HIGH",
                "score_contribution": 0.45,
                "reason": f"Устройство используется несколькими аккаунтами ({int(shared_accs) + 1} шт.).",
            })

        # 7. Graph: Mule / Dropper destination
        dest_in_deg = float(features.get("dest_in_degree", 0.0))
        if dest_in_deg >= self.mule_in_degree_threshold and is_new_rec and tx_type in ("TRANSFER", "CASH_OUT"):
            score += 0.35
            hits.append({
                "rule_name": "MULE_RECIPIENT_PATTERN",
                "severity": "MEDIUM",
                "score_contribution": 0.35,
                "reason": f"Получатель {dest} получает переводы от {int(dest_in_deg)} уникальных отправителей (признак дроппера).",
            })

        # 8. Extreme z-score deviation
        zscore = float(features.get("amount_zscore", 0.0))
        if zscore >= 8.0 and is_new_rec:
            score += 0.30
            hits.append({
                "rule_name": "EXTREME_AMOUNT_DEVIATION",
                "severity": "MEDIUM",
                "score_contribution": 0.30,
                "reason": f"Сумма перевода резко превышает профиль клиента (z-score = {zscore:.1f}) на нового получателя.",
            })

        return RuleEvaluation(
            rule_score=float(min(score, 1.0)),
            hard_block=hard_block,
            hits=hits,
        )

    def score_batch(self, df) -> list[float]:
        """Vectorized/loop helper to compute rule scores on a DataFrame for evaluation & ablations."""
        scores = []
        for row in df.to_dict(orient="records"):
            ev = self.evaluate(row, row)
            scores.append(ev.rule_score)
        return scores
