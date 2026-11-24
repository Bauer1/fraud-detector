"""Stateful, strictly past-only Online Feature Store.

Used identically during chronological batch feature generation and real-time FastAPI scoring,
guaranteeing zero data leakage and zero training-serving skew.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from datetime import datetime
from typing import Any

import networkx as nx
import numpy as np

TX_TYPES = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
CHANNELS = ["MOBILE", "WEB", "API", "ATM"]

# Feature groups for ablation studies
TX_FEATURE_NAMES = [
    "amount",
    "log_amount",
    "hour_of_day",
    "is_night",
    "balance_drain_ratio",
    "orig_balance_zero_after",
    "orig_balance_error",
    "dest_balance_error",
    "is_foreign_ip",
    "type_TRANSFER",
    "type_CASH_OUT",
    "type_PAYMENT",
    "type_CASH_IN",
    "type_DEBIT",
    "channel_MOBILE",
    "channel_WEB",
    "channel_API",
    "channel_ATM",
]

BEHAVIORAL_FEATURE_NAMES = [
    "orig_tx_count_1h",
    "orig_tx_sum_1h",
    "orig_tx_count_24h",
    "orig_tx_sum_24h",
    "orig_tx_count_7d",
    "orig_tx_sum_7d",
    "amount_zscore",
    "amount_to_mean_ratio",
    "is_new_recipient",
    "is_new_device",
    "time_since_last_tx_hours",
    "orig_past_tx_count",
]

GRAPH_FEATURE_NAMES = [
    "dest_in_degree",
    "orig_out_degree",
    "device_shared_accounts",
    "dest_tx_count_24h",
    "is_mule_candidate",
]

ALL_FEATURE_NAMES = TX_FEATURE_NAMES + BEHAVIORAL_FEATURE_NAMES + GRAPH_FEATURE_NAMES

FEATURE_DESCRIPTIONS_RU: dict[str, str] = {
    "amount": "Сумма перевода",
    "log_amount": "Логарифм суммы перевода",
    "hour_of_day": "Час суток",
    "is_night": "Ночное время (00:00–05:59)",
    "balance_drain_ratio": "Доля списания от баланса отправителя",
    "orig_balance_zero_after": "Счёт отправителя обнулён после перевода",
    "orig_balance_error": "Расхождение баланса отправителя",
    "dest_balance_error": "Расхождение баланса получателя",
    "is_foreign_ip": "Иностранный IP-адрес",
    "type_TRANSFER": "Тип операции: перевод (TRANSFER)",
    "type_CASH_OUT": "Тип операции: снятие/вывод (CASH_OUT)",
    "type_PAYMENT": "Тип операции: платёж (PAYMENT)",
    "type_CASH_IN": "Тип операции: пополнение (CASH_IN)",
    "type_DEBIT": "Тип операции: дебет (DEBIT)",
    "channel_MOBILE": "Канал: мобильное приложение",
    "channel_WEB": "Канал: веб-банк",
    "channel_API": "Канал: API",
    "channel_ATM": "Канал: банкомат",
    "orig_tx_count_1h": "Количество переводов клиента за 1 час",
    "orig_tx_sum_1h": "Сумма переводов клиента за 1 час",
    "orig_tx_count_24h": "Количество переводов клиента за 24 часа",
    "orig_tx_sum_24h": "Сумма переводов клиента за 24 часа",
    "orig_tx_count_7d": "Количество переводов клиента за 7 дней",
    "orig_tx_sum_7d": "Сумма переводов клиента за 7 дней",
    "amount_zscore": "Отклонение суммы от нормы клиента (z-score)",
    "amount_to_mean_ratio": "Отношение суммы к среднему чеку клиента",
    "is_new_recipient": "Новый получатель (ранее не встречался)",
    "is_new_device": "Новое устройство клиента",
    "time_since_last_tx_hours": "Время с прошлой операции клиента (часы)",
    "orig_past_tx_count": "Общее число прошлых операций клиента",
    "dest_in_degree": "Входящая степень получателя (уникальных отправителей)",
    "orig_out_degree": "Исходящая степень отправителя (уникальных получателей)",
    "device_shared_accounts": "Число разных аккаунтов на этом устройстве",
    "dest_tx_count_24h": "Входящих переводов получателю за 24 часа",
    "is_mule_candidate": "Признак дроппера (высокая входящая степень получателя)",
}


def _extract_ts_seconds(tx: dict[str, Any]) -> float:
    ts = tx.get("timestamp")
    if isinstance(ts, datetime):
        return ts.timestamp()
    if isinstance(ts, str):
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    # Fallback to step (hours)
    return float(tx.get("step", 0)) * 3600.0


class OnlineFeatureStore:
    """Maintains incremental customer, device, and graph state strictly from past events."""

    def __init__(self, mule_in_degree_threshold: int = 5):
        self.mule_in_degree_threshold = mule_in_degree_threshold

        # Rolling windows per sender: deque of (ts_sec, amount)
        self._orig_window_7d: dict[str, deque[tuple[float, float]]] = defaultdict(deque)
        self._orig_sum_7d: dict[str, float] = defaultdict(float)

        # Welford online mean/variance for sender amount
        self._orig_count: dict[str, int] = defaultdict(int)
        self._orig_mean: dict[str, float] = defaultdict(float)
        self._orig_m2: dict[str, float] = defaultdict(float)
        self._orig_last_ts: dict[str, float] = {}

        # Known recipients and devices per sender
        self._orig_recipients: dict[str, set[str]] = defaultdict(set)
        self._orig_devices: dict[str, set[str]] = defaultdict(set)

        # Graph state (incremental, past-only)
        self._dest_senders: dict[str, set[str]] = defaultdict(set)
        self._device_accounts: dict[str, set[str]] = defaultdict(set)
        self._dest_window_24h: dict[str, deque[float]] = defaultdict(deque)

    def compute_features(self, tx: dict[str, Any], update_state: bool = True) -> dict[str, float]:
        """Compute all features using ONLY state prior to `tx`, then optionally update state."""
        orig = str(tx["nameOrig"])
        dest = str(tx["nameDest"])
        device = str(tx.get("device_id", "DEV_UNKNOWN"))
        tx_type = str(tx["type"])
        channel = str(tx.get("channel", "MOBILE"))
        ip_country = str(tx.get("ip_country", "RU"))
        amount = float(tx["amount"])
        old_orig = float(tx.get("oldbalanceOrg", 0.0))
        new_orig = float(tx.get("newbalanceOrig", 0.0))
        old_dest = float(tx.get("oldbalanceDest", 0.0))
        new_dest = float(tx.get("newbalanceDest", 0.0))
        step = int(tx.get("step", 0))
        ts_sec = _extract_ts_seconds(tx)

        # Determine hour of day
        ts_val = tx.get("timestamp")
        if isinstance(ts_val, datetime):
            hour = ts_val.hour
        elif isinstance(ts_val, str):
            hour = datetime.fromisoformat(ts_val.replace("Z", "+00:00")).hour
        else:
            hour = step % 24

        # --- 1. Transactional features ---
        drain_ratio = amount / (old_orig + 1.0) if old_orig > 0 else 0.0
        drain_ratio = min(drain_ratio, 5.0)
        orig_zero_after = 1.0 if (old_orig > 0 and new_orig <= 1.0) else 0.0
        orig_bal_err = abs((old_orig - amount) - new_orig) if tx_type != "CASH_IN" else abs((old_orig + amount) - new_orig)
        dest_bal_err = abs((old_dest + amount) - new_dest)

        feats: dict[str, float] = {
            "amount": amount,
            "log_amount": math.log1p(max(amount, 0.0)),
            "hour_of_day": float(hour),
            "is_night": 1.0 if hour in (0, 1, 2, 3, 4, 5) else 0.0,
            "balance_drain_ratio": float(drain_ratio),
            "orig_balance_zero_after": orig_zero_after,
            "orig_balance_error": float(orig_bal_err),
            "dest_balance_error": float(dest_bal_err),
            "is_foreign_ip": 0.0 if ip_country == "RU" else 1.0,
        }
        for t in TX_TYPES:
            feats[f"type_{t}"] = 1.0 if tx_type == t else 0.0
        for c in CHANNELS:
            feats[f"channel_{c}"] = 1.0 if channel == c else 0.0

        # --- 2. Behavioral rolling features (STRICTLY past transactions) ---
        win_7d = self._orig_window_7d[orig]
        cutoff_7d = ts_sec - 7 * 86400.0
        while win_7d and win_7d[0][0] < cutoff_7d:
            _, old_amt = win_7d.popleft()
            self._orig_sum_7d[orig] -= old_amt
        if self._orig_sum_7d[orig] < 0:
            self._orig_sum_7d[orig] = 0.0

        cutoff_1h = ts_sec - 3600.0
        cutoff_24h = ts_sec - 86400.0
        cnt_1h = 0
        sum_1h = 0.0
        cnt_24h = 0
        sum_24h = 0.0
        for prev_ts, prev_amt in reversed(win_7d):
            if prev_ts >= cutoff_24h:
                cnt_24h += 1
                sum_24h += prev_amt
                if prev_ts >= cutoff_1h:
                    cnt_1h += 1
                    sum_1h += prev_amt
            else:
                break

        cnt_7d = len(win_7d)
        sum_7d = self._orig_sum_7d[orig]

        # Deviation from customer's historical norm
        past_n = self._orig_count[orig]
        if past_n >= 2:
            mean_amt = self._orig_mean[orig]
            var_amt = self._orig_m2[orig] / max(past_n - 1, 1)
            std_amt = math.sqrt(max(var_amt, 1.0))
            zscore = (amount - mean_amt) / std_amt
            ratio_mean = amount / max(mean_amt, 1.0)
        elif past_n == 1:
            mean_amt = self._orig_mean[orig]
            zscore = (amount - mean_amt) / max(mean_amt * 0.5, 100.0)
            ratio_mean = amount / max(mean_amt, 1.0)
        else:
            zscore = 0.0
            ratio_mean = 1.0

        zscore = float(np.clip(zscore, -5.0, 50.0))
        ratio_mean = float(np.clip(ratio_mean, 0.0, 100.0))

        is_new_recipient = 0.0 if (dest in self._orig_recipients[orig] or past_n == 0) else 1.0
        is_new_device = 0.0 if (device in self._orig_devices[orig] or past_n == 0) else 1.0

        last_ts = self._orig_last_ts.get(orig)
        time_since_last_h = (ts_sec - last_ts) / 3600.0 if last_ts is not None else 168.0
        time_since_last_h = float(np.clip(time_since_last_h, 0.0, 720.0))

        feats.update({
            "orig_tx_count_1h": float(cnt_1h),
            "orig_tx_sum_1h": float(sum_1h),
            "orig_tx_count_24h": float(cnt_24h),
            "orig_tx_sum_24h": float(sum_24h),
            "orig_tx_count_7d": float(cnt_7d),
            "orig_tx_sum_7d": float(sum_7d),
            "amount_zscore": zscore,
            "amount_to_mean_ratio": ratio_mean,
            "is_new_recipient": is_new_recipient,
            "is_new_device": is_new_device,
            "time_since_last_tx_hours": time_since_last_h,
            "orig_past_tx_count": float(past_n),
        })

        # --- 3. Graph features (STRICTLY past state) ---
        dest_win_24h = self._dest_window_24h[dest]
        while dest_win_24h and dest_win_24h[0] < cutoff_24h:
            dest_win_24h.popleft()

        dest_in_deg = len(self._dest_senders[dest])
        orig_out_deg = len(self._orig_recipients[orig])
        device_shared = len(self._device_accounts[device] - {orig})
        dest_cnt_24h = len(dest_win_24h)
        is_mule_cand = 1.0 if dest_in_deg >= self.mule_in_degree_threshold else 0.0

        feats.update({
            "dest_in_degree": float(dest_in_deg),
            "orig_out_degree": float(orig_out_deg),
            "device_shared_accounts": float(device_shared),
            "dest_tx_count_24h": float(dest_cnt_24h),
            "is_mule_candidate": is_mule_cand,
        })

        # --- 4. Update state AFTER computing features ---
        if update_state:
            self._update_state(orig, dest, device, ts_sec, amount)

        return feats

    def _update_state(self, orig: str, dest: str, device: str, ts_sec: float, amount: float) -> None:
        self._orig_window_7d[orig].append((ts_sec, amount))
        self._orig_sum_7d[orig] += amount

        # Welford update
        n = self._orig_count[orig] + 1
        self._orig_count[orig] = n
        delta = amount - self._orig_mean[orig]
        self._orig_mean[orig] += delta / n
        delta2 = amount - self._orig_mean[orig]
        self._orig_m2[orig] += delta * delta2

        self._orig_last_ts[orig] = ts_sec
        self._orig_recipients[orig].add(dest)
        self._orig_devices[orig].add(device)

        self._dest_senders[dest].add(orig)
        self._device_accounts[device].add(orig)
        self._dest_window_24h[dest].append(ts_sec)

    def build_networkx_graph(self, top_dest_min_degree: int = 3) -> nx.DiGraph:
        """Export high-activity recipient subgraphs as a NetworkX DiGraph for analytics/inspection."""
        g = nx.DiGraph()
        for dest, senders in self._dest_senders.items():
            if len(senders) >= top_dest_min_degree:
                g.add_node(dest, role="recipient", in_degree=len(senders))
                for s in senders:
                    g.add_node(s, role="sender")
                    g.add_edge(s, dest)
        return g
