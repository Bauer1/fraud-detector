"""Stateful, thread-safe, strictly point-in-time Online Feature Store with Pre-Decision Availability Contract.

Implements a shared batch/online specification with explicit feature provenance, leakage guards,
and cold-start / missingness observability:

1. Pre-Decision Feature Availability Contract:
   - Raw identifiers (`nameOrig`, `nameDest`, `device_id`) are used strictly as state lookup keys
     and NEVER enter model feature vectors.
   - Forbidden post-factum / target fields (`isFlaggedFraud`, `isFraud`, `chargeback`, `analyst_label`,
     `investigation_result`) are rejected in online scoring (`ForbiddenLeakageFieldError`).
   - Post-settlement balance fields (`newbalanceOrig`, `newbalanceDest`) and unverified destination
     balances (`oldbalanceDest`) are excluded from pre-decision model feature sets by default
     (`ENRICHED_PRE_DECISION_FEATURE_NAMES`, `PAYSIM_PRE_DECISION_FEATURE_NAMES`).
   - Absent telemetry (`device_id=None`, `channel=None`, `ip_country=None`, e.g. in Original PaySim CSV)
     is never fabricated and never links unrelated customers via a placeholder `"DEV_UNKNOWN"` device.

2. Event Ordering, Deduplication & Tie-Breaking:
   - Exact retry (`tx_id` already committed with identical canonical payload): idempotent, returns cached features.
   - Conflicting `tx_id` reuse: raises `DuplicateTransactionConflictError`.
   - Out-of-order events exceeding sender/global watermarks: raises `StaleEventError`.
   - Equal timestamps / steps: deterministic tie-breaking by `(utc_timestamp, tx_id)` in batch mode
     (`(step, source_row_idx)` for PaySim CSV) and `RLock`-serialized arrival order in online mode.
"""
from __future__ import annotations

import bisect
import math
import threading
from collections import OrderedDict, defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd

SIMULATION_EPOCH_UTC = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
COLD_START_TIME_SINCE_LAST_HOURS = 168.0
MAX_TIME_SINCE_LAST_HOURS = 720.0
WINDOW_EPS_SEC = 1e-6
SCHEMA_VERSION = "2.0.0-pre-decision"

TX_TYPES = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
OUTGOING_TX_TYPES = frozenset({"TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"})
CHANNELS = ["MOBILE", "WEB", "API", "ATM"]
UNKNOWN_DEVICE_SENTINELS = frozenset({"", "NONE", "NULL", "UNKNOWN", "DEV_UNKNOWN"})

KNOWN_TX_FIELDS = frozenset({
    "tx_id",
    "step",
    "timestamp",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "device_id",
    "channel",
    "ip_country",
})

FORBIDDEN_ONLINE_LEAKAGE_FIELDS = frozenset({
    "isFlaggedFraud",
    "isFraud",
    "chargeback",
    "analyst_label",
    "label_status",
    "investigation_result",
})

ALLOWED_BATCH_META_FIELDS = frozenset({
    "isFraud",
    "isFlaggedFraud",
    "fraud_scenario",
    "source_row_idx",
    "cust_idx",
    "offset_sec",
    "force_drain",
})

# --- Source-Aware Pre-Decision Feature Groups ---

# 1. Core PaySim pre-decision transactional features (no telemetry, no post-tx balances)
PAYSIM_TX_FEATURE_NAMES = [
    "amount",
    "log_amount",
    "hour_of_day",
    "is_night",
    "balance_drain_ratio",
    "type_TRANSFER",
    "type_CASH_OUT",
    "type_PAYMENT",
    "type_CASH_IN",
    "type_DEBIT",
]

# 2. Enriched pre-decision transactional features (adds channel & IP telemetry; still NO post-tx balances)
ENRICHED_TX_FEATURE_NAMES = PAYSIM_TX_FEATURE_NAMES + [
    "is_foreign_ip",
    "channel_MOBILE",
    "channel_WEB",
    "channel_API",
    "channel_ATM",
]

# 3. Post-settlement balance features (EXCLUDED by default; kept only for explicit leakage-audit ablations)
POST_TX_BALANCE_LEAKAGE_FEATURES = [
    "orig_balance_zero_after",
    "orig_balance_error",
    "dest_balance_error",
]

# Full computed transactional keys (for backward compatibility & leakage audit)
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

PAYSIM_BEHAVIORAL_FEATURE_NAMES = [
    "orig_tx_count_1h",
    "orig_tx_sum_1h",
    "orig_tx_count_24h",
    "orig_tx_sum_24h",
    "orig_tx_count_7d",
    "orig_tx_sum_7d",
    "amount_zscore",
    "amount_to_mean_ratio",
    "is_new_recipient",
    "time_since_last_tx_hours",
    "orig_past_tx_count",
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

PAYSIM_GRAPH_FEATURE_NAMES = [
    "dest_in_degree",
    "orig_out_degree",
    "dest_tx_count_24h",
    "is_mule_candidate",
]

GRAPH_FEATURE_NAMES = [
    "dest_in_degree",
    "orig_out_degree",
    "device_shared_accounts",
    "dest_tx_count_24h",
    "is_mule_candidate",
]

# Clean default feature sets for production / pilot training
PAYSIM_PRE_DECISION_FEATURE_NAMES = (
    PAYSIM_TX_FEATURE_NAMES + PAYSIM_BEHAVIORAL_FEATURE_NAMES + PAYSIM_GRAPH_FEATURE_NAMES
)
ENRICHED_PRE_DECISION_FEATURE_NAMES = (
    ENRICHED_TX_FEATURE_NAMES + BEHAVIORAL_FEATURE_NAMES + GRAPH_FEATURE_NAMES
)

# Complete dictionary key order produced by OnlineFeatureStore.compute_features
ALL_FEATURE_NAMES = TX_FEATURE_NAMES + BEHAVIORAL_FEATURE_NAMES + GRAPH_FEATURE_NAMES


def resolve_model_feature_names(
    dataset_source: str = "enriched_simulation",
    feature_contract_cfg: dict[str, Any] | None = None,
) -> list[str]:
    """Return the strictly allowed model input features for a dataset source and availability contract."""
    fc = feature_contract_cfg or {}
    allow_post = bool(fc.get("allow_post_tx_balances", False))
    allow_dest = bool(fc.get("allow_dest_balances", False))
    allow_pre_orig = bool(fc.get("allow_sender_pre_balance", True))

    if dataset_source == "paysim_csv":
        base = list(PAYSIM_PRE_DECISION_FEATURE_NAMES)
    else:
        base = list(ENRICHED_PRE_DECISION_FEATURE_NAMES)

    if not allow_pre_orig and "balance_drain_ratio" in base:
        base.remove("balance_drain_ratio")

    if allow_post:
        for col in ("orig_balance_zero_after", "orig_balance_error"):
            if col not in base:
                base.append(col)
        if allow_dest and "dest_balance_error" not in base:
            base.append(col := "dest_balance_error")

    return base


def resolve_tx_only_feature_names(
    dataset_source: str = "enriched_simulation",
    feature_contract_cfg: dict[str, Any] | None = None,
) -> list[str]:
    """Return transactional-only baseline features respecting the pre-decision contract."""
    allowed = set(resolve_model_feature_names(dataset_source, feature_contract_cfg))
    candidates = PAYSIM_TX_FEATURE_NAMES if dataset_source == "paysim_csv" else TX_FEATURE_NAMES
    return [f for f in candidates if f in allowed]


FEATURE_DESCRIPTIONS_RU: dict[str, str] = {
    "amount": "Сумма перевода",
    "log_amount": "Логарифм суммы перевода",
    "hour_of_day": "Час суток (UTC)",
    "is_night": "Ночное время (00:00–05:59 UTC)",
    "balance_drain_ratio": "Доля суммы от доступного баланса отправителя до операции",
    "orig_balance_zero_after": "[Аудит утечки] Счёт отправителя обнулён после проводки",
    "orig_balance_error": "[Аудит утечки] Расхождение баланса отправителя до/после",
    "dest_balance_error": "[Аудит утечки] Расхождение баланса получателя до/после",
    "is_foreign_ip": "Иностранный IP-адрес",
    "type_TRANSFER": "Тип операции: перевод (TRANSFER)",
    "type_CASH_OUT": "Тип операции: снятие/вывод (CASH_OUT)",
    "type_PAYMENT": "Тип операции: платёж мерчанту (PAYMENT)",
    "type_CASH_IN": "Тип операции: пополнение счёта (CASH_IN)",
    "type_DEBIT": "Тип операции: прямое списание (DEBIT)",
    "channel_MOBILE": "Канал: мобильное приложение",
    "channel_WEB": "Канал: веб-банк",
    "channel_API": "Канал: API",
    "channel_ATM": "Канал: банкомат",
    "orig_tx_count_1h": "Количество исходящих операций клиента за 1 час",
    "orig_tx_sum_1h": "Сумма исходящих операций клиента за 1 час",
    "orig_tx_count_24h": "Количество исходящих операций клиента за 24 часа",
    "orig_tx_sum_24h": "Сумма исходящих операций клиента за 24 часа",
    "orig_tx_count_7d": "Количество исходящих операций клиента за 7 дней",
    "orig_tx_sum_7d": "Сумма исходящих операций клиента за 7 дней",
    "amount_zscore": "Отклонение суммы от прошлой нормы клиента (z-score)",
    "amount_to_mean_ratio": "Отношение суммы к прошлому среднему чеку клиента",
    "is_new_recipient": "Новый получатель (ранее не встречался у клиента)",
    "is_new_device": "Новое устройство клиента",
    "time_since_last_tx_hours": "Время с прошлой операции клиента (часы)",
    "orig_past_tx_count": "Число прошлых исходящих операций клиента",
    "dest_in_degree": "Входящая степень P2P-получателя (уникальных отправителей)",
    "orig_out_degree": "Исходящая степень отправителя (уникальных получателей)",
    "device_shared_accounts": "Число других аккаунтов на этом устройстве",
    "dest_tx_count_24h": "Входящих P2P-переводов получателю за 24 часа",
    "is_mule_candidate": "Признак дроппера (высокая входящая степень P2P-получателя)",
}

FEATURE_AVAILABILITY_REGISTRY: dict[str, dict[str, str]] = {
    f: {
        "group": (
            "transactional"
            if f in TX_FEATURE_NAMES
            else ("behavioral" if f in BEHAVIORAL_FEATURE_NAMES else "graph")
        ),
        "source": (
            "post_settlement_balance"
            if f in POST_TX_BALANCE_LEAKAGE_FEATURES
            else (
                "sender_pre_balance"
                if f == "balance_drain_ratio"
                else (
                    "extended_telemetry"
                    if f
                    in {
                        "is_foreign_ip",
                        "channel_MOBILE",
                        "channel_WEB",
                        "channel_API",
                        "channel_ATM",
                        "is_new_device",
                        "device_shared_accounts",
                    }
                    else "paysim_core_point_in_time"
                )
            )
        ),
        "availability_status": (
            "EXCLUDED_POST_TX_LEAKAGE"
            if f in POST_TX_BALANCE_LEAKAGE_FEATURES
            else (
                "CONDITIONAL_PRE_DECISION_BALANCE"
                if f == "balance_drain_ratio"
                else (
                    "VERIFIED_PRE_DECISION_TELEMETRY"
                    if f not in PAYSIM_PRE_DECISION_FEATURE_NAMES
                    else "VERIFIED_PRE_DECISION_CORE"
                )
            )
        ),
        "description_ru": FEATURE_DESCRIPTIONS_RU.get(f, f),
    }
    for f in ALL_FEATURE_NAMES
}

FEATURE_CONTRACT: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "timezone": "UTC",
    "naive_datetime_assumption": "UTC",
    "step_fallback_epoch_utc": SIMULATION_EPOCH_UTC.isoformat(),
    "rolling_window_bounds": "[t - window, t] inclusive (strictly past committed events with ts <= t)",
    "same_timestamp_policy": {
        "batch_enriched": "Deterministic sort by (utc_timestamp, tx_id)",
        "batch_paysim_csv": "Deterministic sort by (step, source_row_idx) via PAYSIM_{row_idx:08d}",
        "online": "Arrival-serialized causal order under RLock",
    },
    "excluded_leakage_inputs": sorted(
        list(FORBIDDEN_ONLINE_LEAKAGE_FIELDS) + ["newbalanceOrig", "newbalanceDest", "raw_nameOrig", "raw_nameDest"]
    ),
    "excluded_post_tx_features_by_default": POST_TX_BALANCE_LEAKAGE_FEATURES,
    "paysim_pre_decision_feature_count": len(PAYSIM_PRE_DECISION_FEATURE_NAMES),
    "enriched_pre_decision_feature_count": len(ENRICHED_PRE_DECISION_FEATURE_NAMES),
    "tx_type_semantics": {
        "CASH_IN": "Credit/deposit to sender; drain=0, does not increment outgoing spend velocity or P2P mule degree",
        "PAYMENT": "Outgoing payment to merchant; debits sender, excludes merchant from P2P mule degree",
        "CASH_OUT": "Outgoing withdrawal; debits sender, tracked in sender outgoing windows",
        "TRANSFER": "P2P/account transfer; full sender behavioral and recipient P2P graph tracking",
        "DEBIT": "Outgoing direct debit; full sender outgoing behavioral tracking",
    },
    "cold_start_defaults": {
        "orig_past_tx_count": 0.0,
        "rolling_counts_and_sums": 0.0,
        "amount_zscore": 0.0,
        "amount_to_mean_ratio": 1.0,
        "is_new_recipient": 0.0,
        "is_new_device": 0.0,
        "time_since_last_tx_hours": COLD_START_TIME_SINCE_LAST_HOURS,
    },
}


class FeatureStoreError(ValueError):
    """Base error for feature contract or event ordering violations."""


class StaleEventError(FeatureStoreError):
    """Raised when an incoming transaction timestamp violates event-ordering watermarks."""


class DuplicateTransactionConflictError(FeatureStoreError):
    """Raised when a duplicate tx_id arrives with a different canonical transaction payload."""


class UnknownFeatureFieldError(FeatureStoreError):
    """Raised in strict mode when unknown fields are supplied in a transaction payload."""


class ForbiddenLeakageFieldError(UnknownFeatureFieldError):
    """Raised when forbidden target/post-investigation fields (e.g. isFlaggedFraud, chargeback) are supplied."""


def normalize_event_datetime(tx: dict[str, Any]) -> datetime:
    """Resolve transaction event time as an explicit UTC datetime."""
    ts = tx.get("timestamp")
    if ts is not None and not (isinstance(ts, float) and math.isnan(ts)):
        if isinstance(ts, pd.Timestamp):
            dt = ts.to_pydatetime()
        elif isinstance(ts, datetime):
            dt = ts
        elif isinstance(ts, str) and ts.strip():
            dt = datetime.fromisoformat(ts.strip().replace("Z", "+00:00"))
        else:
            raise FeatureStoreError(f"Unsupported timestamp value: {ts!r}")

        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    step_raw = tx.get("step", 0)
    step = 0 if step_raw is None or (isinstance(step_raw, float) and math.isnan(step_raw)) else int(step_raw)
    if step < 0:
        raise FeatureStoreError(f"Transaction step must be >= 0, got {step}")
    return SIMULATION_EPOCH_UTC + timedelta(hours=step)


def _is_missing(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    if isinstance(val, str) and not val.strip():
        return True
    return False


def _safe_float(val: Any, default: float = 0.0, *, field_name: str = "value", allow_zero: bool = True) -> float:
    if val is None:
        res = default
    else:
        try:
            res = float(val)
        except (TypeError, ValueError) as exc:
            raise FeatureStoreError(f"Invalid numeric value for {field_name}: {val!r}") from exc
        if math.isnan(res):
            res = default
    if not math.isfinite(res):
        raise FeatureStoreError(f"Non-finite numeric value for {field_name}: {val!r}")
    if allow_zero and res < 0.0:
        raise FeatureStoreError(f"Negative value not allowed for {field_name}: {res}")
    if not allow_zero and res <= 0.0:
        raise FeatureStoreError(f"Positive value required for {field_name}, got {res}")
    return res


def _safe_optional_float(val: Any, *, field_name: str) -> float | None:
    if _is_missing(val):
        return None
    return _safe_float(val, 0.0, field_name=field_name, allow_zero=True)


def _safe_str(val: Any, default: str) -> str:
    if val is None:
        return default
    if isinstance(val, float) and math.isnan(val):
        return default
    s = str(val).strip()
    return s if s else default


def _normalize_optional_device(val: Any) -> str | None:
    s = _safe_str(val, "")
    if not s or s.upper() in UNKNOWN_DEVICE_SENTINELS:
        return None
    return s


def canonical_tx_fingerprint(tx: dict[str, Any]) -> tuple[Any, ...]:
    """Build canonical pre-decision tuple used to distinguish exact retries from conflicting tx_id reuse."""
    dt_utc = normalize_event_datetime(tx)
    old_orig = _safe_optional_float(tx.get("oldbalanceOrg"), field_name="oldbalanceOrg")
    dev = _normalize_optional_device(tx.get("device_id"))
    ch = _safe_str(tx.get("channel"), "").upper()
    if ch == "UNKNOWN":
        ch = ""
    ip_c = _safe_str(tx.get("ip_country"), "").upper()
    return (
        _safe_str(tx.get("type"), "UNKNOWN").upper(),
        round(_safe_float(tx.get("amount"), field_name="amount", allow_zero=False), 2),
        _safe_str(tx.get("nameOrig"), ""),
        _safe_str(tx.get("nameDest"), ""),
        round(old_orig, 2) if old_orig is not None else None,
        dev,
        ch or None,
        ip_c or None,
        round(dt_utc.timestamp(), 3),
    )


def _is_merchant_counterparty(dest: str, tx_type: str) -> bool:
    return dest.startswith("M") or tx_type == "PAYMENT"


class OnlineFeatureStore:
    """Thread-safe, incremental customer, device, and graph feature store strictly from past events."""

    def __init__(
        self,
        mule_in_degree_threshold: int = 5,
        max_sender_out_of_order_sec: float | None = 300.0,
        max_global_staleness_sec: float | None = 86400.0,
        max_idempotency_entries: int = 50000,
        state_origin: str = "empty",
        as_of_timestamp: str | None = None,
    ):
        self.mule_in_degree_threshold = int(mule_in_degree_threshold)
        self.max_sender_out_of_order_sec = max_sender_out_of_order_sec
        self.max_global_staleness_sec = max_global_staleness_sec
        self.max_idempotency_entries = max(100, int(max_idempotency_entries))

        # Provenance & observability metadata
        self.state_origin: str = state_origin
        self.as_of_timestamp: str | None = as_of_timestamp
        self.ingested_events: int = 0
        self.cold_start_events: int = 0
        self._missing_field_counts: dict[str, int] = defaultdict(int)

        # Rolling windows per sender: sorted deque of (ts_sec, amount) for outgoing operations
        self._orig_window_7d: dict[str, deque[tuple[float, float]]] = defaultdict(deque)
        self._orig_sum_7d: dict[str, float] = defaultdict(float)

        # Welford online mean/variance for sender outgoing amount
        self._orig_count: dict[str, int] = defaultdict(int)
        self._orig_mean: dict[str, float] = defaultdict(float)
        self._orig_m2: dict[str, float] = defaultdict(float)

        # Activity watermarks (tracks ALL valid events including CASH_IN)
        self._orig_last_ts: dict[str, float] = {}
        self._global_max_ts: float | None = None

        # Known recipients and devices per sender
        self._orig_recipients: dict[str, set[str]] = defaultdict(set)
        self._orig_devices: dict[str, set[str]] = defaultdict(set)

        # Graph state (P2P / non-merchant recipients & device sharing)
        self._dest_senders: dict[str, set[str]] = defaultdict(set)
        self._device_accounts: dict[str, set[str]] = defaultdict(set)
        self._dest_window_24h: dict[str, deque[float]] = defaultdict(deque)

        # Bounded LRU idempotency registry: tx_id -> (fingerprint, features_dict)
        self._processed_tx: OrderedDict[str, tuple[tuple[Any, ...], dict[str, float]]] = OrderedDict()

        self._lock = threading.RLock()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.RLock()
        if not hasattr(self, "_processed_tx"):
            self._processed_tx = OrderedDict()
        if not hasattr(self, "state_origin"):
            self.state_origin = "legacy"
        if not hasattr(self, "as_of_timestamp"):
            self.as_of_timestamp = None
        if not hasattr(self, "ingested_events"):
            self.ingested_events = 0
        if not hasattr(self, "cold_start_events"):
            self.cold_start_events = 0
        if not hasattr(self, "_missing_field_counts"):
            self._missing_field_counts = defaultdict(int)
        if not hasattr(self, "_global_max_ts"):
            self._global_max_ts = max(self._orig_last_ts.values()) if self._orig_last_ts else None

    def get_observability_stats(self) -> dict[str, Any]:
        """Return feature store observability metrics (cold-start rate, missingness, entity cardinality)."""
        with self._lock:
            n = self.ingested_events
            return {
                "schema_version": SCHEMA_VERSION,
                "state_origin": self.state_origin,
                "as_of_timestamp": self.as_of_timestamp,
                "ingested_events": n,
                "cold_start_events": self.cold_start_events,
                "cold_start_rate": round(self.cold_start_events / n, 4) if n > 0 else 0.0,
                "missing_optional_counts": dict(self._missing_field_counts),
                "unique_senders_tracked": len(self._orig_last_ts),
                "unique_p2p_recipients_tracked": len(self._dest_senders),
                "unique_devices_tracked": len(self._device_accounts),
                "idempotency_cache_entries": len(self._processed_tx),
            }

    @classmethod
    def from_history(
        cls,
        history: pd.DataFrame | Iterable[dict[str, Any]],
        *,
        as_of_timestamp: datetime | str | None = None,
        mule_in_degree_threshold: int = 5,
        max_sender_out_of_order_sec: float | None = 300.0,
        max_global_staleness_sec: float | None = 86400.0,
        max_idempotency_entries: int = 50000,
        state_origin: str = "historical_warmup",
    ) -> "OnlineFeatureStore":
        """Construct a clean initial store strictly from historical events up to `as_of_timestamp`."""
        cutoff_dt: datetime | None = None
        if as_of_timestamp is not None:
            cutoff_dt = normalize_event_datetime({"timestamp": as_of_timestamp})

        if isinstance(history, pd.DataFrame):
            records = history.to_dict(orient="records")
        else:
            records = list(history)

        filtered: list[tuple[float, str, int, dict[str, Any]]] = []
        for idx, rec in enumerate(records):
            dt = normalize_event_datetime(rec)
            if cutoff_dt is not None and dt > cutoff_dt:
                continue
            tx_id = _safe_str(rec.get("tx_id"), f"__idx_{idx:09d}")
            filtered.append((dt.timestamp(), tx_id, idx, rec))

        filtered.sort(key=lambda item: (item[0], item[1], item[2]))

        # Use batch-safe out-of-order settings while replaying sorted historical warm-up, then restore live limits
        store = cls(
            mule_in_degree_threshold=mule_in_degree_threshold,
            max_sender_out_of_order_sec=None,
            max_global_staleness_sec=None,
            max_idempotency_entries=max_idempotency_entries,
            state_origin=state_origin,
            as_of_timestamp=cutoff_dt.isoformat() if cutoff_dt else None,
        )
        for _, _, _, rec in filtered:
            store.compute_features(rec, update_state=True)

        store.max_sender_out_of_order_sec = max_sender_out_of_order_sec
        store.max_global_staleness_sec = max_global_staleness_sec
        return store

    def get_cached_features(self, tx: dict[str, Any]) -> dict[str, float] | None:
        """Return cached features if `tx_id` was already committed; raise if payload conflicts."""
        tx_id_raw = tx.get("tx_id")
        if tx_id_raw is None:
            return None
        tx_id = _safe_str(tx_id_raw, "")
        if not tx_id:
            return None

        with self._lock:
            existing = self._processed_tx.get(tx_id)
            if existing is None:
                return None
            prev_fp, prev_feats = existing
            cur_fp = canonical_tx_fingerprint(tx)
            if cur_fp != prev_fp:
                raise DuplicateTransactionConflictError(
                    f"Conflicting transaction payload for already-processed tx_id='{tx_id}'"
                )
            self._processed_tx.move_to_end(tx_id)
            return dict(prev_feats)

    def _validate_fields_and_order(self, tx: dict[str, Any], ts_sec: float, orig: str, strict_fields: bool) -> None:
        if strict_fields:
            leaked = set(tx.keys()) & FORBIDDEN_ONLINE_LEAKAGE_FIELDS
            if leaked:
                raise ForbiddenLeakageFieldError(
                    f"Forbidden target/post-decision leakage fields supplied to online scoring: {sorted(leaked)}"
                )
            unknown = set(tx.keys()) - KNOWN_TX_FIELDS
            if unknown:
                raise UnknownFeatureFieldError(
                    f"Unknown transaction fields are not allowed: {sorted(unknown)}"
                )

        last_orig_ts = self._orig_last_ts.get(orig)
        if (
            self.max_sender_out_of_order_sec is not None
            and last_orig_ts is not None
            and ts_sec < last_orig_ts - self.max_sender_out_of_order_sec - WINDOW_EPS_SEC
        ):
            delta = last_orig_ts - ts_sec
            raise StaleEventError(
                f"Transaction for sender '{orig}' is out of order by {delta:.1f}s "
                f"(max allowed: {self.max_sender_out_of_order_sec:.1f}s)"
            )

        if (
            self.max_global_staleness_sec is not None
            and self._global_max_ts is not None
            and ts_sec < self._global_max_ts - self.max_global_staleness_sec - WINDOW_EPS_SEC
        ):
            delta = self._global_max_ts - ts_sec
            raise StaleEventError(
                f"Transaction timestamp lags stream watermark by {delta:.1f}s "
                f"(max allowed: {self.max_global_staleness_sec:.1f}s)"
            )

    def compute_features(
        self,
        tx: dict[str, Any],
        update_state: bool = True,
        *,
        strict_fields: bool = False,
    ) -> dict[str, float]:
        """Compute all features using ONLY state prior to `tx`, then optionally commit state atomically."""
        with self._lock:
            cached = self.get_cached_features(tx)
            if cached is not None:
                return cached

            orig = _safe_str(tx.get("nameOrig"), "")
            dest = _safe_str(tx.get("nameDest"), "")
            if not orig or not dest:
                raise FeatureStoreError("Both 'nameOrig' and 'nameDest' must be non-empty strings")

            device = _normalize_optional_device(tx.get("device_id"))
            tx_type = _safe_str(tx.get("type"), "UNKNOWN").upper()
            channel_raw = _safe_str(tx.get("channel"), "").upper()
            channel = "" if channel_raw == "UNKNOWN" else channel_raw
            ip_country = _safe_str(tx.get("ip_country"), "").upper()

            amount = _safe_float(tx.get("amount"), field_name="amount", allow_zero=False)
            old_orig_opt = _safe_optional_float(tx.get("oldbalanceOrg"), field_name="oldbalanceOrg")
            new_orig_opt = _safe_optional_float(tx.get("newbalanceOrig"), field_name="newbalanceOrig")
            old_dest_opt = _safe_optional_float(tx.get("oldbalanceDest"), field_name="oldbalanceDest")
            new_dest_opt = _safe_optional_float(tx.get("newbalanceDest"), field_name="newbalanceDest")

            dt_utc = normalize_event_datetime(tx)
            ts_sec = dt_utc.timestamp()
            hour = dt_utc.hour

            self._validate_fields_and_order(tx, ts_sec=ts_sec, orig=orig, strict_fields=strict_fields)

            is_outgoing = tx_type in OUTGOING_TX_TYPES
            is_merchant = _is_merchant_counterparty(dest, tx_type)

            # --- 1. Transactional features (pre-decision safe + optional audit diagnostics) ---
            old_orig = old_orig_opt if old_orig_opt is not None else 0.0
            if tx_type == "CASH_IN":
                drain_ratio = 0.0
                orig_zero_after = 0.0
                orig_bal_err = (
                    abs((old_orig + amount) - new_orig_opt)
                    if (old_orig_opt is not None and new_orig_opt is not None)
                    else 0.0
                )
                dest_bal_err = 0.0
            else:
                drain_ratio = (
                    min(amount / (old_orig + 1.0), 5.0)
                    if (old_orig_opt is not None and old_orig > 0.0)
                    else 0.0
                )
                if old_orig_opt is not None and new_orig_opt is not None:
                    orig_zero_after = 1.0 if (old_orig > 0.0 and new_orig_opt <= 1.0) else 0.0
                    orig_bal_err = abs((old_orig - amount) - new_orig_opt)
                else:
                    orig_zero_after = 0.0
                    orig_bal_err = 0.0

                if old_dest_opt is not None and new_dest_opt is not None:
                    if tx_type == "CASH_OUT" and abs(new_dest_opt - old_dest_opt) <= 1e-2:
                        dest_bal_err = 0.0
                    elif is_merchant and old_dest_opt == 0.0 and new_dest_opt == 0.0:
                        dest_bal_err = 0.0
                    else:
                        dest_bal_err = abs((old_dest_opt + amount) - new_dest_opt)
                else:
                    dest_bal_err = 0.0

            is_foreign_ip = 1.0 if (ip_country and ip_country != "RU") else 0.0

            feats: dict[str, float] = {
                "amount": amount,
                "log_amount": math.log1p(max(amount, 0.0)),
                "hour_of_day": float(hour),
                "is_night": 1.0 if hour in (0, 1, 2, 3, 4, 5) else 0.0,
                "balance_drain_ratio": float(drain_ratio),
                "orig_balance_zero_after": float(orig_zero_after),
                "orig_balance_error": float(orig_bal_err),
                "dest_balance_error": float(dest_bal_err),
                "is_foreign_ip": float(is_foreign_ip),
            }
            for t in TX_TYPES:
                feats[f"type_{t}"] = 1.0 if tx_type == t else 0.0
            for c in CHANNELS:
                feats[f"channel_{c}"] = 1.0 if channel == c else 0.0

            # --- 2. Behavioral rolling features (STRICTLY committed past outgoing events in [t - W, t]) ---
            win_7d = self._orig_window_7d.get(orig)
            cutoff_7d = ts_sec - 7 * 86400.0 - WINDOW_EPS_SEC
            cutoff_24h = ts_sec - 86400.0 - WINDOW_EPS_SEC
            cutoff_1h = ts_sec - 3600.0 - WINDOW_EPS_SEC

            if win_7d:
                evict_ref_ts = min(ts_sec, self._orig_last_ts.get(orig, ts_sec))
                safe_evict_cutoff = evict_ref_ts - 7 * 86400.0 - WINDOW_EPS_SEC
                while win_7d and win_7d[0][0] < safe_evict_cutoff:
                    _, old_amt = win_7d.popleft()
                    self._orig_sum_7d[orig] -= old_amt
                if self._orig_sum_7d[orig] < 0.0:
                    self._orig_sum_7d[orig] = 0.0

                cnt_7d = 0
                sum_7d = 0.0
                cnt_24h = 0
                sum_24h = 0.0
                cnt_1h = 0
                sum_1h = 0.0
                for prev_ts, prev_amt in reversed(win_7d):
                    if prev_ts > ts_sec + WINDOW_EPS_SEC:
                        continue
                    if prev_ts >= cutoff_7d:
                        cnt_7d += 1
                        sum_7d += prev_amt
                        if prev_ts >= cutoff_24h:
                            cnt_24h += 1
                            sum_24h += prev_amt
                            if prev_ts >= cutoff_1h:
                                cnt_1h += 1
                                sum_1h += prev_amt
                    else:
                        break
            else:
                cnt_1h = cnt_24h = cnt_7d = 0
                sum_1h = sum_24h = sum_7d = 0.0

            past_n = self._orig_count.get(orig, 0)
            if not is_outgoing:
                zscore = 0.0
                ratio_mean = 1.0
            elif past_n >= 2:
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

            has_any_history = orig in self._orig_last_ts
            known_recs = self._orig_recipients.get(orig, set())
            known_devs = self._orig_devices.get(orig, set())
            is_new_recipient = (
                0.0
                if (not is_outgoing or not has_any_history or len(known_recs) == 0 or dest in known_recs)
                else 1.0
            )
            is_new_device = (
                0.0
                if (device is None or not has_any_history or len(known_devs) == 0 or device in known_devs)
                else 1.0
            )

            last_ts = self._orig_last_ts.get(orig)
            if last_ts is not None:
                time_since_last_h = max(0.0, (ts_sec - last_ts) / 3600.0)
                time_since_last_h = float(min(time_since_last_h, MAX_TIME_SINCE_LAST_HOURS))
            else:
                time_since_last_h = COLD_START_TIME_SINCE_LAST_HOURS

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

            # --- 3. Graph features (STRICTLY past P2P recipient & verified device state) ---
            if is_merchant or tx_type == "CASH_IN":
                dest_in_deg = 0
                dest_cnt_24h = 0
                is_mule_cand = 0.0
            else:
                dest_win_24h = self._dest_window_24h.get(dest)
                if dest_win_24h:
                    safe_dest_cutoff = min(ts_sec, dest_win_24h[-1]) - 86400.0 - WINDOW_EPS_SEC
                    while dest_win_24h and dest_win_24h[0] < safe_dest_cutoff:
                        dest_win_24h.popleft()
                    dest_cnt_24h = 0
                    for prev_dts in reversed(dest_win_24h):
                        if prev_dts > ts_sec + WINDOW_EPS_SEC:
                            continue
                        if prev_dts >= cutoff_24h:
                            dest_cnt_24h += 1
                        else:
                            break
                else:
                    dest_cnt_24h = 0

                dest_in_deg = len(self._dest_senders.get(dest, set()))
                is_mule_cand = 1.0 if dest_in_deg >= self.mule_in_degree_threshold else 0.0

            orig_out_deg = len(known_recs)
            if device is not None:
                dev_accs = self._device_accounts.get(device)
                device_shared = len(dev_accs - {orig}) if dev_accs else 0
            else:
                device_shared = 0

            feats.update({
                "dest_in_degree": float(dest_in_deg),
                "orig_out_degree": float(orig_out_deg),
                "device_shared_accounts": float(device_shared),
                "dest_tx_count_24h": float(dest_cnt_24h),
                "is_mule_candidate": float(is_mule_cand),
            })

            if update_state:
                self.commit_event(tx, feats)

            return feats

    def commit_event(self, tx: dict[str, Any], features: dict[str, float]) -> None:
        """Atomically commit a validated transaction's state transition & observability counters."""
        with self._lock:
            tx_id_raw = tx.get("tx_id")
            tx_id = _safe_str(tx_id_raw, "") if tx_id_raw is not None else ""
            if tx_id and tx_id in self._processed_tx:
                self.get_cached_features(tx)
                return

            orig = _safe_str(tx.get("nameOrig"), "")
            dest = _safe_str(tx.get("nameDest"), "")
            device = _normalize_optional_device(tx.get("device_id"))
            tx_type = _safe_str(tx.get("type"), "UNKNOWN").upper()
            amount = _safe_float(tx.get("amount"), field_name="amount", allow_zero=False)
            ts_sec = normalize_event_datetime(tx).timestamp()

            self._validate_fields_and_order(tx, ts_sec=ts_sec, orig=orig, strict_fields=False)

            # Track cold-start & missingness observability BEFORE mutating sender history
            if orig not in self._orig_last_ts:
                self.cold_start_events += 1
            if _is_missing(tx.get("timestamp")):
                self._missing_field_counts["timestamp"] += 1
            if _is_missing(tx.get("oldbalanceOrg")):
                self._missing_field_counts["oldbalanceOrg"] += 1
            if device is None:
                self._missing_field_counts["device_id"] += 1
            if _is_missing(tx.get("channel")):
                self._missing_field_counts["channel"] += 1
            if _is_missing(tx.get("ip_country")):
                self._missing_field_counts["ip_country"] += 1

            is_outgoing = tx_type in OUTGOING_TX_TYPES
            is_merchant = _is_merchant_counterparty(dest, tx_type)

            if is_outgoing:
                win_7d = self._orig_window_7d[orig]
                if not win_7d or ts_sec >= win_7d[-1][0]:
                    win_7d.append((ts_sec, amount))
                else:
                    ts_keys = [item[0] for item in win_7d]
                    ins_idx = bisect.bisect_right(ts_keys, ts_sec)
                    win_7d.insert(ins_idx, (ts_sec, amount))
                self._orig_sum_7d[orig] += amount

                n = self._orig_count[orig] + 1
                self._orig_count[orig] = n
                delta = amount - self._orig_mean[orig]
                self._orig_mean[orig] += delta / n
                delta2 = amount - self._orig_mean[orig]
                self._orig_m2[orig] += delta * delta2

                self._orig_recipients[orig].add(dest)

                if not is_merchant:
                    self._dest_senders[dest].add(orig)
                    dest_win = self._dest_window_24h[dest]
                    if not dest_win or ts_sec >= dest_win[-1]:
                        dest_win.append(ts_sec)
                    else:
                        ins_idx = bisect.bisect_right(list(dest_win), ts_sec)
                        dest_win.insert(ins_idx, ts_sec)

            # Only link real non-placeholder devices so absent telemetry never creates false shared-device rings
            if device is not None:
                self._orig_devices[orig].add(device)
                self._device_accounts[device].add(orig)

            prev_last = self._orig_last_ts.get(orig)
            self._orig_last_ts[orig] = ts_sec if prev_last is None else max(prev_last, ts_sec)
            self._global_max_ts = ts_sec if self._global_max_ts is None else max(self._global_max_ts, ts_sec)
            self.ingested_events += 1

            if tx_id:
                self._processed_tx[tx_id] = (canonical_tx_fingerprint(tx), dict(features))
                while len(self._processed_tx) > self.max_idempotency_entries:
                    self._processed_tx.popitem(last=False)
