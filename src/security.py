"""Security utilities: PII pseudonymization, secret redaction, API key auth, and rate limiting."""
from __future__ import annotations

import hashlib
import hmac
import os
import threading
import time
from collections import defaultdict, deque
from typing import Any

SENSITIVE_KEYS = frozenset({
    "api_key",
    "openai_api_key",
    "authorization",
    "token",
    "password",
    "secret",
    "x-api-key",
})


def pseudonymize_id(raw_id: str, salt: str = "pilot-default-salt", keep_entity_prefix: bool = True) -> str:
    """Deterministically pseudonymize an account or device identifier via HMAC-SHA256.

    Preserves leading entity type indicator (`M` for merchant, `C` for customer, `DEV_` for device)
    when `keep_entity_prefix=True` so point-in-time graph semantics remain intact without storing raw PII.
    """
    val = str(raw_id or "").strip()
    if not val:
        return ""
    # Avoid double-hashing already pseudonymized tokens
    if val.startswith(("PSEUDO_C_", "PSEUDO_M_", "PSEUDO_DEV_", "PSEUDO_ID_")):
        return val

    digest = hmac.new(salt.encode("utf-8"), val.encode("utf-8"), hashlib.sha256).hexdigest()[:16]
    if not keep_entity_prefix:
        return f"PSEUDO_ID_{digest}"
    if val.startswith("M"):
        return f"M_PSEUDO_{digest}"
    if val.startswith("DEV"):
        return f"DEV_PSEUDO_{digest}"
    if val.startswith("C"):
        return f"C_PSEUDO_{digest}"
    return f"PSEUDO_ID_{digest}"


def pseudonymize_transaction_dict(tx: dict[str, Any], salt: str) -> dict[str, Any]:
    """Return a shallow copy of transaction dict with sender, recipient, and device pseudonymized."""
    out = dict(tx)
    if out.get("nameOrig"):
        out["nameOrig"] = pseudonymize_id(str(out["nameOrig"]), salt=salt, keep_entity_prefix=True)
    if out.get("nameDest"):
        out["nameDest"] = pseudonymize_id(str(out["nameDest"]), salt=salt, keep_entity_prefix=True)
    if out.get("device_id"):
        out["device_id"] = pseudonymize_id(str(out["device_id"]), salt=salt, keep_entity_prefix=True)
    return out


def redact_for_logging(data: dict[str, Any]) -> dict[str, Any]:
    """Redact secrets and mask raw identifiers before writing to logs."""
    sanitized: dict[str, Any] = {}
    for k, v in data.items():
        if k.lower() in SENSITIVE_KEYS:
            sanitized[k] = "***REDACTED***"
        elif k in ("nameOrig", "nameDest", "device_id") and isinstance(v, str) and len(v) > 4:
            sanitized[k] = f"{v[:2]}***{v[-2:]}"
        elif isinstance(v, dict):
            sanitized[k] = redact_for_logging(v)
        else:
            sanitized[k] = v
    return sanitized


class SlidingWindowRateLimiter:
    """Thread-safe in-memory sliding-window rate limiter per client key."""

    def __init__(self, max_requests_per_minute: int = 1200):
        self.max_requests = max(1, int(max_requests_per_minute))
        self.window_sec = 60.0
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client_key: str = "global") -> bool:
        now = time.monotonic()
        cutoff = now - self.window_sec
        with self._lock:
            q = self._hits[client_key]
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= self.max_requests:
                return False
            q.append(now)
            return True


def resolve_expected_api_key(security_cfg: dict[str, Any] | None = None) -> str | None:
    """Resolve configured service-to-service API key from env or config (never hardcoded)."""
    env_key = os.getenv("FRAUD_API_KEY", "").strip()
    if env_key:
        return env_key
    cfg = security_cfg or {}
    if bool(cfg.get("require_api_key", False)):
        cfg_key = str(cfg.get("api_key", "")).strip()
        return cfg_key or None
    return None
