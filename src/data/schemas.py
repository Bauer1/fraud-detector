"""Pydantic schemas for transactions, scoring decisions, alerts, and feedback."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def ensure_utc_datetime(dt: datetime | None) -> datetime | None:
    """Normalize any datetime to explicit UTC. Naive datetimes are treated as UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class TransactionType(str, Enum):
    TRANSFER = "TRANSFER"
    CASH_OUT = "CASH_OUT"
    PAYMENT = "PAYMENT"
    CASH_IN = "CASH_IN"
    DEBIT = "DEBIT"


class ChannelType(str, Enum):
    MOBILE = "MOBILE"
    WEB = "WEB"
    API = "API"
    ATM = "ATM"


class DecisionAction(str, Enum):
    ALLOW = "ALLOW"
    OTP = "OTP"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


class TransactionRequest(BaseModel):
    """Strict input contract for real-time scoring. Unknown fields are forbidden."""

    model_config = ConfigDict(extra="forbid")

    tx_id: str = Field(..., min_length=1, max_length=128, description="Unique transaction identifier")
    step: int = Field(..., ge=0, description="Time step in hours from simulation start")
    timestamp: datetime | None = Field(default=None, description="Transaction timestamp (normalized to UTC)")
    type: TransactionType = Field(..., description="PaySim-compatible transaction type")
    amount: float = Field(..., gt=0, description="Transfer amount")
    nameOrig: str = Field(..., min_length=1, description="Sender account ID")
    oldbalanceOrg: float = Field(..., ge=0, description="Sender balance before transaction")
    newbalanceOrig: float = Field(..., ge=0, description="Sender balance after transaction")
    nameDest: str = Field(..., min_length=1, description="Recipient account ID")
    oldbalanceDest: float = Field(default=0.0, ge=0, description="Recipient balance before transaction")
    newbalanceDest: float = Field(default=0.0, ge=0, description="Recipient balance after transaction")
    device_id: str = Field(default="DEV_UNKNOWN", min_length=1, description="Device fingerprint ID")
    channel: ChannelType = Field(default=ChannelType.MOBILE, description="Transaction channel")
    ip_country: str = Field(default="RU", min_length=2, max_length=8, description="ISO country code of IP")

    @field_validator("timestamp")
    @classmethod
    def _normalize_timestamp_utc(cls, value: datetime | None) -> datetime | None:
        return ensure_utc_datetime(value)


class RuleHit(BaseModel):
    rule_name: str
    severity: str  # "HARD_BLOCK", "HIGH", "MEDIUM", "LOW"
    score_contribution: float
    reason: str


class FeatureContribution(BaseModel):
    feature: str
    value: float
    shap_value: float
    description: str


class ScoreResponse(BaseModel):
    tx_id: str
    action: DecisionAction
    risk_score: float = Field(..., ge=0.0, le=1.0)
    ml_score: float = Field(..., ge=0.0, le=1.0)
    rule_score: float = Field(..., ge=0.0, le=1.0)
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    triggered_rules: list[RuleHit] = Field(default_factory=list)
    top_factors: list[FeatureContribution] = Field(default_factory=list)
    explanation_summary: str
    latency_ms: float
    is_duplicate: bool = Field(
        default=False,
        description="True if tx_id was already scored with the same payload and returned idempotently",
    )


class FeedbackRequest(BaseModel):
    """Strict input contract for analyst feedback. Unknown fields are forbidden."""

    model_config = ConfigDict(extra="forbid")

    tx_id: str = Field(..., min_length=1, description="Transaction ID to label")
    is_fraud: bool = Field(..., description="Analyst ground-truth label")
    analyst_id: str = Field(default="analyst_1", min_length=1, description="Analyst identifier")
    notes: str = Field(default="", description="Optional analyst investigation notes")


class FeedbackResponse(BaseModel):
    status: str
    tx_id: str
    is_fraud: bool
    labeled_at: datetime


class AlertRecord(BaseModel):
    tx_id: str
    timestamp: datetime | None
    step: int
    type: str
    amount: float
    nameOrig: str
    nameDest: str
    device_id: str
    channel: str
    action: str
    risk_score: float
    ml_score: float
    rule_score: float
    anomaly_score: float
    triggered_rules: list[dict[str, Any]]
    top_factors: list[dict[str, Any]]
    explanation_summary: str
    analyst_label: bool | None = None
    analyst_id: str | None = None
    analyst_notes: str | None = None
    created_at: datetime | None = None
