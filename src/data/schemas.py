"""Pydantic schemas for pre-decision transactions, scoring responses, analyst feedback, and model governance."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    UNKNOWN = "UNKNOWN"


class DecisionAction(str, Enum):
    ALLOW = "ALLOW"
    OTP = "OTP"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


class TechnicalStatus(str, Enum):
    OK = "OK"
    DEGRADED_FALLBACK = "DEGRADED_FALLBACK"


class LabelStatus(str, Enum):
    FRAUD = "fraud"
    NOT_FRAUD = "not_fraud"
    UNKNOWN = "unknown"


class TransactionRequest(BaseModel):
    """Strict input contract for real-time pre-decision scoring.

    Unknown or post-investigation leakage fields (`isFlaggedFraud`, `isFraud`, `chargeback`, etc.)
    are strictly forbidden (`extra="forbid"`). Post-transaction balances (`newbalanceOrig`,
    `newbalanceDest`) and extended telemetry (`device_id`, `channel`, `ip_country`) are optional
    and excluded from pre-decision ML features unless explicitly verified by the data contract.
    """

    model_config = ConfigDict(extra="forbid")

    tx_id: str = Field(..., min_length=1, max_length=128, description="Stable unique transaction identifier")
    step: int = Field(default=0, ge=0, description="Hourly time step (optional if UTC timestamp is supplied)")
    timestamp: datetime | None = Field(default=None, description="Transaction event timestamp (normalized to UTC)")
    type: TransactionType = Field(..., description="Transaction operation type")
    amount: float = Field(..., gt=0, description="Transfer amount (> 0)")
    nameOrig: str = Field(..., min_length=1, description="Sender account key (for point-in-time aggregates only)")
    oldbalanceOrg: float | None = Field(default=None, ge=0, description="Sender balance prior to transaction (if available pre-decision)")
    newbalanceOrig: float | None = Field(
        default=None,
        ge=0,
        description="Optional post-tx sender balance snapshot (NOT used in pre-decision ML features)",
    )
    nameDest: str = Field(..., min_length=1, description="Recipient account key (for point-in-time aggregates only)")
    oldbalanceDest: float | None = Field(
        default=None,
        ge=0,
        description="Optional recipient pre-balance (excluded unless pre-decision availability is confirmed)",
    )
    newbalanceDest: float | None = Field(
        default=None,
        ge=0,
        description="Optional post-tx recipient balance snapshot (NOT used in pre-decision ML features)",
    )
    device_id: str | None = Field(default=None, min_length=1, description="Optional device fingerprint ID")
    channel: ChannelType | None = Field(default=None, description="Optional transaction channel")
    ip_country: str | None = Field(default=None, min_length=2, max_length=8, description="Optional ISO country code of IP")

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
    effective_action: DecisionAction = Field(
        default=DecisionAction.ALLOW,
        description="Action enforced on payment rail (`ALLOW` in shadow pilot mode, equal to `action` in active mode)",
    )
    decision_mode: str = Field(default="ACTIVE", description="ACTIVE or SHADOW")
    technical_status: TechnicalStatus = Field(
        default=TechnicalStatus.OK,
        description="Technical health of scoring (`OK` or `DEGRADED_FALLBACK` for safe upstream handling)",
    )
    fallback_reason: str | None = Field(default=None, description="Populated if deterministic fail-safe policy was used")
    model_version: str = Field(default="unversioned", description="Immutable active model version identifier")
    feature_schema_version: str = Field(default="2.0.0-pre-decision", description="Feature contract version")
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
        description="True if tx_id was already scored with the same canonical payload and returned idempotently",
    )

    @model_validator(mode="after")
    def _sync_effective_action_default(self) -> "ScoreResponse":
        if self.decision_mode == "SHADOW":
            self.effective_action = DecisionAction.ALLOW
        elif "effective_action" not in self.model_fields_set:
            self.effective_action = self.action
        return self


class FeedbackRequest(BaseModel):
    """Strict input contract for analyst / investigation feedback (`fraud` / `not_fraud` / `unknown`)."""

    model_config = ConfigDict(extra="forbid")

    tx_id: str = Field(..., min_length=1, description="Transaction ID to label")
    label: LabelStatus | None = Field(
        default=None,
        description="Tri-state label: `fraud`, `not_fraud`, or `unknown` (immature / under investigation)",
    )
    is_fraud: bool | None = Field(
        default=None,
        description="Backward-compatible boolean alias (`True` -> `fraud`, `False` -> `not_fraud`)",
    )
    label_source: str = Field(
        default="analyst_review",
        min_length=1,
        max_length=64,
        description="Origin of label: analyst_review, chargeback, customer_claim, otp_abandoned, simulation",
    )
    analyst_id: str = Field(default="analyst_1", min_length=1, description="Analyst or upstream system identifier")
    notes: str = Field(default="", description="Analyst investigation commentary")
    labeled_at: datetime | None = Field(default=None, description="Optional UTC timestamp of label determination")

    @field_validator("labeled_at")
    @classmethod
    def _normalize_labeled_at(cls, value: datetime | None) -> datetime | None:
        return ensure_utc_datetime(value)

    @model_validator(mode="after")
    def _resolve_label_status(self) -> "FeedbackRequest":
        if self.label is None and self.is_fraud is None:
            raise ValueError("Either `label` ('fraud'/'not_fraud'/'unknown') or `is_fraud` (bool) must be provided")
        if self.label is not None and self.is_fraud is not None:
            expected = LabelStatus.FRAUD if self.is_fraud else LabelStatus.NOT_FRAUD
            if self.label != expected:
                raise ValueError(f"Conflicting `label={self.label.value}` and `is_fraud={self.is_fraud}`")
        if self.label is None:
            self.label = LabelStatus.FRAUD if self.is_fraud else LabelStatus.NOT_FRAUD
        if self.label == LabelStatus.FRAUD:
            self.is_fraud = True
        elif self.label == LabelStatus.NOT_FRAUD:
            self.is_fraud = False
        else:
            self.is_fraud = None
        return self


class FeedbackResponse(BaseModel):
    status: str
    tx_id: str
    label: LabelStatus
    is_fraud: bool | None
    label_source: str
    labeled_at: datetime
    audit_id: int | None = None


class PromoteModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved_by: str = Field(..., min_length=1, max_length=64, description="Approver identity (risk lead / model owner)")
    reason: str = Field(default="Approved after validation & comparison review", max_length=512)
    override_guardrail: bool = Field(default=False, description="Explicit sign-off if overriding non-critical guardrail")


class RollbackModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_version: str | None = Field(default=None, description="Specific prior approved version; defaults to previous active")
    initiated_by: str = Field(..., min_length=1, max_length=64, description="Operator / owner initiating rollback")
    reason: str = Field(default="Operational rollback to prior verified version", max_length=512)


class AlertRecord(BaseModel):
    tx_id: str
    timestamp: datetime | None
    step: int
    type: str
    amount: float
    nameOrig: str
    nameDest: str
    device_id: str | None = None
    channel: str | None = None
    action: str
    effective_action: str = "ALLOW"
    decision_mode: str = "ACTIVE"
    technical_status: str = "OK"
    model_version: str = "unversioned"
    risk_score: float
    ml_score: float
    rule_score: float
    anomaly_score: float
    triggered_rules: list[dict[str, Any]]
    top_factors: list[dict[str, Any]]
    explanation_summary: str
    label_status: str | None = None
    analyst_label: bool | None = None
    label_source: str | None = None
    analyst_id: str | None = None
    analyst_notes: str | None = None
    labeled_at: datetime | None = None
    created_at: datetime | None = None
