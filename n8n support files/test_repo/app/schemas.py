"""
Pydantic v2 schemas for API request/response validation.
Separate from domain models to maintain clean API contract.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
import uuid


# ── Request Schemas ──────────────────────────────────────────────────────────

class PaymentRequest(BaseModel):
    idempotency_key: str = Field(..., min_length=8, max_length=128)
    from_account: str = Field(..., min_length=3)
    to_account: str = Field(..., min_length=3)
    amount: float = Field(..., gt=0, le=50000)
    currency: str = Field(default="USD", pattern=r"^[A-Z]{3}$")
    method: str = Field(default="card")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, v):
        allowed = {"USD", "EUR", "GBP", "INR", "SGD"}
        if v not in allowed:
            raise ValueError(f"Unsupported currency: {v}. Allowed: {allowed}")
        return v


class RefundRequest(BaseModel):
    transaction_id: str
    reason: str = Field(..., min_length=5, max_length=500)
    amount: Optional[float] = None    # None = full refund


class WebhookPayload(BaseModel):
    webhook_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str
    transaction_id: str
    gateway: str = "stripe_mock"
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    retry_count: int = Field(default=0, ge=0)


# ── Response Schemas ─────────────────────────────────────────────────────────

class TransactionResponse(BaseModel):
    id: str
    idempotency_key: str
    from_account: str
    to_account: str
    amount: float
    currency: str
    method: str
    status: str
    fraud_score: float
    fee: float
    net_amount: float
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None


class AccountResponse(BaseModel):
    id: str
    name: str
    balance: float
    currency: str
    is_active: bool


class FraudCheckResponse(BaseModel):
    transaction_id: str
    account_id: str
    base_score: float
    compounded_score: float
    triggered_rules: List[str]
    is_flagged: bool
    checked_at: datetime


class ReconciliationResponse(BaseModel):
    report_id: str
    ledger_balance: float
    actual_balance: float
    drift: float
    drift_pct: float
    transaction_count: int
    status: str
    generated_at: datetime


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    uptime_seconds: float
    db_pool_available: int
    db_pool_in_use: int
    active_transactions: int
    components: Dict[str, str]


class WebhookResponse(BaseModel):
    webhook_id: str
    status: str
    processing_count: int
    message: str


class AnalyticsSummary(BaseModel):
    total_transactions: int
    successful_transactions: int
    failed_transactions: int
    partial_commits: int
    fraud_blocked: int
    total_volume_usd: float
    total_fees_collected: float
    reconciliation_drift: float
    avg_processing_time_ms: float
    webhook_retry_storms: int
    idempotency_violations: int
    deadlock_events: int
    db_pool_exhaustions: int
    fraud_cascade_events: int


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    transaction_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
