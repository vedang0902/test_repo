"""
Domain models for the PaymentPipeline service.
Uses dataclasses intentionally (not SQLAlchemy ORM) to mimic a service
that evolved from simple scripts into a distributed system without
proper refactoring — a common source of production debt.
"""
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any


class TransactionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL_COMMIT = "partial_commit"   # Deducted but not confirmed
    FRAUD_BLOCKED = "fraud_blocked"
    REFUNDED = "refunded"


class PaymentMethod(str, Enum):
    CARD = "card"
    BANK_TRANSFER = "bank_transfer"
    WALLET = "wallet"
    CRYPTO = "crypto"


class WebhookEventType(str, Enum):
    PAYMENT_CONFIRMED = "payment.confirmed"
    PAYMENT_FAILED = "payment.failed"
    REFUND_INITIATED = "refund.initiated"
    CHARGEBACK_RAISED = "chargeback.raised"
    FRAUD_FLAGGED = "fraud.flagged"


@dataclass
class Account:
    id: str
    name: str
    balance: float         # Float intentionally — source of rounding bugs
    currency: str = "USD"
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if self.balance < 0:
            raise ValueError(f"Account {self.id} cannot have negative balance at creation")


@dataclass
class Transaction:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    idempotency_key: str = ""
    from_account: str = ""
    to_account: str = ""
    amount: float = 0.0           # Float — not Decimal (intentional bug source)
    currency: str = "USD"
    method: PaymentMethod = PaymentMethod.CARD
    status: TransactionStatus = TransactionStatus.PENDING
    fraud_score: float = 0.0
    fee: float = 0.0              # Float fee, also a rounding bug source
    net_amount: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    def mark_completed(self):
        self.status = TransactionStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def mark_failed(self, reason: str):
        self.status = TransactionStatus.FAILED
        self.error_message = reason
        self.updated_at = datetime.utcnow()

    def mark_partial_commit(self, reason: str):
        self.status = TransactionStatus.PARTIAL_COMMIT
        self.error_message = reason
        self.updated_at = datetime.utcnow()


@dataclass
class Order:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    transaction_id: str = ""
    merchant_id: str = ""
    items: list = field(default_factory=list)
    total_amount: float = 0.0
    status: str = "pending"       # pending | confirmed | cancelled
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class WebhookEvent:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    transaction_id: str = ""
    event_type: WebhookEventType = WebhookEventType.PAYMENT_CONFIRMED
    payload: Dict[str, Any] = field(default_factory=dict)
    source_gateway: str = "stripe_mock"
    processing_count: int = 0    # Track how many times processed (for storm detection)
    processed: bool = False      # In-memory flag — not persisted (bug source)
    received_at: datetime = field(default_factory=datetime.utcnow)
    last_processed_at: Optional[datetime] = None


@dataclass
class FraudCheckResult:
    transaction_id: str
    account_id: str
    base_score: float
    compounded_score: float
    triggered_rules: list
    is_flagged: bool
    checked_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ReconciliationReport:
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    period_start: datetime = field(default_factory=datetime.utcnow)
    period_end: datetime = field(default_factory=datetime.utcnow)
    ledger_balance: float = 0.0
    actual_balance: float = 0.0
    drift: float = 0.0
    transaction_count: int = 0
    status: str = "ok"            # ok | mismatch | critical
    generated_at: datetime = field(default_factory=datetime.utcnow)
