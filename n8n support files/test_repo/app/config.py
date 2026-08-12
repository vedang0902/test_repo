"""
Application configuration.
Loads from environment variables with production-safe defaults.
"""
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class DatabaseConfig:
    # Mock PostgreSQL pool settings
    pool_size: int = int(os.getenv("DB_POOL_SIZE", "5"))
    pool_timeout_seconds: float = float(os.getenv("DB_POOL_TIMEOUT", "3.0"))
    connection_failure_rate: float = float(os.getenv("DB_FAILURE_RATE", "0.12"))
    slow_query_threshold_ms: float = float(os.getenv("DB_SLOW_QUERY_MS", "200"))
    # SQLite file backing the mock pool
    sqlite_path: str = os.getenv("SQLITE_PATH", "./paymentpipeline.db")


@dataclass
class PaymentConfig:
    # Idempotency key TTL in seconds
    idempotency_ttl: int = int(os.getenv("IDEMPOTENCY_TTL", "3600"))
    # Rate at which partial commits are injected (realistic: ~5-8%)
    partial_commit_rate: float = float(os.getenv("PARTIAL_COMMIT_RATE", "0.07"))
    # Supported currencies
    supported_currencies: List[str] = field(
        default_factory=lambda: ["USD", "EUR", "GBP", "INR", "SGD"]
    )
    # Fee rate (used in reconciliation - intentionally as float, not Decimal)
    fee_rate: float = float(os.getenv("FEE_RATE", "0.029"))
    max_transaction_amount: float = float(os.getenv("MAX_TX_AMOUNT", "50000.0"))


@dataclass
class FraudConfig:
    threshold: float = float(os.getenv("FRAUD_THRESHOLD", "0.65"))
    # Score bleed factor - causes cascade bug
    score_bleed_factor: float = float(os.getenv("FRAUD_SCORE_BLEED", "0.30"))
    high_amount_threshold: float = float(os.getenv("FRAUD_HIGH_AMOUNT", "5000.0"))
    cascade_check_count: int = int(os.getenv("FRAUD_CASCADE_COUNT", "4"))


@dataclass
class WebhookConfig:
    # Retry storm threshold
    storm_threshold: int = int(os.getenv("WEBHOOK_STORM_THRESHOLD", "3"))
    # Chance that external gateway "retries" a webhook (simulated)
    gateway_retry_rate: float = float(os.getenv("GATEWAY_RETRY_RATE", "0.20"))
    max_retry_count: int = int(os.getenv("WEBHOOK_MAX_RETRIES", "5"))


@dataclass
class GeneratorConfig:
    # Transaction generation interval (seconds between batches)
    min_interval: float = float(os.getenv("GEN_MIN_INTERVAL", "2.0"))
    max_interval: float = float(os.getenv("GEN_MAX_INTERVAL", "6.0"))
    # Concurrent transactions per batch
    batch_size: int = int(os.getenv("GEN_BATCH_SIZE", "3"))
    # Enable refund generation (triggers deadlock scenario)
    refund_rate: float = float(os.getenv("GEN_REFUND_RATE", "0.15"))
    # Memory leak: cache is never pruned beyond this (0 = unbounded)
    cache_max_size: int = int(os.getenv("GEN_CACHE_MAX", "0"))


@dataclass
class ReconciliationConfig:
    # How often to run reconciliation (seconds)
    interval_seconds: int = int(os.getenv("RECON_INTERVAL", "30"))
    # Drift threshold before alerting
    drift_threshold: float = float(os.getenv("RECON_DRIFT_THRESHOLD", "0.005"))


@dataclass
class AppConfig:
    app_name: str = "PaymentPipeline"
    version: str = "2.4.1"
    environment: str = os.getenv("APP_ENV", "production")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    api_port: int = int(os.getenv("API_PORT", "5000"))
    metrics_port: int = int(os.getenv("METRICS_PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    payment: PaymentConfig = field(default_factory=PaymentConfig)
    fraud: FraudConfig = field(default_factory=FraudConfig)
    webhook: WebhookConfig = field(default_factory=WebhookConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    reconciliation: ReconciliationConfig = field(default_factory=ReconciliationConfig)


# Singleton
settings = AppConfig()
