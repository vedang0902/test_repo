"""
Centralized Prometheus metrics registry for PaymentPipeline.
All metrics defined here to avoid duplication across modules.
"""
from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server
import logging

logger = logging.getLogger("metrics")


# ── Payment Transaction Metrics ───────────────────────────────────────────────

payment_transactions_total = Counter(
    "payment_transactions_total",
    "Total payment transactions processed",
    ["status", "method", "currency"],
)

payment_processing_duration_seconds = Histogram(
    "payment_processing_duration_seconds",
    "Time spent processing a payment end-to-end",
    ["method"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

payment_amount_processed_usd = Counter(
    "payment_amount_processed_usd_total",
    "Total USD value of transactions processed (all statuses)",
)

payment_fees_collected_usd = Counter(
    "payment_fees_collected_usd_total",
    "Total USD fees collected from transactions",
)

active_payment_requests = Gauge(
    "payment_active_requests",
    "Number of payment requests currently being processed",
)


# ── Database / Connection Pool Metrics ───────────────────────────────────────

db_pool_connections_available = Gauge(
    "db_pool_connections_available",
    "Number of idle connections in the mock Postgres pool",
)

db_pool_connections_in_use = Gauge(
    "db_pool_connections_in_use",
    "Number of connections currently checked out from pool",
)

db_pool_exhausted_total = Counter(
    "db_pool_exhausted_total",
    "Times the connection pool was fully exhausted (all connections in use)",
)

db_connection_errors_total = Counter(
    "db_connection_errors_total",
    "Total database connection errors (timeouts, failures)",
    ["error_type"],
)

db_slow_queries_total = Counter(
    "db_slow_queries_total",
    "Queries that exceeded the slow-query threshold",
    ["operation"],
)

db_query_duration_seconds = Histogram(
    "db_query_duration_seconds",
    "Time spent executing database queries",
    ["operation"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 3.0],
)


# ── Idempotency / Race Condition Metrics ─────────────────────────────────────

idempotency_violations_total = Counter(
    "payment_idempotency_violations_total",
    "Times two concurrent requests with the same idempotency key both succeeded (double-charge)",
)

idempotency_cache_size = Gauge(
    "payment_idempotency_cache_size",
    "Current number of idempotency keys held in memory",
)


# ── Reconciliation / Float Bug Metrics ───────────────────────────────────────

reconciliation_drift_usd = Gauge(
    "reconciliation_drift_usd",
    "Current absolute drift between ledger and actual balances (USD)",
)

reconciliation_mismatches_total = Counter(
    "reconciliation_mismatches_total",
    "Number of reconciliation runs that detected a balance mismatch",
)

reconciliation_drift_exceeded_total = Counter(
    "reconciliation_drift_exceeded_total",
    "Times reconciliation drift exceeded the configured threshold",
)

ledger_balance_usd = Gauge(
    "ledger_balance_usd",
    "Current ledger balance (float arithmetic — drifts over time)",
)

actual_balance_usd = Gauge(
    "actual_balance_usd",
    "Current actual balance (rounded)",
)


# ── Webhook / Retry Storm Metrics ─────────────────────────────────────────────

webhook_received_total = Counter(
    "webhook_received_total",
    "Total webhook events received from payment gateway",
    ["event_type", "gateway"],
)

webhook_duplicate_processing_total = Counter(
    "webhook_duplicate_processing_total",
    "Webhooks processed more than once (idempotency failure)",
)

webhook_retry_storm_total = Counter(
    "webhook_retry_storm_total",
    "Webhook retry storm events detected (same webhook processed >3 times)",
)

webhook_processing_rate = Gauge(
    "webhook_processing_rate",
    "Current rate of webhook events being processed per second",
)


# ── Partial Commit Metrics ────────────────────────────────────────────────────

partial_commits_total = Counter(
    "payment_partial_commits_total",
    "Transactions where phase-1 (debit) succeeded but phase-2 (order confirm) failed",
)

orphaned_debits_total = Counter(
    "payment_orphaned_debits_total",
    "Accounts debited without a corresponding confirmed order",
)


# ── Fraud Detection Metrics ───────────────────────────────────────────────────

fraud_checks_total = Counter(
    "fraud_checks_total",
    "Total fraud checks performed",
    ["result"],  # flagged | cleared | error
)

fraud_false_positives_total = Counter(
    "fraud_false_positives_total",
    "Legitimate transactions incorrectly flagged as fraud (detected post-hoc)",
)

fraud_cascade_events_total = Counter(
    "fraud_cascade_events_total",
    "Times fraud score compounding caused a cascade (score runaway)",
)

fraud_score_histogram = Histogram(
    "fraud_score_distribution",
    "Distribution of raw fraud scores across all transactions",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


# ── Deadlock / Concurrency Metrics ────────────────────────────────────────────

deadlock_events_total = Counter(
    "payment_deadlock_events_total",
    "Times a deadlock was detected (lock acquisition timeout)",
    ["lock_type"],  # account_lock | order_lock
)

lock_wait_duration_seconds = Histogram(
    "payment_lock_wait_duration_seconds",
    "Time spent waiting to acquire processing locks",
    ["lock_type"],
    buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
)


# ── Memory Leak Metrics ───────────────────────────────────────────────────────

transaction_cache_size = Gauge(
    "payment_transaction_cache_size",
    "Number of entries in the in-memory transaction history cache (never evicted)",
)

webhook_dedup_cache_size = Gauge(
    "webhook_dedup_cache_size",
    "Number of webhook IDs in in-memory dedup store",
)


# ── General App Health Metrics ───────────────────────────────────────────────

app_errors_total = Counter(
    "app_errors_total",
    "Total unhandled application errors",
    ["component", "error_type"],
)

app_error_rate = Gauge(
    "app_error_rate",
    "Current error rate (rolling; 1 = error state, 0 = healthy)",
)

app_uptime_seconds = Gauge(
    "app_uptime_seconds",
    "Application uptime in seconds",
)


def start_metrics_server(port: int = 8000):
    """Start Prometheus metrics HTTP server."""
    start_http_server(port, addr="0.0.0.0")
    logger.info(f"Prometheus metrics server started on :{port}/metrics")
