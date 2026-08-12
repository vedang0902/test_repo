"""
Autonomous Transaction Generator.

Simulates a production payment pipeline under continuous load:
  - Generates synthetic payment requests at configurable intervals
  - Triggers concurrent requests to expose race conditions
  - Simulates external gateway retrying webhooks (retry storm)
  - Seeds an initial account pool on startup

=============================================================================
BUG: Memory Leak — Unbounded Transaction History Cache
=============================================================================
Root cause:
  `_transaction_history_cache` is a dict that grows indefinitely.
  Every generated transaction is added but NEVER evicted.
  At 3 transactions/batch × every 4 seconds = ~45 entries/min.
  Over 8 hours = ~21,600 entries holding full Transaction objects.

  In Python this is ~2-5 MB/hr which in a long-running container causes
  gradual OOM — the kind of bug that only surfaces after a weekend.

Fix (NOT applied): Use functools.lru_cache, TTLCache from cachetools,
  or explicit eviction when len > N.

Prometheus metrics:
  payment_transaction_cache_size ↑ monotonically
"""
import asyncio
import logging
import random
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from app.config import settings
from app.database import db_pool, DBConnectionError
from app.models import Transaction, Account, PaymentMethod, WebhookEvent, WebhookEventType
from app.metrics import prometheus_metrics as m
from app.services.payment_processor import payment_processor
from app.services.webhook_service import webhook_service

logger = logging.getLogger("transaction_generator")

cfg_gen = settings.generator
cfg_wh = settings.webhook

# Realistic synthetic account pool
SYNTHETIC_ACCOUNTS = [
    Account(id="acct_001", name="Arjun Sharma",     balance=50000.0, currency="USD"),
    Account(id="acct_002", name="Priya Patel",      balance=32000.0, currency="USD"),
    Account(id="acct_003", name="Rohan Mehta",      balance=18500.0, currency="USD"),
    Account(id="acct_004", name="Ananya Singh",     balance=75000.0, currency="USD"),
    Account(id="acct_005", name="Vikram Nair",      balance=12000.0, currency="USD"),
    Account(id="acct_006", name="Kavya Reddy",      balance=95000.0, currency="USD"),
    Account(id="acct_007", name="Siddharth Joshi",  balance=28000.0, currency="USD"),
    Account(id="acct_008", name="Neha Gupta",       balance=61000.0, currency="USD"),
    Account(id="acct_merchant_A", name="Shopify Store A", balance=200000.0, currency="USD"),
    Account(id="acct_merchant_B", name="Razorpay Merchant B", balance=150000.0, currency="USD"),
]

CURRENCIES = ["USD", "EUR", "GBP", "INR", "SGD"]
METHODS = list(PaymentMethod)

# Realistic transaction amount distributions
AMOUNT_PROFILES = [
    (0.40, lambda: round(random.uniform(5, 150), 2)),          # Small retail
    (0.30, lambda: round(random.uniform(150, 1500), 2)),       # Mid-tier
    (0.15, lambda: round(random.uniform(1500, 8000), 2)),      # Large transaction
    (0.10, lambda: round(random.choice([100, 500, 1000, 2000, 5000]), 2)),  # Round amounts (fraud signal)
    (0.05, lambda: round(random.uniform(8000, 50000), 2)),     # Enterprise
]


def sample_amount() -> float:
    r = random.random()
    cumulative = 0.0
    for weight, generator in AMOUNT_PROFILES:
        cumulative += weight
        if r < cumulative:
            return generator()
    return 99.99


class TransactionGenerator:
    """
    Background service that auto-generates payment traffic.

    Designed to continuously exercise all buggy code paths:
      - Normal transactions → fraud scoring, reconciliation drift
      - Concurrent identical keys → idempotency race condition
      - Intermittent high amounts → fraud cascade trigger
      - Refund generation → deadlock trigger
      - Webhook re-delivery → retry storm trigger
    """

    def __init__(self):
        # BUG: Never cleared — memory leak
        self._transaction_history_cache: Dict[str, Transaction] = {}
        # BUG: Completed transaction IDs for refund targeting — also unbounded
        self._completed_tx_ids: List[str] = []
        self._accounts_seeded = False
        self._total_generated = 0
        self._total_failed = 0

    async def seed_accounts(self):
        """Insert synthetic accounts into the DB on startup."""
        if self._accounts_seeded:
            return
        try:
            with db_pool.connection() as conn:
                for acct in SYNTHETIC_ACCOUNTS:
                    conn.execute(
                        """INSERT OR IGNORE INTO accounts (id, name, balance, currency, is_active, created_at)
                           VALUES (?, ?, ?, ?, 1, ?)""",
                        (acct.id, acct.name, acct.balance, acct.currency,
                         datetime.utcnow().isoformat()),
                    )
                conn.commit()
            self._accounts_seeded = True
            logger.info(f"[Generator] Seeded {len(SYNTHETIC_ACCOUNTS)} synthetic accounts")
        except DBConnectionError as e:
            logger.error(f"[Generator] Failed to seed accounts: {e}")

    def _make_transaction(
        self,
        idempotency_key: Optional[str] = None,
        force_high_amount: bool = False,
    ) -> Transaction:
        """Create a randomised synthetic transaction."""
        from_acct = random.choice(SYNTHETIC_ACCOUNTS[:8])   # Exclude merchants as senders
        to_acct = random.choice(SYNTHETIC_ACCOUNTS[8:])     # Merchants as receivers

        amount = sample_amount()
        if force_high_amount:
            amount = random.uniform(5000, 25000)

        currency = random.choices(CURRENCIES, weights=[0.60, 0.15, 0.10, 0.10, 0.05])[0]

        return Transaction(
            idempotency_key=idempotency_key or f"idem_{uuid.uuid4().hex[:16]}",
            from_account=from_acct.id,
            to_account=to_acct.id,
            amount=round(amount, 2),
            currency=currency,
            method=random.choice(METHODS),
            metadata={
                "generated": True,
                "source": "transaction_generator",
                "batch_time": datetime.utcnow().isoformat(),
            },
        )

    def _cache_transaction(self, tx: Transaction):
        """
        Cache transaction in history.
        BUG: Cache never evicted — grows indefinitely.
        """
        self._transaction_history_cache[tx.id] = tx
        # Update memory leak metric
        cache_size = len(self._transaction_history_cache)
        m.transaction_cache_size.set(cache_size)

        if cache_size % 100 == 0 and cache_size > 0:
            logger.warning(
                f"[Generator] Transaction cache size: {cache_size} entries "
                f"(memory leak — cache is never evicted)"
            )

    async def _process_with_webhook(self, tx: Transaction):
        """
        Process a transaction then simulate gateway webhook delivery.
        Sometimes simulates the retry storm by re-delivering the webhook.
        """
        try:
            result = await payment_processor.process_payment(tx)
            self._cache_transaction(result)

            if result.status.value in ("completed", "partial_commit"):
                self._completed_tx_ids.append(result.id)
                # Keep list bounded (only cache of TX objects leaks)
                if len(self._completed_tx_ids) > 500:
                    self._completed_tx_ids = self._completed_tx_ids[-500:]

            # ── Simulate gateway webhook delivery ─────────────────────────────
            event_type = (
                WebhookEventType.PAYMENT_CONFIRMED
                if result.status == "completed"
                else WebhookEventType.PAYMENT_FAILED
            )
            webhook = WebhookEvent(
                transaction_id=result.id,
                event_type=event_type,
                source_gateway=random.choice(["stripe_mock", "razorpay_mock", "paypal_mock"]),
                payload={"amount": tx.amount, "currency": tx.currency, "status": result.status.value},
            )

            await webhook_service.handle(webhook)

            # BUG: Simulate gateway retry (20% chance) — causes retry storm
            if random.random() < cfg_wh.gateway_retry_rate:
                await asyncio.sleep(random.uniform(0.3, 1.5))
                logger.debug(f"[Generator] Simulating gateway retry for webhook={webhook.id}")
                await webhook_service.handle(webhook)

                # Second retry (10% of the already-retried ones)
                if random.random() < 0.5:
                    await asyncio.sleep(random.uniform(0.5, 2.0))
                    await webhook_service.handle(webhook)

            self._total_generated += 1

        except Exception as e:
            self._total_failed += 1
            logger.debug(f"[Generator] Transaction failed (expected): {e}")

    async def run_normal_batch(self):
        """Generate a normal batch of independent transactions."""
        tasks = []
        for _ in range(cfg_gen.batch_size):
            tx = self._make_transaction()
            tasks.append(self._process_with_webhook(tx))
        await asyncio.gather(*tasks, return_exceptions=True)

    async def run_idempotency_race(self):
        """
        BUG TRIGGER: Send two concurrent requests with the same idempotency key.
        This triggers the race condition → potential double charge.
        """
        shared_key = f"idem_{uuid.uuid4().hex[:16]}"
        tx1 = self._make_transaction(idempotency_key=shared_key)
        tx2 = self._make_transaction(idempotency_key=shared_key)
        tx2.id = str(uuid.uuid4())  # Different TX IDs, same idempotency key

        logger.info(
            f"[Generator] Triggering idempotency race: key={shared_key} "
            f"tx1={tx1.id} tx2={tx2.id}"
        )

        # Launch concurrently — race condition window
        await asyncio.gather(
            self._process_with_webhook(tx1),
            self._process_with_webhook(tx2),
            return_exceptions=True,
        )

    async def run_refund_batch(self):
        """
        BUG TRIGGER: Process refunds concurrently with payments.
        Concurrent lock acquisition in opposite order → deadlock.
        """
        if not self._completed_tx_ids:
            return

        # Pick a random completed transaction to refund
        orig_tx_id = random.choice(self._completed_tx_ids)
        cached_tx = self._transaction_history_cache.get(orig_tx_id)

        if not cached_tx or cached_tx.status.value != "completed":
            return

        refund_amount = round(cached_tx.amount * random.uniform(0.1, 1.0), 2)
        reason = random.choice([
            "Customer requested refund",
            "Duplicate charge detected",
            "Product not delivered",
            "Chargeback initiated",
        ])

        # Launch refund concurrently with normal payments to trigger deadlock
        normal_tx = self._make_transaction()
        logger.info(
            f"[Generator] Concurrent refund + payment: "
            f"refund orig={orig_tx_id} | payment tx={normal_tx.id}"
        )
        await asyncio.gather(
            payment_processor.process_refund(cached_tx, reason, refund_amount),
            self._process_with_webhook(normal_tx),
            return_exceptions=True,
        )

    async def run_fraud_cascade_trigger(self):
        """
        BUG TRIGGER: Send multiple high-value transactions from same account.
        Each successive fraud check compounds the previous score → cascade.
        """
        target_account = random.choice(SYNTHETIC_ACCOUNTS[:4])
        merchant = random.choice(SYNTHETIC_ACCOUNTS[8:])

        for i in range(random.randint(3, 6)):
            tx = Transaction(
                idempotency_key=f"idem_{uuid.uuid4().hex[:16]}",
                from_account=target_account.id,
                to_account=merchant.id,
                amount=round(random.uniform(3000, 15000), 2),
                currency="USD",
                method=PaymentMethod.CARD,
                metadata={"cascade_trigger": True, "sequence": i},
            )
            await self._process_with_webhook(tx)
            await asyncio.sleep(random.uniform(0.1, 0.5))

    async def run(self):
        """Main generator loop. Runs forever in background."""
        logger.info("[Generator] Starting autonomous transaction generator")
        await self.seed_accounts()

        tick = 0
        while True:
            try:
                interval = random.uniform(cfg_gen.min_interval, cfg_gen.max_interval)
                await asyncio.sleep(interval)

                tick += 1

                # Normal transactions every tick
                await self.run_normal_batch()

                # Every 5 ticks: trigger idempotency race condition
                if tick % 5 == 0:
                    await self.run_idempotency_race()

                # Every 7 ticks: trigger refund + concurrent payment (deadlock)
                if tick % 7 == 0 and random.random() < cfg_gen.refund_rate:
                    await self.run_refund_batch()

                # Every 12 ticks: trigger fraud cascade
                if tick % 12 == 0:
                    await self.run_fraud_cascade_trigger()

                m.app_uptime_seconds.set(
                    (datetime.utcnow() - _start_time).total_seconds()
                )

            except Exception as e:
                logger.error(f"[Generator] Unexpected error in main loop: {e}", exc_info=True)
                m.app_errors_total.labels(
                    component="transaction_generator", error_type="loop_error"
                ).inc()
                await asyncio.sleep(2)


_start_time = datetime.utcnow()

# Singleton
transaction_generator = TransactionGenerator()
