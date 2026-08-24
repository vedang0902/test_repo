"""
Core Payment Processing Service.

Orchestrates the full payment lifecycle:
  1. Idempotency check  (FIXED: atomic per-key lock)
  2. Fraud scoring
  3. Database debit + order confirmation in a single atomic transaction (FIXED)
  4. Reconciliation recording

Fixes applied
─────────────
BUG 1 – Race condition on idempotency key
  Per-key asyncio.Lock held for the entire check-and-set window makes
  the read-modify-write atomic within the process.  An in-flight key
  causes the second caller to wait and then receive the cached result.

BUG 2 – Partial transaction commit / orphaned debit
  Phase 1 (debit) and Phase 2 (order insert + tx record) are now
  executed inside a single DB transaction.  If phase 2 fails the DB
  rolls back the debit automatically.  The artificial partial-commit
  injection that simulated the bug has been removed.

BUG 3 – Deadlock on lock ordering
  Both process_payment and process_refund now acquire locks in the
  same order: _acct_lock → _order_lock, eliminating the hold-and-wait
  cycle.
"""
import asyncio
import json
import logging
import random
import time
import uuid
from datetime import datetime
from typing import Dict, Optional as TypingOptional

from app.config import settings
from app.database import db_pool, DBConnectionError, PoolExhaustedError
from app.models import (
    Transaction, TransactionStatus, Order, PaymentMethod
)
from app.metrics import prometheus_metrics as m
from app.services.fraud_detector import fraud_detector
from app.services.reconciliation import reconciliation_service

logger = logging.getLogger("payment_processor")

cfg = settings.payment

# ── Idempotency store ────────────────────────────────────────────────────────
# Key → Transaction result
_idempotency_store: Dict[str, Transaction] = {}

# Per-key locks guarantee atomic check-and-set (FIX for BUG 1).
# A defaultdict would leak lock objects; we manage lifecycle explicitly.
_idempotency_key_locks: Dict[str, asyncio.Lock] = {}
_idempotency_registry_lock = asyncio.Lock()   # guards _idempotency_key_locks


async def _get_key_lock(key: str) -> asyncio.Lock:
    """Return (creating if necessary) the per-idempotency-key lock."""
    async with _idempotency_registry_lock:
        if key not in _idempotency_key_locks:
            _idempotency_key_locks[key] = asyncio.Lock()
        return _idempotency_key_locks[key]


# ── Ordered locks (FIX for BUG 3) ────────────────────────────────────────────
# RULE: always acquire _acct_lock before _order_lock, in both payment
#       and refund paths.
_acct_lock = asyncio.Lock()
_order_lock = asyncio.Lock()


class PaymentProcessorError(Exception):
    pass


class PartialCommitError(PaymentProcessorError):
    pass


class IdempotencyViolationError(PaymentProcessorError):
    pass


class InsufficientFundsError(PaymentProcessorError):
    pass


class PaymentProcessor:

    async def process_payment(self, tx: Transaction) -> Transaction:
        """
        Full payment processing pipeline.

        Idempotency is now atomic (per-key lock).
        DB writes are wrapped in a single transaction (no partial commits).
        Lock order matches process_refund (no deadlocks).
        """
        start = time.monotonic()
        m.active_payment_requests.inc()

        try:
            tx.status = TransactionStatus.PROCESSING

            # ── Step 1: Atomic idempotency check-and-reserve ─────────────────
            # Acquire the per-key lock BEFORE reading the store so that
            # concurrent callers with the same key serialise here.
            key_lock = await _get_key_lock(tx.idempotency_key)
            async with key_lock:
                existing = _idempotency_store.get(tx.idempotency_key)
                if existing:
                    logger.info(
                        f"[Payment] Idempotency hit: key={tx.idempotency_key} "
                        f"returning cached tx={existing.id}"
                    )
                    return existing

                # Reserve the slot immediately so any concurrent coroutine
                # that acquires the lock next will see a result and bail out.
                # We store the in-progress transaction; it will be updated
                # in-place to its final status before the lock is released.
                _idempotency_store[tx.idempotency_key] = tx
                m.idempotency_cache_size.set(len(_idempotency_store))

                try:
                    result = await self._execute_payment(tx)
                except Exception:
                    # Remove reservation so the caller can retry with a new tx.
                    _idempotency_store.pop(tx.idempotency_key, None)
                    m.idempotency_cache_size.set(len(_idempotency_store))
                    raise

            return result

        except PaymentProcessorError:
            tx.mark_failed("Internal processing error")
            m.payment_transactions_total.labels(
                status="failed",
                method=tx.method.value,
                currency=tx.currency,
            ).inc()
            raise

        except Exception as e:
            tx.mark_failed(str(e))
            m.payment_transactions_total.labels(
                status="failed",
                method=tx.method.value,
                currency=tx.currency,
            ).inc()
            m.app_errors_total.labels(
                component="payment_processor", error_type="unexpected"
            ).inc()
            logger.error(f"[Payment] Unexpected error for tx={tx.id}: {e}", exc_info=True)
            raise

        finally:
            elapsed = time.monotonic() - start
            m.payment_processing_duration_seconds.labels(
                method=tx.method.value
            ).observe(elapsed)
            m.active_payment_requests.dec()

    async def _execute_payment(self, tx: Transaction) -> Transaction:
        """
        Inner pipeline: fraud check → lock acquisition → atomic DB write.
        Called only when we hold the per-idempotency-key lock.
        """
        # ── Step 2: Fraud check ───────────────────────────────────────────────
        fraud_result = fraud_detector.check(tx)
        tx.fraud_score = fraud_result.compounded_score

        if fraud_result.is_flagged:
            tx.status = TransactionStatus.FRAUD_BLOCKED
            tx.error_message = (
                f"Fraud score {tx.fraud_score:.3f} exceeds threshold "
                f"{cfg.fee_rate} | rules={fraud_result.triggered_rules}"
            )
            m.payment_transactions_total.labels(
                status="fraud_blocked",
                method=tx.method.value,
                currency=tx.currency,
            ).inc()
            logger.warning(
                f"[Payment] BLOCKED (fraud): tx={tx.id} "
                f"score={tx.fraud_score:.3f} account={tx.from_account}"
            )
            # tx is already in _idempotency_store (reserved by caller)
            return tx

        # ── Step 3: Acquire locks in canonical order: acct → order ───────────
        # (FIX for BUG 3: process_refund uses the same order)
        try:
            lock_start = time.monotonic()
            await asyncio.wait_for(_acct_lock.acquire(), timeout=3.0)
            m.lock_wait_duration_seconds.labels(lock_type="account_lock").observe(
                time.monotonic() - lock_start
            )
        except asyncio.TimeoutError:
            m.deadlock_events_total.labels(lock_type="account_lock").inc()
            m.app_errors_total.labels(
                component="payment_processor", error_type="deadlock"
            ).inc()
            logger.error(
                f"[Payment] Timeout acquiring account_lock tx={tx.id}"
            )
            raise PaymentProcessorError(
                f"Timeout acquiring account lock for tx {tx.id}"
            )

        try:
            order_start = time.monotonic()
            await asyncio.wait_for(_order_lock.acquire(), timeout=2.0)
            m.lock_wait_duration_seconds.labels(lock_type="order_lock").observe(
                time.monotonic() - order_start
            )
        except asyncio.TimeoutError:
            _acct_lock.release()
            m.deadlock_events_total.labels(lock_type="order_lock").inc()
            m.app_errors_total.labels(
                component="payment_processor", error_type="deadlock"
            ).inc()
            logger.error(
                f"[Payment] Timeout acquiring order_lock tx={tx.id}"
            )
            raise PaymentProcessorError(
                f"Timeout acquiring order lock for tx {tx.id}"
            )

        # ── Step 4: Atomic two-phase DB write (FIX for BUG 2) ────────────────
        # Both the account debit and the order/transaction insert happen
        # inside a single DB transaction.  A failure in phase 2 causes the
        # DB to roll back the phase-1 debit automatically — no orphaned debits.
        conn = None
        try:
            conn = db_pool.acquire()

            # Begin explicit transaction (autocommit off)
            conn.execute("BEGIN")

            # Phase 1: Debit account
            conn.execute(
                "UPDATE accounts SET balance = balance - ? WHERE id = ? AND is_active = 1",
                (tx.amount, tx.from_account),
            )
            logger.debug(
                f"[Payment] Phase 1 staged: will debit ${tx.amount:.2f} "
                f"from {tx.from_account}"
            )

            # Phase 2: Confirm order + insert transaction record
            order_id = str(uuid.uuid4())
            merchant_id = f"merchant_{random.randint(100, 999)}"

            conn.execute(
                """INSERT INTO orders
                       (id, transaction_id, merchant_id, total_amount, status, created_at)
                   VALUES (?, ?, ?, ?, 'confirmed', ?)""",
                (
                    order_id, tx.id, merchant_id, tx.amount,
                    datetime.utcnow().isoformat(),
                ),
            )

            conn.execute(
                """INSERT INTO transactions
                       (id, idempotency_key, from_account, to_account, amount, currency,
                        method, status, fraud_score, fee, net_amount, metadata,
                        created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    tx.id, tx.idempotency_key, tx.from_account, tx.to_account,
                    tx.amount, tx.currency, tx.method.value,
                    TransactionStatus.COMPLETED.value, tx.fraud_score,
                    tx.fee, tx.net_amount,
                    json.dumps(tx.metadata),
                    tx.created_at.isoformat(), datetime.utcnow().isoformat(),
                ),
            )

            # Single commit — both phases succeed or neither does.
            conn.commit()

            tx.mark_completed()
            reconciliation_service.record_transaction(tx)

            m.payment_transactions_total.labels(
                status="completed",
                method=tx.method.value,
                currency=tx.currency,
            ).inc()
            m.payment_amount_processed_usd.inc(tx.amount)
            m.payment_fees_collected_usd.inc(tx.fee)

            logger.info(
                f"[Payment] SUCCESS: tx={tx.id} amount=${tx.amount:.2f} "
                f"fee=${tx.fee:.4f} net=${tx.net_amount:.4f} "
                f"from={tx.from_account} to={tx.to_account}"
            )

        except (DBConnectionError, PoolExhaustedError) as e:
            if conn:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
            tx.mark_failed(str(e))
            m.payment_transactions_total.labels(
                status="failed_db",
                method=tx.method.value,
                currency=tx.currency,
            ).inc()
            m.app_errors_total.labels(
                component="payment_processor", error_type="db_error"
            ).inc()
            logger.error(f"[Payment] DB error for tx={tx.id}: {e}")
            raise

        except Exception as e:
            if conn:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
            tx.mark_failed(str(e))
            raise

        finally:
            if conn:
                try:
                    db_pool.release(conn)
                except Exception:
                    pass
            if _acct_lock.locked():
                try:
                    _acct_lock.release()
                except RuntimeError:
                    pass
            if _order_lock.locked():
                try:
                    _order_lock.release()
                except RuntimeError:
                    pass

        # _idempotency_store already holds tx (reserved slot); it is now
        # updated in-place to the completed state — no separate write needed.
        m.idempotency_cache_size.set(len(_idempotency_store))
        return tx

    async def process_refund(
        self,
        original_tx: Transaction,
        reason: str,
        amount: TypingOptional[float] = None,
    ) -> Transaction:
        """
        Process a refund for a completed transaction.

        Lock order is now _acct_lock → _order_lock, matching process_payment
        (FIX for BUG 3).
        """
        refund_amount = amount or original_tx.amount

        refund_tx = Transaction(
            idempotency_key=f"refund_{original_tx.id}",
            from_account=original_tx.to_account,
            to_account=original_tx.from_account,
            amount=refund_amount,
            currency=original_tx.currency,
            method=original_tx.method,
            metadata={"refund_for": original_tx.id, "reason": reason},
        )

        m.active_payment_requests.inc()
        start = time.monotonic()

        try:
            # ── Idempotency guard for refunds ─────────────────────────────────
            key_lock = await _get_key_lock(refund_tx.idempotency_key)
            async with key_lock:
                existing = _idempotency_store.get(refund_tx.idempotency_key)
                if existing:
                    logger.info(
                        f"[Refund] Idempotency hit: key={refund_tx.idempotency_key} "
                        f"returning cached refund_tx={existing.id}"
                    )
                    return existing

                _idempotency_store[refund_tx.idempotency_key] = refund_tx
                m.idempotency_cache_size.set(len(_idempotency_store))

                try:
                    result = await self._execute_refund(refund_tx, original_tx, refund_amount)
                except Exception:
                    _idempotency_store.pop(refund_tx.idempotency_key, None)
                    m.idempotency_cache_size.set(len(_idempotency_store))
                    raise

            return result

        finally:
            elapsed = time.monotonic() - start
            m.payment_processing_duration_seconds.labels(
                method=refund_tx.method.value
            ).observe(elapsed)
            m.active_payment_requests.dec()

    async def _execute_refund(
        self,
        refund_tx: Transaction,
        original_tx: Transaction,
        refund_amount: float,
    ) -> Transaction:
        """Inner refund pipeline — called while holding the per-key lock."""

        # ── Acquire locks in canonical order: acct → order (FIX for BUG 3) ──
        try:
            lock_start = time.monotonic()
            await asyncio.wait_for(_acct_lock.acquire(), timeout=2.5)
            m.lock_wait_duration_seconds.labels(lock_type="account_lock").observe(
                time.monotonic() - lock_start
            )
        except asyncio.TimeoutError:
            m.deadlock_events_total.labels(lock_type="account_lock").inc()
            m.app_errors_total.labels(
                component="payment_processor", error_type="deadlock"
            ).inc()
            logger.error(
                f"[Refund] Timeout acquiring account_lock "
                f"refund_tx={refund_tx.id} original_tx={original_tx.id}"
            )
            raise PaymentProcessorError(
                "Timeout acquiring account lock for refund"
            )

        try:
            order_start = time.monotonic()
            await asyncio.wait_for(_order_lock.acquire(), timeout=2.5)
            m.lock_wait_duration_seconds.labels(lock_type="order_lock").observe(
                time.monotonic() - order_start
            )
        except asyncio.TimeoutError:
            _acct_lock.release()
            m.deadlock_events_total.labels(lock_type="order_lock").inc()
            m.app_errors_total.labels(
                component="payment_processor", error_type="deadlock"
            ).inc()
            logger.error(
                f"[Refund] Timeout acquiring order_lock "
                f"refund_tx={refund_tx.id} original_tx={original_tx.id}"
            )
            raise PaymentProcessorError(
                "Timeout acquiring order lock during refund"
            )

        conn = None
        try:
            conn = db_pool.acquire()
            conn.execute("BEGIN")

            conn.execute(
                "UPDATE accounts SET balance = balance + ? WHERE id = ?",
                (refund_amount, refund_tx.to_account),
            )
            conn.execute(
                """INSERT INTO transactions
                       (id, idempotency_key, from_account, to_account, amount, currency,
                        method, status, fraud_score, fee, net_amount, metadata,
                        created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0.0, 0.0, ?, ?, ?, ?)""",
                (
                    refund_tx.id, refund_tx.idempotency_key,
                    refund_tx.from_account, refund_tx.to_account,
                    refund_amount, refund_tx.currency,
                    refund_tx.method.value,
                    TransactionStatus.COMPLETED.value,
                    refund_amount,
                    json.dumps(refund_tx.metadata),
                    refund_tx.created_at.isoformat(),
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
            refund_tx.mark_completed()

            m.payment_transactions_total.labels(
                status="refunded",
                method=refund_tx.method.value,
                currency=refund_tx.currency,
            ).inc()
            logger.info(
                f"[Refund] SUCCESS: refund_tx={refund_tx.id} "
                f"original={original_tx.id} amount=${refund_amount:.2f}"
            )

        except DBConnectionError as e:
            if conn:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
            refund_tx.mark_failed(str(e))
            raise

        except Exception as e:
            if conn:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
            refund_tx.mark_failed(str(e))
            raise

        finally:
            if conn:
                try:
                    db_pool.release(conn)
                except Exception:
                    pass
            if _acct_lock.locked():
                try:
                    _acct_lock.release()
                except RuntimeError:
                    pass
            if _order_lock.locked():
                try:
                    _order_lock.release()
                except RuntimeError:
                    pass

        m.idempotency_cache_size.set(len(_idempotency_store))
        return refund_tx


# Singleton
payment_processor = PaymentProcessor()
