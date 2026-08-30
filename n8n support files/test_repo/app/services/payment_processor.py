"""
Core Payment Processing Service.

Orchestrates the full payment lifecycle:
  1. Idempotency check (atomic, per-key lock)
  2. Fraud scoring
  3. Database debit + order confirmation (single atomic transaction)
  4. Reconciliation recording

Fixes applied:
  FIX 1: Idempotency check is now atomic via per-key asyncio.Lock.
  FIX 2: Both DB phases wrapped in a single transaction; compensating credit
          issued on Phase 2 failure to prevent orphaned debits.
  FIX 3: process_refund now acquires locks in the same order as
          process_payment (_acct_lock then _order_lock) to prevent deadlocks.
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
# FIX 1: Access is serialised per-key via _idempotency_key_locks.
# The store itself is still in-memory; for production replace with a DB
# unique constraint + INSERT OR IGNORE + read-back for cross-process safety.
_idempotency_store: Dict[str, Transaction] = {}
_idempotency_key_locks: Dict[str, asyncio.Lock] = {}
_idempotency_registry_lock = asyncio.Lock()  # guards _idempotency_key_locks


async def _get_idempotency_lock(key: str) -> asyncio.Lock:
    """Return (and lazily create) a per-key lock, thread-safely."""
    async with _idempotency_registry_lock:
        if key not in _idempotency_key_locks:
            _idempotency_key_locks[key] = asyncio.Lock()
        return _idempotency_key_locks[key]


# ── Locks (consistent acquisition order: _acct_lock then _order_lock) ────────
# FIX 3: process_refund now uses the same order, eliminating the deadlock.
_acct_lock  = asyncio.Lock()
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

        FIX 1 – atomic idempotency: hold per-key lock across check + write.
        FIX 2 – atomic DB writes: single transaction with rollback +
                 compensating credit on Phase-2 failure.
        FIX 3 – consistent lock order: _acct_lock then _order_lock.
        """
        start = time.monotonic()
        m.active_payment_requests.inc()

        try:
            tx.status = TransactionStatus.PROCESSING

            # ── Step 1: Idempotency check (FIX 1: atomic via per-key lock) ───
            idem_lock = await _get_idempotency_lock(tx.idempotency_key)
            async with idem_lock:
                existing = _idempotency_store.get(tx.idempotency_key)
                if existing:
                    logger.info(
                        f"[Payment] Idempotency hit: key={tx.idempotency_key} "
                        f"returning cached tx={existing.id}"
                    )
                    return existing

                # Mark the key as in-flight immediately while we still hold
                # the per-key lock so no other coroutine can slip through.
                _idempotency_store[tx.idempotency_key] = tx
                m.idempotency_cache_size.set(len(_idempotency_store))

            # ── Step 2: Fraud check ──────────────────────────────────────────
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
                # Update idempotency store with final fraud-blocked state
                async with await _get_idempotency_lock(tx.idempotency_key):
                    _idempotency_store[tx.idempotency_key] = tx
                return tx

            # ── Step 3: Acquire locks (FIX 3: _acct_lock then _order_lock) ──
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
                    f"[Payment] LOCK TIMEOUT: could not acquire account_lock "
                    f"tx={tx.id}"
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
                    f"[Payment] LOCK TIMEOUT: could not acquire order_lock "
                    f"tx={tx.id}"
                )
                raise PaymentProcessorError(
                    f"Timeout acquiring order lock for tx {tx.id}"
                )

            # ── Step 4: Atomic two-phase DB write (FIX 2) ────────────────────
            # Both the account debit and the order insertion run inside a
            # single DB transaction.  If Phase 2 fails we ROLLBACK the debit
            # (no orphan) AND issue an explicit compensating credit as a
            # belt-and-suspenders guard for engines without full ACID rollback.
            conn = None
            try:
                conn = db_pool.acquire()

                # Begin explicit transaction (autocommit must be off).
                conn.execute("BEGIN")

                # ── Phase 1: Debit account ───────────────────────────────────
                conn.execute(
                    "UPDATE accounts SET balance = balance - ? WHERE id = ? AND is_active = 1",
                    (tx.amount, tx.from_account),
                )
                logger.debug(
                    f"[Payment] Phase 1 staged: will debit ${tx.amount:.2f} "
                    f"from {tx.from_account} (not yet committed)"
                )

                # ── Phase 2: Confirm order ───────────────────────────────────
                order_id   = str(uuid.uuid4())
                merchant_id = f"merchant_{random.randint(100, 999)}"

                # Simulate intermittent Phase-2 failure BEFORE we commit so
                # the rollback keeps both phases consistent.
                if random.random() < settings.payment.partial_commit_rate:
                    conn.execute("ROLLBACK")
                    # Belt-and-suspenders: explicit compensating credit in case
                    # the engine had already flushed the debit page.
                    try:
                        conn.execute(
                            "UPDATE accounts SET balance = balance + ? WHERE id = ?",
                            (tx.amount, tx.from_account),
                        )
                        conn.execute("COMMIT")
                        logger.warning(
                            f"[Payment] COMPENSATING CREDIT applied: "
                            f"tx={tx.id} credited ${tx.amount:.2f} back to "
                            f"{tx.from_account} after Phase-2 failure."
                        )
                    except Exception as comp_err:
                        logger.critical(
                            f"[Payment] COMPENSATION FAILED: tx={tx.id} "
                            f"manual review required. err={comp_err}"
                        )

                    tx.mark_failed(
                        f"Phase 2 write failed after debit of ${tx.amount:.2f} "
                        f"(intermittent DB error) — debit rolled back"
                    )
                    m.payment_transactions_total.labels(
                        status="failed_phase2",
                        method=tx.method.value,
                        currency=tx.currency,
                    ).inc()
                    m.app_errors_total.labels(
                        component="payment_processor",
                        error_type="phase2_failure_compensated",
                    ).inc()
                    logger.error(
                        f"[Payment] Phase-2 failure for tx={tx.id}: debit "
                        f"rolled back, no orphaned debit created."
                    )
                    # Update idempotency with failed state so callers can retry
                    async with await _get_idempotency_lock(tx.idempotency_key):
                        _idempotency_store[tx.idempotency_key] = tx
                    return tx

                conn.execute(
                    """INSERT INTO orders
                         (id, transaction_id, merchant_id, total_amount, status, created_at)
                       VALUES (?, ?, ?, ?, 'confirmed', ?)""",
                    (
                        order_id, tx.id, merchant_id,
                        tx.amount, datetime.utcnow().isoformat(),
                    ),
                )

                # Insert transaction record
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

                # Single commit — both phases land atomically.
                conn.execute("COMMIT")

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
                # Attempt rollback so no partial state remains.
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

            # Persist final completed state to idempotency store.
            async with await _get_idempotency_lock(tx.idempotency_key):
                _idempotency_store[tx.idempotency_key] = tx
                m.idempotency_cache_size.set(len(_idempotency_store))

            return tx

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
            logger.error(
                f"[Payment] Unexpected error for tx={tx.id}: {e}", exc_info=True
            )
            raise

        finally:
            elapsed = time.monotonic() - start
            m.payment_processing_duration_seconds.labels(
                method=tx.method.value
            ).observe(elapsed)
            m.active_payment_requests.dec()

    async def process_refund(
        self,
        original_tx: Transaction,
        reason: str,
        amount: TypingOptional[float] = None,
    ) -> Transaction:
        """
        Process a refund for a completed transaction.

        FIX 3: Locks are now acquired in the same order as process_payment
        (_acct_lock then _order_lock) to eliminate the classic deadlock.
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
            # FIX 3: Acquire _acct_lock FIRST, then _order_lock —
            # identical order to process_payment.
            try:
                lock_start = time.monotonic()
                await asyncio.wait_for(_acct_lock.acquire(), timeout=2.5)
                m.lock_wait_duration_seconds.labels(lock_type="account_lock").observe(
                    time.monotonic() - lock_start
                )
            except asyncio.TimeoutError:
                m.deadlock_events_total.labels(lock_type="account_lock").inc()
                m.app_errors_total.labels(
                    component="payment_processor", error_type="lock_timeout"
                ).inc()
                logger.error(
                    f"[Refund] LOCK TIMEOUT: account_lock "
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
                    component="payment_processor", error_type="lock_timeout"
                ).inc()
                logger.error(
                    f"[Refund] LOCK TIMEOUT: order_lock "
                    f"refund_tx={refund_tx.id} original_tx={original_tx.id}"
                )
                raise PaymentProcessorError(
                    "Timeout acquiring order lock for refund"
                )

            # Process refund credit inside a single atomic transaction.
            try:
                conn = db_pool.acquire()
                try:
                    conn.execute("BEGIN")
                    conn.execute(
                        "UPDATE accounts SET balance = balance + ? WHERE id = ?",
                        (refund_amount, refund_tx.to_account),
                    )
                    conn.execute(
                        """INSERT INTO transactions
                             (id, idempotency_key, from_account, to_account, amount,
                              currency, method, status, fraud_score, fee, net_amount,
                              metadata, created_at, updated_at)
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
                    conn.execute("COMMIT")
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

                except Exception:
                    try:
                        conn.execute("ROLLBACK")
                    except Exception:
                        pass
                    raise

                finally:
                    db_pool.release(conn)

            except DBConnectionError as e:
                refund_tx.mark_failed(str(e))
                raise

            finally:
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

            return refund_tx

        finally:
            elapsed = time.monotonic() - start
            m.payment_processing_duration_seconds.labels(
                method=refund_tx.method.value
            ).observe(elapsed)
            m.active_payment_requests.dec()


# Singleton
payment_processor = PaymentProcessor()
