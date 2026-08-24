"""Core Payment Processing Service.

Orchestrates the full payment lifecycle:
  1. Idempotency check  (atomic via per-key lock)
  2. Fraud scoring
  3. DB debit + order confirmation inside a single ACID transaction (phase 1+2)
  4. Reconciliation recording

Fixes applied
─────────────
BUG 1 – Race condition on idempotency key
  A per-key asyncio.Lock is acquired before the store is read and released
  only after the result is written back.  Concurrent requests with the same
  key block on the lock; the second caller finds the cached result and
  returns it without touching the database.

BUG 2 – Partial commit / orphaned debit
  Both the account-debit UPDATE and the order/transaction INSERTs are now
  issued inside a single DB transaction.  conn.commit() is called once at
  the end; any failure triggers conn.rollback() so the debit is reversed
  atomically.

BUG 3 – Deadlock on lock ordering
  process_refund now acquires _acct_lock first, then _order_lock — the
  same order used by process_payment — eliminating the hold-and-wait cycle.
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

# Per-key locks that make the idempotency check-then-set atomic.
# A defaultdict-style helper keeps the surface area small.
_idempotency_key_locks: Dict[str, asyncio.Lock] = {}
_idempotency_registry_lock = asyncio.Lock()   # guards _idempotency_key_locks itself


async def _get_key_lock(key: str) -> asyncio.Lock:
    """Return (creating if necessary) the per-idempotency-key Lock."""
    async with _idempotency_registry_lock:
        if key not in _idempotency_key_locks:
            _idempotency_key_locks[key] = asyncio.Lock()
        return _idempotency_key_locks[key]


# ── Shared coarse-grained locks (payment + refund must acquire in THIS order) ─
# FIX BUG 3: both code paths now acquire _acct_lock before _order_lock.
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

        Idempotency is guaranteed by holding a per-key asyncio.Lock around
        the read-check-then-write so no two coroutines can race past the
        guard.  The two-phase DB write is wrapped in a single ACID
        transaction to prevent orphaned debits.
        """
        start = time.monotonic()
        m.active_payment_requests.inc()

        try:
            tx.status = TransactionStatus.PROCESSING

            # ── Step 1: Atomic idempotency check-and-reserve ─────────────────
            # FIX BUG 1: acquire the per-key lock BEFORE reading the store so
            # that concurrent coroutines for the same key are serialised here.
            key_lock = await _get_key_lock(tx.idempotency_key)
            await key_lock.acquire()
            try:
                existing = _idempotency_store.get(tx.idempotency_key)
                if existing:
                    logger.info(
                        f"[Payment] Idempotency hit: key={tx.idempotency_key} "
                        f"returning cached tx={existing.id}"
                    )
                    return existing

                # Reserve the key with a PROCESSING sentinel immediately so
                # any concurrent coroutine that acquires the lock next will
                # see an entry and wait for the real result via the lock.
                # (The lock itself is the primary barrier; the sentinel is
                # belt-and-suspenders.)
                _idempotency_store[tx.idempotency_key] = tx
                m.idempotency_cache_size.set(len(_idempotency_store))
            except Exception:
                key_lock.release()
                raise
            # key_lock is released after the final result is written (see
            # the finally block at the end of the processing section below).

            try:
                # ── Step 2: Fraud check ───────────────────────────────────────
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
                    # Update idempotency store with final fraud-blocked result
                    _idempotency_store[tx.idempotency_key] = tx
                    m.idempotency_cache_size.set(len(_idempotency_store))
                    return tx

                # ── Step 3: Acquire coarse locks (acct first, order second) ───
                # FIX BUG 3: consistent lock order prevents deadlock.
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
                        f"[Payment] DEADLOCK: timeout acquiring account_lock "
                        f"tx={tx.id}"
                    )
                    raise PaymentProcessorError(
                        f"Deadlock: timeout acquiring account lock for tx {tx.id}"
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
                        f"[Payment] DEADLOCK: timeout acquiring order_lock "
                        f"tx={tx.id}"
                    )
                    raise PaymentProcessorError(
                        f"Deadlock: timeout acquiring order lock for tx {tx.id}"
                    )

                # ── Step 4: ACID two-phase DB write ───────────────────────────
                # FIX BUG 2: both phases run inside ONE transaction; any error
                # after the debit rolls it back automatically — no orphaned debit.
                conn = None
                try:
                    conn = db_pool.acquire()

                    # Begin explicit transaction (autocommit must be off,
                    # which is the default for most DB drivers).
                    conn.execute("BEGIN")

                    # Phase 1: Debit account
                    conn.execute(
                        "UPDATE accounts SET balance = balance - ? "
                        "WHERE id = ? AND is_active = 1",
                        (tx.amount, tx.from_account),
                    )
                    logger.debug(
                        f"[Payment] Phase 1 staged: will debit "
                        f"${tx.amount:.2f} from {tx.from_account}"
                    )

                    # Phase 2: Confirm order
                    order_id = str(uuid.uuid4())
                    merchant_id = f"merchant_{random.randint(100, 999)}"

                    conn.execute(
                        """INSERT INTO orders
                               (id, transaction_id, merchant_id, total_amount, status, created_at)
                           VALUES (?, ?, ?, ?, 'confirmed', ?)""",
                        (
                            order_id, tx.id, merchant_id,
                            tx.amount, datetime.utcnow().isoformat(),
                        ),
                    )

                    conn.execute(
                        """INSERT INTO transactions
                               (id, idempotency_key, from_account, to_account, amount, currency,
                                method, status, fraud_score, fee, net_amount, metadata,
                                created_at, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            tx.id, tx.idempotency_key,
                            tx.from_account, tx.to_account,
                            tx.amount, tx.currency, tx.method.value,
                            TransactionStatus.COMPLETED.value, tx.fraud_score,
                            tx.fee, tx.net_amount,
                            json.dumps(tx.metadata),
                            tx.created_at.isoformat(),
                            datetime.utcnow().isoformat(),
                        ),
                    )

                    # Single commit covers both phases atomically.
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
                    # Roll back so the debit does not persist.
                    if conn:
                        try:
                            conn.rollback()
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
                    # Unexpected error — roll back to prevent orphaned debit.
                    if conn:
                        try:
                            conn.rollback()
                        except Exception:
                            pass
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

                # Write final result into idempotency store.
                _idempotency_store[tx.idempotency_key] = tx
                m.idempotency_cache_size.set(len(_idempotency_store))
                return tx

            finally:
                # Always release the per-key idempotency lock so waiting
                # coroutines can proceed (and find the cached result).
                try:
                    key_lock.release()
                except RuntimeError:
                    pass

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

        FIX BUG 3: Lock order is now _acct_lock → _order_lock, identical to
        process_payment, eliminating the deadlock.
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
            # FIX BUG 3: acquire _acct_lock first, then _order_lock.
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
                    f"[Refund] DEADLOCK: timeout acquiring account_lock "
                    f"refund_tx={refund_tx.id} original_tx={original_tx.id}"
                )
                raise PaymentProcessorError(
                    "Deadlock: timeout acquiring account lock for refund"
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
                    f"[Refund] DEADLOCK: timeout acquiring order_lock "
                    f"while account_lock held. original_tx={original_tx.id}"
                )
                raise PaymentProcessorError(
                    "Deadlock: timeout acquiring order lock during refund"
                )

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

                except Exception:
                    try:
                        conn.rollback()
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
