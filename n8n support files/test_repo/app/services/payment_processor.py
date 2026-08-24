"""
Core Payment Processing Service.

Orchestrates the full payment lifecycle:
  1. Idempotency check  (FIXED: per-key lock makes check-and-set atomic)
  2. Fraud scoring
  3. Atomic DB transaction: debit + order confirm  (FIXED: single commit)
  4. Compensating credit on Phase 2 failure        (FIXED: saga rollback)
  5. Reconciliation recording
  6. Consistent lock ordering in refund path        (FIXED: deadlock eliminated)
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

# ---------------------------------------------------------------------------
# Idempotency store
# FIX: Access is now serialised through a per-key asyncio.Lock so that the
#      check and the write are effectively atomic within this process.
#      A production deployment should back this with a DB unique constraint
#      (INSERT OR IGNORE + read-back) for multi-replica safety.
# ---------------------------------------------------------------------------
_idempotency_store: Dict[str, Transaction] = {}
_idempotency_key_locks: Dict[str, asyncio.Lock] = {}
_idempotency_meta_lock = asyncio.Lock()   # guards _idempotency_key_locks dict


async def _get_idempotency_lock(key: str) -> asyncio.Lock:
    """Return (creating if necessary) the per-key lock in a thread-safe way."""
    async with _idempotency_meta_lock:
        if key not in _idempotency_key_locks:
            _idempotency_key_locks[key] = asyncio.Lock()
        return _idempotency_key_locks[key]


# ---------------------------------------------------------------------------
# Locks
# FIX: Both process_payment AND process_refund now acquire in the same order:
#        _acct_lock  →  _order_lock
#      This eliminates the lock-ordering deadlock entirely.
# ---------------------------------------------------------------------------
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
        Full payment processing pipeline — all three bugs fixed.
        """
        start = time.monotonic()
        m.active_payment_requests.inc()

        try:
            tx.status = TransactionStatus.PROCESSING

            # ── Step 1: Idempotency check — now atomic via per-key lock ──────
            idem_lock = await _get_idempotency_lock(tx.idempotency_key)
            async with idem_lock:
                existing = _idempotency_store.get(tx.idempotency_key)
                if existing:
                    logger.info(
                        f"[Payment] Idempotency hit: key={tx.idempotency_key} "
                        f"returning cached tx={existing.id}"
                    )
                    return existing

                # Mark the key as in-flight immediately (inside the lock) so
                # that any racing coroutine that acquires the lock next will
                # see the entry and return the cached result once we update it.
                # We store a sentinel now and overwrite with the real tx later.
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
                async with idem_lock:
                    _idempotency_store[tx.idempotency_key] = tx
                m.idempotency_cache_size.set(len(_idempotency_store))
                return tx

            # ── Step 3: Acquire locks in CONSISTENT order (_acct → _order) ──
            try:
                lock_start = time.monotonic()
                acquired = await asyncio.wait_for(_acct_lock.acquire(), timeout=3.0)
                m.lock_wait_duration_seconds.labels(lock_type="account_lock").observe(
                    time.monotonic() - lock_start
                )
            except asyncio.TimeoutError:
                m.deadlock_events_total.labels(lock_type="account_lock").inc()
                m.app_errors_total.labels(
                    component="payment_processor", error_type="deadlock"
                ).inc()
                logger.error(
                    f"[Payment] TIMEOUT acquiring account_lock tx={tx.id}"
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
                    f"[Payment] TIMEOUT acquiring order_lock tx={tx.id}"
                )
                raise PaymentProcessorError(
                    f"Timeout acquiring order lock for tx {tx.id}"
                )

            # ── Step 4: ATOMIC two-phase DB write ────────────────────────────
            # FIX: Both the debit (Phase 1) and the order insert (Phase 2) are
            #      executed inside a single database transaction.  conn.commit()
            #      is called ONCE, after both writes succeed.  If Phase 2 fails
            #      we call conn.rollback() which undoes the debit atomically —
            #      no orphaned debit is possible.
            #
            #      For extra safety a compensating credit (saga rollback) is
            #      also attempted should the rollback itself fail (e.g. the
            #      connection is broken).
            conn = None
            try:
                conn = db_pool.acquire()

                # Begin explicit transaction (autocommit must be off, which is
                # the default for most DB-API 2.0 drivers including sqlite3).
                # If the driver uses autocommit mode, call conn.begin() here.

                # Phase 1: debit (not committed yet)
                conn.execute(
                    "UPDATE accounts SET balance = balance - ? WHERE id = ? AND is_active = 1",
                    (tx.amount, tx.from_account),
                )
                logger.debug(
                    f"[Payment] Phase 1 staged: debit ${tx.amount:.2f} "
                    f"from {tx.from_account} (not committed)"
                )

                # Simulate intermittent Phase 2 failure (same probability as
                # before, but now the debit is NOT committed so no orphan).
                if random.random() < settings.payment.partial_commit_rate:
                    # Roll back the staged debit — account balance is restored.
                    conn.rollback()
                    tx.mark_failed(
                        f"Phase 2 write failed — debit of ${tx.amount:.2f} "
                        f"rolled back cleanly (no orphaned debit)"
                    )
                    m.payment_transactions_total.labels(
                        status="failed_rolled_back",
                        method=tx.method.value,
                        currency=tx.currency,
                    ).inc()
                    m.app_errors_total.labels(
                        component="payment_processor",
                        error_type="phase2_failure_rolled_back",
                    ).inc()
                    logger.warning(
                        f"[Payment] Phase 2 failure — tx={tx.id} debit "
                        f"rolled back cleanly. No orphaned debit."
                    )
                    async with idem_lock:
                        _idempotency_store[tx.idempotency_key] = tx
                    m.idempotency_cache_size.set(len(_idempotency_store))
                    return tx

                # Phase 2: confirm order (still inside the same transaction)
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
                        tx.id, tx.idempotency_key, tx.from_account, tx.to_account,
                        tx.amount, tx.currency, tx.method.value,
                        TransactionStatus.COMPLETED.value, tx.fraud_score,
                        tx.fee, tx.net_amount,
                        json.dumps(tx.metadata),
                        tx.created_at.isoformat(), datetime.utcnow().isoformat(),
                    ),
                )

                # Single commit — both phases land atomically.
                conn.commit()
                logger.debug(
                    f"[Payment] Atomic commit: debit + order confirmed for tx={tx.id}"
                )

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
                # Attempt rollback; if that also fails we log but do not
                # raise a second exception over the original.
                if conn:
                    try:
                        conn.rollback()
                        logger.info(
                            f"[Payment] DB error — transaction rolled back for tx={tx.id}"
                        )
                    except Exception as rb_err:
                        logger.error(
                            f"[Payment] Rollback failed for tx={tx.id}: {rb_err}. "
                            f"Attempting compensating credit."
                        )
                        # Last-resort saga compensation: issue a credit to undo
                        # any partially visible debit.
                        try:
                            comp_conn = db_pool.acquire()
                            try:
                                comp_conn.execute(
                                    "UPDATE accounts SET balance = balance + ? "
                                    "WHERE id = ? AND is_active = 1",
                                    (tx.amount, tx.from_account),
                                )
                                comp_conn.commit()
                                logger.info(
                                    f"[Payment] Compensating credit applied for tx={tx.id} "
                                    f"amount=${tx.amount:.2f}"
                                )
                            finally:
                                db_pool.release(comp_conn)
                        except Exception as comp_err:
                            logger.critical(
                                f"[Payment] COMPENSATING CREDIT FAILED for tx={tx.id}: "
                                f"{comp_err}. Manual intervention required."
                            )
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

            # Update idempotency store with the completed transaction.
            async with idem_lock:
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
            logger.error(f"[Payment] Unexpected error for tx={tx.id}: {e}", exc_info=True)
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

        FIX: Lock acquisition order changed to _acct_lock → _order_lock,
             matching process_payment.  The classic deadlock is eliminated.
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
            # Acquire _acct_lock FIRST, then _order_lock — same order as
            # process_payment.  This eliminates the lock-ordering deadlock.
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
                    f"[Refund] TIMEOUT acquiring account_lock "
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
                    f"[Refund] TIMEOUT acquiring order_lock "
                    f"refund_tx={refund_tx.id} original_tx={original_tx.id}"
                )
                raise PaymentProcessorError(
                    "Timeout acquiring order lock for refund"
                )

            try:
                conn = db_pool.acquire()
                try:
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
