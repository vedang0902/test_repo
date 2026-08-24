"""
Core Payment Processing Service.

Orchestrates the full payment lifecycle:
  1. Idempotency check  (atomic via per-key lock)
  2. Fraud scoring
  3. DB debit + order confirm in a single ACID transaction (phase 1+2)
  4. Compensating credit on phase-2 failure (saga pattern)
  5. Reconciliation recording

All three original bugs have been fixed:
  BUG 1 (race condition)   → per-key asyncio.Lock makes check+write atomic
  BUG 2 (partial commit)   → single conn.execute / conn.rollback wraps both phases;
                              compensating UPDATE credits the account on failure
  BUG 3 (deadlock)         → process_refund now acquires _acct_lock then _order_lock
                              (same order as process_payment)
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
# Idempotency store — now protected by a per-key asyncio.Lock (fixes BUG 1).
# The global _idempotency_meta_lock guards mutations to _idempotency_locks so
# that lock-object creation is itself race-free.
# ---------------------------------------------------------------------------
_idempotency_store: Dict[str, Transaction] = {}
_idempotency_locks: Dict[str, asyncio.Lock] = {}
_idempotency_meta_lock = asyncio.Lock()


async def _get_idempotency_lock(key: str) -> asyncio.Lock:
    """Return (creating if necessary) the per-key idempotency lock."""
    async with _idempotency_meta_lock:
        if key not in _idempotency_locks:
            _idempotency_locks[key] = asyncio.Lock()
        return _idempotency_locks[key]


# ---------------------------------------------------------------------------
# Deadlock-free shared locks.
# BOTH process_payment AND process_refund now acquire in the order:
#   _acct_lock  →  _order_lock                          (fixes BUG 3)
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

            # ── Step 1: Atomic idempotency check+write (BUG 1 fixed) ─────────
            # Acquire the per-key lock BEFORE reading the store so that no two
            # coroutines can simultaneously pass the "not in store" check.
            idem_lock = await _get_idempotency_lock(tx.idempotency_key)
            async with idem_lock:
                existing = _idempotency_store.get(tx.idempotency_key)
                if existing:
                    logger.info(
                        f"[Payment] Idempotency hit: key={tx.idempotency_key} "
                        f"returning cached tx={existing.id}"
                    )
                    return existing

                # Reserve the slot immediately so any concurrent coroutine that
                # obtains the lock next will see the entry and return early.
                # We will overwrite this with the final Transaction object once
                # processing completes (or leave the terminal status on failure).
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
                # Idempotency slot was already written above; update status.
                async with idem_lock:
                    _idempotency_store[tx.idempotency_key] = tx
                return tx

            # ── Step 3: Acquire locks in canonical order (BUG 3 fixed) ───────
            # Order: _acct_lock → _order_lock  (same in process_refund below)
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

            # ── Step 4: Single atomic DB transaction (BUG 2 fixed) ───────────
            # Both the account debit and the order/transaction insert are
            # executed inside ONE database transaction.  If anything fails
            # after the debit, we roll back (the DB undoes the debit), so
            # there is no orphaned debit.  We also add an explicit
            # compensating UPDATE as a belt-and-suspenders safety net in case
            # the driver has already auto-committed phase 1 somehow.
            conn = None
            try:
                conn = db_pool.acquire()

                # Begin explicit transaction (most DB drivers default to
                # autocommit=False when using execute(); call explicitly for
                # clarity and safety).
                conn.execute("BEGIN")

                # Phase 1: Debit account
                conn.execute(
                    "UPDATE accounts SET balance = balance - ? WHERE id = ? AND is_active = 1",
                    (tx.amount, tx.from_account),
                )
                logger.debug(
                    f"[Payment] Phase 1 staged: debit ${tx.amount:.2f} "
                    f"from {tx.from_account} (not yet committed)"
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

                # Both phases succeeded — commit atomically.
                conn.commit()
                logger.debug(
                    f"[Payment] Atomic commit succeeded: tx={tx.id} "
                    f"debit=${tx.amount:.2f} order={order_id}"
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
                # Roll back the entire unit of work — no orphaned debit.
                if conn:
                    try:
                        conn.rollback()
                        logger.warning(
                            f"[Payment] DB error — rolled back tx={tx.id}: {e}"
                        )
                    except Exception as rb_err:
                        # Last-resort compensating credit if rollback itself
                        # failed (e.g. connection already closed after a
                        # partial write).  Belt-and-suspenders only.
                        logger.critical(
                            f"[Payment] Rollback failed for tx={tx.id}: {rb_err}. "
                            f"Attempting compensating credit of ${tx.amount:.2f} "
                            f"to {tx.from_account}."
                        )
                        try:
                            comp_conn = db_pool.acquire()
                            comp_conn.execute(
                                "UPDATE accounts SET balance = balance + ? WHERE id = ?",
                                (tx.amount, tx.from_account),
                            )
                            comp_conn.commit()
                            db_pool.release(comp_conn)
                            logger.info(
                                f"[Payment] Compensating credit issued: "
                                f"${tx.amount:.2f} → {tx.from_account}"
                            )
                        except Exception as comp_err:
                            logger.critical(
                                f"[Payment] COMPENSATING CREDIT ALSO FAILED for "
                                f"tx={tx.id}: {comp_err}. Manual intervention required."
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

            except Exception as e:
                # Unexpected error — roll back so no partial state leaks.
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
                # Release locks in reverse acquisition order.
                if _order_lock.locked():
                    try:
                        _order_lock.release()
                    except RuntimeError:
                        pass
                if _acct_lock.locked():
                    try:
                        _acct_lock.release()
                    except RuntimeError:
                        pass

            # Persist final state back to idempotency store.
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

        BUG 3 fixed: lock acquisition order is now identical to process_payment:
          _acct_lock  →  _order_lock
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
            # ── Acquire locks in canonical order: _acct_lock → _order_lock ──
            # This matches process_payment and eliminates the deadlock.
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

            # Process refund credit inside a single DB transaction.
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
                # Release in reverse acquisition order.
                if _order_lock.locked():
                    try:
                        _order_lock.release()
                    except RuntimeError:
                        pass
                if _acct_lock.locked():
                    try:
                        _acct_lock.release()
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
