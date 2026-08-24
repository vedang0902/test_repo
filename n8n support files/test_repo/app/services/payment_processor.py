"""Core Payment Processing Service.

Orchestrates the full payment lifecycle:
  1. Idempotency check (atomic, per-key lock)
  2. Fraud scoring
  3. Atomic DB transaction: debit + order confirmation (single commit)
  4. Reconciliation recording

Fixes applied:
  BUG 1 (Race on idempotency): per-key asyncio.Lock prevents concurrent
         requests with the same key from both passing the read check.
  BUG 2 (Partial commit): both DB writes now run inside a single
         transaction; the intermediate conn.commit() is removed so a
         failure in phase 2 automatically rolls back phase 1.
  BUG 3 (Deadlock): process_refund now acquires locks in the same order
         as process_payment (_acct_lock then _order_lock).
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
# FIX BUG 1: Access to the store is serialised through a per-key asyncio.Lock.
# The lock is acquired before the read and held until after the write, making
# the check-then-set atomic within the event loop.
_idempotency_store: Dict[str, Transaction] = {}
_idempotency_key_locks: Dict[str, asyncio.Lock] = {}
_idempotency_registry_lock = asyncio.Lock()  # protects _idempotency_key_locks


async def _get_idempotency_key_lock(key: str) -> asyncio.Lock:
    """Return (creating if necessary) the per-key lock for *key*."""
    async with _idempotency_registry_lock:
        if key not in _idempotency_key_locks:
            _idempotency_key_locks[key] = asyncio.Lock()
        return _idempotency_key_locks[key]


# ── Shared locks (consistent acquisition order everywhere) ───────────────────
# FIX BUG 3: Both process_payment and process_refund now acquire in the
# same order: _acct_lock first, _order_lock second.
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
        """Full payment processing pipeline (all three bugs fixed)."""
        start = time.monotonic()
        m.active_payment_requests.inc()

        try:
            tx.status = TransactionStatus.PROCESSING

            # ── Step 1: Idempotency check (FIX BUG 1 — atomic per-key lock) ──
            key_lock = await _get_idempotency_key_lock(tx.idempotency_key)
            async with key_lock:
                existing = _idempotency_store.get(tx.idempotency_key)
                if existing:
                    logger.info(
                        f"[Payment] Idempotency hit: key={tx.idempotency_key} "
                        f"returning cached tx={existing.id}"
                    )
                    return existing

                # Mark the key as in-flight immediately so any concurrent
                # coroutine that acquires the lock after us sees a result and
                # returns early.  We overwrite with the final tx object once
                # processing completes.
                _idempotency_store[tx.idempotency_key] = tx
                m.idempotency_cache_size.set(len(_idempotency_store))

            # ── Step 2: Fraud check ───────────────────────────────────────────
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
                async with key_lock:
                    _idempotency_store[tx.idempotency_key] = tx
                    m.idempotency_cache_size.set(len(_idempotency_store))
                return tx

            # ── Step 3: Acquire locks (_acct_lock then _order_lock) ───────────
            # FIX BUG 3: consistent lock order with process_refund.
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
                    f"[Payment] DEADLOCK: timeout acquiring account_lock tx={tx.id}"
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
                    f"[Payment] DEADLOCK: timeout acquiring order_lock tx={tx.id}"
                )
                raise PaymentProcessorError(
                    f"Deadlock: timeout acquiring order lock for tx {tx.id}"
                )

            # ── Step 4: Atomic two-phase DB write (FIX BUG 2) ────────────────
            # Both the debit (phase 1) and the order insertion (phase 2) run
            # inside a single transaction.  conn.commit() is called only once,
            # after both writes succeed.  Any exception in phase 2 causes the
            # implicit rollback in the finally block, which undoes the debit.
            conn = None
            try:
                conn = db_pool.acquire()

                # Begin explicit transaction (rollback is automatic on error).
                conn.execute("BEGIN")

                # ── Phase 1: Debit account ────────────────────────────────────
                conn.execute(
                    "UPDATE accounts SET balance = balance - ? WHERE id = ? AND is_active = 1",
                    (tx.amount, tx.from_account),
                )
                logger.debug(
                    f"[Payment] Phase 1 staged: will debit ${tx.amount:.2f} "
                    f"from {tx.from_account} (not yet committed)"
                )

                # Simulate intermittent infrastructure failure.
                # FIX BUG 2: because we have NOT committed yet, raising here
                # means the debit is rolled back automatically — no orphaned
                # debit can occur.
                if random.random() < settings.payment.partial_commit_rate:
                    raise PartialCommitError(
                        f"Simulated phase-2 failure for tx={tx.id} "
                        f"(would have debited ${tx.amount:.2f})"
                    )

                # ── Phase 2: Confirm order (same transaction) ─────────────────
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

                # Single commit — both writes land together or not at all.
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

            except PartialCommitError as e:
                # Phase-2 simulated failure: roll back so the debit never lands.
                if conn:
                    try:
                        conn.execute("ROLLBACK")
                    except Exception:
                        pass
                tx.mark_failed(str(e))
                m.payment_transactions_total.labels(
                    status="failed_rolled_back",
                    method=tx.method.value,
                    currency=tx.currency,
                ).inc()
                m.app_errors_total.labels(
                    component="payment_processor",
                    error_type="partial_commit_prevented",
                ).inc()
                logger.error(
                    f"[Payment] Phase-2 failure for tx={tx.id} — "
                    f"transaction rolled back, no orphaned debit created. "
                    f"Reason: {e}"
                )
                raise PaymentProcessorError(str(e)) from e

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

            # Update idempotency store with the completed transaction object.
            async with key_lock:
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
        """Process a refund for a completed transaction.

        FIX BUG 3: locks are now acquired in the same order as
        process_payment (_acct_lock first, then _order_lock), eliminating
        the deadlock that occurred when the two paths held locks in
        opposite order.
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
            # FIX BUG 3: acquire _acct_lock before _order_lock (same order as
            # process_payment).
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
                    f"[Refund] timeout acquiring account_lock "
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
                    f"[Refund] timeout acquiring order_lock "
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
