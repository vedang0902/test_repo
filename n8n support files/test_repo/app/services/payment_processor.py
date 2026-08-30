"""
Core Payment Processing Service.

Orchestrates the full payment lifecycle:
  1. Idempotency check  (atomic via per-key lock)
  2. Fraud scoring
  3. Database debit + order confirmation in ONE atomic transaction
  4. Reconciliation recording

All three bugs from the original file have been fixed:
  BUG 1 - Race condition on idempotency key  → per-key asyncio.Lock
  BUG 2 - Partial transaction commit          → single DB transaction + compensating rollback
  BUG 3 - Deadlock on lock ordering           → canonical lock order (_acct_lock then _order_lock)
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
# Idempotency store — now protected by a per-key lock so the check+write is
# atomic within a single asyncio event loop.
# ---------------------------------------------------------------------------
# Key → Transaction result
_idempotency_store: Dict[str, Transaction] = {}

# Per-key locks: while a coroutine holds the lock for a given idempotency_key
# no other coroutine can pass the guard for the same key.
_idempotency_key_locks: Dict[str, asyncio.Lock] = {}
_idempotency_registry_lock: asyncio.Lock = asyncio.Lock()  # guards the dict of locks


async def _get_key_lock(key: str) -> asyncio.Lock:
    """Return (creating if necessary) the per-idempotency-key Lock."""
    async with _idempotency_registry_lock:
        if key not in _idempotency_key_locks:
            _idempotency_key_locks[key] = asyncio.Lock()
        return _idempotency_key_locks[key]


# ---------------------------------------------------------------------------
# Shared structural locks.
# FIX (BUG 3): BOTH process_payment and process_refund now acquire locks in
# the SAME canonical order: _acct_lock first, then _order_lock.
# ---------------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # process_payment
    # -----------------------------------------------------------------------
    async def process_payment(self, tx: Transaction) -> Transaction:
        """
        Full payment processing pipeline — all three bugs fixed.
        """
        start = time.monotonic()
        m.active_payment_requests.inc()

        try:
            tx.status = TransactionStatus.PROCESSING

            # ── Step 1: Atomic idempotency check+set (FIX for BUG 1) ─────────
            # Acquire the per-key lock BEFORE reading the store.  Any concurrent
            # coroutine that arrives with the same key will block here until the
            # first one has either returned the cached result or finished writing
            # the new result to the store.
            key_lock = await _get_key_lock(tx.idempotency_key)
            async with key_lock:
                existing = _idempotency_store.get(tx.idempotency_key)
                if existing:
                    logger.info(
                        f"[Payment] Idempotency hit: key={tx.idempotency_key} "
                        f"returning cached tx={existing.id}"
                    )
                    return existing

                # No cached result — process the payment while still holding
                # the key lock so no concurrent duplicate can slip through.
                result = await self._execute_payment(tx)

                # Persist result atomically before releasing the key lock.
                _idempotency_store[tx.idempotency_key] = result
                m.idempotency_cache_size.set(len(_idempotency_store))

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

    # -----------------------------------------------------------------------
    # _execute_payment  (internal — called only while key_lock is held)
    # -----------------------------------------------------------------------
    async def _execute_payment(self, tx: Transaction) -> Transaction:
        """
        Fraud check + atomic two-phase DB write.
        Called only from within the per-key idempotency lock.
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
            return tx

        # ── Step 3: Acquire structural locks (canonical order: acct → order) ──
        # FIX (BUG 3): always acquire _acct_lock before _order_lock.
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
        # Both the account debit and the order insertion are executed inside a
        # single DB transaction.  If phase 2 fails the whole transaction is
        # rolled back — no orphaned debit.  We also issue an explicit
        # compensating UPDATE (credit back) as a belt-and-suspenders guard for
        # databases that do not support transactional DDL.
        conn = None
        try:
            conn = db_pool.acquire()

            # Begin explicit transaction (autocommit must be off).
            conn.execute("BEGIN")

            # ── Phase 1: Debit account ────────────────────────────────────────
            conn.execute(
                "UPDATE accounts SET balance = balance - ? WHERE id = ? AND is_active = 1",
                (tx.amount, tx.from_account),
            )
            logger.debug(
                f"[Payment] Phase 1 (within txn): debited ${tx.amount:.2f} "
                f"from {tx.from_account}"
            )

            # ── Phase 2: Confirm order ────────────────────────────────────────
            order_id   = str(uuid.uuid4())
            merchant_id = f"merchant_{random.randint(100, 999)}"

            conn.execute(
                """INSERT INTO orders
                       (id, transaction_id, merchant_id, total_amount, status, created_at)
                   VALUES (?, ?, ?, ?, 'confirmed', ?)""",
                (order_id, tx.id, merchant_id, tx.amount,
                 datetime.utcnow().isoformat()),
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

            # Commit both phases atomically.
            conn.commit()
            logger.debug(
                f"[Payment] Phases 1+2 committed atomically for tx={tx.id}"
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

        except Exception as db_exc:
            # Roll back the entire transaction so the debit is reversed.
            if conn:
                try:
                    conn.execute("ROLLBACK")
                    logger.warning(
                        f"[Payment] Transaction rolled back for tx={tx.id}: {db_exc}"
                    )
                except Exception as rb_exc:
                    # Rollback itself failed (e.g. connection lost).  Issue a
                    # compensating credit so the account is not left debited.
                    logger.critical(
                        f"[Payment] ROLLBACK FAILED for tx={tx.id}: {rb_exc}. "
                        f"Issuing compensating credit of ${tx.amount:.2f} to "
                        f"{tx.from_account}."
                    )
                    try:
                        conn2 = db_pool.acquire()
                        try:
                            conn2.execute(
                                "UPDATE accounts SET balance = balance + ? WHERE id = ?",
                                (tx.amount, tx.from_account),
                            )
                            conn2.commit()
                            logger.info(
                                f"[Payment] Compensating credit applied for tx={tx.id}"
                            )
                        finally:
                            db_pool.release(conn2)
                    except Exception as comp_exc:
                        logger.critical(
                            f"[Payment] COMPENSATING CREDIT ALSO FAILED for tx={tx.id}: "
                            f"{comp_exc}. Manual intervention required."
                        )
                        m.app_errors_total.labels(
                            component="payment_processor",
                            error_type="compensation_failed",
                        ).inc()

            if isinstance(db_exc, (DBConnectionError, PoolExhaustedError)):
                tx.mark_failed(str(db_exc))
                m.payment_transactions_total.labels(
                    status="failed_db",
                    method=tx.method.value,
                    currency=tx.currency,
                ).inc()
                m.app_errors_total.labels(
                    component="payment_processor", error_type="db_error"
                ).inc()
                logger.error(f"[Payment] DB error for tx={tx.id}: {db_exc}")
            else:
                tx.mark_failed(str(db_exc))
                m.payment_transactions_total.labels(
                    status="failed",
                    method=tx.method.value,
                    currency=tx.currency,
                ).inc()
                m.app_errors_total.labels(
                    component="payment_processor", error_type="unexpected"
                ).inc()
                logger.error(
                    f"[Payment] Unexpected DB-phase error for tx={tx.id}: {db_exc}",
                    exc_info=True,
                )
            raise

        finally:
            if conn:
                try:
                    db_pool.release(conn)
                except Exception:
                    pass
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

        return tx

    # -----------------------------------------------------------------------
    # process_refund
    # -----------------------------------------------------------------------
    async def process_refund(
        self,
        original_tx: Transaction,
        reason: str,
        amount: TypingOptional[float] = None,
    ) -> Transaction:
        """
        Process a refund for a completed transaction.

        FIX (BUG 3): Lock order is now identical to process_payment:
          _acct_lock first, then _order_lock.
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
            # ── Acquire locks in canonical order: acct → order ────────────────
            # FIX (BUG 3): previously this was order → acct (reversed).
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
                    f"[Refund] Timeout acquiring order_lock while account_lock held. "
                    f"original_tx={original_tx.id}"
                )
                raise PaymentProcessorError(
                    "Timeout acquiring order lock during refund"
                )

            # ── Process refund credit ─────────────────────────────────────────
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
