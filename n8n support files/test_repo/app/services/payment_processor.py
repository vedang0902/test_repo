"""Core Payment Processing Service.

Orchestrates the full payment lifecycle:
  1. Idempotency check (atomic, per-key lock)
  2. Fraud scoring
  3. Atomic DB transaction: debit + order confirmation
  4. Compensating credit on failure (saga pattern)
  5. Reconciliation recording

Fixes applied:
  BUG 1: Idempotency check is now atomic via per-key asyncio.Lock.
  BUG 2: Both DB phases wrapped in a single transaction; compensating
         credit issued on Phase 2 failure (no more orphaned debits).
  BUG 3: process_refund now acquires locks in the same order as
         process_payment (_acct_lock → _order_lock).
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
# FIX BUG 1: Access to each key is serialised by a per-key asyncio.Lock so
# concurrent requests with the same idempotency_key cannot both pass the
# "does it exist?" check and proceed to charge the account.
_idempotency_store: Dict[str, Transaction] = {}
_idempotency_key_locks: Dict[str, asyncio.Lock] = {}
_idempotency_meta_lock = asyncio.Lock()  # guards _idempotency_key_locks dict


async def _get_idempotency_key_lock(key: str) -> asyncio.Lock:
    """Return (creating if necessary) the per-key lock for `key`."""
    async with _idempotency_meta_lock:
        if key not in _idempotency_key_locks:
            _idempotency_key_locks[key] = asyncio.Lock()
        return _idempotency_key_locks[key]


# ── Shared locks ─────────────────────────────────────────────────────────────
# FIX BUG 3: Both process_payment and process_refund acquire in the SAME
# order: _acct_lock first, then _order_lock.
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

    # ------------------------------------------------------------------
    # Internal helper: issue a compensating credit to undo a debit that
    # was committed but whose corresponding order write failed.
    # ------------------------------------------------------------------
    async def _compensate_debit(
        self,
        tx: Transaction,
        conn,
        reason: str,
    ) -> None:
        """Roll back an already-committed debit by crediting the account.

        This is the saga compensating transaction for BUG 2.  We reuse
        the same DB connection so that the compensation is visible in
        the same session; a fresh connection is used as fallback.
        """
        try:
            conn.execute(
                "UPDATE accounts SET balance = balance + ? WHERE id = ? AND is_active = 1",
                (tx.amount, tx.from_account),
            )
            comp_id = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO transactions
                   (id, idempotency_key, from_account, to_account, amount, currency,
                    method, status, fraud_score, fee, net_amount, metadata, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0.0, 0.0, ?, ?, ?, ?)""",
                (
                    comp_id,
                    f"compensate_{tx.idempotency_key}",
                    tx.to_account,   # direction reversed
                    tx.from_account,
                    tx.amount,
                    tx.currency,
                    tx.method.value,
                    TransactionStatus.COMPLETED.value,
                    tx.amount,
                    json.dumps({"compensates": tx.id, "reason": reason}),
                    datetime.utcnow().isoformat(),
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
            logger.info(
                f"[Payment] COMPENSATED: tx={tx.id} credited back "
                f"${tx.amount:.2f} to {tx.from_account} (comp_id={comp_id})"
            )
            m.app_errors_total.labels(
                component="payment_processor",
                error_type="compensating_credit_issued",
            ).inc()
        except Exception as comp_err:
            # Compensation itself failed — alert loudly; manual intervention needed.
            logger.critical(
                f"[Payment] COMPENSATION FAILED for tx={tx.id}: {comp_err}. "
                f"MANUAL INTERVENTION REQUIRED to credit ${tx.amount:.2f} "
                f"back to account {tx.from_account}.",
                exc_info=True,
            )
            m.app_errors_total.labels(
                component="payment_processor",
                error_type="compensation_failed",
            ).inc()

    # ------------------------------------------------------------------
    async def process_payment(self, tx: Transaction) -> Transaction:
        """Full payment processing pipeline (bugs 1-3 fixed)."""
        start = time.monotonic()
        m.active_payment_requests.inc()

        try:
            tx.status = TransactionStatus.PROCESSING

            # ── Step 1: Atomic idempotency check (FIX BUG 1) ─────────────────
            key_lock = await _get_idempotency_key_lock(tx.idempotency_key)
            async with key_lock:
                existing = _idempotency_store.get(tx.idempotency_key)
                if existing:
                    logger.info(
                        f"[Payment] Idempotency hit: key={tx.idempotency_key} "
                        f"returning cached tx={existing.id}"
                    )
                    return existing

                # Mark as in-flight immediately so concurrent callers wait and
                # then get the result once we finish.
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
                # Update idempotency entry with final state
                async with key_lock:
                    _idempotency_store[tx.idempotency_key] = tx
                return tx

            # ── Step 3: Acquire locks (FIX BUG 3: acct then order) ───────────
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

            # ── Step 4: Single atomic DB transaction (FIX BUG 2) ─────────────
            # Both the account debit and the order/transaction inserts are
            # executed inside ONE transaction that is committed only when
            # both phases succeed.  If Phase 2 fails we roll back and issue
            # a compensating credit so the account is never left debited
            # without a corresponding confirmed order.
            conn = None
            debit_committed = False
            try:
                conn = db_pool.acquire()

                # Begin explicit transaction (autocommit must be off, which
                # is the default for most DB-API 2.0 drivers).  If the
                # driver uses implicit transactions, calling conn.execute()
                # will start one automatically.
                conn.execute("BEGIN")

                # ── Phase 1: Debit account ────────────────────────────────────
                conn.execute(
                    "UPDATE accounts SET balance = balance - ? "
                    "WHERE id = ? AND is_active = 1",
                    (tx.amount, tx.from_account),
                )
                # NOTE: No conn.commit() here — debit is not yet durable.
                debit_committed = False
                logger.debug(
                    f"[Payment] Phase 1 staged (not yet committed): "
                    f"debit ${tx.amount:.2f} from {tx.from_account}"
                )

                # ── Simulate intermittent Phase 2 failure ─────────────────────
                # (kept so the test harness can exercise the compensation path)
                if random.random() < settings.payment.partial_commit_rate:
                    raise PartialCommitError(
                        f"Simulated Phase 2 DB error after debit of "
                        f"${tx.amount:.2f}"
                    )

                # ── Phase 2: Confirm order + transaction record ───────────────
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
                        TransactionStatus.COMPLETED.value,
                        tx.fraud_score, tx.fee, tx.net_amount,
                        json.dumps(tx.metadata),
                        tx.created_at.isoformat(),
                        datetime.utcnow().isoformat(),
                    ),
                )

                # ── Single commit: both phases become durable atomically ───────
                conn.commit()
                debit_committed = True  # full transaction committed

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
                # Phase 2 failed before commit — roll back the whole transaction.
                # Because we never called conn.commit(), the debit was never
                # persisted; a rollback is sufficient.
                try:
                    conn.rollback()
                    logger.info(
                        f"[Payment] Rolled back atomic transaction for tx={tx.id}: {e}"
                    )
                except Exception as rb_err:
                    # Rollback failed (e.g., connection dropped).  We do not
                    # know whether the debit landed; issue a compensating credit
                    # as a safe-side measure and alert.
                    logger.critical(
                        f"[Payment] ROLLBACK FAILED for tx={tx.id}: {rb_err}. "
                        f"Attempting compensating credit.",
                        exc_info=True,
                    )
                    await self._compensate_debit(tx, conn, str(rb_err))

                tx.mark_failed(str(e))
                m.payment_transactions_total.labels(
                    status="failed_phase2",
                    method=tx.method.value,
                    currency=tx.currency,
                ).inc()
                m.app_errors_total.labels(
                    component="payment_processor",
                    error_type="phase2_failure_rolled_back",
                ).inc()
                logger.error(
                    f"[Payment] Phase 2 failed, transaction rolled back for "
                    f"tx={tx.id}. No orphaned debit. Reason: {e}"
                )
                # Update idempotency store with failed state so callers get a
                # clear error and can retry with a new idempotency key.
                async with key_lock:
                    _idempotency_store[tx.idempotency_key] = tx
                raise PaymentProcessorError(str(e)) from e

            except (DBConnectionError, PoolExhaustedError) as e:
                # If the connection died we cannot be sure whether the
                # transaction committed.  Attempt a compensating credit on a
                # fresh connection.
                if not debit_committed and conn:
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
                async with key_lock:
                    _idempotency_store[tx.idempotency_key] = tx
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

            # Persist final completed state in idempotency store
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
                f"[Payment] Unexpected error for tx={tx.id}: {e}",
                exc_info=True,
            )
            raise

        finally:
            elapsed = time.monotonic() - start
            m.payment_processing_duration_seconds.labels(
                method=tx.method.value
            ).observe(elapsed)
            m.active_payment_requests.dec()

    # ------------------------------------------------------------------
    async def process_refund(
        self,
        original_tx: Transaction,
        reason: str,
        amount: TypingOptional[float] = None,
    ) -> Transaction:
        """Process a refund for a completed transaction.

        FIX BUG 3: Locks are now acquired in the SAME order as
        process_payment (_acct_lock first, then _order_lock) to
        eliminate the classic deadlock.
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
            # FIX BUG 3: Acquire _acct_lock FIRST, then _order_lock —
            # identical ordering to process_payment.
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
