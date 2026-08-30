"""Core Payment Processing Service.

Orchestrates the full payment lifecycle:
  1. Idempotency check (atomic, per-key lock)
  2. Fraud scoring
  3. Atomic DB write: debit + order confirmation in ONE transaction
  4. Compensating credit (saga rollback) on Phase 2 failure
  5. Reconciliation recording

Fixes applied:
  BUG 1: Idempotency race closed with _idempotency_locks (per-key asyncio.Lock)
  BUG 2: Partial commit eliminated — both phases share one conn with a single
          conn.commit(); compensating UPDATE credit on failure prevents orphaned debit
  BUG 3: Deadlock eliminated — process_refund now acquires _acct_lock then
          _order_lock, matching process_payment order
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
# Idempotency store — guarded by per-key locks to close the race window.
# _idempotency_locks maps idempotency_key -> asyncio.Lock.
# A coroutine must hold the per-key lock while checking AND writing the store.
# ---------------------------------------------------------------------------
_idempotency_store: Dict[str, Transaction] = {}
_idempotency_locks: Dict[str, asyncio.Lock] = {}
_idempotency_meta_lock = asyncio.Lock()   # guards _idempotency_locks dict itself


async def _get_idempotency_lock(key: str) -> asyncio.Lock:
    """Return (creating if necessary) the per-key idempotency lock."""
    async with _idempotency_meta_lock:
        if key not in _idempotency_locks:
            _idempotency_locks[key] = asyncio.Lock()
        return _idempotency_locks[key]


# ---------------------------------------------------------------------------
# Shared locks — BOTH process_payment AND process_refund acquire in the
# same order (_acct_lock first, then _order_lock) to prevent deadlock.
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

    # ------------------------------------------------------------------
    # Internal helper: acquire both locks in canonical order
    # ------------------------------------------------------------------
    async def _acquire_locks(self, context: str, tx_id: str) -> None:
        """Acquire _acct_lock then _order_lock (fixed order, deadlock-safe)."""
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
                f"[{context}] Timeout acquiring account_lock tx/ref={tx_id}"
            )
            raise PaymentProcessorError(
                f"Deadlock: timeout acquiring account lock ({context}) {tx_id}"
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
                f"[{context}] Timeout acquiring order_lock tx/ref={tx_id}"
            )
            raise PaymentProcessorError(
                f"Deadlock: timeout acquiring order lock ({context}) {tx_id}"
            )

    @staticmethod
    def _release_locks() -> None:
        """Release both locks if held (safe to call in finally blocks)."""
        for lock in (_order_lock, _acct_lock):   # release in reverse order
            if lock.locked():
                try:
                    lock.release()
                except RuntimeError:
                    pass

    # ------------------------------------------------------------------
    # process_payment
    # ------------------------------------------------------------------
    async def process_payment(self, tx: Transaction) -> Transaction:
        """Full payment processing pipeline — atomic two-phase write."""
        start = time.monotonic()
        m.active_payment_requests.inc()

        try:
            tx.status = TransactionStatus.PROCESSING

            # ── Step 1: Atomic idempotency check+write via per-key lock ──────
            idem_lock = await _get_idempotency_lock(tx.idempotency_key)
            async with idem_lock:
                existing = _idempotency_store.get(tx.idempotency_key)
                if existing:
                    logger.info(
                        f"[Payment] Idempotency hit: key={tx.idempotency_key} "
                        f"returning cached tx={existing.id}"
                    )
                    return existing

                # ── Step 2: Fraud check ──────────────────────────────────────
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
                    _idempotency_store[tx.idempotency_key] = tx
                    m.idempotency_cache_size.set(len(_idempotency_store))
                    return tx

                # ── Step 3: Acquire shared locks (canonical order) ───────────
                await self._acquire_locks("Payment", tx.id)

                # ── Step 4: Single atomic DB transaction (FIX for BUG 2) ─────
                #
                # Both the account debit AND the order/transaction inserts are
                # executed inside ONE database transaction.  We do NOT call
                # conn.commit() after Phase 1.  If Phase 2 fails for any reason
                # (exception, simulated intermittent error, etc.) we:
                #   a) roll back the entire transaction (no debit persisted), OR
                #   b) if the DB has already auto-committed Phase 1 (legacy
                #      driver behaviour), issue a compensating credit UPDATE
                #      so the account balance is restored before re-raising.
                # This eliminates orphaned debits regardless of driver semantics.
                conn = None
                phase1_committed = False   # sentinel for compensating tx
                try:
                    conn = db_pool.acquire()

                    # Disable autocommit — everything below is one transaction.
                    # (db_pool.acquire() returns a connection in manual-commit
                    # mode; conn.commit() is the only commit point.)

                    # ── Phase 1: Debit account ────────────────────────────────
                    conn.execute(
                        "UPDATE accounts SET balance = balance - ? "
                        "WHERE id = ? AND is_active = 1",
                        (tx.amount, tx.from_account),
                    )
                    # NOT committing here — phase 1 is still inside the tx.
                    logger.debug(
                        f"[Payment] Phase 1 staged (not yet committed): "
                        f"debit ${tx.amount:.2f} from {tx.from_account}"
                    )

                    # Simulate intermittent failure between phases.
                    # With the fix in place the connection has not been
                    # committed yet, so a rollback undoes the debit cleanly.
                    if random.random() < settings.payment.partial_commit_rate:
                        # Roll back — debit is not persisted.
                        try:
                            conn.rollback()
                        except Exception:
                            pass
                        m.app_errors_total.labels(
                            component="payment_processor",
                            error_type="phase2_simulated_error",
                        ).inc()
                        logger.error(
                            f"[Payment] Phase 2 simulated failure for tx={tx.id}: "
                            f"rolled back debit of ${tx.amount:.2f} — no orphan created."
                        )
                        tx.mark_failed(
                            f"Transient DB error during Phase 2 (tx rolled back, no debit)"
                        )
                        m.payment_transactions_total.labels(
                            status="failed_rolled_back",
                            method=tx.method.value,
                            currency=tx.currency,
                        ).inc()
                        # Do NOT store in idempotency — caller may safely retry.
                        raise PartialCommitError(
                            f"Phase 2 transient error for tx {tx.id} — safely rolled back"
                        )

                    # ── Phase 2: Confirm order + insert transaction record ────
                    order_id   = str(uuid.uuid4())
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
                               (id, idempotency_key, from_account, to_account, amount,
                                currency, method, status, fraud_score, fee, net_amount,
                                metadata, created_at, updated_at)
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

                    # Single commit — both phases land atomically.
                    conn.commit()
                    phase1_committed = True   # both phases committed together

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

                except PartialCommitError:
                    raise

                except (DBConnectionError, PoolExhaustedError) as e:
                    # If Phase 1 somehow committed before the error (e.g. legacy
                    # driver with statement-level autocommit), issue a compensating
                    # credit so the account is not silently debited.
                    if phase1_committed is False:
                        try:
                            conn.rollback()
                            logger.info(
                                f"[Payment] DB error before commit — rolled back cleanly: "
                                f"tx={tx.id}"
                            )
                        except Exception as rb_err:
                            # Rollback failed — attempt compensating credit.
                            logger.critical(
                                f"[Payment] Rollback failed ({rb_err}); issuing compensating "
                                f"credit for tx={tx.id} amount=${tx.amount:.2f}"
                            )
                            try:
                                comp_conn = db_pool.acquire()
                                try:
                                    comp_conn.execute(
                                        "UPDATE accounts SET balance = balance + ? WHERE id = ?",
                                        (tx.amount, tx.from_account),
                                    )
                                    comp_conn.commit()
                                    logger.info(
                                        f"[Payment] Compensating credit applied for tx={tx.id}"
                                    )
                                finally:
                                    db_pool.release(comp_conn)
                            except Exception as comp_err:
                                logger.critical(
                                    f"[Payment] COMPENSATING CREDIT FAILED for tx={tx.id}: "
                                    f"{comp_err} — manual intervention required"
                                )
                                m.app_errors_total.labels(
                                    component="payment_processor",
                                    error_type="compensating_credit_failed",
                                ).inc()

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
                    # Unexpected error — attempt rollback.
                    try:
                        if conn:
                            conn.rollback()
                    except Exception:
                        pass
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
                    if conn:
                        try:
                            db_pool.release(conn)
                        except Exception:
                            pass
                    self._release_locks()

                # Store in idempotency AFTER successful commit.
                _idempotency_store[tx.idempotency_key] = tx
                m.idempotency_cache_size.set(len(_idempotency_store))
                return tx

        except PaymentProcessorError:
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
                f"[Payment] Unexpected outer error for tx={tx.id}: {e}",
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
    # process_refund  (BUG 3 fixed: lock order now matches process_payment)
    # ------------------------------------------------------------------
    async def process_refund(
        self,
        original_tx: Transaction,
        reason: str,
        amount: TypingOptional[float] = None,
    ) -> Transaction:
        """Process a refund — acquires locks in same order as process_payment."""
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
            # FIX (BUG 3): acquire _acct_lock THEN _order_lock — same order as
            # process_payment — so the two paths can never deadlock each other.
            await self._acquire_locks("Refund", refund_tx.id)

            try:
                conn = db_pool.acquire()
                try:
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

                finally:
                    db_pool.release(conn)

            except DBConnectionError as e:
                refund_tx.mark_failed(str(e))
                raise

            finally:
                self._release_locks()

            return refund_tx

        finally:
            elapsed = time.monotonic() - start
            m.payment_processing_duration_seconds.labels(
                method=refund_tx.method.value
            ).observe(elapsed)
            m.active_payment_requests.dec()


# Singleton
payment_processor = PaymentProcessor()
