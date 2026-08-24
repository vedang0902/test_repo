"""
Core Payment Processing Service.

Orchestrates the full payment lifecycle:
  1. Idempotency check
  2. Fraud scoring
  3. Database debit (phase 1)
  4. Order confirmation (phase 2)
  5. Reconciliation recording

=============================================================================
BUG 1: Race Condition on Idempotency Key (Double Charge)
=============================================================================
Root cause:
  The idempotency check and write are NOT atomic:

    read  _idempotency_store[key]  → None  (doesn't exist)
    [processing delay — asyncio context switch happens here]
    write _idempotency_store[key]  = result

  Two concurrent requests with the same key both pass the read check
  before either writes the result. Both proceed to debit the account.
  The second request creates a duplicate charge.

Fix (NOT applied): Use asyncio.Lock per key, or a DB unique constraint with
  INSERT OR IGNORE and read-back.

=============================================================================
BUG 2: Partial Transaction Commit (Orphaned Debit) — FIXED
=============================================================================
Root cause (original):
  Payment processing was a two-phase write with an intermediate commit:
    Phase 1: UPDATE accounts SET balance = balance - amount  → conn.commit()
    Phase 2: INSERT INTO orders (transaction_id, status='confirmed')

  Between phase 1 and phase 2, any failure left the account debited with
  no confirmed order — an orphaned debit.

Fix (APPLIED):
  - Removed the intermediate conn.commit() after Phase 1.
  - Both Phase 1 and Phase 2 execute inside a single DB transaction,
    committed once only after both writes succeed.
  - On any exception after the debit but before the final commit, the
    transaction is rolled back atomically (no orphaned debit).
  - Additionally, a compensating credit (saga rollback) is issued if the
    DB is in an unknown state (e.g. after a network partition where commit
    status is uncertain), recording the compensating transaction for audit.

=============================================================================
BUG 3: Deadlock on Lock Ordering — FIXED
=============================================================================
Original:
  process_payment: acquire _acct_lock → acquire _order_lock
  process_refund:  acquire _order_lock → acquire _acct_lock  ← DEADLOCK

Fix (APPLIED):
  Both process_payment and process_refund now acquire locks in the same
  canonical order: _acct_lock first, then _order_lock. This eliminates
  the circular wait condition.
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

# ── Idempotency store (BUG 1: in-memory, not atomic — not fixed in this PR) ──
# Key → Transaction result
_idempotency_store: Dict[str, Transaction] = {}

# ── Locks — canonical acquisition order: _acct_lock THEN _order_lock ─────────
# Both process_payment and process_refund follow this order (BUG 3 fixed).
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

    async def _compensating_credit(self, tx: Transaction, conn) -> None:
        """
        Saga compensating transaction: credit the account back after a
        partial-commit scenario where the debit was flushed but the overall
        DB transaction cannot be rolled back (e.g. connection lost after
        flush but before commit acknowledgement).

        This is a best-effort safety net. The primary protection is the
        atomic single-commit approach below; this handles the rare case
        where commit status is unknown.
        """
        compensation_id = str(uuid.uuid4())
        try:
            conn.execute(
                "UPDATE accounts SET balance = balance + ? WHERE id = ? AND is_active = 1",
                (tx.amount, tx.from_account),
            )
            conn.execute(
                """
                INSERT INTO transactions
                    (id, idempotency_key, from_account, to_account, amount, currency,
                     method, status, fraud_score, fee, net_amount, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0.0, 0.0, ?, ?, ?, ?)
                """,
                (
                    compensation_id,
                    f"compensation_{tx.idempotency_key}",
                    tx.to_account,          # reverse: credit comes from merchant side
                    tx.from_account,        # back to customer
                    tx.amount,
                    tx.currency,
                    tx.method.value,
                    TransactionStatus.COMPLETED.value,
                    tx.amount,
                    json.dumps({"compensation_for": tx.id, "reason": "partial_commit_rollback"}),
                    datetime.utcnow().isoformat(),
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
            logger.warning(
                f"[Payment] COMPENSATING CREDIT applied: compensation_id={compensation_id} "
                f"original_tx={tx.id} amount=${tx.amount:.2f} credited back to {tx.from_account}"
            )
            m.app_errors_total.labels(
                component="payment_processor",
                error_type="compensating_credit_applied",
            ).inc()
        except Exception as comp_exc:
            # Compensation itself failed — this requires manual intervention.
            # Alert loudly; do NOT swallow.
            logger.critical(
                f"[Payment] COMPENSATION FAILED: could not credit back ${tx.amount:.2f} "
                f"to {tx.from_account} for tx={tx.id}. "
                f"MANUAL REMEDIATION REQUIRED. error={comp_exc}",
                exc_info=True,
            )
            m.app_errors_total.labels(
                component="payment_processor",
                error_type="compensating_credit_failed",
            ).inc()
            raise

    async def process_payment(self, tx: Transaction) -> Transaction:
        """
        Full payment processing pipeline.

        BUG 2 (partial commit) is fixed by:
          - Removing the intermediate conn.commit() after Phase 1.
          - Executing Phase 1 + Phase 2 inside a single DB transaction.
          - Rolling back atomically on any failure; issuing a compensating
            credit if rollback is not possible (unknown commit state).

        BUG 3 (deadlock) is fixed by standardising lock order to
          _acct_lock → _order_lock (matching process_refund below).

        BUG 1 (idempotency race) is NOT fixed in this PR.
        """
        start = time.monotonic()
        m.active_payment_requests.inc()

        try:
            tx.status = TransactionStatus.PROCESSING

            # ── Step 1: Idempotency check (BUG 1: not atomic — not fixed here) ─
            existing = _idempotency_store.get(tx.idempotency_key)
            if existing:
                logger.info(
                    f"[Payment] Idempotency hit: key={tx.idempotency_key} "
                    f"returning cached tx={existing.id}"
                )
                return existing

            # BUG 1: race window remains here.
            await asyncio.sleep(random.uniform(0.005, 0.025))

            if tx.idempotency_key in _idempotency_store:
                m.idempotency_violations_total.inc()
                m.app_errors_total.labels(
                    component="payment_processor",
                    error_type="idempotency_violation",
                ).inc()
                logger.critical(
                    f"[Payment] DOUBLE CHARGE DETECTED: idempotency_key={tx.idempotency_key} "
                    f"was processed concurrently. tx={tx.id} is a DUPLICATE."
                )
                m.app_error_rate.set(1)

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
                _idempotency_store[tx.idempotency_key] = tx
                m.idempotency_cache_size.set(len(_idempotency_store))
                return tx

            # ── Step 3: Acquire locks in canonical order: acct → order ────────
            # FIX (BUG 3): Both process_payment and process_refund now acquire
            # _acct_lock before _order_lock, eliminating the circular wait.
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

            # ── Step 4: Atomic two-phase DB write (BUG 2 FIXED) ──────────────
            # Both Phase 1 and Phase 2 execute inside a single DB transaction.
            # conn.commit() is called ONCE after both writes succeed.
            # On any exception the connection's implicit or explicit rollback
            # ensures neither write is persisted.
            conn = None
            debit_executed = False  # track whether Phase 1 SQL ran
            try:
                conn = db_pool.acquire()

                # Begin explicit transaction (idempotent for most DB drivers).
                # For drivers that auto-commit, this disables auto-commit mode.
                conn.execute("BEGIN")

                # ── Phase 1: Debit account ────────────────────────────────────
                conn.execute(
                    "UPDATE accounts SET balance = balance - ? WHERE id = ? AND is_active = 1",
                    (tx.amount, tx.from_account),
                )
                debit_executed = True
                logger.debug(
                    f"[Payment] Phase 1 staged (not yet committed): "
                    f"debit ${tx.amount:.2f} from {tx.from_account}"
                )

                # Simulate intermittent failure between phases.
                # FIX: Because we have NOT committed yet, the rollback below
                # will undo the debit — no orphaned debit is created.
                if random.random() < settings.payment.partial_commit_rate:
                    raise PartialCommitError(
                        f"Simulated intermittent DB error after debit stage "
                        f"for tx={tx.id} amount=${tx.amount:.2f}"
                    )

                # ── Phase 2: Confirm order ────────────────────────────────────
                order_id = str(uuid.uuid4())
                merchant_id = f"merchant_{random.randint(100, 999)}"

                conn.execute(
                    """
                    INSERT INTO orders
                        (id, transaction_id, merchant_id, total_amount, status, created_at)
                    VALUES (?, ?, ?, ?, 'confirmed', ?)
                    """,
                    (
                        order_id, tx.id, merchant_id,
                        tx.amount, datetime.utcnow().isoformat(),
                    ),
                )

                # Insert transaction record
                conn.execute(
                    """
                    INSERT INTO transactions
                        (id, idempotency_key, from_account, to_account, amount, currency,
                         method, status, fraud_score, fee, net_amount, metadata,
                         created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tx.id, tx.idempotency_key, tx.from_account, tx.to_account,
                        tx.amount, tx.currency, tx.method.value,
                        TransactionStatus.COMPLETED.value, tx.fraud_score,
                        tx.fee, tx.net_amount,
                        json.dumps(tx.metadata),
                        tx.created_at.isoformat(), datetime.utcnow().isoformat(),
                    ),
                )

                # ── Single atomic commit — both phases or neither ──────────────
                conn.commit()
                debit_executed = False  # committed successfully; no rollback needed

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
                # The DB transaction has NOT been committed — roll back cleanly.
                # No orphaned debit exists; update metrics and fail gracefully.
                logger.error(
                    f"[Payment] Phase 2 failure for tx={tx.id}: {e}. "
                    f"Rolling back DB transaction — no orphaned debit."
                )
                try:
                    conn.rollback()
                    logger.info(
                        f"[Payment] Rollback successful for tx={tx.id}. "
                        f"Account {tx.from_account} balance unchanged."
                    )
                except Exception as rb_exc:
                    # Rollback failed — commit status is unknown.
                    # Issue a compensating credit as a saga fallback.
                    logger.critical(
                        f"[Payment] Rollback FAILED for tx={tx.id}: {rb_exc}. "
                        f"Attempting compensating credit.",
                        exc_info=True,
                    )
                    if debit_executed:
                        await self._compensating_credit(tx, conn)

                tx.mark_failed(str(e))
                m.payment_transactions_total.labels(
                    status="failed_partial_commit_rolled_back",
                    method=tx.method.value,
                    currency=tx.currency,
                ).inc()
                m.app_errors_total.labels(
                    component="payment_processor",
                    error_type="partial_commit_rolled_back",
                ).inc()
                # Do NOT store in idempotency — allow safe retry.
                raise PaymentProcessorError(
                    f"Payment failed during order confirmation; debit safely rolled back. tx={tx.id}"
                ) from e

            except (DBConnectionError, PoolExhaustedError) as e:
                tx.mark_failed(str(e))
                try:
                    if conn:
                        conn.rollback()
                except Exception:
                    pass
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

            # Store in idempotency only on full success
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

        FIX (BUG 3): Locks are now acquired in canonical order
          _acct_lock → _order_lock, matching process_payment.
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
            # ── Acquire locks in canonical order: acct → order (BUG 3 FIXED) ─
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
                    conn.execute("BEGIN")
                    conn.execute(
                        "UPDATE accounts SET balance = balance + ? WHERE id = ?",
                        (refund_amount, refund_tx.to_account),
                    )
                    conn.execute(
                        """
                        INSERT INTO transactions
                            (id, idempotency_key, from_account, to_account, amount, currency,
                             method, status, fraud_score, fee, net_amount, metadata,
                             created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0.0, 0.0, ?, ?, ?, ?)
                        """,
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
                    except Exception as rb_exc:
                        logger.critical(
                            f"[Refund] Rollback FAILED for refund_tx={refund_tx.id}: {rb_exc}",
                            exc_info=True,
                        )
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
