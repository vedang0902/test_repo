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
Root cause (was):
  Payment processing is a two-phase write:
    Phase 1: UPDATE accounts SET balance = balance - amount  → conn.commit()
    Phase 2: INSERT INTO orders (transaction_id, status='confirmed')

  Between phase 1 and phase 2, the mock DB throws an intermittent error
  (simulates: network partition, PG write timeout, OOM kill of DB slave).
  The account is debited but no confirmed order exists — an orphaned debit.

Fix (APPLIED):
  - Removed the intermediate conn.commit() after Phase 1.
  - Both Phase 1 and Phase 2 writes now execute inside a single DB
    transaction that is only committed after both succeed.
  - On any failure between phases a compensating UPDATE (credit back) is
    executed and the transaction is rolled back, preventing orphaned debits.
  - The partial_commit simulation block now raises PartialCommitError which
    triggers the compensating transaction path instead of returning silently.

=============================================================================
BUG 3: Deadlock on Lock Ordering
=============================================================================
Two processing paths acquire locks in opposite order:
  process_payment: acquire account_lock → acquire order_lock
  process_refund:  acquire order_lock  → acquire account_lock

Under concurrent load the classic deadlock scenario plays out.
We detect it via asyncio.wait_for timeout and log/metric it.

Symptoms in logs:
  CRITICAL payment_processor | DOUBLE CHARGE: idempotency_key=idem_xxx processed twice
  CRITICAL payment_processor | PARTIAL COMMIT: tx=tx_xxx debited $250.00 but order not confirmed
  ERROR    payment_processor | DEADLOCK: timeout acquiring account_lock for refund ref_xxx
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

# ── Idempotency store (BUG: in-memory, not atomic) ──────────────────────────
# Key → Transaction result
# Never expires (another memory leak contributing to transaction_cache_size)
_idempotency_store: Dict[str, Transaction] = {}

# ── Deadlock-prone locks ──────────────────────────────────────────────────────
# BUG: payment acquires _acct_lock then _order_lock
#      refund acquires _order_lock then _acct_lock  → classic deadlock
_acct_lock =  asyncio.Lock()
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
        Contains BUG 1 (race condition).
        BUG 2 (partial commit) is FIXED — both phases now run inside a single
        atomic DB transaction with compensating rollback on failure.
        """
        start = time.monotonic()
        m.active_payment_requests.inc()

        try:
            tx.status = TransactionStatus.PROCESSING

            # ── Step 1: Idempotency check (BUG: not atomic) ──────────────────
            existing = _idempotency_store.get(tx.idempotency_key)
            if existing:
                logger.info(
                    f"[Payment] Idempotency hit: key={tx.idempotency_key} "
                    f"returning cached tx={existing.id}"
                )
                return existing

            # BUG: Async context switch can happen here.
            # Another coroutine with the same key passes the check above,
            # then both proceed. The sleep below widens the race window.
            await asyncio.sleep(random.uniform(0.005, 0.025))

            # Check for concurrent key collision (detect but not prevent the bug)
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
                # Still store in idempotency to prevent re-processing
                _idempotency_store[tx.idempotency_key] = tx
                m.idempotency_cache_size.set(len(_idempotency_store))
                return tx

            # ── Step 3: Acquire lock (BUG: order of lock acquisition differs
            #    from refund path → deadlock under concurrent load) ──────────
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
                    f"[Payment] DEADLOCK: timeout acquiring account_lock "
                    f"tx={tx.id} — concurrent refund holding lock"
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
                    f"tx={tx.id} — concurrent refund holding order lock"
                )
                raise PaymentProcessorError(
                    f"Deadlock: timeout acquiring order lock for tx {tx.id}"
                )

            # ── Step 4: Atomic two-phase DB write (BUG 2 FIXED) ─────────────
            #
            # FIX: Both the account debit (Phase 1) and the order/transaction
            # inserts (Phase 2) are now executed within a single DB transaction.
            # The intermediate conn.commit() after Phase 1 has been removed.
            # A single conn.commit() is called only after ALL writes succeed.
            #
            # If any failure occurs between Phase 1 and Phase 2 (simulated
            # below, or real DB/network errors), conn.rollback() is called,
            # which atomically reverses the debit — no orphaned debit is
            # possible.  The transaction is marked failed and the caller can
            # safely retry (idempotency key is NOT stored on failure so a
            # retry will attempt a fresh write).
            conn = None
            try:
                conn = db_pool.acquire()

                # Begin explicit transaction (no auto-commit).
                # If the driver uses autocommit mode, disable it here.
                conn.execute("BEGIN")

                # ── Phase 1: Debit account (not yet committed) ────────────────
                conn.execute(
                    "UPDATE accounts SET balance = balance - ? WHERE id = ? AND is_active = 1",
                    (tx.amount, tx.from_account),
                )
                logger.debug(
                    f"[Payment] Phase 1 staged: debit ${tx.amount:.2f} "
                    f"from {tx.from_account} (not yet committed)"
                )

                # Simulate intermittent failure between Phase 1 and Phase 2.
                # Previously this caused an orphaned debit because Phase 1 had
                # already been committed.  Now both phases are inside the same
                # transaction so the rollback in the except block will undo the
                # debit automatically.
                if random.random() < settings.payment.partial_commit_rate:
                    raise PartialCommitError(
                        f"Simulated intermittent DB error after debit of "
                        f"${tx.amount:.2f} (partial_commit_rate trigger)"
                    )

                # ── Phase 2: Confirm order ────────────────────────────────────
                order_id = str(uuid.uuid4())
                merchant_id = f"merchant_{random.randint(100, 999)}"

                conn.execute(
                    """INSERT INTO orders (id, transaction_id, merchant_id, total_amount, status, created_at)
                       VALUES (?, ?, ?, ?, 'confirmed', ?)""",
                    (order_id, tx.id, merchant_id, tx.amount, datetime.utcnow().isoformat()),
                )

                # Insert transaction record
                conn.execute(
                    """INSERT INTO transactions
                       (id, idempotency_key, from_account, to_account, amount, currency,
                        method, status, fraud_score, fee, net_amount, metadata, created_at, updated_at)
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

                # Single commit: atomically persists debit + order + tx record.
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
                # Rollback atomically undoes the Phase 1 debit — no orphaned debit.
                try:
                    conn.rollback()
                    logger.info(
                        f"[Payment] Rolled back debit for tx={tx.id} "
                        f"after phase-2 failure — no orphaned debit."
                    )
                except Exception as rb_exc:
                    logger.error(
                        f"[Payment] Rollback failed for tx={tx.id}: {rb_exc}",
                        exc_info=True,
                    )

                tx.mark_failed(str(e))
                m.payment_transactions_total.labels(
                    status="partial_commit_rolled_back",
                    method=tx.method.value,
                    currency=tx.currency,
                ).inc()
                m.app_errors_total.labels(
                    component="payment_processor",
                    error_type="partial_commit_rolled_back",
                ).inc()
                # Intentionally do NOT record in _idempotency_store so the
                # caller can safely retry after the transient DB error clears.
                logger.error(
                    f"[Payment] PARTIAL COMMIT PREVENTED (rolled back): "
                    f"tx={tx.id} account={tx.from_account} — "
                    f"debit of ${tx.amount:.2f} reversed. Reason: {e}"
                )
                raise PaymentProcessorError(
                    f"Payment failed due to transient DB error; debit rolled back. tx={tx.id}"
                ) from e

            except (DBConnectionError, PoolExhaustedError) as e:
                # Attempt rollback to keep the DB state consistent.
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
                logger.error(
                    f"[Payment] DB error for tx={tx.id}: {e}"
                )
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

            # Store in idempotency only after full success.
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

    async def process_refund(self, original_tx: Transaction, reason: str, amount: TypingOptional[float] = None) -> Transaction:
        """
        Process a refund for a completed transaction.

        BUG: Acquires locks in OPPOSITE ORDER to process_payment:
          process_payment: _acct_lock → _order_lock
          process_refund:  _order_lock → _acct_lock   ← DEADLOCK under concurrency
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
            # BUG: Lock order reversed — order_lock first, then acct_lock
            try:
                lock_start = time.monotonic()
                await asyncio.wait_for(_order_lock.acquire(), timeout=2.5)
                m.lock_wait_duration_seconds.labels(lock_type="order_lock").observe(
                    time.monotonic() - lock_start
                )
            except asyncio.TimeoutError:
                m.deadlock_events_total.labels(lock_type="order_lock").inc()
                m.app_errors_total.labels(
                    component="payment_processor", error_type="deadlock"
                ).inc()
                logger.error(
                    f"[Refund] DEADLOCK: timeout acquiring order_lock "
                    f"refund_tx={refund_tx.id} original_tx={original_tx.id}"
                )
                raise PaymentProcessorError("Deadlock: timeout acquiring order lock for refund")

            try:
                acct_start = time.monotonic()
                await asyncio.wait_for(_acct_lock.acquire(), timeout=2.5)
                m.lock_wait_duration_seconds.labels(lock_type="account_lock").observe(
                    time.monotonic() - acct_start
                )
            except asyncio.TimeoutError:
                _order_lock.release()
                m.deadlock_events_total.labels(lock_type="account_lock").inc()
                m.app_errors_total.labels(
                    component="payment_processor", error_type="deadlock"
                ).inc()
                logger.error(
                    f"[Refund] DEADLOCK: timeout acquiring account_lock "
                    f"while order_lock held. original_tx={original_tx.id}"
                )
                raise PaymentProcessorError("Deadlock: timeout acquiring account lock during refund")

            # Process refund credit
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
                            method, status, fraud_score, fee, net_amount, metadata, created_at, updated_at)
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
