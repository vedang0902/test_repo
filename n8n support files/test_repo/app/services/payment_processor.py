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
Root cause:
  Payment processing is a two-phase write:
    Phase 1: UPDATE accounts SET balance = balance - amount
    Phase 2: INSERT INTO orders (transaction_id, status='confirmed')

  Between phase 1 and phase 2, the mock DB throws an intermittent error
  (simulates: network partition, PG write timeout, OOM kill of DB slave).
  The account is debited but no confirmed order exists — an orphaned debit.

Fix (APPLIED): Both phases are now executed within a single DB transaction.
  conn.commit() is called only after ALL writes succeed. On any failure
  between phase 1 and phase 2, conn.rollback() is called to reverse the
  debit atomically. A compensating credit is attempted as a safety net if
  rollback itself fails.

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

# ── Idempotency store (BUG: in-memory, not atomic) ───────────────────────────────────────
# Key → Transaction result
# Never expires (another memory leak contributing to transaction_cache_size)
_idempotency_store: Dict[str, Transaction] = {}

# ── Deadlock-prone locks ──────────────────────────────────────────────
# BUG: payment acquires _acct_lock then _order_lock
#      refund acquires _order_lock then _acct_lock  → classic deadlock
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
        BUG 2 (partial commit) is fixed: both DB phases now execute inside
        a single atomic transaction that is rolled back on any mid-flight
        failure.
        Contains BUG 1 (race condition) and BUG 3 (deadlock) — not fixed here.
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

            # ── Step 2: Fraud check ────────────────────────────────────────────
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
            #    from refund path → deadlock under concurrent load) ────────
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

            # ── Step 4: Atomic two-phase DB write (BUG 2 FIXED) ──────────────
            #
            # FIX: Both the account debit (phase 1) and the order/transaction
            # inserts (phase 2) are now executed within a single database
            # transaction. conn.commit() is deferred until every write has
            # succeeded. If any step raises — including the simulated
            # intermittent failure — conn.rollback() is called, which
            # atomically reverses the debit and prevents an orphaned debit
            # from ever being persisted.
            conn = None
            try:
                conn = db_pool.acquire()

                try:
                    # ── Phase 1: Debit account (NOT committed yet) ──────────
                    conn.execute(
                        "UPDATE accounts SET balance = balance - ? WHERE id = ? AND is_active = 1",
                        (tx.amount, tx.from_account),
                    )
                    logger.debug(
                        f"[Payment] Phase 1 staged (not committed): "
                        f"debit ${tx.amount:.2f} from {tx.from_account}"
                    )

                    # Simulate intermittent failure between phase 1 and phase 2.
                    # With the fix in place this will now trigger a rollback
                    # instead of leaving an orphaned debit.
                    if random.random() < settings.payment.partial_commit_rate:
                        # Raise an exception to force rollback of the debit
                        # staged above. No money leaves the account.
                        raise PartialCommitError(
                            f"Simulated intermittent DB error after debit of "
                            f"${tx.amount:.2f} — rolling back entire transaction"
                        )

                    # ── Phase 2: Confirm order ──────────────────────────────
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

                    # All writes succeeded — commit the entire transaction atomically.
                    conn.commit()
                    logger.debug(
                        f"[Payment] Atomic commit successful: "
                        f"debit + order + transaction record persisted for tx={tx.id}"
                    )

                except PartialCommitError as pce:
                    # Roll back the staged debit so no money is taken.
                    try:
                        conn.rollback()
                        logger.warning(
                            f"[Payment] Rolled back staged debit for tx={tx.id} "
                            f"after phase-2 failure — no orphaned debit created. "
                            f"reason={pce}"
                        )
                    except Exception as rb_err:
                        # Rollback itself failed (extremely rare — e.g. connection
                        # dropped mid-transaction). Emit a compensating credit as
                        # a last-resort safety net so the customer is made whole.
                        logger.critical(
                            f"[Payment] ROLLBACK FAILED for tx={tx.id}: {rb_err}. "
                            f"Attempting compensating credit of ${tx.amount:.2f} "
                            f"to account {tx.from_account}."
                        )
                        try:
                            comp_conn = db_pool.acquire()
                            try:
                                comp_conn.execute(
                                    "UPDATE accounts "
                                    "SET balance = balance + ? "
                                    "WHERE id = ? AND is_active = 1",
                                    (tx.amount, tx.from_account),
                                )
                                comp_conn.commit()
                                logger.info(
                                    f"[Payment] Compensating credit of "
                                    f"${tx.amount:.2f} applied to {tx.from_account} "
                                    f"for tx={tx.id}"
                                )
                            finally:
                                db_pool.release(comp_conn)
                        except Exception as comp_err:
                            logger.critical(
                                f"[Payment] Compensating credit ALSO FAILED for "
                                f"tx={tx.id}: {comp_err}. "
                                f"Manual remediation required."
                            )

                    # Whether rollback succeeded or the compensating credit was
                    # applied, mark the transaction as failed — do NOT return a
                    # partial-commit result to the caller.
                    tx.mark_failed(
                        f"Phase 2 write failed — transaction rolled back, "
                        f"no debit applied. ({pce})"
                    )
                    m.payment_transactions_total.labels(
                        status="failed",
                        method=tx.method.value,
                        currency=tx.currency,
                    ).inc()
                    m.app_errors_total.labels(
                        component="payment_processor",
                        error_type="phase2_rollback",
                    ).inc()
                    logger.error(
                        f"[Payment] Phase 2 failure rolled back cleanly for "
                        f"tx={tx.id}. No orphaned debit."
                    )
                    # Store failed tx in idempotency so retries don't loop.
                    _idempotency_store[tx.idempotency_key] = tx
                    m.idempotency_cache_size.set(len(_idempotency_store))
                    return tx

                except (DBConnectionError, PoolExhaustedError) as e:
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

            # Reaching here means the atomic commit succeeded.
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

            # Store in idempotency
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
