"""
Core Payment Processing Service.

Orchestrates the full payment lifecycle:
  1. Idempotency check  (FIXED: atomic via per-key lock)
  2. Fraud scoring
  3. Database debit (phase 1)
  4. Order confirmation (phase 2)
  5. Reconciliation recording

=============================================================================
BUG 1: Race Condition on Idempotency Key (Double Charge)  ← FIXED
=============================================================================
Fix applied: A per-key asyncio.Lock is acquired before the idempotency
  read-check-write, making the entire sequence atomic within a single
  asyncio event loop. A second coroutine arriving with the same key will
  block on the lock, then find the completed result in the store and
  return it without re-processing.

=============================================================================
BUG 2: Partial Transaction Commit (Orphaned Debit)
=============================================================================
Root cause (unchanged — out of scope for this incident):
  Payment processing is a two-phase write:
    Phase 1: UPDATE accounts SET balance = balance - amount
    Phase 2: INSERT INTO orders (transaction_id, status='confirmed')

  Between phase 1 and phase 2, the mock DB throws an intermittent error.
  The account is debited but no confirmed order exists — an orphaned debit.

Fix (NOT applied): Wrap both phases in a DB transaction with ACID guarantees.
  Use saga pattern with compensating transaction (credit back) on failure.

=============================================================================
BUG 3: Deadlock on Lock Ordering  ← FIXED
=============================================================================
Fix applied: process_refund now acquires _acct_lock then _order_lock,
  matching the order used by process_payment, eliminating the deadlock.
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
# Never expires (another memory leak contributing to transaction_cache_size)
_idempotency_store: Dict[str, Transaction] = {}

# ── Per-key idempotency locks (FIX for BUG 1) ───────────────────────────────
# Holding the per-key lock while performing the read-check-write makes the
# entire sequence atomic within the asyncio event loop.  A second coroutine
# with the same key blocks here, then finds the result already stored.
_idempotency_locks: Dict[str, asyncio.Lock] = {}
_idempotency_locks_meta_lock = asyncio.Lock()  # protects _idempotency_locks dict


async def _get_idempotency_lock(key: str) -> asyncio.Lock:
    """Return (creating if necessary) the per-key idempotency lock."""
    async with _idempotency_locks_meta_lock:
        if key not in _idempotency_locks:
            _idempotency_locks[key] = asyncio.Lock()
        return _idempotency_locks[key]


# ── Global DB-operation locks (consistent order: _acct_lock → _order_lock) ──
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
        BUG 1 (idempotency race) is FIXED via per-key locking.
        BUG 2 (partial commit) remains and is documented above.
        BUG 3 (deadlock) is FIXED by consistent lock ordering.
        """
        start = time.monotonic()
        m.active_payment_requests.inc()

        try:
            tx.status = TransactionStatus.PROCESSING

            # ── Step 1: Idempotency check — now atomic (FIX for BUG 1) ───────
            # Acquire the per-key lock before reading so that no two coroutines
            # can simultaneously pass the "key not found" gate.
            idempotency_lock = await _get_idempotency_lock(tx.idempotency_key)
            async with idempotency_lock:
                existing = _idempotency_store.get(tx.idempotency_key)
                if existing:
                    logger.info(
                        f"[Payment] Idempotency hit: key={tx.idempotency_key} "
                        f"returning cached tx={existing.id}"
                    )
                    return existing

                # The sleep that previously widened the race window is kept here
                # intentionally to prove the fix works under the same timing:
                # the second coroutine is now blocked on idempotency_lock above.
                await asyncio.sleep(random.uniform(0.005, 0.025))

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
                    _idempotency_store[tx.idempotency_key] = tx
                    m.idempotency_cache_size.set(len(_idempotency_store))
                    return tx

                # ── Step 3: Acquire DB locks (consistent order: acct → order) ─
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

                # ── Step 4: Two-phase DB write (BUG 2 partial commit unchanged) ─
                conn = None
                try:
                    conn = db_pool.acquire()

                    # ── Phase 1: Debit account ────────────────────────────────
                    conn.execute(
                        "UPDATE accounts SET balance = balance - ? WHERE id = ? AND is_active = 1",
                        (tx.amount, tx.from_account),
                    )
                    conn.commit()
                    logger.debug(f"[Payment] Phase 1 complete: debited ${tx.amount:.2f} from {tx.from_account}")

                    # BUG 2: Intermittent failure between phase 1 and phase 2.
                    if random.random() < settings.payment.partial_commit_rate:
                        tx.mark_partial_commit(
                            f"Phase 2 write failed after debit of ${tx.amount:.2f} "
                            f"(intermittent DB error)"
                        )
                        m.partial_commits_total.inc()
                        m.orphaned_debits_total.inc()
                        m.payment_transactions_total.labels(
                            status="partial_commit",
                            method=tx.method.value,
                            currency=tx.currency,
                        ).inc()
                        m.app_errors_total.labels(
                            component="payment_processor",
                            error_type="partial_commit",
                        ).inc()
                        m.app_error_rate.set(1)
                        logger.critical(
                            f"[Payment] PARTIAL COMMIT: tx={tx.id} account={tx.from_account} "
                            f"debited ${tx.amount:.2f} but order confirmation FAILED. "
                            f"ORPHANED DEBIT created."
                        )
                        _idempotency_store[tx.idempotency_key] = tx
                        m.idempotency_cache_size.set(len(_idempotency_store))
                        return tx

                    # ── Phase 2: Confirm order ────────────────────────────────
                    order_id = str(uuid.uuid4())
                    merchant_id = f"merchant_{random.randint(100, 999)}"

                    conn.execute(
                        """INSERT INTO orders (id, transaction_id, merchant_id, total_amount, status, created_at)
                           VALUES (?, ?, ?, ?, 'confirmed', ?)""",
                        (order_id, tx.id, merchant_id, tx.amount, datetime.utcnow().isoformat()),
                    )

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

                # Write to idempotency store while still holding idempotency_lock
                # so the result is visible to any waiter before the lock drops.
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

        FIX (BUG 3): Lock order is now _acct_lock → _order_lock, matching
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
            # ── Acquire locks in the SAME order as process_payment (FIX BUG 3) ─
            # Order: _acct_lock first, then _order_lock.
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
                raise PaymentProcessorError("Timeout acquiring account lock for refund")

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
                    f"[Refund] timeout acquiring order_lock while account_lock held. "
                    f"original_tx={original_tx.id}"
                )
                raise PaymentProcessorError("Timeout acquiring order lock during refund")

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
