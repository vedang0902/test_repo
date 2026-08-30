"""Core Payment Processing Service.

Orchestrates the full payment lifecycle:
  1. Idempotency check (atomic, per-key lock)
  2. Fraud scoring
  3. Atomic DB write: debit + order confirmation in one transaction
  4. Compensating credit on failure (saga pattern)
  5. Reconciliation recording

Fixes applied
─────────────
BUG 1 (Race / double-charge):  Per-key asyncio.Lock makes the
  idempotency check-and-set atomic within the process.

BUG 2 (Partial commit / orphaned debit):  Both the account UPDATE
  and the order INSERT are now executed inside a single DB
  transaction (one conn.commit()).  If Phase 2 fails we roll back
  and issue a compensating credit so the account balance is never
  left in an inconsistent state.

BUG 3 (Deadlock):  process_refund now acquires locks in the same
  order as process_payment (_acct_lock → _order_lock).
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
# FIX (BUG 1): Access is serialised through a per-key asyncio.Lock so that
# the check-and-set is atomic within the event loop.  The global dict and
# the per-key lock dict are both protected by a single "registry" lock that
# is held only long enough to allocate/fetch the per-key lock — avoiding a
# coarse-grained bottleneck while still preventing races.
_idempotency_store: Dict[str, Transaction] = {}
_idempotency_key_locks: Dict[str, asyncio.Lock] = {}
_idempotency_registry_lock: asyncio.Lock = asyncio.Lock()


async def _get_idempotency_lock(key: str) -> asyncio.Lock:
    """Return (and lazily create) the per-key idempotency lock."""
    async with _idempotency_registry_lock:
        if key not in _idempotency_key_locks:
            _idempotency_key_locks[key] = asyncio.Lock()
        return _idempotency_key_locks[key]


# ── Ordered locks (FIX BUG 3: both paths use _acct_lock → _order_lock) ───────
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

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    async def _compensate_debit(
        self, tx: Transaction, amount: float, reason: str
    ) -> None:
        """Issue a compensating credit to undo an already-committed debit.

        Called only when Phase 1 committed but Phase 2 failed AND we were
        unable to roll back (e.g. the connection died).  Under normal
        circumstances the single-transaction fix means this path is never
        reached, but it is kept as a safety net.
        """
        try:
            conn = db_pool.acquire()
            try:
                conn.execute(
                    "UPDATE accounts SET balance = balance + ? WHERE id = ?",
                    (amount, tx.from_account),
                )
                conn.commit()
                logger.warning(
                    f"[Payment] COMPENSATING CREDIT applied: tx={tx.id} "
                    f"credited ${amount:.2f} back to {tx.from_account}. reason={reason}"
                )
                m.app_errors_total.labels(
                    component="payment_processor",
                    error_type="compensating_credit_applied",
                ).inc()
            finally:
                db_pool.release(conn)
        except Exception as comp_err:
            # Compensation itself failed — surface loudly for manual remediation.
            logger.critical(
                f"[Payment] COMPENSATION FAILED: tx={tx.id} "
                f"could not credit ${amount:.2f} back to {tx.from_account}. "
                f"MANUAL REMEDIATION REQUIRED. error={comp_err}"
            )
            m.app_errors_total.labels(
                component="payment_processor",
                error_type="compensation_failed",
            ).inc()

    # ------------------------------------------------------------------ #
    #  process_payment                                                     #
    # ------------------------------------------------------------------ #

    async def process_payment(self, tx: Transaction) -> Transaction:
        """Full payment processing pipeline (all three bugs fixed)."""
        start = time.monotonic()
        m.active_payment_requests.inc()

        try:
            tx.status = TransactionStatus.PROCESSING

            # ── Step 1: Atomic idempotency check-and-set (FIX BUG 1) ─────────
            idem_lock = await _get_idempotency_lock(tx.idempotency_key)
            async with idem_lock:
                existing = _idempotency_store.get(tx.idempotency_key)
                if existing:
                    logger.info(
                        f"[Payment] Idempotency hit: key={tx.idempotency_key} "
                        f"returning cached tx={existing.id}"
                    )
                    return existing

                # Reserve the slot immediately — no async gap before we write.
                # We will overwrite with the real Transaction once processing
                # completes (or remove it on unrecoverable failure so the
                # caller can retry with a new key if desired).
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
                # Idempotency slot already written above — result is final.
                return tx

            # ── Step 3: Acquire locks (_acct_lock → _order_lock) ─────────────
            # FIX (BUG 3): consistent lock order in both payment and refund.
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

            # ── Step 4: Single atomic DB transaction (FIX BUG 2) ─────────────
            # Both the account UPDATE and the order/transaction INSERT are
            # sent to the DB before conn.commit() is called.  The DB engine
            # guarantees that either both writes persist or neither does.
            # There is therefore no window in which the account is debited
            # but the order does not exist.
            conn = None
            debit_committed = False  # tracks whether we need compensation
            try:
                conn = db_pool.acquire()

                # Begin explicit transaction (most DB drivers auto-begin, but
                # be explicit for clarity).
                try:
                    conn.execute("BEGIN")
                except Exception:
                    pass  # driver may not support explicit BEGIN — that is fine

                # Phase 1: Debit account (not yet committed).
                conn.execute(
                    "UPDATE accounts SET balance = balance - ? WHERE id = ? AND is_active = 1",
                    (tx.amount, tx.from_account),
                )
                logger.debug(
                    f"[Payment] Phase 1 staged (not committed): "
                    f"debit ${tx.amount:.2f} from {tx.from_account}"
                )

                # Simulate intermittent Phase-2 infrastructure failure.
                # With the single-transaction fix, a failure here causes a
                # rollback of Phase 1 as well — no orphaned debit.
                if random.random() < settings.payment.partial_commit_rate:
                    raise PartialCommitError(
                        f"Simulated intermittent DB error during Phase 2 "
                        f"write for tx={tx.id} (amount=${tx.amount:.2f})"
                    )

                # Phase 2: Confirm order + record transaction.
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
                           (id, idempotency_key, from_account, to_account, amount, currency,
                            method, status, fraud_score, fee, net_amount, metadata,
                            created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        tx.id, tx.idempotency_key,
                        tx.from_account, tx.to_account,
                        tx.amount, tx.currency,
                        tx.method.value, TransactionStatus.COMPLETED.value,
                        tx.fraud_score, tx.fee, tx.net_amount,
                        json.dumps(tx.metadata),
                        tx.created_at.isoformat(),
                        datetime.utcnow().isoformat(),
                    ),
                )

                # Single commit — atomically persists debit + order.
                conn.commit()
                debit_committed = True  # both phases succeeded together
                logger.debug(
                    f"[Payment] Atomic commit successful: tx={tx.id} "
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

            except PartialCommitError as e:
                # The single-transaction approach means the DB rolled back
                # Phase 1 automatically — no orphaned debit, no compensation
                # needed.  We still record the metric so ops can track the
                # underlying infrastructure instability.
                try:
                    conn.rollback()
                except Exception:
                    pass
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
                logger.error(
                    f"[Payment] Phase-2 failure rolled back atomically — "
                    f"NO orphaned debit. tx={tx.id} error={e}"
                )
                # Remove the idempotency reservation so the caller can retry.
                async with await _get_idempotency_lock(tx.idempotency_key):
                    _idempotency_store.pop(tx.idempotency_key, None)
                    m.idempotency_cache_size.set(len(_idempotency_store))
                raise PaymentProcessorError(str(e)) from e

            except (DBConnectionError, PoolExhaustedError) as e:
                try:
                    conn.rollback()
                except Exception:
                    pass
                # If the connection died after commit we cannot be certain
                # of DB state — apply compensating credit as a safety net.
                if debit_committed:
                    await self._compensate_debit(tx, tx.amount, str(e))
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

            # Finalise idempotency slot with the completed transaction.
            async with await _get_idempotency_lock(tx.idempotency_key):
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
                f"[Payment] Unexpected error for tx={tx.id}: {e}", exc_info=True
            )
            raise

        finally:
            elapsed = time.monotonic() - start
            m.payment_processing_duration_seconds.labels(
                method=tx.method.value
            ).observe(elapsed)
            m.active_payment_requests.dec()

    # ------------------------------------------------------------------ #
    #  process_refund                                                      #
    # ------------------------------------------------------------------ #

    async def process_refund(
        self,
        original_tx: Transaction,
        reason: str,
        amount: TypingOptional[float] = None,
    ) -> Transaction:
        """Process a refund for a completed transaction.

        FIX (BUG 3): locks are now acquired in the same order as
        process_payment (_acct_lock → _order_lock), eliminating the
        deadlock.
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
            # FIX (BUG 3): acquire _acct_lock FIRST, then _order_lock —
            # identical order to process_payment.
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
                    f"[Refund] DEADLOCK: timeout acquiring account_lock "
                    f"refund_tx={refund_tx.id} original_tx={original_tx.id}"
                )
                raise PaymentProcessorError(
                    "Deadlock: timeout acquiring account lock for refund"
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
                    f"[Refund] DEADLOCK: timeout acquiring order_lock "
                    f"refund_tx={refund_tx.id} original_tx={original_tx.id}"
                )
                raise PaymentProcessorError(
                    "Deadlock: timeout acquiring order lock for refund"
                )

            try:
                conn = db_pool.acquire()
                try:
                    try:
                        conn.execute("BEGIN")
                    except Exception:
                        pass

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
