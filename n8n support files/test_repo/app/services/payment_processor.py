"""
Core Payment Processing Service.

Orchestrates the full payment lifecycle:
  1. Idempotency check (atomic, per-key lock)
  2. Fraud scoring
  3. Atomic DB transaction: debit account + confirm order (phase 1+2)
  4. Compensating credit on phase-2 failure (saga pattern)
  5. Reconciliation recording

All three bugs from the original implementation have been fixed:
  BUG 1 (race condition / double charge)  — fixed with per-key asyncio.Lock
  BUG 2 (orphaned debit / partial commit) — fixed with atomic DB transaction + compensating credit
  BUG 3 (deadlock)                        — fixed by enforcing consistent lock order (_acct_lock -> _order_lock)
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
# Idempotency store — FIX 1: protect each key with its own asyncio.Lock so
# the check-then-set is atomic within a single event-loop thread.
# ---------------------------------------------------------------------------
_idempotency_store: Dict[str, Transaction] = {}
_idempotency_key_locks: Dict[str, asyncio.Lock] = {}
_idempotency_meta_lock = asyncio.Lock()   # guards the dict of per-key locks


async def _get_idempotency_lock(key: str) -> asyncio.Lock:
    """Return (and lazily create) the per-key lock for *key*."""
    async with _idempotency_meta_lock:
        if key not in _idempotency_key_locks:
            _idempotency_key_locks[key] = asyncio.Lock()
        return _idempotency_key_locks[key]


# ---------------------------------------------------------------------------
# Shared locks — FIX 3: BOTH process_payment AND process_refund now acquire
# in the SAME order: _acct_lock first, then _order_lock.
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

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    async def _acquire_processing_locks(self, label: str, tx_id: str) -> None:
        """
        Acquire _acct_lock then _order_lock (consistent order — no deadlock).
        Raises PaymentProcessorError on timeout.
        """
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
                f"[{label}] DEADLOCK: timeout acquiring account_lock tx={tx_id}"
            )
            raise PaymentProcessorError(
                f"Deadlock: timeout acquiring account lock for {tx_id}"
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
                f"[{label}] DEADLOCK: timeout acquiring order_lock tx={tx_id}"
            )
            raise PaymentProcessorError(
                f"Deadlock: timeout acquiring order lock for {tx_id}"
            )

    @staticmethod
    def _release_processing_locks() -> None:
        """Release order_lock then acct_lock (reverse of acquisition order)."""
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

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    async def process_payment(self, tx: Transaction) -> Transaction:
        """
        Full payment processing pipeline — all three bugs fixed.
        """
        start = time.monotonic()
        m.active_payment_requests.inc()

        try:
            tx.status = TransactionStatus.PROCESSING

            # ── Step 1: Idempotency check — FIX 1: atomic under per-key lock ──
            idem_lock = await _get_idempotency_lock(tx.idempotency_key)
            async with idem_lock:
                existing = _idempotency_store.get(tx.idempotency_key)
                if existing:
                    logger.info(
                        f"[Payment] Idempotency hit: key={tx.idempotency_key} "
                        f"returning cached tx={existing.id}"
                    )
                    return existing

                # Mark the slot immediately so any concurrent coroutine that
                # acquires this lock next will see the in-progress transaction
                # and return it rather than starting a duplicate.
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
                # Idempotency store already holds tx; update its status.
                _idempotency_store[tx.idempotency_key] = tx
                return tx

            # ── Step 3: Acquire locks (consistent order — FIX 3) ─────────────
            await self._acquire_processing_locks("Payment", tx.id)

            # ── Step 4: Atomic DB transaction — FIX 2 ────────────────────────
            # Both the account debit (phase 1) and the order insert (phase 2)
            # are committed in a single transaction.  If phase 2 fails we roll
            # back the debit and issue a compensating credit so the account
            # balance is always consistent.
            conn = None
            try:
                conn = db_pool.acquire()

                # Begin explicit transaction (connection-level autocommit must
                # be disabled — assumed by db_pool contract).
                conn.execute("BEGIN")

                # Phase 1: debit
                conn.execute(
                    "UPDATE accounts SET balance = balance - ? WHERE id = ? AND is_active = 1",
                    (tx.amount, tx.from_account),
                )
                logger.debug(
                    f"[Payment] Phase 1 staged: debit ${tx.amount:.2f} "
                    f"from {tx.from_account} (not yet committed)"
                )

                # Phase 2: confirm order
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

                # Single commit — both phases succeed or both are rolled back.
                conn.commit()

                tx.mark_completed()
                # Update idempotency store with final state.
                _idempotency_store[tx.idempotency_key] = tx
                m.idempotency_cache_size.set(len(_idempotency_store))

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
                # Roll back whatever was staged in this transaction so the
                # account balance is never left in an inconsistent state.
                if conn:
                    try:
                        conn.execute("ROLLBACK")
                        logger.warning(
                            f"[Payment] Transaction rolled back for tx={tx.id}: {db_exc}"
                        )
                    except Exception as rb_exc:
                        # Rollback itself failed — last-resort compensating credit.
                        logger.critical(
                            f"[Payment] ROLLBACK FAILED for tx={tx.id}: {rb_exc}. "
                            f"Issuing compensating credit of ${tx.amount:.2f} "
                            f"to {tx.from_account}."
                        )
                        try:
                            conn.execute(
                                "UPDATE accounts SET balance = balance + ? WHERE id = ?",
                                (tx.amount, tx.from_account),
                            )
                            conn.commit()
                            logger.info(
                                f"[Payment] Compensating credit applied for tx={tx.id}."
                            )
                        except Exception as comp_exc:
                            logger.critical(
                                f"[Payment] COMPENSATING CREDIT ALSO FAILED for tx={tx.id}: "
                                f"{comp_exc}. Manual remediation required."
                            )

                tx.mark_failed(str(db_exc))
                # Remove the provisional idempotency entry so the caller can
                # safely retry once the underlying issue is resolved.
                _idempotency_store.pop(tx.idempotency_key, None)
                m.idempotency_cache_size.set(len(_idempotency_store))

                if isinstance(db_exc, (DBConnectionError, PoolExhaustedError)):
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
                    m.app_errors_total.labels(
                        component="payment_processor", error_type="unexpected"
                    ).inc()
                    logger.error(
                        f"[Payment] Unexpected error for tx={tx.id}: {db_exc}",
                        exc_info=True,
                    )
                raise

            finally:
                if conn:
                    try:
                        db_pool.release(conn)
                    except Exception:
                        pass
                self._release_processing_locks()

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

        FIX 3: Lock order is now _acct_lock -> _order_lock, identical to
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
            # FIX 3: Acquire in the same order as process_payment.
            await self._acquire_processing_locks("Refund", refund_tx.id)

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
                    except Exception as rb_exc:
                        logger.critical(
                            f"[Refund] ROLLBACK FAILED for refund_tx={refund_tx.id}: "
                            f"{rb_exc}. Manual remediation required."
                        )
                    raise

                finally:
                    db_pool.release(conn)

            except DBConnectionError as e:
                refund_tx.mark_failed(str(e))
                raise

            finally:
                self._release_processing_locks()

            return refund_tx

        finally:
            elapsed = time.monotonic() - start
            m.payment_processing_duration_seconds.labels(
                method=refund_tx.method.value
            ).observe(elapsed)
            m.active_payment_requests.dec()


# Singleton
payment_processor = PaymentProcessor()
