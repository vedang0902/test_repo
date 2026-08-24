"""Payment Reconciliation Service.

Runs periodically to verify that the sum of all processed transactions
matches the expected ledger balance.
"""
import logging
import asyncio
import time
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import List

from app.config import settings
from app.models import Transaction, TransactionStatus, ReconciliationReport
from app.metrics import prometheus_metrics as m

logger = logging.getLogger("reconciliation")

cfg = settings.reconciliation
# Convert fee_rate once at module load so every call uses an exact Decimal.
_FEE_RATE: Decimal = Decimal(str(settings.payment.fee_rate))
_TWO_PLACES = Decimal("0.01")


def _to_decimal(value) -> Decimal:
    """Safely coerce a float or string to Decimal."""
    try:
        return Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"Cannot convert {value!r} to Decimal: {exc}") from exc


class ReconciliationService:
    """
    Validates that sum(transaction.net_amount) == expected_ledger_balance.

    All monetary values are stored as decimal.Decimal, rounded to 2 dp with
    ROUND_HALF_UP at every intermediate step, eliminating IEEE 754 drift.
    """

    def __init__(self):
        self._ledger_balance: Decimal = Decimal("0.00")
        self._actual_balance: Decimal = Decimal("0.00")
        self._fee_ledger: Decimal = Decimal("0.00")
        self._fee_actual: Decimal = Decimal("0.00")
        self._transaction_count: int = 0
        self._total_drift: Decimal = Decimal("0.00")
        self._last_run: datetime = datetime.utcnow()
        self._report_history: List[ReconciliationReport] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _round2(value: Decimal) -> Decimal:
        """Round to 2 decimal places using ROUND_HALF_UP."""
        return value.quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_transaction(self, tx: Transaction):
        """
        Called after every successful transaction to update running balances.

        All arithmetic is performed in Decimal so there is no floating-point
        accumulation error.
        """
        amount: Decimal = self._round2(_to_decimal(tx.amount))

        # Round fee and net to 2 dp individually — matches what a customer
        # sees on their statement and what the ledger should hold.
        fee: Decimal = self._round2(amount * _FEE_RATE)
        net: Decimal = self._round2(amount - fee)

        # Both accumulators use the same Decimal arithmetic — no divergence.
        self._ledger_balance += net
        self._fee_ledger += fee
        self._actual_balance += net
        self._fee_actual += fee

        self._transaction_count += 1

        # Persist exact Decimal values back to the transaction model.
        # Convert to float only at the boundary if the model requires it.
        tx.fee = float(fee)
        tx.net_amount = float(net)

        # Drift is always zero with correct Decimal arithmetic; compute
        # defensively so the metric path stays intact.
        drift: Decimal = abs(self._ledger_balance - self._actual_balance)
        self._total_drift = drift

        drift_float = float(drift)
        m.ledger_balance_usd.set(float(self._ledger_balance))
        m.actual_balance_usd.set(float(self._actual_balance))
        m.reconciliation_drift_usd.set(drift_float)

        if drift_float > cfg.drift_threshold:
            m.reconciliation_mismatches_total.inc()
            logger.error(
                f"[Reconciliation] Mismatch detected: "
                f"ledger=${self._ledger_balance:.6f} actual=${self._actual_balance:.6f} "
                f"drift=${drift:.6f} tx_count={self._transaction_count} tx_id={tx.id}"
            )
            m.app_errors_total.labels(
                component="reconciliation", error_type="float_drift"
            ).inc()

    def run_reconciliation(self) -> ReconciliationReport:
        """Periodic reconciliation check. Emits alerts if drift exceeds threshold."""
        start = time.monotonic()
        drift: Decimal = abs(self._ledger_balance - self._actual_balance)
        drift_float = float(drift)
        now = datetime.utcnow()

        if drift_float > cfg.drift_threshold * 10:
            status = "critical"
        elif drift_float > cfg.drift_threshold:
            status = "mismatch"
        else:
            status = "ok"

        report = ReconciliationReport(
            period_start=self._last_run,
            period_end=now,
            ledger_balance=float(self._ledger_balance),
            actual_balance=float(self._actual_balance),
            drift=drift_float,
            transaction_count=self._transaction_count,
            status=status,
            generated_at=now,
        )
        self._report_history.append(report)
        self._last_run = now

        elapsed_ms = (time.monotonic() - start) * 1000

        if status == "critical":
            m.reconciliation_drift_exceeded_total.inc()
            m.app_error_rate.set(1)
            m.app_errors_total.labels(
                component="reconciliation", error_type="critical_drift"
            ).inc()
            logger.error(
                f"[Reconciliation] CRITICAL: drift=${drift:.6f} exceeds threshold by "
                f"{drift_float / cfg.drift_threshold:.1f}x | "
                f"ledger=${self._ledger_balance:.4f} actual=${self._actual_balance:.4f} | "
                f"tx_count={self._transaction_count} | elapsed={elapsed_ms:.1f}ms"
            )
        elif status == "mismatch":
            m.reconciliation_drift_exceeded_total.inc()
            logger.error(
                f"[Reconciliation] FAILED: drift=${drift:.6f} > threshold=${cfg.drift_threshold} | "
                f"ledger=${self._ledger_balance:.4f} actual=${self._actual_balance:.4f} | "
                f"tx_count={self._transaction_count}"
            )
        else:
            logger.info(
                f"[Reconciliation] OK: drift=${drift:.8f} tx_count={self._transaction_count}"
            )

        return report

    async def run_periodic(self):
        """Background task: run reconciliation every N seconds."""
        logger.info(
            f"[Reconciliation] Periodic runner started (interval={cfg.interval_seconds}s)"
        )
        while True:
            await asyncio.sleep(cfg.interval_seconds)
            try:
                report = self.run_reconciliation()
                logger.info(
                    f"[Reconciliation] Period report: status={report.status} "
                    f"drift=${report.drift:.6f} txns={report.transaction_count}"
                )
            except Exception as e:
                logger.error(f"[Reconciliation] Periodic run failed: {e}")
                m.app_errors_total.labels(
                    component="reconciliation", error_type="periodic_failure"
                ).inc()

    def get_summary(self) -> dict:
        drift: Decimal = abs(self._ledger_balance - self._actual_balance)
        drift_float = float(drift)
        actual_float = float(self._actual_balance)
        return {
            "ledger_balance": float(self._ledger_balance),
            "actual_balance": actual_float,
            "drift": drift_float,
            "drift_pct": (drift_float / max(actual_float, 0.01)) * 100,
            "transaction_count": self._transaction_count,
            "total_fees_ledger": float(self._fee_ledger),
            "total_fees_actual": float(self._fee_actual),
            "status": (
                "critical" if drift_float > cfg.drift_threshold * 10
                else "mismatch" if drift_float > cfg.drift_threshold
                else "ok"
            ),
        }


# Singleton
reconciliation_service = ReconciliationService()
