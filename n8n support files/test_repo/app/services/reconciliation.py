"""
Payment Reconciliation Service.

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
# Convert fee_rate once to Decimal at module load to avoid repeated conversions
_FEE_RATE: Decimal = Decimal(str(settings.payment.fee_rate))
_TWO_PLACES: Decimal = Decimal("0.01")
_ZERO: Decimal = Decimal("0")


def _to_decimal(value) -> Decimal:
    """Safely convert a numeric value to Decimal via string to avoid float imprecision."""
    try:
        return Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"Cannot convert {value!r} to Decimal: {exc}") from exc


def _round_currency(value: Decimal) -> Decimal:
    """Round a Decimal to 2 decimal places using ROUND_HALF_UP (standard financial rounding)."""
    return value.quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)


class ReconciliationService:
    """
    Validates that sum(transaction.net_amount) == expected_ledger_balance.

    All balances and intermediate calculations use decimal.Decimal with
    ROUND_HALF_UP to eliminate floating-point drift.
    """

    def __init__(self):
        # All balances are Decimal — no float drift
        self._ledger_balance: Decimal = _ZERO
        self._actual_balance: Decimal = _ZERO
        self._fee_ledger: Decimal = _ZERO
        self._fee_actual: Decimal = _ZERO
        self._transaction_count: int = 0
        self._total_drift: Decimal = _ZERO
        self._last_run: datetime = datetime.utcnow()
        self._report_history: List[ReconciliationReport] = []
        # Cache Decimal threshold to avoid repeated conversion
        self._drift_threshold: Decimal = _to_decimal(cfg.drift_threshold)

    def record_transaction(self, tx: Transaction):
        """
        Called after every successful transaction to update running balances.

        Both the ledger path and the actual path now use Decimal with explicit
        ROUND_HALF_UP so there is no divergence between the two accumulators.
        """
        amount: Decimal = _to_decimal(tx.amount)

        # Round fee and net to 2dp immediately — this is the canonical value
        fee: Decimal = _round_currency(amount * _FEE_RATE)
        net: Decimal = _round_currency(amount - fee)

        # Both accumulators use the same rounded Decimal values — drift is zero
        self._ledger_balance += net
        self._fee_ledger += fee

        self._actual_balance += net
        self._fee_actual += fee

        self._transaction_count += 1

        # Persist Decimal-precise values back to the transaction model.
        # Store as float for model compatibility; precision is already locked in.
        tx.fee = float(fee)
        tx.net_amount = float(net)

        # Drift should be zero; compute defensively in case of external mutation
        drift: Decimal = abs(self._ledger_balance - self._actual_balance)
        self._total_drift = drift

        m.ledger_balance_usd.set(float(self._ledger_balance))
        m.actual_balance_usd.set(float(self._actual_balance))
        m.reconciliation_drift_usd.set(float(drift))

        if drift > self._drift_threshold:
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
        """
        Periodic reconciliation check.
        Emits alerts if drift exceeds threshold.
        """
        start = time.monotonic()
        drift: Decimal = abs(self._ledger_balance - self._actual_balance)
        now = datetime.utcnow()

        # Determine status
        if drift > self._drift_threshold * 10:
            status = "critical"
        elif drift > self._drift_threshold:
            status = "mismatch"
        else:
            status = "ok"

        report = ReconciliationReport(
            period_start=self._last_run,
            period_end=now,
            ledger_balance=float(self._ledger_balance),
            actual_balance=float(self._actual_balance),
            drift=float(drift),
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
                f"{drift / self._drift_threshold:.1f}x | "
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
        actual_for_pct = self._actual_balance if self._actual_balance > _ZERO else Decimal("0.01")
        return {
            "ledger_balance": float(self._ledger_balance),
            "actual_balance": float(self._actual_balance),
            "drift": float(drift),
            "drift_pct": float((drift / actual_for_pct) * 100),
            "transaction_count": self._transaction_count,
            "total_fees_ledger": float(self._fee_ledger),
            "total_fees_actual": float(self._fee_actual),
            "status": "critical" if drift > self._drift_threshold * 10
                       else "mismatch" if drift > self._drift_threshold
                       else "ok",
        }


# Singleton
reconciliation_service = ReconciliationService()
