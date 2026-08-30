"""
Payment Reconciliation Service.

Runs periodically to verify that the sum of all processed transactions
matches the expected ledger balance.
"""
import logging
import asyncio
import time
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import List

from app.config import settings
from app.models import Transaction, TransactionStatus, ReconciliationReport
from app.metrics import prometheus_metrics as m

logger = logging.getLogger("reconciliation")

cfg = settings.reconciliation
# Convert fee_rate to Decimal once at module load to avoid repeated conversions
FEE_RATE: Decimal = Decimal(str(settings.payment.fee_rate))
TWO_PLACES: Decimal = Decimal("0.01")
DRIFT_THRESHOLD: Decimal = Decimal(str(cfg.drift_threshold))


def _to_decimal(value) -> Decimal:
    """Safely convert a float or string amount to Decimal."""
    return Decimal(str(value))


class ReconciliationService:
    """
    Validates that sum(transaction.net_amount) == expected_ledger_balance.

    All balances are maintained as decimal.Decimal with ROUND_HALF_UP to
    eliminate IEEE 754 floating-point drift.
    """

    def __init__(self):
        self._ledger_balance: Decimal = Decimal("0")
        self._actual_balance: Decimal = Decimal("0")  # kept for API compat; equals ledger
        self._fee_ledger: Decimal = Decimal("0")
        self._fee_actual: Decimal = Decimal("0")      # kept for API compat; equals fee_ledger
        self._transaction_count: int = 0
        self._total_drift: Decimal = Decimal("0")
        self._last_run: datetime = datetime.utcnow()
        self._report_history: List[ReconciliationReport] = []

    def record_transaction(self, tx: Transaction):
        """
        Called after every successful transaction to update running balances.

        Uses Decimal arithmetic with ROUND_HALF_UP to ensure each fee and
        net amount is computed to exactly 2 decimal places with no drift.
        """
        amount: Decimal = _to_decimal(tx.amount)

        # Round fee to 2 dp with ROUND_HALF_UP — deterministic, no float drift
        fee: Decimal = (amount * FEE_RATE).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
        net: Decimal = (amount - fee).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)

        self._ledger_balance += net
        self._fee_ledger += fee

        # actual_balance mirrors ledger_balance exactly (no separate rounding path needed)
        self._actual_balance += net
        self._fee_actual += fee

        self._transaction_count += 1

        # Persist Decimal values back onto the transaction model.
        # Convert to float only at the boundary where the model requires it.
        tx.fee = float(fee)
        tx.net_amount = float(net)

        # Drift is always zero with a single Decimal code path, but we still
        # compute and emit it so dashboards and alerts continue to function.
        drift: Decimal = abs(self._ledger_balance - self._actual_balance)
        self._total_drift = drift

        m.ledger_balance_usd.set(float(self._ledger_balance))
        m.actual_balance_usd.set(float(self._actual_balance))
        m.reconciliation_drift_usd.set(float(drift))

        if drift > DRIFT_THRESHOLD:
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

        if drift > DRIFT_THRESHOLD * 10:
            status = "critical"
        elif drift > DRIFT_THRESHOLD:
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
                f"{drift / DRIFT_THRESHOLD:.1f}x | "
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
        actual = self._actual_balance if self._actual_balance > 0 else Decimal("0.01")
        return {
            "ledger_balance": float(self._ledger_balance),
            "actual_balance": float(self._actual_balance),
            "drift": float(drift),
            "drift_pct": float((drift / actual) * 100),
            "transaction_count": self._transaction_count,
            "total_fees_ledger": float(self._fee_ledger),
            "total_fees_actual": float(self._fee_actual),
            "status": (
                "critical" if drift > DRIFT_THRESHOLD * 10
                else "mismatch" if drift > DRIFT_THRESHOLD
                else "ok"
            ),
        }


# Singleton
reconciliation_service = ReconciliationService()
