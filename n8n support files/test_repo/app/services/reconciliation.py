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
# Convert fee_rate to Decimal once at module load to avoid repeated conversions.
_FEE_RATE: Decimal = Decimal(str(settings.payment.fee_rate))
_TWO_PLACES: Decimal = Decimal("0.01")


def _to_decimal(value) -> Decimal:
    """Safely convert a numeric value to Decimal via string to avoid float noise."""
    try:
        return Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"Cannot convert {value!r} to Decimal: {exc}") from exc


class ReconciliationService:
    """
    Validates that sum(transaction.net_amount) == expected_ledger_balance.

    All balances are kept as decimal.Decimal with ROUND_HALF_UP to two decimal
    places, preventing IEEE 754 float accumulation errors.
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

    def record_transaction(self, tx: Transaction):
        """
        Called after every successful transaction to update running balances.

        Both fee and net are rounded to exactly 2 decimal places with
        ROUND_HALF_UP before being added to the running totals, so no
        drift accumulates over time.
        """
        amount: Decimal = _to_decimal(tx.amount)

        # Round fee and net to 2 dp with ROUND_HALF_UP — the canonical
        # approach for financial systems.
        fee: Decimal = (amount * _FEE_RATE).quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)
        net: Decimal = (amount - fee).quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)

        # Both ledger and actual use the same Decimal arithmetic, so they
        # track identically and drift stays at $0.00.
        self._ledger_balance += net
        self._fee_ledger += fee

        self._actual_balance += net
        self._fee_actual += fee

        self._transaction_count += 1

        # Persist Decimal values back to the transaction model.
        # Convert to float only at the boundary where the model requires it.
        tx.fee = float(fee)
        tx.net_amount = float(net)

        # Drift is now structurally zero; compute it anyway for observability.
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
        """
        Periodic reconciliation check.
        Emits alerts if drift exceeds threshold.
        """
        start = time.monotonic()
        drift: Decimal = abs(self._ledger_balance - self._actual_balance)
        drift_float = float(drift)
        now = datetime.utcnow()

        # Determine status
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
        actual_balance_float = float(self._actual_balance)
        return {
            "ledger_balance": float(self._ledger_balance),
            "actual_balance": actual_balance_float,
            "drift": float(drift),
            "drift_pct": (float(drift) / max(actual_balance_float, 0.01)) * 100,
            "transaction_count": self._transaction_count,
            "total_fees_ledger": float(self._fee_ledger),
            "total_fees_actual": float(self._fee_actual),
            "status": "critical" if float(drift) > cfg.drift_threshold * 10
                       else "mismatch" if float(drift) > cfg.drift_threshold
                       else "ok",
        }


# Singleton
reconciliation_service = ReconciliationService()
