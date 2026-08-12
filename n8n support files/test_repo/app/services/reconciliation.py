"""
Payment Reconciliation Service.

Runs periodically to verify that the sum of all processed transactions
matches the expected ledger balance.

=============================================================================
BUG: Float Arithmetic Currency Drift
=============================================================================
Root cause:
  Transaction amounts, fees, and net amounts are stored as Python `float`
  (IEEE 754 double-precision). The correct approach for financial systems is
  `decimal.Decimal` with explicit rounding modes.

  Examples of the drift:
    0.1 + 0.2  → 0.30000000000000004   (not 0.30)
    29.99 * 0.029 → 0.86971            (should be 0.87 rounded to 2dp)

  Over hundreds of transactions, the ledger accumulates a systematic drift
  of several cents. This crosses the $0.005 threshold, triggering alerts.

Fix (NOT applied here):
  Use `from decimal import Decimal, ROUND_HALF_UP` throughout.

Symptoms in logs:
  ERROR reconciliation | Drift exceeded threshold: ledger=1023.741200 actual=1023.750000 drift=$0.008800
  ERROR reconciliation | Reconciliation FAILED: cumulative drift=$0.0312 over 847 transactions

Prometheus metrics:
  reconciliation_drift_usd              ↑ over time
  reconciliation_mismatches_total       ↑
  reconciliation_drift_exceeded_total   ↑
"""
import logging
import asyncio
import time
from datetime import datetime
from typing import List, Tuple

from app.config import settings
from app.models import Transaction, TransactionStatus, ReconciliationReport
from app.metrics import prometheus_metrics as m

logger = logging.getLogger("reconciliation")

cfg = settings.reconciliation
fee_rate = settings.payment.fee_rate


class ReconciliationService:
    """
    Validates that sum(transaction.net_amount) == expected_ledger_balance.

    The ledger is an in-memory running total (float) that accumulates
    rounding errors over time — a genuine production-grade floating-point bug.
    """

    def __init__(self):
        # BUG: Both balances are float — should be Decimal
        self._ledger_balance: float = 0.0       # Running float sum (drifts)
        self._actual_balance: float = 0.0       # Rounded to 2dp each transaction
        self._fee_ledger: float = 0.0
        self._fee_actual: float = 0.0
        self._transaction_count: int = 0
        self._total_drift: float = 0.0
        self._last_run: datetime = datetime.utcnow()
        self._report_history: List[ReconciliationReport] = []

    def record_transaction(self, tx: Transaction):
        """
        Called after every successful transaction to update running balances.

        BUG: fee = amount * fee_rate  →  floating-point multiplication error
             net = amount - fee       →  accumulates rounding errors per tx
        """
        # BUG: Using float multiplication for financial amounts
        fee = tx.amount * fee_rate                   # e.g. 100.05 * 0.029 = 2.9014499...
        net = tx.amount - fee                        # Subtraction of two floats

        # Ledger accumulates raw floats — drift grows
        self._ledger_balance += net
        self._fee_ledger += fee

        # Actual uses round() — gets the "right" answer 2dp
        net_rounded = round(tx.amount - round(tx.amount * fee_rate, 2), 2)
        fee_rounded = round(tx.amount * fee_rate, 2)
        self._actual_balance += net_rounded
        self._fee_actual += fee_rounded

        self._transaction_count += 1

        # Update tx model with float values (preserves bug)
        tx.fee = fee
        tx.net_amount = net

        # Compute drift and emit metrics
        drift = abs(self._ledger_balance - self._actual_balance)
        self._total_drift = drift

        m.ledger_balance_usd.set(self._ledger_balance)
        m.actual_balance_usd.set(self._actual_balance)
        m.reconciliation_drift_usd.set(drift)

        if drift > cfg.drift_threshold:
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
        drift = abs(self._ledger_balance - self._actual_balance)
        now = datetime.utcnow()

        # Determine status
        if drift > cfg.drift_threshold * 10:
            status = "critical"
        elif drift > cfg.drift_threshold:
            status = "mismatch"
        else:
            status = "ok"

        report = ReconciliationReport(
            period_start=self._last_run,
            period_end=now,
            ledger_balance=self._ledger_balance,
            actual_balance=self._actual_balance,
            drift=drift,
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
                f"{drift / cfg.drift_threshold:.1f}x | "
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
        drift = abs(self._ledger_balance - self._actual_balance)
        return {
            "ledger_balance": self._ledger_balance,
            "actual_balance": self._actual_balance,
            "drift": drift,
            "drift_pct": (drift / max(self._actual_balance, 0.01)) * 100,
            "transaction_count": self._transaction_count,
            "total_fees_ledger": self._fee_ledger,
            "total_fees_actual": self._fee_actual,
            "status": "critical" if drift > cfg.drift_threshold * 10
                       else "mismatch" if drift > cfg.drift_threshold
                       else "ok",
        }


# Singleton
reconciliation_service = ReconciliationService()
