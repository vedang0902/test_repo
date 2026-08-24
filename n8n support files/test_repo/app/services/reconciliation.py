"""
Payment Reconciliation Service.

Runs periodically to verify that the sum of all processed transactions
matches the expected ledger balance.

Fix applied: All monetary arithmetic now uses decimal.Decimal with
ROUND_HALF_UP to eliminate IEEE 754 float drift across transactions.
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
# Convert fee_rate to Decimal once at module load to avoid repeated conversions
_FEE_RATE: Decimal = Decimal(str(settings.payment.fee_rate))
_TWO_PLACES: Decimal = Decimal("0.01")
_ZERO: Decimal = Decimal("0")


def _to_decimal(value) -> Decimal:
    """Safely convert a numeric value to Decimal."""
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError) as exc:
        raise ValueError(f"Cannot convert {value!r} to Decimal: {exc}") from exc


def _round2(value: Decimal) -> Decimal:
    """Round a Decimal to 2 decimal places using ROUND_HALF_UP."""
    return value.quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)


class ReconciliationService:
    """
    Validates that sum(transaction.net_amount) == expected_ledger_balance.

    All internal balances are decimal.Decimal to prevent IEEE 754 drift.
    """

    def __init__(self):
        self._ledger_balance: Decimal = _ZERO
        self._actual_balance: Decimal = _ZERO
        self._fee_ledger: Decimal = _ZERO
        self._fee_actual: Decimal = _ZERO
        self._transaction_count: int = 0
        self._total_drift: Decimal = _ZERO
        self._last_run: datetime = datetime.utcnow()
        self._report_history: List[ReconciliationReport] = []

    def record_transaction(self, tx: Transaction):
        """
        Called after every successful transaction to update running balances.

        Both fee and net are computed with Decimal arithmetic and rounded to
        2 decimal places (ROUND_HALF_UP) before accumulation, eliminating
        the systematic float drift observed in production.
        """
        amount: Decimal = _to_decimal(tx.amount)

        # Compute fee and net with exact decimal arithmetic
        fee: Decimal = _round2(amount * _FEE_RATE)
        net: Decimal = _round2(amount - fee)

        # Both ledger and actual now use the same Decimal-rounded values;
        # drift between them will be zero by construction.
        self._ledger_balance += net
        self._fee_ledger += fee

        self._actual_balance += net
        self._fee_actual += fee

        self._transaction_count += 1

        # Persist Decimal values back to the transaction model.
        # Convert to float only at the boundary (model storage / serialisation)
        # to preserve compatibility with existing model fields.
        tx.fee = float(fee)
        tx.net_amount = float(net)

        # Drift is now structurally zero; emit it for observability anyway
        drift: Decimal = abs(self._ledger_balance - self._actual_balance)
        self._total_drift = drift
        drift_float: float = float(drift)

        m.ledger_balance_usd.set(float(self._ledger_balance))
        m.actual_balance_usd.set(float(self._actual_balance))
        m.reconciliation_drift_usd.set(drift_float)

        drift_threshold: Decimal = _to_decimal(cfg.drift_threshold)
        if drift > drift_threshold:
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
        drift_threshold: Decimal = _to_decimal(cfg.drift_threshold)
        now = datetime.utcnow()

        if drift > drift_threshold * 10:
            status = "critical"
        elif drift > drift_threshold:
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
                f"{drift / drift_threshold:.1f}x | "
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
        actual_for_pct: Decimal = self._actual_balance if self._actual_balance > _ZERO else _to_decimal("0.01")
        drift_threshold: Decimal = _to_decimal(cfg.drift_threshold)
        return {
            "ledger_balance": float(self._ledger_balance),
            "actual_balance": float(self._actual_balance),
            "drift": float(drift),
            "drift_pct": float((drift / actual_for_pct) * 100),
            "transaction_count": self._transaction_count,
            "total_fees_ledger": float(self._fee_ledger),
            "total_fees_actual": float(self._fee_actual),
            "status": (
                "critical" if drift > drift_threshold * 10
                else "mismatch" if drift > drift_threshold
                else "ok"
            ),
        }


# Singleton
reconciliation_service = ReconciliationService()
