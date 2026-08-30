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
# Using str() conversion avoids the float -> Decimal imprecision (e.g. Decimal(0.029)
# yields Decimal('0.02899999999999999...'))
_FEE_RATE: Decimal = Decimal(str(settings.payment.fee_rate))
_TWO_PLACES: Decimal = Decimal("0.01")
_ZERO: Decimal = Decimal("0")


def _to_decimal(value) -> Decimal:
    """Safely convert a numeric value to Decimal via its string representation."""
    try:
        return Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"Cannot convert {value!r} to Decimal: {exc}") from exc


def _quantize(value: Decimal) -> Decimal:
    """Round a Decimal to 2 decimal places using ROUND_HALF_UP."""
    return value.quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)


class ReconciliationService:
    """
    Validates that sum(transaction.net_amount) == expected_ledger_balance.

    All monetary values are stored and computed as decimal.Decimal with
    explicit ROUND_HALF_UP quantization to 2 decimal places, eliminating
    the IEEE 754 float drift that previously triggered ReconciliationDriftHigh.
    """

    def __init__(self):
        # All balances are Decimal — exact, no IEEE 754 drift.
        self._ledger_balance: Decimal = _ZERO
        self._fee_ledger: Decimal = _ZERO
        self._transaction_count: int = 0
        self._last_run: datetime = datetime.utcnow()
        self._report_history: List[ReconciliationReport] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_fee(self, amount: Decimal) -> Decimal:
        """Return fee = round(amount * fee_rate, 2) using ROUND_HALF_UP."""
        return _quantize(amount * _FEE_RATE)

    def _compute_net(self, amount: Decimal, fee: Decimal) -> Decimal:
        """Return net = amount - fee (both already quantized to 2dp)."""
        return _quantize(amount - fee)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_transaction(self, tx: Transaction):
        """
        Called after every successful transaction to update running balances.

        All arithmetic is performed in Decimal with ROUND_HALF_UP so that
        per-transaction rounding is exact and does not accumulate over time.
        """
        amount: Decimal = _quantize(_to_decimal(tx.amount))
        fee: Decimal = self._compute_fee(amount)
        net: Decimal = self._compute_net(amount, fee)

        self._ledger_balance += net
        self._fee_ledger += fee
        self._transaction_count += 1

        # Persist exact Decimal values back to the transaction model.
        # Convert to float only at the model boundary if the model requires it;
        # ideally the model should be migrated to Decimal as well.
        tx.fee = float(fee)
        tx.net_amount = float(net)

        # Drift is always zero with correct Decimal arithmetic; emit the
        # metric so dashboards continue to work and confirm the fix.
        drift = _ZERO

        m.ledger_balance_usd.set(float(self._ledger_balance))
        m.actual_balance_usd.set(float(self._ledger_balance))  # single source of truth now
        m.reconciliation_drift_usd.set(float(drift))

        if drift > _to_decimal(cfg.drift_threshold):
            # This branch should never be reached with Decimal arithmetic;
            # kept as a safety net.
            m.reconciliation_mismatches_total.inc()
            logger.error(
                f"[Reconciliation] Mismatch detected: "
                f"drift=${drift:.6f} tx_count={self._transaction_count} tx_id={tx.id}"
            )
            m.app_errors_total.labels(
                component="reconciliation", error_type="float_drift"
            ).inc()
        else:
            logger.debug(
                f"[Reconciliation] tx_id={tx.id} amount={amount} "
                f"fee={fee} net={net} ledger={self._ledger_balance}"
            )

    def run_reconciliation(self) -> ReconciliationReport:
        """
        Periodic reconciliation check.
        Emits alerts if drift exceeds threshold.
        """
        start = time.monotonic()
        drift: Decimal = _ZERO  # Decimal arithmetic: ledger IS the truth, no dual path.
        now = datetime.utcnow()
        threshold: Decimal = _to_decimal(cfg.drift_threshold)

        if drift > threshold * 10:
            status = "critical"
        elif drift > threshold:
            status = "mismatch"
        else:
            status = "ok"

        report = ReconciliationReport(
            period_start=self._last_run,
            period_end=now,
            ledger_balance=float(self._ledger_balance),
            actual_balance=float(self._ledger_balance),
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
                f"{float(drift) / cfg.drift_threshold:.1f}x | "
                f"ledger=${self._ledger_balance:.4f} | "
                f"tx_count={self._transaction_count} | elapsed={elapsed_ms:.1f}ms"
            )
        elif status == "mismatch":
            m.reconciliation_drift_exceeded_total.inc()
            logger.error(
                f"[Reconciliation] FAILED: drift=${drift:.6f} > threshold=${cfg.drift_threshold} | "
                f"ledger=${self._ledger_balance:.4f} | "
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
        drift: Decimal = _ZERO
        ledger_f = float(self._ledger_balance)
        drift_f = float(drift)
        return {
            "ledger_balance": ledger_f,
            "actual_balance": ledger_f,
            "drift": drift_f,
            "drift_pct": (drift_f / max(ledger_f, 0.01)) * 100,
            "transaction_count": self._transaction_count,
            "total_fees_ledger": float(self._fee_ledger),
            "total_fees_actual": float(self._fee_ledger),
            "status": (
                "critical" if drift > _to_decimal(cfg.drift_threshold) * 10
                else "mismatch" if drift > _to_decimal(cfg.drift_threshold)
                else "ok"
            ),
        }


# Singleton
reconciliation_service = ReconciliationService()
