"""
Simulated AI Fraud Detection Agent.

Architecture: rule-based scoring engine with a learned risk model (mocked).
In a real system this would call an ML inference endpoint or a feature store.
"""
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

from app.config import settings
from app.models import Transaction, FraudCheckResult
from app.metrics import prometheus_metrics as m

logger = logging.getLogger("fraud_detector")

cfg = settings.fraud

# How long per-account state is considered valid before it is discarded.
# Transactions arriving after this window are scored independently.
_ACCUMULATOR_TTL_MINUTES: int = 30


class _AccountState:
    """Holds per-account fraud-scoring state with an expiry timestamp."""

    def __init__(self):
        self.check_count: int = 0
        self.flag_count: int = 0
        self.recent_amounts: List[float] = []
        self.expires_at: datetime = datetime.utcnow() + timedelta(
            minutes=_ACCUMULATOR_TTL_MINUTES
        )

    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() >= self.expires_at

    def refresh_expiry(self):
        self.expires_at = datetime.utcnow() + timedelta(
            minutes=_ACCUMULATOR_TTL_MINUTES
        )


class FraudDetectionAgent:
    """
    Multi-rule fraud scoring engine.
    Mimics a production ML-backed fraud service.

    Each transaction is scored independently — there is no cross-transaction
    score bleed. Per-account state (velocity window, check count) is retained
    for a rolling TTL window and then automatically discarded so that old
    activity cannot permanently elevate an account's risk baseline.
    """

    def __init__(self):
        self._state: Dict[str, _AccountState] = {}

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _get_state(self, account_id: str) -> _AccountState:
        """Return the live state for *account_id*, creating or expiring as needed."""
        existing: Optional[_AccountState] = self._state.get(account_id)
        if existing is None or existing.is_expired:
            # Expired or first-seen: start fresh so old data cannot compound.
            if existing is not None and existing.is_expired:
                logger.debug(
                    f"[Fraud] Account state expired and reset for account={account_id}"
                )
            self._state[account_id] = _AccountState()
        return self._state[account_id]

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def check(self, tx: Transaction) -> FraudCheckResult:
        """
        Score a transaction for fraud.

        The fraud score for each transaction is derived solely from the
        rules evaluated against *that* transaction plus recent velocity
        data.  There is no bleed from previous scores.
        """
        account_id = tx.from_account
        triggered_rules: List[str] = []

        state = self._get_state(account_id)

        # ── Update velocity window before rule evaluation ─────────────────
        state.recent_amounts.append(tx.amount)
        if len(state.recent_amounts) > 20:
            state.recent_amounts = state.recent_amounts[-20:]

        # ── Base rule scoring (independent per-transaction) ───────────────
        base_score, rules = self._evaluate_rules(tx, state.recent_amounts)
        triggered_rules.extend(rules)

        # FIX: score is purely the base score; no cross-transaction bleed.
        transaction_score = base_score
        transaction_score = min(transaction_score, 1.0)

        # Track check count and refresh the TTL on each activity.
        state.check_count += 1
        state.refresh_expiry()

        # ── Cascade detection ─────────────────────────────────────────────
        # A cascade is only meaningful when there has been sustained,
        # *recent* high-scoring activity.  Because scores no longer compound
        # this guard acts as an early-warning for genuinely repeated
        # high-risk behaviour rather than a self-fulfilling prophecy.
        is_flagged = transaction_score >= cfg.threshold
        cascade_detected = False

        if state.check_count >= cfg.cascade_check_count and transaction_score > 0.40:
            cascade_detected = True
            m.fraud_cascade_events_total.inc()
            triggered_rules.append("cascade:repeated_high_score")
            logger.error(
                f"[Fraud] CASCADE DETECTED account={account_id} "
                f"base_score={base_score:.3f} transaction_score={transaction_score:.3f} "
                f"check_count={state.check_count} tx_id={tx.id}"
            )
            m.app_error_rate.set(1)
            m.app_errors_total.labels(
                component="fraud_detector", error_type="cascade"
            ).inc()

        # ── False positive tracking ───────────────────────────────────────
        if is_flagged:
            state.flag_count += 1

            # A false positive is a flag that was triggered by the compounding
            # effect rather than the transaction's own signals.  Because we no
            # longer compound, we only record a false positive when the
            # transaction's standalone base score is below the threshold but
            # the account has been flagged multiple times in the current window
            # — suggesting the rules themselves are over-sensitive for this
            # account rather than a cascade artefact.
            if state.flag_count > 2 and base_score < cfg.threshold:
                m.fraud_false_positives_total.inc()
                logger.warning(
                    f"[Fraud] Likely FALSE POSITIVE account={account_id} "
                    f"base_score={base_score:.3f} "
                    f"transaction_score={transaction_score:.3f} "
                    f"flag_count={state.flag_count} tx_id={tx.id}"
                )

        # ── Prometheus metrics ────────────────────────────────────────────
        result_label = "flagged" if is_flagged else "cleared"
        m.fraud_checks_total.labels(result=result_label).inc()
        m.fraud_score_histogram.observe(transaction_score)

        if is_flagged:
            logger.warning(
                f"[Fraud] FLAGGED tx={tx.id} account={account_id} "
                f"score={transaction_score:.3f} rules={triggered_rules}"
            )
        else:
            logger.debug(
                f"[Fraud] cleared tx={tx.id} account={account_id} "
                f"score={transaction_score:.3f}"
            )

        return FraudCheckResult(
            transaction_id=tx.id,
            account_id=account_id,
            base_score=base_score,
            compounded_score=transaction_score,  # field kept for schema compat
            triggered_rules=triggered_rules,
            is_flagged=is_flagged,
        )

    def _evaluate_rules(
        self, tx: Transaction, recent_amounts: List[float]
    ) -> Tuple[float, List[str]]:
        """Apply deterministic + stochastic fraud rules."""
        score = 0.0
        rules: List[str] = []

        # Rule 1: High transaction amount
        if tx.amount > cfg.high_amount_threshold:
            score += 0.30
            rules.append("high_amount")
        elif tx.amount > cfg.high_amount_threshold * 0.4:
            score += 0.12
            rules.append("medium_amount")

        # Rule 2: Off-hours transaction (04:00–06:00 UTC is high-risk window)
        hour = datetime.utcnow().hour
        if 4 <= hour < 6:
            score += 0.18
            rules.append("off_hours")

        # Rule 3: Velocity check — many transactions in recent window
        # Note: recent_amounts already includes the current tx amount.
        prior_count = len(recent_amounts) - 1  # exclude the tx we just appended
        if prior_count >= 8:
            score += 0.15
            rules.append("high_velocity")
        elif prior_count >= 4:
            score += 0.07
            rules.append("medium_velocity")

        # Rule 4: Cross-currency transaction
        if tx.currency != "USD":
            score += 0.08
            rules.append("cross_currency")

        # Rule 5: Simulated ML model score (Gaussian noise around 0.08)
        ml_score = random.gauss(0.08, 0.04)
        ml_score = max(0.0, min(ml_score, 0.30))
        score += ml_score
        if ml_score > 0.15:
            rules.append("ml_model_elevated")

        # Rule 6: Round-number amount (common in fraud)
        if tx.amount == int(tx.amount) and tx.amount >= 100:
            score += 0.06
            rules.append("round_amount")

        # Rule 7: Self-transfer
        if tx.from_account == tx.to_account:
            score += 0.25
            rules.append("self_transfer")

        return min(score, 0.75), rules  # Cap base score at 0.75

    def get_account_risk_profile(self, account_id: str) -> dict:
        state = self._state.get(account_id)
        if state is None or state.is_expired:
            return {
                "account_id": account_id,
                "accumulated_score": 0.0,
                "check_count": 0,
                "flag_count": 0,
                "state": "no_active_window",
            }
        return {
            "account_id": account_id,
            "accumulated_score": 0.0,  # no longer tracked; score is per-tx
            "check_count": state.check_count,
            "flag_count": state.flag_count,
            "state": "active",
            "expires_at": state.expires_at.isoformat(),
        }

    def reset_account(self, account_id: str):
        """Reset fraud state for an account (used by ops team manually)."""
        self._state.pop(account_id, None)
        logger.info(f"[Fraud] Account {account_id} fraud state reset")


# Singleton
fraud_detector = FraudDetectionAgent()
