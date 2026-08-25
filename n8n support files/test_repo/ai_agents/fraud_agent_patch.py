"""
Simulated AI Fraud Detection Agent.

Architecture:
- Rule-based scoring engine
- Learned risk model (mocked)

In a real system this would call an ML inference endpoint
or a feature store.
"""

import random
import logging

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from app.config import settings
from app.models import Transaction, FraudCheckResult
from app.metrics import prometheus_metrics as m


logger = logging.getLogger("fraud_detector")

cfg = settings.fraud


# -------------------------------------------------------------------
# Accumulator configuration
# -------------------------------------------------------------------

# How long a per-account accumulator entry remains valid.
#
# After this window, the accumulated score is discarded
# and the account starts fresh.
#
# Default = 300 seconds = 5 minutes.

_ACCUMULATOR_TTL_SECONDS: int = getattr(
    cfg,
    "accumulator_ttl_seconds",
    300
)


# -------------------------------------------------------------------
# Accumulator Entry
# -------------------------------------------------------------------

class _AccumulatorEntry:
    """
    Stores a score together with the timestamp
    at which it was last updated.
    """

    __slots__ = ("score", "updated_at")

    def __init__(self, score: float):
        self.score: float = score
        self.updated_at: datetime = datetime.utcnow()

    def is_expired(self, ttl_seconds: int) -> bool:
        return (
            datetime.utcnow() - self.updated_at
        ) > timedelta(seconds=ttl_seconds)


# -------------------------------------------------------------------
# Fraud Detection Agent
# -------------------------------------------------------------------

class FraudDetectionAgent:
    """
    Multi-rule fraud scoring engine.

    Mimics a production ML-backed fraud service.

    Each transaction is scored independently.

    A short-lived accumulator is maintained per account
    only to support burst / velocity context within a
    narrow time window.

    The accumulator cannot compound indefinitely.
    """

    def __init__(self):

        # Accumulator now stores _AccumulatorEntry objects
        # so that entries can expire.

        self._score_accumulator: Dict[
            str,
            _AccumulatorEntry
        ] = {}

        self._check_count: Dict[str, int] = {}

        self._flagged_accounts: Dict[str, int] = {}

        self._recent_amounts: Dict[
            str,
            List[float]
        ] = {}


    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------

    def _get_previous_score(
        self,
        account_id: str
    ) -> float:

        """
        Return the stored score only if it is still
        within the TTL window.
        """

        entry: Optional[_AccumulatorEntry] = (
            self._score_accumulator.get(account_id)
        )

        if (
            entry is None
            or entry.is_expired(
                _ACCUMULATOR_TTL_SECONDS
            )
        ):

            # Entry is absent or stale.
            # Treat as a clean slate.

            self._score_accumulator.pop(
                account_id,
                None
            )

            return 0.0

        return entry.score


    def _update_accumulator(
        self,
        account_id: str,
        score: float
    ) -> None:

        """
        Overwrite the accumulator with the
        latest score.

        Do NOT add the new score to the old score.
        """

        self._score_accumulator[
            account_id
        ] = _AccumulatorEntry(score)


    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def check(
        self,
        tx: Transaction
    ) -> FraudCheckResult:

        """
        Score a transaction for fraud.

        The compounded score is now simply the
        base score for this transaction.

        The accumulator is retained only for
        short-window burst context and expires
        automatically.
        """

        account_id = tx.from_account

        triggered_rules: List[str] = []


        # ------------------------------------------------------------
        # Base rule scoring
        # ------------------------------------------------------------

        base_score, rules = self._evaluate_rules(tx)

        triggered_rules.extend(rules)


        # ------------------------------------------------------------
        # FIX: No cross-transaction score bleed
        # ------------------------------------------------------------

        # Each transaction is evaluated on its own merits.

        # We intentionally DO NOT add a fraction
        # of the previous score.

        _ = self._get_previous_score(
            account_id
        )

        compounded_score = min(
            base_score,
            1.0
        )


        # Store current score for the current window.

        self._update_accumulator(
            account_id,
            compounded_score
        )


        # ------------------------------------------------------------
        # Track check count
        # ------------------------------------------------------------

        count = (
            self._check_count.get(
                account_id,
                0
            )
            + 1
        )

        self._check_count[
            account_id
        ] = count


        # ------------------------------------------------------------
        # Update velocity window
        # ------------------------------------------------------------

        amounts = self._recent_amounts.get(
            account_id,
            []
        )

        amounts.append(tx.amount)

        if len(amounts) > 20:
            amounts = amounts[-20:]

        self._recent_amounts[
            account_id
        ] = amounts


        # ------------------------------------------------------------
        # Cascade detection
        # ------------------------------------------------------------

        is_flagged = (
            compounded_score >= cfg.threshold
        )

        cascade_detected = False


        if (
            count >= cfg.cascade_check_count
            and compounded_score > 0.40
        ):

            cascade_detected = True

            m.fraud_cascade_events_total.inc()

            triggered_rules.append(
                "cascade:score_runaway"
            )

            logger.error(
                f"[Fraud] CASCADE DETECTED "
                f"account={account_id} "
                f"base_score={base_score:.3f} "
                f"compounded={compounded_score:.3f} "
                f"check_count={count} "
                f"tx_id={tx.id}"
            )

            m.app_error_rate.set(1)

            m.app_errors_total.labels(
                component="fraud_detector",
                error_type="cascade"
            ).inc()


        # ------------------------------------------------------------
        # False positive tracking
        # ------------------------------------------------------------

        if is_flagged:

            flag_count = (
                self._flagged_accounts.get(
                    account_id,
                    0
                )
                + 1
            )

            self._flagged_accounts[
                account_id
            ] = flag_count


            # Only classify as false positive when:

            # 1. The account has been flagged more than
            #    cascade_check_count times

            # AND

            # 2. The current transaction's base score
            #    is below the threshold.

            if (
                flag_count > cfg.cascade_check_count
                and base_score < cfg.threshold
            ):

                m.fraud_false_positives_total.inc()

                logger.warning(
                    f"[Fraud] Likely FALSE POSITIVE "
                    f"account={account_id} "
                    f"base_score={base_score:.3f} "
                    f"compounded={compounded_score:.3f} "
                    f"flag_count={flag_count} "
                    f"tx_id={tx.id}"
                )


        # ------------------------------------------------------------
        # Prometheus metrics
        # ------------------------------------------------------------

        result_label = (
            "flagged"
            if is_flagged
            else "cleared"
        )

        m.fraud_checks_total.labels(
            result=result_label
        ).inc()

        m.fraud_score_histogram.observe(
            compounded_score
        )


        # ------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------

        if is_flagged:

            logger.warning(
                f"[Fraud] FLAGGED "
                f"tx={tx.id} "
                f"account={account_id} "
                f"score={compounded_score:.3f} "
                f"rules={triggered_rules}"
            )

        else:

            logger.debug(
                f"[Fraud] cleared "
                f"tx={tx.id} "
                f"account={account_id} "
                f"score={compounded_score:.3f}"
            )


        # ------------------------------------------------------------
        # Result
        # ------------------------------------------------------------

        return FraudCheckResult(
            transaction_id=tx.id,
            account_id=account_id,
            base_score=base_score,
            compounded_score=compounded_score,
            triggered_rules=triggered_rules,
            is_flagged=is_flagged,
        )


    # ----------------------------------------------------------------
    # Rule evaluation
    # ----------------------------------------------------------------

    def _evaluate_rules(
        self,
        tx: Transaction
    ) -> Tuple[float, List[str]]:

        """
        Apply deterministic + stochastic fraud rules.
        """

        score = 0.0

        rules: List[str] = []


        # ------------------------------------------------------------
        # Rule 1: High transaction amount
        # ------------------------------------------------------------

        if tx.amount > cfg.high_amount_threshold:

            score += 0.30

            rules.append(
                "high_amount"
            )

        elif (
            tx.amount
            > cfg.high_amount_threshold * 0.4
        ):

            score += 0.12

            rules.append(
                "medium_amount"
            )


        # ------------------------------------------------------------
        # Rule 2: Off-hours transaction
        # ------------------------------------------------------------

        hour = datetime.utcnow().hour

        if 4 <= hour < 6:

            score += 0.18

            rules.append(
                "off_hours"
            )


        # ------------------------------------------------------------
        # Rule 3: Velocity check
        # ------------------------------------------------------------

        recent = self._recent_amounts.get(
            tx.from_account,
            []
        )

        if len(recent) >= 8:

            score += 0.15

            rules.append(
                "high_velocity"
            )

        elif len(recent) >= 4:

            score += 0.07

            rules.append(
                "medium_velocity"
            )


        # ------------------------------------------------------------
        # Rule 4: Cross-currency transaction
        # ------------------------------------------------------------

        if tx.currency != "USD":

            score += 0.08

            rules.append(
                "cross_currency"
            )


        # ------------------------------------------------------------
        # Rule 5: Simulated ML model
        # ------------------------------------------------------------

        ml_score = random.gauss(
            0.08,
            0.04
        )

        ml_score = max(
            0.0,
            min(ml_score, 0.30)
        )

        score += ml_score

        if ml_score > 0.15:

            rules.append(
                "ml_model_elevated"
            )


        # ------------------------------------------------------------
        # Rule 6: Round-number amount
        # ------------------------------------------------------------

        if (
            tx.amount == int(tx.amount)
            and tx.amount >= 100
        ):

            score += 0.06

            rules.append(
                "round_amount"
            )


        # ------------------------------------------------------------
        # Rule 7: Self-transfer
        # ------------------------------------------------------------

        if (
            tx.from_account
            == tx.to_account
        ):

            score += 0.25

            rules.append(
                "self_transfer"
            )


        # Cap base score at 0.75

        return min(score, 0.75), rules


    # ----------------------------------------------------------------
    # Account risk profile
    # ----------------------------------------------------------------

    def get_account_risk_profile(
        self,
        account_id: str
    ) -> dict:

        entry: Optional[_AccumulatorEntry] = (
            self._score_accumulator.get(
                account_id
            )
        )

        accumulated = 0.0

        if (
            entry
            and not entry.is_expired(
                _ACCUMULATOR_TTL_SECONDS
            )
        ):

            accumulated = entry.score


        return {
            "account_id": account_id,
            "accumulated_score": accumulated,
            "check_count": self._check_count.get(
                account_id,
                0
            ),
            "flag_count": self._flagged_accounts.get(
                account_id,
                0
            ),
        }


    # ----------------------------------------------------------------
    # Manual reset
    # ----------------------------------------------------------------

    def reset_account(
        self,
        account_id: str
    ):

        """
        Reset fraud state for an account.
        Used manually by the operations team.
        """

        self._score_accumulator.pop(
            account_id,
            None
        )

        self._check_count.pop(
            account_id,
            None
        )

        self._flagged_accounts.pop(
            account_id,
            None
        )

        logger.info(
            f"[Fraud] Account {account_id} "
            f"fraud state reset"
        )


# -------------------------------------------------------------------
# Singleton
# -------------------------------------------------------------------

fraud_detector = FraudDetectionAgent()