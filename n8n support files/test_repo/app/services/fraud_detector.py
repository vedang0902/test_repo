"""
Simulated AI Fraud Detection Agent.

Architecture: rule-based scoring engine with a learned risk model (mocked).
In a real system this would call an ML inference endpoint or a feature store.
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

# How long a score entry is considered valid before it is discarded.
# Transactions older than this window no longer influence future checks.
_ACCUMULATOR_TTL_MINUTES: int = 30


class _ScoreEntry:
    """Holds a score value together with the timestamp it was recorded."""

    __slots__ = ("score", "recorded_at")

    def __init__(self, score: float) -> None:
        self.score: float = score
        self.recorded_at: datetime = datetime.utcnow()

    def is_expired(self, ttl_minutes: int = _ACCUMULATOR_TTL_MINUTES) -> bool:
        return datetime.utcnow() - self.recorded_at > timedelta(minutes=ttl_minutes)


class FraudDetectionAgent:
    """
    Multi-rule fraud scoring engine.
    Mimics a production ML-backed fraud service.

    Design principles (post-fix):
      - Each transaction is scored independently; no score bleeds between checks.
      - The accumulator stores the *most recent* score only for cascade-rate
        detection (not for compounding) and expires after TTL minutes.
      - False-positive tracking fires only when base_score < threshold AND
        the cascade rule was NOT triggered by the current check.
    """

    def __init__(self) -> None:
        # Stores the last scored entry per account for cascade-rate detection.
        # Entries expire after _ACCUMULATOR_TTL_MINUTES and are NOT used to
        # inflate future scores.
        self._last_score: Dict[str, _ScoreEntry] = {}
        self._check_count: Dict[str, int] = {}
        self._flagged_accounts: Dict[str, int] = {}
        self._recent_amounts: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, tx: Transaction) -> FraudCheckResult:
        """
        Score a transaction for fraud.

        Each transaction is evaluated on its own merits; prior transaction
        scores are NOT added to the current score (no bleed / cascade).
        """
        account_id = tx.from_account
        triggered_rules: List[str] = []

        # ── Base rule scoring ────────────────────────────────────────────────
        base_score, rules = self._evaluate_rules(tx)
        triggered_rules.extend(rules)

        # ── FIX: Per-transaction independent score ───────────────────────────
        # compounded_score is now identical to base_score.
        # We intentionally do NOT add any fraction of the previous score so
        # that a single mildly-elevated transaction cannot permanently raise
        # an account's risk baseline.
        compounded_score = min(base_score, 1.0)

        # Persist the score with a timestamp so it can expire.
        self._last_score[account_id] = _ScoreEntry(compounded_score)

        # Track check count
        count = self._check_count.get(account_id, 0) + 1
        self._check_count[account_id] = count

        # Update velocity window (bounded rolling window — unchanged)
        amounts = self._recent_amounts.get(account_id, [])
        amounts.append(tx.amount)
        if len(amounts) > 20:
            amounts = amounts[-20:]
        self._recent_amounts[account_id] = amounts

        # ── Cascade detection ────────────────────────────────────────────────
        # A cascade is now defined as: the account has been checked many times
        # AND the *current independent* score is genuinely high — meaning the
        # transaction itself looks suspicious, not just because of score bleed.
        cascade_detected = False
        if count >= cfg.cascade_check_count and compounded_score > cfg.threshold:
            cascade_detected = True
            m.fraud_cascade_events_total.inc()
            triggered_rules.append("cascade:repeated_high_score")
            logger.error(
                f"[Fraud] CASCADE DETECTED account={account_id} "
                f"base_score={base_score:.3f} score={compounded_score:.3f} "
                f"check_count={count} tx_id={tx.id}"
            )
            m.app_error_rate.set(1)
            m.app_errors_total.labels(
                component="fraud_detector", error_type="cascade"
            ).inc()

        # ── Flag decision ────────────────────────────────────────────────────
        is_flagged = compounded_score >= cfg.threshold

        # ── False positive tracking (fixed logic) ────────────────────────────
        # A false positive is recorded only when:
        #   1. The transaction was flagged.
        #   2. The base score alone is below the threshold (rules didn't truly
        #      warrant a flag — previously this was caused by score bleed).
        #   3. The cascade rule was NOT triggered (cascade = genuinely repeated
        #      suspicious behaviour and is not a false positive by definition).
        if is_flagged:
            flag_count = self._flagged_accounts.get(account_id, 0) + 1
            self._flagged_accounts[account_id] = flag_count

            if base_score < cfg.threshold and not cascade_detected:
                m.fraud_false_positives_total.inc()
                logger.warning(
                    f"[Fraud] Likely FALSE POSITIVE account={account_id} "
                    f"base_score={base_score:.3f} score={compounded_score:.3f} "
                    f"flag_count={flag_count} tx_id={tx.id}"
                )

        # ── Prometheus metrics ────────────────────────────────────────────────
        result_label = "flagged" if is_flagged else "cleared"
        m.fraud_checks_total.labels(result=result_label).inc()
        m.fraud_score_histogram.observe(compounded_score)

        if is_flagged:
            logger.warning(
                f"[Fraud] FLAGGED tx={tx.id} account={account_id} "
                f"score={compounded_score:.3f} rules={triggered_rules}"
            )
        else:
            logger.debug(
                f"[Fraud] cleared tx={tx.id} account={account_id} "
                f"score={compounded_score:.3f}"
            )

        return FraudCheckResult(
            transaction_id=tx.id,
            account_id=account_id,
            base_score=base_score,
            compounded_score=compounded_score,
            triggered_rules=triggered_rules,
            is_flagged=is_flagged,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_rules(self, tx: Transaction) -> Tuple[float, List[str]]:
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
        recent = self._recent_amounts.get(tx.from_account, [])
        if len(recent) >= 8:
            score += 0.15
            rules.append("high_velocity")
        elif len(recent) >= 4:
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

    def _evict_expired_scores(self) -> None:
        """Remove stale accumulator entries to prevent unbounded memory growth."""
        expired = [
            acct
            for acct, entry in self._last_score.items()
            if entry.is_expired()
        ]
        for acct in expired:
            self._last_score.pop(acct, None)

    # ------------------------------------------------------------------
    # Ops / introspection
    # ------------------------------------------------------------------

    def get_account_risk_profile(self, account_id: str) -> dict:
        entry: Optional[_ScoreEntry] = self._last_score.get(account_id)
        last_score = 0.0 if entry is None or entry.is_expired() else entry.score
        return {
            "account_id": account_id,
            "last_score": last_score,
            "check_count": self._check_count.get(account_id, 0),
            "flag_count": self._flagged_accounts.get(account_id, 0),
        }

    def reset_account(self, account_id: str) -> None:
        """Reset fraud state for an account (used by ops team manually)."""
        self._last_score.pop(account_id, None)
        self._check_count.pop(account_id, None)
        self._flagged_accounts.pop(account_id, None)
        self._recent_amounts.pop(account_id, None)
        logger.info(f"[Fraud] Account {account_id} fraud state reset")


# Singleton
fraud_detector = FraudDetectionAgent()
