"""
Simulated AI Fraud Detection Agent.

Architecture: rule-based scoring engine with a learned risk model (mocked).
In a real system this would call an ML inference endpoint or a feature store.

=============================================================================
FIX: Fraud Score Cascade / Compounding
=============================================================================
Root cause (fixed):
  The FraudDetector kept an in-memory `_score_accumulator` per account that
  added 30% of the previous compounded score to every new check, and was
  never reset. This caused legitimate accounts to accumulate scores over time
  until they were permanently flagged.

Fix applied:
  - Accumulator entries now carry a timestamp.
  - If the last update for an account is older than `cfg.score_decay_ttl_seconds`
    (default 900 s / 15 min) the accumulator and check-counter are treated as
    zero (cold start) before the new score is written.
  - `compounded_score` is still computed for the *current window* so genuine
    burst activity within 15 minutes is still caught, but cross-session bleed
    is eliminated.
  - False-positive tracking logic corrected: only increment when the *current
    window* flag count exceeds the threshold, not the all-time count.
"""
import random
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple

from app.config import settings
from app.models import Transaction, FraudCheckResult
from app.metrics import prometheus_metrics as m

logger = logging.getLogger("fraud_detector")

cfg = settings.fraud

# Default TTL if not present in config (seconds)
_DEFAULT_SCORE_DECAY_TTL = 900  # 15 minutes


def _decay_ttl() -> float:
    """Return the configured score decay TTL in seconds."""
    return float(getattr(cfg, "score_decay_ttl_seconds", _DEFAULT_SCORE_DECAY_TTL))


class _AccountState:
    """
    Holds per-account mutable fraud state with a last-updated timestamp.
    All fields are reset automatically when the TTL has elapsed since the
    last transaction, eliminating cross-session score compounding.
    """

    __slots__ = (
        "accumulated_score",
        "check_count",
        "flag_count",
        "last_updated",
    )

    def __init__(self):
        self.accumulated_score: float = 0.0
        self.check_count: int = 0
        self.flag_count: int = 0
        self.last_updated: float = 0.0  # epoch seconds

    def maybe_decay(self, ttl: float) -> None:
        """Reset state if TTL has elapsed since the last update."""
        if self.last_updated > 0 and (time.monotonic() - self.last_updated) >= ttl:
            self.accumulated_score = 0.0
            self.check_count = 0
            self.flag_count = 0
            self.last_updated = 0.0
            logger.debug("[Fraud] Score state decayed (TTL expired)")

    def touch(self) -> None:
        self.last_updated = time.monotonic()


class FraudDetectionAgent:
    """
    Multi-rule fraud scoring engine.
    Mimics a production ML-backed fraud service.

    Score compounding is bounded to a sliding TTL window so that stale
    accumulated scores do not bleed into future sessions.
    """

    def __init__(self):
        self._account_state: Dict[str, _AccountState] = {}
        self._recent_amounts: Dict[str, List[float]] = {}  # velocity check

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_state(self, account_id: str) -> _AccountState:
        """Return (and lazily create) the _AccountState for an account."""
        if account_id not in self._account_state:
            self._account_state[account_id] = _AccountState()
        return self._account_state[account_id]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, tx: Transaction) -> FraudCheckResult:
        """
        Score a transaction for fraud.

        Returns FraudCheckResult with base + compounded scores.
        Compounded score is what gets used for the block decision.

        Score compounding is limited to the current activity window
        (cfg.score_decay_ttl_seconds).  Once the window expires the
        accumulated score is zeroed before the new base score is stored,
        preventing indefinite runaway.
        """
        account_id = tx.from_account
        triggered_rules: List[str] = []
        ttl = _decay_ttl()

        state = self._get_state(account_id)

        # ── Decay stale state before doing anything else ─────────────────────
        state.maybe_decay(ttl)

        # ── Base rule scoring ────────────────────────────────────────────────
        base_score, rules = self._evaluate_rules(tx)
        triggered_rules.extend(rules)

        # ── Score compounding (window-bounded) ───────────────────────────────
        # Within the active window we still allow limited bleed so that a burst
        # of suspicious transactions in quick succession accumulates correctly.
        # Once the TTL expires (handled above) accumulated_score is 0.0, so
        # there is no cross-session compounding.
        previous_score = state.accumulated_score
        bleed_factor = float(getattr(cfg, "score_bleed_factor", 0.0))
        compounded_score = base_score + (previous_score * bleed_factor)
        compounded_score = min(compounded_score, 1.0)

        # Persist the updated score and increment counters
        state.accumulated_score = compounded_score
        state.check_count += 1
        state.touch()

        count = state.check_count

        # Update velocity window (bounded ring buffer)
        amounts = self._recent_amounts.get(account_id, [])
        amounts.append(tx.amount)
        if len(amounts) > 20:
            amounts = amounts[-20:]
        self._recent_amounts[account_id] = amounts

        # ── Cascade detection ────────────────────────────────────────────────
        is_flagged = compounded_score >= cfg.threshold
        cascade_detected = False

        if count >= cfg.cascade_check_count and compounded_score > 0.40:
            cascade_detected = True
            m.fraud_cascade_events_total.inc()
            triggered_rules.append("cascade:score_runaway")
            logger.error(
                f"[Fraud] CASCADE DETECTED account={account_id} "
                f"base_score={base_score:.3f} compounded={compounded_score:.3f} "
                f"check_count={count} tx_id={tx.id}"
            )
            m.app_error_rate.set(1)
            m.app_errors_total.labels(
                component="fraud_detector", error_type="cascade"
            ).inc()

        # ── False positive tracking (corrected logic) ─────────────────────────
        # Only use the *current window* flag count so that a decayed account
        # does not inherit stale flag counts from previous sessions.
        if is_flagged:
            state.flag_count += 1
            flag_count = state.flag_count

            # A false positive is when the window flag count exceeds the
            # repetition threshold but the current base score alone would not
            # trigger a flag — meaning the cascade (not genuine fraud) is
            # responsible for the block.
            if flag_count > 2 and base_score < cfg.threshold:
                m.fraud_false_positives_total.inc()
                logger.warning(
                    f"[Fraud] Likely FALSE POSITIVE account={account_id} "
                    f"base_score={base_score:.3f} compounded={compounded_score:.3f} "
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

    def get_account_risk_profile(self, account_id: str) -> dict:
        state = self._account_state.get(account_id)
        if state is None:
            return {
                "account_id": account_id,
                "accumulated_score": 0.0,
                "check_count": 0,
                "flag_count": 0,
                "last_updated": None,
            }
        return {
            "account_id": account_id,
            "accumulated_score": state.accumulated_score,
            "check_count": state.check_count,
            "flag_count": state.flag_count,
            "last_updated": state.last_updated or None,
        }

    def reset_account(self, account_id: str) -> None:
        """Reset fraud state for an account (used by ops team manually)."""
        self._account_state.pop(account_id, None)
        self._recent_amounts.pop(account_id, None)
        logger.info(f"[Fraud] Account {account_id} fraud state reset")


# Singleton
fraud_detector = FraudDetectionAgent()
