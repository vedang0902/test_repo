"""
Simulated AI Fraud Detection Agent.

Architecture: rule-based scoring engine with a learned risk model (mocked).
In a real system this would call an ML inference endpoint or a feature store.

=============================================================================
FIX: Fraud Score Cascade / Compounding
=============================================================================
Previous root cause:
  The FraudDetector kept an in-memory `_score_accumulator` per account that
  was never reset or decayed. 30% of the previous compounded score bled into
  every new check, causing monotonically increasing scores and eventual
  cascade false-positives for legitimate accounts.

Fix applied:
  - Replaced raw accumulator with a time-stamped, exponentially-decayed
    score store (`_score_state`). Each entry records (score, timestamp).
  - On each check the stored score is decayed by `exp(-elapsed / half_life)`
    before being added to the base score. After `score_ttl_seconds` the
    stored score is treated as 0, fully isolating long-idle accounts.
  - `_check_count` is also scoped to the same TTL window to prevent the
    cascade detection counter from accumulating across unrelated sessions.
  - False-positive tracking logic corrected: only increment when the
    per-transaction base score alone is below threshold (not compounded).
"""
import random
import logging
import math
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

from app.config import settings
from app.models import Transaction, FraudCheckResult
from app.metrics import prometheus_metrics as m

logger = logging.getLogger("fraud_detector")

cfg = settings.fraud

# How long (seconds) before a stored score fully expires.
# Falls back to 300 s (5 min) if not present in config.
_SCORE_TTL_SECONDS: float = getattr(cfg, "score_ttl_seconds", 300.0)
# Half-life for exponential decay (seconds). Score halves every N seconds.
_DECAY_HALF_LIFE: float = getattr(cfg, "score_decay_half_life", 60.0)


class _AccountScoreState:
    """Holds the time-stamped, decayed fraud score for a single account."""

    __slots__ = ("score", "timestamp", "check_count", "window_start")

    def __init__(self):
        self.score: float = 0.0
        self.timestamp: float = 0.0  # epoch seconds
        self.check_count: int = 0
        self.window_start: float = 0.0  # epoch seconds of first check in window

    def decayed_score(self, now: float) -> float:
        """Return the stored score after exponential time-decay."""
        if self.score == 0.0 or self.timestamp == 0.0:
            return 0.0
        elapsed = now - self.timestamp
        if elapsed >= _SCORE_TTL_SECONDS:
            return 0.0
        # Exponential decay: S(t) = S0 * 2^(-elapsed / half_life)
        return self.score * math.pow(2.0, -elapsed / _DECAY_HALF_LIFE)

    def is_window_expired(self, now: float) -> bool:
        """True when the check-count window has exceeded the TTL."""
        if self.window_start == 0.0:
            return True
        return (now - self.window_start) >= _SCORE_TTL_SECONDS


class FraudDetectionAgent:
    """
    Multi-rule fraud scoring engine.
    Mimics a production ML-backed fraud service.

    Score state per account is time-decayed so that historical scores
    from unrelated sessions cannot compound into a cascade.
    """

    def __init__(self):
        # Keyed by account_id; values are _AccountScoreState objects.
        self._state: Dict[str, _AccountScoreState] = {}
        self._flagged_accounts: Dict[str, int] = {}  # account_id -> flag count
        self._recent_amounts: Dict[str, List[float]] = {}  # for velocity check

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_state(self, account_id: str) -> _AccountScoreState:
        if account_id not in self._state:
            self._state[account_id] = _AccountScoreState()
        return self._state[account_id]

    def _now(self) -> float:
        return datetime.now(timezone.utc).timestamp()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, tx: Transaction) -> FraudCheckResult:
        """
        Score a transaction for fraud.

        Returns FraudCheckResult with base + compounded scores.
        Compounded score is what gets used for the block decision.
        """
        account_id = tx.from_account
        triggered_rules: List[str] = []
        now = self._now()

        # ── Base rule scoring ────────────────────────────────────────────────
        base_score, rules = self._evaluate_rules(tx)
        triggered_rules.extend(rules)

        # ── Time-decayed score compounding ───────────────────────────────────
        state = self._get_state(account_id)
        decayed_previous = state.decayed_score(now)

        # Blend: current base score + decayed residual from recent activity.
        # Because the residual decays to zero over _SCORE_TTL_SECONDS, old
        # sessions cannot permanently inflate future scores.
        bleed_factor = getattr(cfg, "score_bleed_factor", 0.30)
        compounded_score = base_score + (decayed_previous * bleed_factor)
        compounded_score = min(compounded_score, 1.0)

        # Persist updated state
        state.score = compounded_score
        state.timestamp = now

        # ── Check count (TTL-scoped) ─────────────────────────────────────────
        if state.is_window_expired(now):
            # Start a fresh counting window
            state.check_count = 1
            state.window_start = now
        else:
            state.check_count += 1
        count = state.check_count

        # ── Velocity window ──────────────────────────────────────────────────
        amounts = self._recent_amounts.get(account_id, [])
        amounts.append(tx.amount)
        if len(amounts) > 20:
            amounts = amounts[-20:]
        self._recent_amounts[account_id] = amounts

        # ── Cascade detection ────────────────────────────────────────────────
        cascade_check_count = getattr(cfg, "cascade_check_count", 10)
        is_flagged = compounded_score >= cfg.threshold
        cascade_detected = False

        if count >= cascade_check_count and compounded_score > 0.40:
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

        # ── False positive tracking ───────────────────────────────────────────
        # A flag is a *likely* false positive only when the isolated base score
        # (no cross-transaction bleed) is below threshold. We no longer use the
        # compounded score here because the cascade itself used to cause the
        # inflation that was then incorrectly attributed to a false positive.
        if is_flagged:
            flag_count = self._flagged_accounts.get(account_id, 0) + 1
            self._flagged_accounts[account_id] = flag_count

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

        # Rule 2: Off-hours transaction (04:00-06:00 UTC is high-risk window)
        hour = datetime.now(timezone.utc).hour
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
        state = self._state.get(account_id)
        now = self._now()
        return {
            "account_id": account_id,
            "accumulated_score": state.decayed_score(now) if state else 0.0,
            "raw_stored_score": state.score if state else 0.0,
            "score_age_seconds": (now - state.timestamp) if (state and state.timestamp) else None,
            "check_count_in_window": state.check_count if state else 0,
            "flag_count": self._flagged_accounts.get(account_id, 0),
        }

    def reset_account(self, account_id: str):
        """Reset fraud state for an account (used by ops team manually)."""
        self._state.pop(account_id, None)
        self._flagged_accounts.pop(account_id, None)
        self._recent_amounts.pop(account_id, None)
        logger.info(f"[Fraud] Account {account_id} fraud state reset")


# Singleton
fraud_detector = FraudDetectionAgent()
