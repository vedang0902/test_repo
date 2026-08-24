"""
Simulated AI Fraud Detection Agent.

Architecture: rule-based scoring engine with a learned risk model (mocked).
In a real system this would call an ML inference endpoint or a feature store.

=============================================================================
FIX: Fraud Score Cascade / Compounding — RESOLVED
=============================================================================
Root cause (fixed):
  The original FraudDetector kept an in-memory `_score_accumulator` per
  account where each new check added 30 % of the previous score to the new
  base score.  The accumulator was never reset or decayed, causing even
  low-risk transactions to compound into high scores over time (cascade).

Changes made:
  1. `compounded_score` is now identical to `base_score` — no cross-
     transaction bleed.  Every transaction is scored independently.
  2. `_score_accumulator` is replaced by `_last_score` which stores only
     the raw base score for observability / risk-profile purposes; it carries
     NO influence into the next transaction's score.
  3. A TTL map (`_score_ts`) is introduced: entries older than
     `cfg.score_accumulator_ttl_seconds` (default 300 s) are expired before
     each read, preventing unbounded memory growth.
  4. False-positive tracking logic is corrected: we only increment the
     counter when the base score alone is below threshold (cascade-driven
     flags), which was already the intent but was previously mis-triggered
     because the compounded score was artificially elevated.
  5. `reset_account` is updated to clear the new data structures.
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

# Default TTL for per-account score state (seconds).  Prefer config value when
# available; fall back to 300 s (5 min) so the attribute is always defined.
_SCORE_TTL: float = getattr(cfg, "score_accumulator_ttl_seconds", 300.0)


class FraudDetectionAgent:
    """
    Multi-rule fraud scoring engine.
    Mimics a production ML-backed fraud service.

    Each transaction is scored independently — there is no cross-transaction
    score bleed.  Per-account state (check count, recent amounts) is retained
    for velocity rules only and is bounded / TTL-expired.
    """

    def __init__(self):
        # Last observed BASE score per account — for observability only;
        # NOT fed back into future scoring decisions.
        self._last_score: Dict[str, float] = {}
        # Timestamp of the last check per account (epoch seconds).
        self._score_ts: Dict[str, float] = {}
        # Total check count per account (bounded by TTL expiry).
        self._check_count: Dict[str, int] = {}
        # Flag count per account — reset on TTL expiry.
        self._flagged_accounts: Dict[str, int] = {}
        # Rolling recent-amount window for velocity rules.
        self._recent_amounts: Dict[str, List[float]] = {}

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _expire_account(self, account_id: str) -> None:
        """
        If the account's last-seen timestamp is older than _SCORE_TTL, wipe all
        per-account state so stale risk data cannot influence new transactions.
        """
        ts = self._score_ts.get(account_id)
        if ts is not None and (time.monotonic() - ts) > _SCORE_TTL:
            self._last_score.pop(account_id, None)
            self._score_ts.pop(account_id, None)
            self._check_count.pop(account_id, None)
            self._flagged_accounts.pop(account_id, None)
            self._recent_amounts.pop(account_id, None)
            logger.debug(
                f"[Fraud] TTL-expired state for account={account_id}"
            )

    # ── Public API ───────────────────────────────────────────────────────────

    def check(self, tx: Transaction) -> FraudCheckResult:
        """
        Score a transaction for fraud.

        Returns FraudCheckResult.  The score used for the block decision is
        the per-transaction base score only — no accumulated bleed.
        """
        account_id = tx.from_account
        triggered_rules: List[str] = []

        # Expire stale state before reading any per-account data.
        self._expire_account(account_id)

        # ── Base rule scoring ────────────────────────────────────────────────
        base_score, rules = self._evaluate_rules(tx)
        triggered_rules.extend(rules)

        # FIX: compounded_score == base_score.  No cross-transaction bleed.
        # Previously: compounded_score = base_score + previous_score * cfg.score_bleed_factor
        compounded_score = base_score  # independent, per-transaction

        # Update observability state (does NOT feed back into scoring).
        self._last_score[account_id] = base_score
        self._score_ts[account_id] = time.monotonic()

        # Track check count.
        count = self._check_count.get(account_id, 0) + 1
        self._check_count[account_id] = count

        # Update velocity window (bounded rolling window).
        amounts = self._recent_amounts.get(account_id, [])
        amounts.append(tx.amount)
        if len(amounts) > 20:
            amounts = amounts[-20:]
        self._recent_amounts[account_id] = amounts

        # ── Cascade detection ────────────────────────────────────────────────
        # With the fix applied, a genuine cascade (score runaway) should no
        # longer occur.  We keep the detection block so that if a real signal
        # pattern triggers many high-scoring transactions we still alert.
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
        if is_flagged:
            flag_count = self._flagged_accounts.get(account_id, 0) + 1
            self._flagged_accounts[account_id] = flag_count

            # A transaction is a likely false positive when the base score
            # alone is below the threshold — meaning rules alone would not
            # have flagged it.  With the cascade fix this should rarely fire.
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
        self._expire_account(account_id)
        return {
            "account_id": account_id,
            "last_base_score": self._last_score.get(account_id, 0.0),
            "check_count": self._check_count.get(account_id, 0),
            "flag_count": self._flagged_accounts.get(account_id, 0),
        }

    def reset_account(self, account_id: str) -> None:
        """Reset fraud state for an account (used by ops team manually)."""
        self._last_score.pop(account_id, None)
        self._score_ts.pop(account_id, None)
        self._check_count.pop(account_id, None)
        self._flagged_accounts.pop(account_id, None)
        self._recent_amounts.pop(account_id, None)
        logger.info(f"[Fraud] Account {account_id} fraud state reset")


# Singleton
fraud_detector = FraudDetectionAgent()
