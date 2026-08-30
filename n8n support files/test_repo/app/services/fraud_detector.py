"""
Simulated AI Fraud Detection Agent.

Architecture: rule-based scoring engine with a learned risk model (mocked).
In a real system this would call an ML inference endpoint or a feature store.

=============================================================================
FIX: Fraud Score Cascade / Compounding — RESOLVED
=============================================================================
Root cause (fixed):
  The FraudDetector was keeping an in-memory `_score_accumulator` per account
  and bleeding 30% of the previous compounded score into every new check.
  The accumulator was never reset or decayed, so even low-risk transactions
  compounded into high scores over time, causing a cascade.

Fix applied:
  - `compounded_score` is now equal to `base_score` (per-transaction,
    no cross-transaction bleed).
  - `_score_accumulator` is retained only for the ops risk-profile endpoint
    but is updated to store the raw base_score and is NOT fed back into
    future scoring decisions.
  - `cfg.score_bleed_factor` is no longer used in the scoring path.

Symptoms resolved:
  fraud_cascade_events_total    ↓
  fraud_false_positives_total   ↓
  fraud_score_distribution      returns to expected distribution
"""
import random
import logging
from datetime import datetime
from typing import Dict, List, Tuple

from app.config import settings
from app.models import Transaction, FraudCheckResult
from app.metrics import prometheus_metrics as m

logger = logging.getLogger("fraud_detector")

cfg = settings.fraud


class FraudDetectionAgent:
    """
    Multi-rule fraud scoring engine.
    Mimics a production ML-backed fraud service.

    Each transaction is scored independently; there is no cross-transaction
    score bleed.  The _score_accumulator is kept solely for the ops
    risk-profile endpoint and is not fed back into scoring.
    """

    def __init__(self):
        # Stores last observed base_score per account — read-only for ops;
        # NOT fed back into future scoring decisions.
        self._score_accumulator: Dict[str, float] = {}
        self._check_count: Dict[str, int] = {}
        self._flagged_accounts: Dict[str, int] = {}  # account_id → flag count
        self._recent_amounts: Dict[str, List[float]] = {}  # for velocity check

    def check(self, tx: Transaction) -> FraudCheckResult:
        """
        Score a transaction for fraud.

        Each transaction is scored independently.  The final score equals the
        base score produced by _evaluate_rules with no bleed from prior checks.
        """
        account_id = tx.from_account
        triggered_rules: List[str] = []

        # ── Base rule scoring ────────────────────────────────────────────────
        base_score, rules = self._evaluate_rules(tx)
        triggered_rules.extend(rules)

        # ── FIX: Score is per-transaction only; no cross-transaction bleed ───
        # Previously: compounded_score = base_score + (previous_score * cfg.score_bleed_factor)
        # Now:        compounded_score = base_score  (independent per transaction)
        compounded_score = min(base_score, 1.0)

        # Update accumulator for ops visibility only (not used in scoring)
        self._score_accumulator[account_id] = compounded_score

        # Track check count
        count = self._check_count.get(account_id, 0) + 1
        self._check_count[account_id] = count

        # Update velocity window
        amounts = self._recent_amounts.get(account_id, [])
        amounts.append(tx.amount)
        if len(amounts) > 20:
            amounts = amounts[-20:]  # Rolling window — bounded
        self._recent_amounts[account_id] = amounts

        # ── Cascade detection ────────────────────────────────────────────────
        # With the bleed removed this path should no longer fire in normal
        # operation; it is retained as a safety net for genuine rule spikes.
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

        # ── False positive tracking ───────────────────────────────────────────
        if is_flagged:
            flag_count = self._flagged_accounts.get(account_id, 0) + 1
            self._flagged_accounts[account_id] = flag_count

            # A flagged transaction whose base score alone is below threshold
            # suggests the rule set is producing a false positive.
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
        return {
            "account_id": account_id,
            "accumulated_score": self._score_accumulator.get(account_id, 0.0),
            "check_count": self._check_count.get(account_id, 0),
            "flag_count": self._flagged_accounts.get(account_id, 0),
        }

    def reset_account(self, account_id: str):
        """Reset fraud state for an account (used by ops team manually)."""
        self._score_accumulator.pop(account_id, None)
        self._check_count.pop(account_id, None)
        self._flagged_accounts.pop(account_id, None)
        logger.info(f"[Fraud] Account {account_id} fraud state reset")


# Singleton
fraud_detector = FraudDetectionAgent()
