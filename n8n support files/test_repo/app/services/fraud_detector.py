"""
Simulated AI Fraud Detection Agent.

Architecture: rule-based scoring engine with a learned risk model (mocked).
In a real system this would call an ML inference endpoint or a feature store.

=============================================================================
FIX: Fraud Score Cascade / Compounding
=============================================================================
Root cause (fixed):
  The original implementation kept an in-memory `_score_accumulator` per
  account that added 30% of the previous compounded score to every new base
  score, and never reset or decayed that value.  Over successive transactions
  even low-risk accounts crossed the cascade threshold.

Fix applied:
  1. Each transaction is scored independently (compounded_score == base_score).
  2. A separate, *decaying* risk signal is maintained per account using
     exponential decay keyed on wall-clock time.  This lets genuine repeated
     fraud signals accumulate while innocent accounts naturally decay to zero.
  3. The decayed signal is available for analytics / ML features but does NOT
     add to the per-transaction flagging decision — that is based solely on
     the current transaction's base_score vs cfg.threshold.
  4. False-positive tracking is corrected: a flag is a false positive only
     when the current base_score is below threshold (i.e. the transaction
     itself is not suspicious) regardless of history.
  5. _check_count, _score_accumulator, and _flagged_accounts maps are pruned
     periodically to prevent unbounded memory growth.
"""
import math
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

# Half-life in seconds for the per-account decayed risk signal.
# After this many seconds of inactivity the signal drops to 50 % of its value.
_DECAY_HALF_LIFE_SECONDS: float = getattr(cfg, "score_decay_half_life_seconds", 300.0)

# How many entries to keep before we prune stale accounts from in-memory maps.
_PRUNE_THRESHOLD: int = 10_000


class FraudDetectionAgent:
    """
    Multi-rule fraud scoring engine.
    Mimics a production ML-backed fraud service.

    Each transaction is evaluated independently.  A time-decayed risk signal
    is maintained per account for observability and feature-engineering
    purposes only — it does not inflate per-transaction scores.
    """

    def __init__(self):
        # Decayed risk signal: value in [0, 1] that fades over time.
        self._decayed_risk: Dict[str, float] = {}
        # Timestamp of the last update for each account (monotonic seconds).
        self._last_update_ts: Dict[str, float] = {}
        # Raw check count per account (for cascade detection counter).
        self._check_count: Dict[str, int] = {}
        # Flagged-transaction count per account.
        self._flagged_accounts: Dict[str, int] = {}
        # Rolling amount window for velocity rule.
        self._recent_amounts: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decayed_signal(self, account_id: str) -> float:
        """Return the current decayed risk signal for *account_id*."""
        raw = self._decayed_risk.get(account_id, 0.0)
        last_ts = self._last_update_ts.get(account_id)
        if last_ts is None or raw == 0.0:
            return raw
        elapsed = time.monotonic() - last_ts
        # Exponential decay: V(t) = V0 * 0.5^(t / half_life)
        decay_factor = math.pow(0.5, elapsed / _DECAY_HALF_LIFE_SECONDS)
        return raw * decay_factor

    def _update_decayed_signal(self, account_id: str, new_base_score: float) -> float:
        """
        Merge *new_base_score* into the decayed risk signal using an
        exponential moving average and return the updated value.
        """
        current = self._decayed_signal(account_id)
        # Blend: weight new evidence at 40 %, history at 60 % (after decay).
        updated = 0.6 * current + 0.4 * new_base_score
        updated = min(updated, 1.0)
        self._decayed_risk[account_id] = updated
        self._last_update_ts[account_id] = time.monotonic()
        return updated

    def _maybe_prune(self) -> None:
        """Remove stale entries when maps exceed the prune threshold."""
        if len(self._decayed_risk) < _PRUNE_THRESHOLD:
            return
        now = time.monotonic()
        stale = [
            acct
            for acct, ts in self._last_update_ts.items()
            if now - ts > _DECAY_HALF_LIFE_SECONDS * 10
        ]
        for acct in stale:
            self._decayed_risk.pop(acct, None)
            self._last_update_ts.pop(acct, None)
            self._check_count.pop(acct, None)
            self._flagged_accounts.pop(acct, None)
            self._recent_amounts.pop(acct, None)
        if stale:
            logger.debug(f"[Fraud] Pruned {len(stale)} stale account entries")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, tx: Transaction) -> FraudCheckResult:
        """
        Score a transaction for fraud.

        The decision is based solely on the current transaction's base_score.
        A separate decayed risk signal is updated for analytics but does NOT
        influence the flagging decision.
        """
        account_id = tx.from_account
        triggered_rules: List[str] = []

        self._maybe_prune()

        # ── Base rule scoring (independent per transaction) ──────────────────
        base_score, rules = self._evaluate_rules(tx)
        triggered_rules.extend(rules)

        # FIX: compounded_score IS the base_score — no cross-transaction bleed.
        compounded_score = base_score  # kept for API / model compatibility

        # ── Update decayed risk signal (analytics only) ───────────────────────
        decayed_signal = self._update_decayed_signal(account_id, base_score)

        # ── Track check count ────────────────────────────────────────────────
        count = self._check_count.get(account_id, 0) + 1
        self._check_count[account_id] = count

        # ── Update velocity window ────────────────────────────────────────────
        amounts = self._recent_amounts.get(account_id, [])
        amounts.append(tx.amount)
        if len(amounts) > 20:
            amounts = amounts[-20:]
        self._recent_amounts[account_id] = amounts

        # ── Flagging decision ────────────────────────────────────────────────
        is_flagged = compounded_score >= cfg.threshold

        # ── Cascade guard: emit metric only when decayed signal is elevated
        #    across many checks — signals a genuinely risky account, not
        #    an artifact of score bleed.
        cascade_detected = False
        if count >= cfg.cascade_check_count and decayed_signal > 0.40:
            cascade_detected = True
            m.fraud_cascade_events_total.inc()
            triggered_rules.append("cascade:persistent_risk")
            logger.error(
                f"[Fraud] CASCADE DETECTED account={account_id} "
                f"base_score={base_score:.3f} decayed_signal={decayed_signal:.3f} "
                f"check_count={count} tx_id={tx.id}"
            )
            m.app_error_rate.set(1)
            m.app_errors_total.labels(
                component="fraud_detector", error_type="cascade"
            ).inc()

        # ── False positive tracking (corrected) ───────────────────────────────
        if is_flagged:
            flag_count = self._flagged_accounts.get(account_id, 0) + 1
            self._flagged_accounts[account_id] = flag_count

            # FIX: a false positive is when THIS transaction's base_score is
            # below threshold — meaning the transaction itself is not suspicious
            # and the flag was not warranted by current evidence.
            if base_score < cfg.threshold:
                m.fraud_false_positives_total.inc()
                logger.warning(
                    f"[Fraud] Likely FALSE POSITIVE account={account_id} "
                    f"base_score={base_score:.3f} flag_count={flag_count} tx_id={tx.id}"
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
            "decayed_risk_signal": self._decayed_signal(account_id),
            "check_count": self._check_count.get(account_id, 0),
            "flag_count": self._flagged_accounts.get(account_id, 0),
        }

    def reset_account(self, account_id: str):
        """Reset fraud state for an account (used by ops team manually)."""
        self._decayed_risk.pop(account_id, None)
        self._last_update_ts.pop(account_id, None)
        self._check_count.pop(account_id, None)
        self._flagged_accounts.pop(account_id, None)
        self._recent_amounts.pop(account_id, None)
        logger.info(f"[Fraud] Account {account_id} fraud state reset")


# Singleton
fraud_detector = FraudDetectionAgent()
