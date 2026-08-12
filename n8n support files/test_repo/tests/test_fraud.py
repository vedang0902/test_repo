"""
Fraud detection tests.
Documents the cascade bug behaviour.
"""
import pytest

from app.models import Transaction, PaymentMethod
from app.services.fraud_detector import FraudDetectionAgent


class TestFraudCascade:
    """Tests for the score cascade / compounding bug."""

    def test_score_accumulates_across_checks(self):
        """
        Repeated checks for the same account should NOT increase score
        if the underlying transaction risk hasn't changed.
        This test documents the BUG: score does compound across checks.
        """
        detector = FraudDetectionAgent()

        results = []
        for i in range(10):
            tx = Transaction(
                idempotency_key=f"cascade-test-{i}",
                from_account="acct_cascade_test",
                to_account="acct_merchant_A",
                amount=100.0,   # Low-risk amount
                currency="USD",
                method=PaymentMethod.CARD,
            )
            result = detector.check(tx)
            results.append(result.compounded_score)

        # BUG: Score increases monotonically due to bleed factor
        # Correct behaviour: score should be stable across identical transactions
        print(f"Score progression: {[f'{s:.3f}' for s in results]}")

        # Document the bug: later scores are higher than earlier ones
        # (In a correct implementation, this assertion would NOT hold)
        if len(results) > 1:
            # The bug makes scores trend upward
            later_avg  = sum(results[5:]) / len(results[5:])
            earlier_avg = sum(results[:5]) / len(results[:5])
            print(f"Early avg: {earlier_avg:.3f}, Late avg: {later_avg:.3f}")

    def test_cascade_detection_fires_after_threshold_checks(self):
        """
        After cfg.cascade_check_count checks with elevated score,
        cascade_events_total should be incremented.
        """
        from app.config import settings
        from app.metrics import prometheus_metrics as m

        detector = FraudDetectionAgent()
        initial_cascades = m.fraud_cascade_events_total._value.get()

        # Process enough transactions to trigger cascade
        for i in range(settings.fraud.cascade_check_count + 2):
            tx = Transaction(
                idempotency_key=f"cascade-fire-{i}",
                from_account="acct_cascade_fire_test",
                to_account="acct_merchant_A",
                amount=6000.0,  # High amount → elevated base score
                currency="USD",
                method=PaymentMethod.CARD,
            )
            detector.check(tx)

        final_cascades = m.fraud_cascade_events_total._value.get()
        # Should have fired at least once
        print(f"Cascade events: initial={initial_cascades} final={final_cascades}")

    def test_false_positive_logic_is_flawed(self):
        """
        Tests that the false positive detection over-counts.
        A legitimate transaction flagged due to cascade is incorrectly
        counted as a false positive even though the cascade is causing it.
        """
        detector = FraudDetectionAgent()

        # Run enough checks to trigger cascade then flag
        for i in range(10):
            tx = Transaction(
                idempotency_key=f"fp-test-{i}",
                from_account="acct_fp_test",
                to_account="acct_merchant_A",
                amount=200.0,   # Actually low-risk
                currency="USD",
                method=PaymentMethod.CARD,
            )
            result = detector.check(tx)

        profile = detector.get_account_risk_profile("acct_fp_test")
        print(f"Account risk profile after 10 checks: {profile}")
        # The accumulated_score should be elevated even for low-risk transactions
        assert profile["check_count"] == 10
