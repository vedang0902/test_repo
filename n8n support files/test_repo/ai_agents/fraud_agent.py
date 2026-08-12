"""
Fraud Detection Agent — Configuration & Orchestration Layer.

In a production system this module would:
  - Load a trained ML model from a model registry (MLflow, SageMaker)
  - Serve inference via a feature store (Feast, Tecton)
  - Emit predictions to a decision bus (Kafka topic: fraud.decisions)
  - Log explainability traces (SHAP values) to an observability backend

Here we simulate all of that with a rule-based engine + Gaussian noise.
The configuration below mirrors how a real fraud ML system is parameterised.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("fraud_agent")


class FraudAgentOrchestrator:
    """
    Top-level orchestrator for the fraud detection subsystem.

    Responsibilities:
      - Load agent configuration from config.json
      - Provide a unified interface for calling the detection engine
      - (In production) manage model versioning and A/B testing
      - Emit decisions to downstream systems
    """

    CONFIG_PATH = Path(__file__).parent / "config.json"

    def __init__(self):
        self.config = self._load_config()
        self.model_version = self.config.get("model_version", "v2.4-rules")
        self.feature_store = self.config.get("feature_store", "mock")
        self.decision_threshold = self.config["scoring"]["block_threshold"]
        self.review_threshold = self.config["scoring"]["review_threshold"]
        logger.info(
            f"[FraudAgent] Initialised: model={self.model_version} "
            f"threshold={self.decision_threshold} "
            f"feature_store={self.feature_store}"
        )

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.CONFIG_PATH) as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("[FraudAgent] config.json not found — using defaults")
            return self._default_config()

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        return {
            "model_version": "v2.4-rules-fallback",
            "feature_store": "mock",
            "scoring": {
                "block_threshold": 0.65,
                "review_threshold": 0.45,
                "score_bleed_factor": 0.30,
            },
            "rules": {
                "high_amount_threshold": 5000,
                "velocity_window_minutes": 60,
                "off_hours_start": 4,
                "off_hours_end": 6,
            },
        }

    def get_decision(self, score: float) -> str:
        """Map a fraud score to a routing decision."""
        if score >= self.decision_threshold:
            return "block"
        if score >= self.review_threshold:
            return "review"
        return "allow"

    def explain_score(self, score: float, rules: list) -> dict:
        """
        Produce a human-readable explanation of a fraud score.
        Mimics SHAP-style feature attribution.
        """
        return {
            "score": score,
            "decision": self.get_decision(score),
            "model_version": self.model_version,
            "triggered_rules": rules,
            "explanation": {
                rule: f"Contributed to elevated risk score ({rule})"
                for rule in rules
            },
            "confidence": min(score * 1.2, 1.0),
        }


# Singleton
fraud_agent_orchestrator = FraudAgentOrchestrator()
