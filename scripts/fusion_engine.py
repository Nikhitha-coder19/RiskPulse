import json


# ============================================================
# RISK FUSION CONFIGURATION
# ============================================================

TRADITIONAL_WEIGHT = 0.40
BEHAVIORAL_WEIGHT = 0.35
MERCHANT_WEIGHT = 0.25

ACTION_THRESHOLDS = {
    "allow": 0.30,
    "review": 0.60,
    "challenge": 0.85,
}


class FusionEngine:
    """
    Deterministic RiskPulse fusion engine.

    Responsibilities:
        1. Accept the three frozen model outputs.
        2. Validate their probabilities.
        3. Calculate the weighted final risk score.
        4. Convert the score into an operational action.
        5. Provide transparent model-level explainability.

    This layer:
        - performs NO machine learning
        - performs NO training
        - does NOT modify model artifacts
        - does NOT update runtime state
    """

    def __init__(self):
        self.weights = {
            "traditional": TRADITIONAL_WEIGHT,
            "behavioral": BEHAVIORAL_WEIGHT,
            "merchant": MERCHANT_WEIGHT,
        }

        weight_sum = sum(self.weights.values())

        if abs(weight_sum - 1.0) > 1e-9:
            raise ValueError(
                f"Fusion weights must sum to 1.0, got {weight_sum}"
            )

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _validate_model_result(result, expected_model):
        if not isinstance(result, dict):
            raise TypeError(
                f"{expected_model} result must be a dictionary."
            )

        if result.get("model") != expected_model:
            raise ValueError(
                f"Expected model '{expected_model}', "
                f"got '{result.get('model')}'."
            )

        if "fraud_probability" not in result:
            raise ValueError(
                f"{expected_model} result is missing "
                "'fraud_probability'."
            )

        probability = result["fraud_probability"]

        try:
            probability = float(probability)
        except (TypeError, ValueError):
            raise ValueError(
                f"{expected_model} fraud probability must be numeric."
            )

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"{expected_model} fraud probability must be "
                f"between 0 and 1, got {probability}."
            )

        return probability

    # ========================================================
    # ACTION
    # ========================================================

    @staticmethod
    def _determine_action(risk_score):
        if risk_score < ACTION_THRESHOLDS["allow"]:
            return "ALLOW"

        if risk_score < ACTION_THRESHOLDS["review"]:
            return "REVIEW"

        if risk_score < ACTION_THRESHOLDS["challenge"]:
            return "CHALLENGE"

        return "BLOCK"

    # ========================================================
    # EXPLAINABILITY
    # ========================================================

    @staticmethod
    def _build_reasons(
        traditional_probability,
        behavioral_probability,
        merchant_probability,
        final_score,
        action,
    ):
        probabilities = {
            "traditional": traditional_probability,
            "behavioral": behavioral_probability,
            "merchant": merchant_probability,
        }

        strongest_model = max(
            probabilities,
            key=probabilities.get,
        )

        strongest_probability = probabilities[strongest_model]

        reasons = []

        # Overall risk level
        if final_score >= ACTION_THRESHOLDS["challenge"]:
            reasons.append(
                "Combined model evidence indicates very high fraud risk."
            )
        elif final_score >= ACTION_THRESHOLDS["review"]:
            reasons.append(
                "Combined model evidence indicates elevated fraud risk."
            )
        elif final_score >= ACTION_THRESHOLDS["allow"]:
            reasons.append(
                "Combined model evidence indicates moderate fraud risk."
            )
        else:
            reasons.append(
                "Combined model evidence indicates low fraud risk."
            )

        # Strongest model signal
        reasons.append(
            f"The strongest model-level signal came from the "
            f"{strongest_model} model "
            f"(probability={strongest_probability:.3f})."
        )

        # Behavioral signal
        if behavioral_probability >= 0.60:
            reasons.append(
                "Behavioral risk is elevated, indicating suspicious "
                "transaction-history or coordinated-activity signals."
            )

        # Merchant signal
        if merchant_probability >= 0.60:
            reasons.append(
                "Merchant risk is elevated based on merchant-level "
                "historical and transaction-pattern signals."
            )

        # Traditional signal
        if traditional_probability >= 0.60:
            reasons.append(
                "Traditional transaction-level fraud risk is elevated."
            )

        # Explicit action explanation
        if action == "BLOCK":
            reasons.append(
                "The final score exceeds the blocking threshold."
            )
        elif action == "CHALLENGE":
            reasons.append(
                "The final score falls in the challenge range."
            )
        elif action == "REVIEW":
            reasons.append(
                "The final score falls in the manual-review range."
            )
        else:
            reasons.append(
                "The final score remains below the review threshold."
            )

        return reasons

    # ========================================================
    # FUSION
    # ========================================================

    def fuse(
        self,
        traditional_result,
        behavioral_result,
        merchant_result,
    ):
        """
        Combine the three frozen model outputs.

        Returns:
            {
                "traditional_probability": float,
                "behavioral_probability": float,
                "merchant_probability": float,
                "final_risk_score": float,
                "action": str,
                "weights": dict,
                "reasons": list
            }
        """

        traditional_probability = self._validate_model_result(
            traditional_result,
            "traditional",
        )

        behavioral_probability = self._validate_model_result(
            behavioral_result,
            "behavioral",
        )

        merchant_probability = self._validate_model_result(
            merchant_result,
            "merchant",
        )

        # ----------------------------------------------------
        # Weighted deterministic fusion
        # ----------------------------------------------------

        final_score = (
            TRADITIONAL_WEIGHT * traditional_probability
            + BEHAVIORAL_WEIGHT * behavioral_probability
            + MERCHANT_WEIGHT * merchant_probability
        )

        # Numerical safety
        final_score = max(
            0.0,
            min(1.0, float(final_score)),
        )

        action = self._determine_action(final_score)

        reasons = self._build_reasons(
            traditional_probability,
            behavioral_probability,
            merchant_probability,
            final_score,
            action,
        )

        return {
            "traditional_probability": traditional_probability,
            "behavioral_probability": behavioral_probability,
            "merchant_probability": merchant_probability,
            "final_risk_score": final_score,
            "action": action,
            "weights": dict(self.weights),
            "action_thresholds": dict(ACTION_THRESHOLDS),
            "reasons": reasons,
        }


# ============================================================
# SMOKE TEST
# ============================================================

if __name__ == "__main__":

    # These are the actual probabilities returned by the
    # three frozen adapter smoke tests.
    traditional_result = {
        "model": "traditional",
        "fraud_probability": 0.50,
    }

    behavioral_result = {
        "model": "behavioral",
        "fraud_probability": 0.5133333333333333,
    }

    merchant_result = {
        "model": "merchant",
        "fraud_probability": 0.9647348387667181,
    }

    engine = FusionEngine()

    result = engine.fuse(
        traditional_result,
        behavioral_result,
        merchant_result,
    )

    print("RiskPulse fusion result:")
    print(json.dumps(result, indent=2))