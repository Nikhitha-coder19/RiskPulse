import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from behavioral_adapter import BehavioralAdapter
from merchant_adapter import MerchantAdapter
from traditional_adapter import TraditionalAdapter
from fusion_engine import FusionEngine
from runtime_state import RiskState
from ui.database import insert_risk_decision


class RiskEngine:
    """
    Runtime orchestration layer for RiskPulse.

    Runtime flow:

        TRANSACTION
             ↓
        MODEL SCORING
             ↓
          FUSION
             ↓
          ACTION

    Important:
        - Model artifacts are frozen.
        - Adapters do not update state.
        - Fusion does not update state.
        - State mutation happens only after the decision.
    """

    def __init__(self):
        self.traditional = TraditionalAdapter()
        self.behavioral = BehavioralAdapter()
        self.merchant = MerchantAdapter()
        self.fusion = FusionEngine()
        self.state = RiskState()

    # ============================================================
    # TRANSACTION VALIDATION
    # ============================================================

    @staticmethod
    def _validate_transaction(tx):
        required_fields = [
            "transaction_id",
            "customer_id",
            "merchant_id",
            "timestamp",
            "transaction_amount",
            "quantity",
            "payment_method",
            "product_category",
            "customer_age",
            "account_age_days",
            "customer_location",
            "device_id",
            "device_type",
            "ip_id",
            "shipping_address_id",
            "billing_address_id",
        ]

        missing = [
            field
            for field in required_fields
            if field not in tx
        ]

        if missing:
            raise ValueError(
                f"RiskEngine transaction missing fields: {missing}"
            )

    # ============================================================
    # SCORE TRANSACTION
    # ============================================================

    def score_transaction(self, tx):
        """
        Score one incoming transaction.

        IMPORTANT:
            This method does NOT update runtime state.

        Flow:
            validate
              ↓
            traditional
              ↓
            behavioral
              ↓
            merchant
              ↓
            fusion
              ↓
            decision
        """

        self._validate_transaction(tx)

        # --------------------------------------------------------
        # STEP 1 — SCORE THE THREE INDEPENDENT RISK LAYERS
        # --------------------------------------------------------

        traditional_result = self.traditional.score(tx)

        behavioral_result = self.behavioral.score(tx)

        merchant_result = self.merchant.score(tx)

        # --------------------------------------------------------
        # STEP 2 — FUSE MODEL PROBABILITIES
        # --------------------------------------------------------

        fusion_result = self.fusion.fuse(
            traditional_result,
            behavioral_result,
            merchant_result,
        )

        # --------------------------------------------------------
        # STEP 3 — RETURN COMPLETE RISK DECISION
        # --------------------------------------------------------

        return {
            "transaction_id": tx["transaction_id"],

            "model_scores": {
                "traditional": traditional_result,
                "behavioral": behavioral_result,
                "merchant": merchant_result,
            },

            "fusion": fusion_result,

            "decision": fusion_result["action"],
        }
    def process_transaction(self, tx, fraud_label=None):
        """
        Complete transaction lifecycle.

        Flow:

            READ STATE
                    ↓
            SCORE MODELS
                    ↓
            FUSION
                    ↓
            ACTION
                    ↓
            UPDATE STATE

        The transaction is added to runtime state only AFTER
        all model scoring and the final decision are complete.

        fraud_label:
            Optional confirmed/observed fraud label.
            This must NOT be RiskPulse's prediction.
        """

        self._validate_transaction(tx)

        # --------------------------------------------------------
        # SCORE BEFORE STATE UPDATE
        # --------------------------------------------------------

        result = self.score_transaction(tx)

        # --------------------------------------------------------
        # PERSIST DECISION BEFORE STATE UPDATE
        # --------------------------------------------------------

        fusion_result = result["fusion"]

        insert_risk_decision(
            transaction_id=tx["transaction_id"],
            final_risk_score=fusion_result["final_risk_score"],
            action=fusion_result["action"],
            traditional_probability=fusion_result[
                "traditional_probability"
            ],
            behavioral_probability=fusion_result[
                "behavioral_probability"
            ],
            merchant_probability=fusion_result[
                "merchant_probability"
            ],
            created_at=tx["timestamp"],
        )

        # --------------------------------------------------------
        # UPDATE STATE AFTER DECISION PERSISTENCE
        # --------------------------------------------------------

        self.state.update_after_transaction(
            tx,
            fraud_label=fraud_label,
        )

        # --------------------------------------------------------
        # STATE UPDATE CONFIRMATION
        # --------------------------------------------------------

        result["state_update"] = {
            "updated": True,
            "fraud_label_recorded": fraud_label is not None,
        }

        return result


# ================================================================
# RUNTIME SMOKE TEST
# ================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------
    # Simulated incoming transaction.
    #
    # This is test input only.
    # The probabilities are NOT hardcoded here.
    # They are generated by the three actual adapters.
    # ------------------------------------------------------------

    transaction = {
        "transaction_id": "runtime_demo_003",

        "customer_id": "runtime_customer_001",
        "merchant_id": "runtime_merchant_001",

        "timestamp": "2026-09-03T15:30:00",

        "transaction_amount": 2500.0,
        "quantity": 1,

        "payment_method": "credit card",
        "product_category": "electronics",

        "customer_age": 24,
        "account_age_days": 180,

        "customer_location": "Hyderabad",

        "device_id": "runtime_device_001",
        "device_type": "mobile",

        "ip_id": "runtime_ip_001",

        "shipping_address_id": "runtime_ship_001",
        "billing_address_id": "runtime_bill_001",
    }

    engine = RiskEngine()

    result = engine.process_transaction(transaction)

    print("RiskPulse runtime decision:")
    print(json.dumps(result, indent=2))