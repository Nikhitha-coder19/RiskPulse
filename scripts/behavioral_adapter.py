import json
from pathlib import Path

import joblib
import pandas as pd

from runtime_state import RiskState


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"


class BehavioralAdapter:

    def __init__(self):

        self.model_path = MODEL_DIR / "behavioral_random_forest.pkl"
        self.imputer_path = MODEL_DIR / "behavioral_imputer.pkl"
        self.results_path = MODEL_DIR / "behavioral_final_results.json"

        # ---------------------------------------------------------
        # Load artifacts
        # ---------------------------------------------------------

        self.model = joblib.load(self.model_path)
        self.imputer = joblib.load(self.imputer_path)

        with open(self.results_path, "r", encoding="utf-8") as f:
            self.results = json.load(f)

        # ---------------------------------------------------------
        # Feature contract
        # ---------------------------------------------------------

        self.features = self.results["features"]
        self.threshold = float(self.results["selected_threshold"])

        if len(self.features) != 26:
            raise ValueError(
                f"Expected 26 behavioral features, got {len(self.features)}"
            )

        # ---------------------------------------------------------
        # Runtime state
        # ---------------------------------------------------------

        self.state = RiskState()

    # =============================================================
    # HELPERS
    # =============================================================

    @staticmethod
    def _hours_since(previous_timestamp, current_timestamp):

        if not previous_timestamp:
            return None

        previous = pd.Timestamp(previous_timestamp)
        current = pd.Timestamp(current_timestamp)

        return (current - previous).total_seconds() / 3600.0

    # =============================================================
    # FEATURE CONSTRUCTION
    # =============================================================

    def build_features(self, tx):

        timestamp = tx["timestamp"]

        customer_id = tx.get("customer_id")
        merchant_id = tx.get("merchant_id")
        device_id = tx.get("device_id")
        ip_id = tx.get("ip_id")

        shipping_address_id = tx.get("shipping_address_id")
        billing_address_id = tx.get("billing_address_id")

        amount = float(tx["transaction_amount"])
        quantity = int(tx["quantity"])
        customer_age = float(tx["customer_age"])
        account_age_days = float(tx["account_age_days"])

        # ---------------------------------------------------------
        # READ PRIOR CUSTOMER STATE
        # ---------------------------------------------------------

        customer = self.state.get_customer_state(customer_id)

        customer_count = customer["transaction_count"]
        customer_avg = customer["avg_amount"]
        customer_max = customer["max_amount"]

        customer_amount_deviation = (
            amount - customer_avg
            if customer_count > 0
            else 0.0
        )

        customer_24h = self.state.get_transaction_count_since(
            "customer_id",
            customer_id,
            timestamp,
            24,
        )

        customer_7d = self.state.get_transaction_count_since(
            "customer_id",
            customer_id,
            timestamp,
            24 * 7,
        )

        hours_since_customer_previous = self._hours_since(
            customer["previous_timestamp"],
            timestamp,
        )

        # ---------------------------------------------------------
        # READ PRIOR MERCHANT STATE
        # ---------------------------------------------------------

        merchant = self.state.get_merchant_state(merchant_id)

        merchant_count = merchant["transaction_count"]
        merchant_avg = merchant["avg_amount"]

        merchant_24h = self.state.get_transaction_count_since(
            "merchant_id",
            merchant_id,
            timestamp,
            24,
        )

        # ---------------------------------------------------------
        # DEVICE / IP HISTORY
        # ---------------------------------------------------------

        device_count = self.state.get_related_count(
            "device",
            device_id,
            "customer",
        )

        # Transaction count is required here, not unique customers.
        # entity_relationships stores unique relationships, so use
        # transaction history for the actual count.
        device_transaction_count = self.state.get_transaction_count_since(
            "device_id",
            device_id,
            timestamp,
            24 * 365 * 100,
        )

        ip_transaction_count = self.state.get_transaction_count_since(
            "ip_id",
            ip_id,
            timestamp,
            24 * 365 * 100,
        )

        ip_unique_customers = self.state.get_related_count(
            "ip",
            ip_id,
            "customer",
        )

        ip_unique_merchants = self.state.get_related_count(
            "ip",
            ip_id,
            "merchant",
        )

        # ---------------------------------------------------------
        # ADDRESS HISTORY
        # ---------------------------------------------------------

        shipping_transaction_count = (
            self.state.get_address_transaction_count_before(
                "shipping_address_id",
                shipping_address_id,
                timestamp,
            )
        )

        billing_transaction_count = (
            self.state.get_address_transaction_count_before(
                "billing_address_id",
                billing_address_id,
                timestamp,
            )
        )

        shipping_unique_customers = (
            self.state.get_address_unique_count_before(
                "shipping_address_id",
                shipping_address_id,
                "customer_id",
                timestamp,
            )
        )

        billing_unique_customers = (
            self.state.get_address_unique_count_before(
                "billing_address_id",
                billing_address_id,
                "customer_id",
                timestamp,
            )
        )

        shipping_unique_merchants = (
            self.state.get_address_unique_count_before(
                "shipping_address_id",
                shipping_address_id,
                "merchant_id",
                timestamp,
            )
        )

        billing_unique_merchants = (
            self.state.get_address_unique_count_before(
                "billing_address_id",
                billing_address_id,
                "merchant_id",
                timestamp,
            )
        )

        shipping_24h = (
            self.state.get_address_transaction_count_since(
                "shipping_address_id",
                shipping_address_id,
                timestamp,
                24,
            )
        )

        billing_24h = (
            self.state.get_address_transaction_count_since(
                "billing_address_id",
                billing_address_id,
                timestamp,
                24,
            )
        )

        # ---------------------------------------------------------
        # BUILD EXACT 26-FEATURE CONTRACT
        # ---------------------------------------------------------

        feature_row = {
            "transaction_amount": amount,
            "quantity": quantity,
            "customer_age": customer_age,
            "account_age_days": account_age_days,

            "customer_transaction_count_before": customer_count,
            "customer_avg_amount_before": customer_avg,
            "customer_max_amount_before": customer_max,
            "customer_amount_deviation": customer_amount_deviation,

            "customer_transactions_last_24h": customer_24h,
            "customer_transactions_last_7d": customer_7d,

            "merchant_transaction_count_before": merchant_count,
            "merchant_avg_amount_before": merchant_avg,
            "merchant_transactions_last_24h": merchant_24h,

            "device_transaction_count_before": device_transaction_count,

            "ip_transaction_count_before": ip_transaction_count,
            "ip_unique_customers_before": ip_unique_customers,
            "ip_unique_merchants_before": ip_unique_merchants,

            "shipping_address_transaction_count_before":
                shipping_transaction_count,

            "billing_address_transaction_count_before":
                billing_transaction_count,

            "shipping_address_unique_customers_before":
                shipping_unique_customers,

            "billing_address_unique_customers_before":
                billing_unique_customers,

            "shipping_address_unique_merchants_before":
                shipping_unique_merchants,

            "billing_address_unique_merchants_before":
                billing_unique_merchants,

            "shipping_address_transactions_last_24h":
                shipping_24h,

            "billing_address_transactions_last_24h":
                billing_24h,

            "hours_since_customer_previous_transaction":
                hours_since_customer_previous,
        }

        # ---------------------------------------------------------
        # Enforce exact training feature order
        # ---------------------------------------------------------

        X = pd.DataFrame(
            [[feature_row[feature] for feature in self.features]],
            columns=self.features,
        )

        return X

    # =============================================================
    # SCORE
    # =============================================================

    def score(self, tx):

        X = self.build_features(tx)

        # ---------------------------------------------------------
        # Imputation
        # ---------------------------------------------------------

        X_imputed = self.imputer.transform(X)

        # IMPORTANT:
        # SimpleImputer returns a NumPy array.
        # Reconstruct a DataFrame with the ORIGINAL feature names
        # before passing it to the Random Forest.
        #
        # This eliminates:
        # "X does not have valid feature names..."
        # ---------------------------------------------------------

        X_imputed = pd.DataFrame(
            X_imputed,
            columns=self.features,
            index=X.index,
        )

        # ---------------------------------------------------------
        # Model probability
        # ---------------------------------------------------------

        probability = float(
            self.model.predict_proba(X_imputed)[0, 1]
        )

        prediction = int(
            probability >= self.threshold
        )

        return {
            "model": "behavioral",
            "model_type": type(self.model).__name__,
            "fraud_probability": probability,
            "threshold": self.threshold,
            "prediction": prediction,
            "feature_count": len(self.features),
        }


# =============================================================
# SMOKE TEST
# =============================================================

if __name__ == "__main__":

    adapter = BehavioralAdapter()

    test_transaction = {
        "transaction_id": "behavioral_test_001",

        "customer_id": "customer_10001",
        "merchant_id": "merchant_001",
        "timestamp": "2024-12-31T12:00:00",

        "transaction_amount": 2500.0,
        "quantity": 1,

        "customer_age": 24,
        "account_age_days": 180,

        "device_id": "device_123",
        "ip_id": "ip_456",

        "shipping_address_id": "ship_789",
        "billing_address_id": "bill_321",
    }

    result = adapter.score(test_transaction)

    print(json.dumps(result, indent=2))