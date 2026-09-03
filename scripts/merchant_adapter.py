import json
import sqlite3
from pathlib import Path

import joblib
import pandas as pd

from runtime_state import RiskState


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
STATE_DB = BASE_DIR / "runtime" / "riskpulse_state.db"

MODEL_PATH = MODEL_DIR / "merchant_logistic_regression.pkl"
IMPUTER_PATH = MODEL_DIR / "merchant_imputer.pkl"
METADATA_PATH = MODEL_DIR / "merchant_preprocessing_metadata.json"
RESULTS_PATH = MODEL_DIR / "merchant_final_results.json"


EXPECTED_FEATURES = [
    "transaction_amount",
    "merchant_transaction_count_before",
    "merchant_avg_amount_before",
    "merchant_previous_amount",
    "merchant_unique_customers_before",
    "merchant_unique_ips_before",
    "merchant_unique_devices_before",
    "merchant_transactions_last_24h",
    "seconds_since_merchant_previous",
    "ip_unique_merchants_before",
    "ip_unique_customers_before",
    "merchant_fraud_count_before",
    "merchant_fraud_rate_before",
    "is_new_merchant",
    "merchant_history_available",
    "amount_deviation_ratio",
    "high_amount_deviation",
    "high_velocity",
    "merchant_high_historical_fraud",
]


class MerchantAdapter:
    """
    Runtime adapter for the frozen merchant risk model.

    Responsibilities:
        1. Read prior state.
        2. Reconstruct the exact 19 merchant features.
        3. Apply the frozen preprocessing artifact.
        4. Score using the frozen Logistic Regression model.
        5. Apply the saved production threshold.

    This adapter does NOT update runtime state.
    """

    def __init__(self):
        self.state = RiskState()

        self.model = joblib.load(MODEL_PATH)
        self.imputer = joblib.load(IMPUTER_PATH)

        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            self.results = json.load(f)

        self.features = self._load_feature_contract()
        self.threshold = self._load_threshold()

        self._validate_artifacts()

    # ============================================================
    # ARTIFACT VALIDATION
    # ============================================================

    def _load_feature_contract(self):
        """
        Load the canonical 19-feature contract from merchant metadata.
        """
        features = self.metadata.get("features")

        if features is None:
            features = self.metadata.get("feature_cols")

        if not features:
            raise ValueError(
                "Merchant preprocessing metadata does not contain a feature list."
            )

        return list(features)

    def _load_threshold(self):
        """
        Load the frozen merchant production threshold
        from merchant_preprocessing_metadata.json.
        """

        threshold_info = self.metadata.get(
            "threshold_selection",
            {}
        )

        threshold = threshold_info.get("threshold")

        if threshold is None:
            raise ValueError(
                "Could not find the saved merchant production threshold."
            )

        threshold = float(threshold)

        if not 0.0 < threshold < 1.0:
            raise ValueError(
                f"Invalid merchant production threshold: {threshold}"
            )

        return threshold

    def _validate_artifacts(self):
        """
        Fail closed if the frozen artifact contract does not match
        the adapter contract.
        """
        if len(self.features) != 19:
            raise ValueError(
                f"Expected 19 merchant features, found {len(self.features)}."
            )

        if self.features != EXPECTED_FEATURES:
            raise ValueError(
                "Merchant feature contract mismatch.\n"
                f"Expected: {EXPECTED_FEATURES}\n"
                f"Artifact: {self.features}"
            )

        if getattr(self.imputer, "n_features_in_", 19) != 19:
            raise ValueError(
                "Merchant imputer does not expect exactly 19 features."
            )

        if getattr(self.model, "n_features_in_", 19) != 19:
            raise ValueError(
                "Merchant model does not expect exactly 19 features."
            )

    # ============================================================
    # DIRECT TRANSACTION HISTORY QUERIES
    # ============================================================

    def _get_previous_merchant_transaction(self, merchant_id, timestamp):
        """
        Return the immediately previous transaction for this merchant.

        Strictly prior to the current timestamp.

        Returns:
            {
                "amount": float | None,
                "timestamp": str | None
            }
        """
        conn = sqlite3.connect(STATE_DB)

        try:
            row = conn.execute(
                """
                SELECT amount, timestamp
                FROM transactions
                WHERE merchant_id = ?
                  AND timestamp < ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (merchant_id, timestamp),
            ).fetchone()

            if row is None:
                return {
                    "amount": None,
                    "timestamp": None,
                }

            return {
                "amount": float(row[0]),
                "timestamp": row[1],
            }

        finally:
            conn.close()

    def _get_prior_merchant_fraud_count(self, merchant_id, timestamp):
        """
        Count confirmed/observed fraudulent transactions for this merchant
        strictly before the current transaction.

        IMPORTANT:
        RiskPulse predictions are NEVER written into this count.
        """
        conn = sqlite3.connect(STATE_DB)

        try:
            row = conn.execute(
                """
                SELECT COUNT(*)
                FROM transactions
                WHERE merchant_id = ?
                  AND timestamp < ?
                  AND is_fraud = 1
                """,
                (merchant_id, timestamp),
            ).fetchone()

            return int(row[0])

        finally:
            conn.close()

    # ============================================================
    # PRIOR 24H MERCHANT VELOCITY
    # ============================================================

    def _get_merchant_transactions_last_24h(
        self,
        merchant_id,
        timestamp,
    ):
        """
        Reproduce the generator's rolling_previous_24h() behavior.

        Generator logic:

            while timestamp_current - timestamp_left > 24h:
                left += 1

        Therefore:
            - strictly prior transactions only
            - exactly 24h old transactions ARE counted
            - current transaction is excluded
        """
        conn = sqlite3.connect(STATE_DB)

        try:
            row = conn.execute(
                """
                SELECT COUNT(*)
                FROM transactions
                WHERE merchant_id = ?
                  AND timestamp >= datetime(?, '-24 hours')
                  AND timestamp < ?
                """,
                (
                    merchant_id,
                    timestamp,
                    timestamp,
                ),
            ).fetchone()

            return int(row[0])

        finally:
            conn.close()

    # ============================================================
    # IP NETWORK HISTORY
    # ============================================================

    def _get_ip_unique_counts_before(
        self,
        ip_id,
        timestamp,
    ):
        """
        Reproduce:

            ip_unique_merchants_before
            ip_unique_customers_before

        using only transactions strictly before the current transaction.
        """
        conn = sqlite3.connect(STATE_DB)

        try:
            merchant_row = conn.execute(
                """
                SELECT COUNT(DISTINCT merchant_id)
                FROM transactions
                WHERE ip_id = ?
                  AND timestamp < ?
                """,
                (ip_id, timestamp),
            ).fetchone()

            customer_row = conn.execute(
                """
                SELECT COUNT(DISTINCT customer_id)
                FROM transactions
                WHERE ip_id = ?
                  AND timestamp < ?
                """,
                (ip_id, timestamp),
            ).fetchone()

            return (
                int(merchant_row[0] or 0),
                int(customer_row[0] or 0),
            )

        finally:
            conn.close()

    # ============================================================
    # MERCHANT FEATURE CALCULATION
    # ============================================================

    def _build_features(self, tx):
        """
        Construct the exact 19 merchant features used during training.
        """

        merchant_id = tx["merchant_id"]
        timestamp = tx["timestamp"]
        ip_id = tx["ip_id"]

        amount = float(tx["transaction_amount"])

        # --------------------------------------------------------
        # PRIOR MERCHANT STATE
        # --------------------------------------------------------

        merchant_state = self.state.get_merchant_state(
            merchant_id
        )

        merchant_count = int(
            merchant_state["transaction_count"]
        )

        merchant_total = float(
            merchant_state["total_amount"]
        )

        if merchant_count > 0:
            merchant_avg = (
                merchant_total / merchant_count
            )
        else:
            merchant_avg = 0.0

        # --------------------------------------------------------
        # PREVIOUS MERCHANT TRANSACTION
        # --------------------------------------------------------

        previous = self._get_previous_merchant_transaction(
            merchant_id,
            timestamp,
        )

        previous_amount = previous["amount"]

        if previous_amount is None:
            previous_amount_feature = None
        else:
            previous_amount_feature = float(previous_amount)

        # --------------------------------------------------------
        # UNIQUE MERCHANT RELATIONSHIPS
        #
        # RiskState relationships represent unique related IDs.
        # --------------------------------------------------------

        unique_customers = self.state.get_related_count(
            "merchant",
            merchant_id,
            "customer",
        )

        unique_ips = self.state.get_related_count(
            "merchant",
            merchant_id,
            "ip",
        )

        unique_devices = self.state.get_related_count(
            "merchant",
            merchant_id,
            "device",
        )

        # --------------------------------------------------------
        # MERCHANT VELOCITY
        # --------------------------------------------------------

        transactions_last_24h = (
            self._get_merchant_transactions_last_24h(
                merchant_id,
                timestamp,
            )
        )

        # --------------------------------------------------------
        # SECONDS SINCE PREVIOUS MERCHANT TRANSACTION
        # --------------------------------------------------------

        if previous["timestamp"] is None:
            seconds_since_previous = -1.0
        else:
            current_time = pd.Timestamp(timestamp)
            previous_time = pd.Timestamp(
                previous["timestamp"]
            )

            seconds_since_previous = (
                current_time - previous_time
            ).total_seconds()

        # --------------------------------------------------------
        # IP NETWORK FEATURES
        # --------------------------------------------------------

        (
            ip_unique_merchants,
            ip_unique_customers,
        ) = self._get_ip_unique_counts_before(
            ip_id,
            timestamp,
        )

        # --------------------------------------------------------
        # HISTORICAL FRAUD
        # --------------------------------------------------------

        fraud_count = self._get_prior_merchant_fraud_count(
            merchant_id,
            timestamp,
        )

        if merchant_count > 0:
            fraud_rate = (
                fraud_count / merchant_count
            )
        else:
            fraud_rate = 0.0

        # --------------------------------------------------------
        # NEW MERCHANT FLAGS
        # --------------------------------------------------------

        is_new_merchant = int(
            merchant_count == 0
        )

        merchant_history_available = int(
            merchant_count > 0
        )

        # --------------------------------------------------------
        # AMOUNT DEVIATION
        # --------------------------------------------------------

        if merchant_avg > 0:
            amount_deviation_ratio = (
                amount / merchant_avg
            )
        else:
            amount_deviation_ratio = 1.0

        high_amount_deviation = int(
            amount_deviation_ratio >= 2.0
        )

        # Generator uses >= 10 for the final feature.
        high_velocity = int(
            transactions_last_24h >= 10
        )

        merchant_high_historical_fraud = int(
            fraud_rate >= 0.15
        )

        # --------------------------------------------------------
        # EXACT FEATURE DICTIONARY
        # --------------------------------------------------------

        features = {
            "transaction_amount": amount,
            "merchant_transaction_count_before": merchant_count,
            "merchant_avg_amount_before": merchant_avg,
            "merchant_previous_amount": previous_amount_feature,
            "merchant_unique_customers_before": unique_customers,
            "merchant_unique_ips_before": unique_ips,
            "merchant_unique_devices_before": unique_devices,
            "merchant_transactions_last_24h": transactions_last_24h,
            "seconds_since_merchant_previous": seconds_since_previous,
            "ip_unique_merchants_before": ip_unique_merchants,
            "ip_unique_customers_before": ip_unique_customers,
            "merchant_fraud_count_before": fraud_count,
            "merchant_fraud_rate_before": fraud_rate,
            "is_new_merchant": is_new_merchant,
            "merchant_history_available": merchant_history_available,
            "amount_deviation_ratio": amount_deviation_ratio,
            "high_amount_deviation": high_amount_deviation,
            "high_velocity": high_velocity,
            "merchant_high_historical_fraud": merchant_high_historical_fraud,
        }

        return features

    # ============================================================
    # SCORING
    # ============================================================

    def score(self, tx):
        """
        Score one canonical transaction.

        Does NOT update RiskState.
        """

        features = self._build_features(tx)

        # --------------------------------------------------------
        # Preserve exact feature order.
        # --------------------------------------------------------

        X = pd.DataFrame(
            [[features[name] for name in self.features]],
            columns=self.features,
        )

        # --------------------------------------------------------
        # Frozen imputer.
        # --------------------------------------------------------

        X_imputed = self.imputer.transform(X)

        probability = float(
            self.model.predict_proba(
                X_imputed
            )[0, 1]
        )

        prediction = int(
            probability >= self.threshold
        )

        return {
            "model": "merchant",
            "model_type": type(self.model).__name__,
            "fraud_probability": probability,
            "threshold": self.threshold,
            "prediction": prediction,
            "feature_count": len(self.features),
        }


# ================================================================
# SMOKE TEST
# ================================================================

if __name__ == "__main__":

    adapter = MerchantAdapter()

    test_transaction = {
        "transaction_id": "adapter_test_merchant_001",
        "customer_id": "customer_test_001",
        "merchant_id": "merchant_test_001",
        "timestamp": "2024-06-15 12:00:00",
        "transaction_amount": 2500.0,
        "quantity": 1,
        "payment_method": "credit card",
        "product_category": "electronics",
        "customer_age": 24,
        "account_age_days": 180,
        "customer_location": "Hyderabad",
        "device_id": "device_test_001",
        "device_type": "mobile",
        "ip_id": "ip_test_001",
        "shipping_address_id": "ship_test_001",
        "billing_address_id": "bill_test_001",
    }

    result = adapter.score(test_transaction)

    print(json.dumps(result, indent=2))