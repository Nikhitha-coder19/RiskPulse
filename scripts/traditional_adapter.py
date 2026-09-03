from pathlib import Path
import json
import joblib

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"


class TraditionalAdapter:
    """
    Runtime adapter for the frozen traditional fraud model.

    Responsibilities:
    - Accept the canonical RiskPulse transaction.
    - Build the exact 21-feature training contract.
    - Load the frozen Random Forest.
    - Load persisted preprocessing metadata.
    - Load the persisted customer-location frequency map.
    - Load the production threshold selected on validation data.
    - Return an actual probability and thresholded prediction.

    No training happens here.
    """

    def __init__(self):
        self.model_path = MODEL_DIR / "random_forest.pkl"
        self.metadata_path = MODEL_DIR / "preprocessing_metadata.json"
        self.location_map_path = MODEL_DIR / "customer_location_freq_map.json"
        self.results_path = MODEL_DIR / "traditional_final_results.json"

        self._load_artifacts()

    def _load_artifacts(self):
        # Frozen production model
        self.model = joblib.load(self.model_path)

        # Exact feature contract
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.feature_cols = self.metadata["feature_cols"]

        # Persisted training-time location frequency map
        with open(self.location_map_path, "r", encoding="utf-8") as f:
            self.location_freq_map = json.load(f)

        # Frozen production threshold
        with open(self.results_path, "r", encoding="utf-8") as f:
            self.final_results = json.load(f)

        self.threshold = float(
            self.final_results["selected_threshold"]
        )

    def _validate_transaction(self, tx):
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
        ]

        missing = [
            field
            for field in required_fields
            if field not in tx
        ]

        if missing:
            raise ValueError(
                f"Traditional adapter missing fields: {missing}"
            )

    def _build_features(self, tx):
        self._validate_transaction(tx)

        timestamp = pd.to_datetime(tx["timestamp"])

        # Start with every expected feature at zero.
        # This guarantees that the final DataFrame follows
        # the frozen training contract.
        features = {
            column: 0.0
            for column in self.feature_cols
        }

        # Numeric features
        numeric_values = {
            "Transaction Amount": float(
                tx["transaction_amount"]
            ),
            "Quantity": float(tx["quantity"]),
            "Customer Age": float(
                tx["customer_age"]
            ),
            "Account Age Days": float(
                tx["account_age_days"]
            ),
            "Transaction Hour": float(
                timestamp.hour
            ),
            "transaction_day_of_week": float(
                timestamp.dayofweek
            ),
            "transaction_day_of_month": float(
                timestamp.day
            ),
            "transaction_month": float(
                timestamp.month
            ),
        }

        for column, value in numeric_values.items():
            if column in features:
                features[column] = value

        # Customer-location frequency encoding.
        #
        # For a previously unseen location we use 0.0 rather
        # than inventing a frequency.
        location = str(tx["customer_location"])

        location_frequency = self.location_freq_map.get(
            location,
            0.0
        )

        if "customer_location_freq" in features:
            features["customer_location_freq"] = float(
                location_frequency
            )

        # One-hot categorical features
        payment_method = str(
            tx["payment_method"]
        )

        payment_column = (
            f"Payment Method_{payment_method}"
        )

        if payment_column in features:
            features[payment_column] = 1.0

        product_category = str(
            tx["product_category"]
        )

        product_column = (
            f"Product Category_{product_category}"
        )

        if product_column in features:
            features[product_column] = 1.0

        device_type = str(
            tx["device_type"]
        )

        device_column = (
            f"Device Used_{device_type}"
        )

        if device_column in features:
            features[device_column] = 1.0

        # Build DataFrame in EXACT frozen feature order.
        X = pd.DataFrame(
            [[features[column] for column in self.feature_cols]],
            columns=self.feature_cols,
        )

        return X

    def score(self, tx):
        """
        Score one canonical transaction.

        Returns:
            {
                "model": "traditional",
                "model_type": "...",
                "fraud_probability": float,
                "threshold": float,
                "prediction": int,
                "feature_count": int
            }
        """

        X = self._build_features(tx)

        # Final contract guard
        if list(X.columns) != self.feature_cols:
            raise RuntimeError(
                "Traditional feature contract mismatch."
            )

        probability = float(
            self.model.predict_proba(X)[0, 1]
        )

        prediction = int(
            probability >= self.threshold
        )

        return {
            "model": "traditional",
            "model_type": type(self.model).__name__,
            "fraud_probability": probability,
            "threshold": self.threshold,
            "prediction": prediction,
            "feature_count": len(X.columns),
        }


if __name__ == "__main__":
    # Runtime smoke test
    sample_transaction = {
        "transaction_id": "tx_demo_001",
        "customer_id": "customer_demo_001",
        "merchant_id": "merchant_demo_001",
        "timestamp": "2026-09-01T14:30:00",
        "transaction_amount": 2500.0,
        "quantity": 1,
        "payment_method": "credit card",
        "product_category": "electronics",
        "customer_age": 24,
        "account_age_days": 180,
        "customer_location": "Hyderabad",
        "device_id": "device_demo_001",
        "device_type": "mobile",
        "ip_id": "ip_demo_001",
        "shipping_address_id": "ship_demo_001",
        "billing_address_id": "bill_demo_001",
    }

    adapter = TraditionalAdapter()
    result = adapter.score(sample_transaction)

    print("Traditional adapter result:")
    print(json.dumps(result, indent=2))