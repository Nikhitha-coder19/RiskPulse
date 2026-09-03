import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "Data" / "behavioral" / "behavioral_transactions.csv"
MODEL_PATH = ROOT / "model" / "behavioral_random_forest.pkl"

TARGET = "is_fraud"

EXCLUDED = {
    "transaction_id",
    "customer_id",
    "merchant_id",
    "device_id",
    "ip_id",
    "shipping_address_id",
    "billing_address_id",
    "scenario_type",
}

# -----------------------------
# Business-cost assumptions
# -----------------------------
# These are explicit assumptions for scenario analysis,
# not claims about real Razorpay costs.

CHARGEBACK_RATE = 0.10
CHARGEBACK_FIXED_COST = 100.0

VERIFICATION_COST = 10.0
ABANDONMENT_RATE = 0.05

# Expected future customer value lost when a genuine
# customer is incorrectly declined.
FUTURE_CUSTOMER_VALUE_RATE = 0.10

THRESHOLDS = [0.20, 0.30, 0.40, 0.50]


# -----------------------------
# Load data and model
# -----------------------------
df = pd.read_csv(DATA_PATH)
df = df.sort_values("timestamp", kind="mergesort").copy()

split_index = int(len(df) * 0.8)
test_df = df.iloc[split_index:].copy()

X = test_df.drop(columns=[TARGET], errors="ignore")
X = X[[c for c in X.columns if c not in EXCLUDED]]
X = X.select_dtypes(include=[np.number]).copy()
X = X.drop(columns=["timestamp"], errors="ignore")

X["hours_since_customer_previous_transaction"] = pd.to_numeric(
    X["hours_since_customer_previous_transaction"],
    errors="coerce",
).fillna(-1.0)

y = test_df[TARGET].astype(int)
amounts = test_df["transaction_amount"].astype(float).to_numpy()

model = joblib.load(MODEL_PATH)
probabilities = model.predict_proba(X)[:, 1]


# -----------------------------
# Threshold cost analysis
# -----------------------------
print("Behavioral Random Forest — Transaction-Sensitive Cost Analysis")
print()
print("Assumptions:")
print(f"  Chargeback rate on missed fraud: {CHARGEBACK_RATE:.2%}")
print(f"  Chargeback fixed cost: ₹{CHARGEBACK_FIXED_COST:.2f}")
print(f"  Verification cost: ₹{VERIFICATION_COST:.2f}")
print(f"  Customer abandonment rate after FP: {ABANDONMENT_RATE:.2%}")
print(f"  Future customer value rate: {FUTURE_CUSTOMER_VALUE_RATE:.2%}")
print()

print(
    "Threshold | FP | FN | Fraud loss | FP friction | Total cost"
)

for threshold in THRESHOLDS:

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    ).ravel()

    # False negatives:
    # Fraud transaction was allowed through.
    fn_mask = (y.to_numpy() == 1) & (predictions == 0)
    fn_amounts = amounts[fn_mask]

    fraud_loss = (
        fn_amounts.sum() * CHARGEBACK_RATE
        + fn * CHARGEBACK_FIXED_COST
    )

    # False positives:
    # Genuine transaction incorrectly flagged.
    fp_mask = (y.to_numpy() == 0) & (predictions == 1)
    fp_amounts = amounts[fp_mask]

    fp_friction = (
        fp * VERIFICATION_COST
        + fp_amounts.sum() * ABANDONMENT_RATE
        + fp_amounts.sum() * FUTURE_CUSTOMER_VALUE_RATE
    )

    total_cost = fraud_loss + fp_friction

    print(
        f"{threshold:.2f} | "
        f"{fp} | "
        f"{fn} | "
        f"₹{fraud_loss:,.2f} | "
        f"₹{fp_friction:,.2f} | "
        f"₹{total_cost:,.2f}"
    )
