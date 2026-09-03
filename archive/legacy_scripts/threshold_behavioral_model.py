import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

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
    errors="coerce"
).fillna(-1.0)

y = test_df[TARGET].astype(int)

model = joblib.load(MODEL_PATH)
probs = model.predict_proba(X)[:, 1]

print("Threshold analysis — Behavioral Random Forest")
print("Threshold | Precision | Recall | F1 | FP | FN | FPR")

for threshold in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
    preds = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, preds, labels=[0, 1]
    ).ravel()

    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    print(
        f"{threshold:.2f} | "
        f"{precision:.3f} | "
        f"{recall:.3f} | "
        f"{f1:.3f} | "
        f"{fp} | {fn} | {fpr:.6f}"
    )
