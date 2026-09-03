import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

DATA_PATH = Path("Data/merchant/merchant_transactions.csv")
MODEL_PATH = Path("model/merchant_logistic_regression.pkl")
IMPUTER_PATH = Path("model/merchant_imputer.pkl")

OUTPUT_PATH = Path("model/merchant_threshold_results.json")

TARGET = "is_fraud"

EXCLUDED = [
    "transaction_id",
    "timestamp",
    "merchant_id",
    "customer_id",
    "ip_id",
    "device_id",
    "merchant_scenario",
    "is_fraud",
]


def main():
    print("=" * 70)
    print("MERCHANT THRESHOLD ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)

    # Reproduce chronological held-out test split
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    split_idx = int(len(df) * 0.80)
    test = df.iloc[split_idx:].copy()

    X_test = test.drop(columns=EXCLUDED)
    y_test = test[TARGET]

    print(f"Test rows: {len(test):,}")
    print(f"Fraud rate: {y_test.mean():.4f}")

    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)

    X_test_imp = imputer.transform(X_test)

    probabilities = model.predict_proba(X_test_imp)[:, 1]

    roc_auc = roc_auc_score(y_test, probabilities)

    print(f"ROC-AUC: {roc_auc:.4f}")
    print()
    print("-" * 70)
    print("THRESHOLD COMPARISON")
    print("-" * 70)

    results = []

    for threshold in np.arange(0.10, 0.91, 0.05):
        y_pred = (probabilities >= threshold).astype(int)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        tp = int(((y_test == 1) & (y_pred == 1)).sum())
        fp = int(((y_test == 0) & (y_pred == 1)).sum())
        tn = int(((y_test == 0) & (y_pred == 0)).sum())
        fn = int(((y_test == 1) & (y_pred == 0)).sum())

        fpr = fp / (fp + tn) if (fp + tn) else 0

        result = {
            "threshold": round(float(threshold), 2),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "fpr": round(float(fpr), 6),
        }

        results.append(result)

        print(
            f"{threshold:.2f} | "
            f"Precision {precision:.3f} | "
            f"Recall {recall:.3f} | "
            f"F1 {f1:.3f} | "
            f"FP {fp:5d} | "
            f"FN {fn:5d}"
        )

    output = {
        "model": "merchant_logistic_regression",
        "test_rows": len(test),
        "fraud_rate": float(y_test.mean()),
        "roc_auc": float(roc_auc),
        "thresholds": results,
        "note": (
            "Thresholds are evaluated on the chronological held-out test set. "
            "Use validation data for final threshold selection if available; "
            "these results are primarily for comparison."
        ),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()