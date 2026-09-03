import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

DATA_PATH = Path("Data/merchant/merchant_transactions.csv")

MODEL_PATH = Path("model/merchant_logistic_regression.pkl")
IMPUTER_PATH = Path("model/merchant_imputer.pkl")
METADATA_PATH = Path("model/merchant_preprocessing_metadata.json")
OUTPUT_PATH = Path("model/merchant_final_results.json")

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


def evaluate(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_true, predictions, zero_division=0
    )
    recall = recall_score(
        y_true, predictions, zero_division=0
    )
    f1 = f1_score(
        y_true, predictions, zero_division=0
    )

    fpr = fp / (fp + tn) if (fp + tn) else 0

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "fpr": float(fpr),
    }


def main():

    print("=" * 70)
    print("FINALIZING MERCHANT RISK MODEL")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)

    # 64% train / 16% validation / 20% final test
    train_end = int(n * 0.64)
    validation_end = int(n * 0.80)

    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:validation_end].copy()
    test = df.iloc[validation_end:].copy()

    print(f"Total rows      : {n:,}")
    print(f"Training rows   : {len(train):,}")
    print(f"Validation rows : {len(validation):,}")
    print(f"Final test rows : {len(test):,}")

    print()
    print("Fraud rates:")
    print(f"  Train      : {train[TARGET].mean():.4f}")
    print(f"  Validation : {validation[TARGET].mean():.4f}")
    print(f"  Final test : {test[TARGET].mean():.4f}")

    X_train = train.drop(columns=EXCLUDED)
    y_train = train[TARGET].values

    X_val = validation.drop(columns=EXCLUDED)
    y_val = validation[TARGET].values

    X_test = test.drop(columns=EXCLUDED)
    y_test = test[TARGET].values

    print()
    print("Merchant features:", X_train.shape[1])

    # ------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------

    imputer = SimpleImputer(strategy="median")

    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    # ------------------------------------------------------------
    # Train Logistic Regression
    # ------------------------------------------------------------

    print()
    print("Training final Logistic Regression...")

    model = LogisticRegression(
        class_weight="balanced",
        solver="lbfgs",
        max_iter=5000,
        random_state=42,
    )

    model.fit(X_train_imp, y_train)

    train_prob = model.predict_proba(X_train_imp)[:, 1]
    val_prob = model.predict_proba(X_val_imp)[:, 1]
    test_prob = model.predict_proba(X_test_imp)[:, 1]

    # ------------------------------------------------------------
    # Threshold selection on VALIDATION ONLY
    # ------------------------------------------------------------

    print()
    print("-" * 70)
    print("THRESHOLD SELECTION ON VALIDATION SET")
    print("-" * 70)

    fn_cost = 10
    fp_cost = 1

    best = None

    for threshold in np.arange(0.10, 0.91, 0.01):

        predictions = (val_prob >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_val,
            predictions,
            labels=[0, 1]
        ).ravel()

        total_cost = (
            fn * fn_cost +
            fp * fp_cost
        )

        if best is None or total_cost < best["total_cost"]:
            best = {
                "threshold": round(float(threshold), 2),
                "total_cost": int(total_cost),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "tn": int(tn),
            }

    threshold = best["threshold"]

    print(f"FN:FP cost ratio : {fn_cost}:1")
    print(f"Selected threshold: {threshold:.2f}")
    print(f"Validation cost  : {best['total_cost']:,}")
    print(f"Validation FP    : {best['fp']:,}")
    print(f"Validation FN    : {best['fn']:,}")

    # ------------------------------------------------------------
    # FINAL TEST — untouched during threshold selection
    # ------------------------------------------------------------

    print()
    print("-" * 70)
    print("FINAL HELD-OUT TEST")
    print("-" * 70)

    final_metrics = evaluate(
        y_test,
        test_prob,
        threshold
    )

    print(f"Threshold : {threshold:.2f}")
    print(f"Precision : {final_metrics['precision']:.4f}")
    print(f"Recall    : {final_metrics['recall']:.4f}")
    print(f"F1        : {final_metrics['f1']:.4f}")
    print(f"ROC-AUC   : {final_metrics['roc_auc']:.4f}")
    print(f"TP        : {final_metrics['tp']:,}")
    print(f"FP        : {final_metrics['fp']:,}")
    print(f"TN        : {final_metrics['tn']:,}")
    print(f"FN        : {final_metrics['fn']:,}")
    print(f"FPR       : {final_metrics['fpr']:.6f}")

    # ------------------------------------------------------------
    # Save model + preprocessing
    # ------------------------------------------------------------

    joblib.dump(model, MODEL_PATH)
    joblib.dump(imputer, IMPUTER_PATH)

    metadata = {
        "model": "merchant_logistic_regression",
        "features": list(X_train.columns),
        "target": TARGET,
        "imputation": "median",
        "split": {
            "train_rows": len(train),
            "validation_rows": len(validation),
            "final_test_rows": len(test),
            "method": "chronological",
        },
        "threshold_selection": {
            "fn_cost": fn_cost,
            "fp_cost": fp_cost,
            "threshold": threshold,
            "selection_set": "validation",
        },
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    output = {
        "model": "merchant_logistic_regression",
        "dataset_rows": n,
        "fraud_rate": float(df[TARGET].mean()),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "final_test_rows": len(test),
        "threshold": threshold,
        "cost_policy": {
            "fn_cost": fn_cost,
            "fp_cost": fp_cost,
            "description": (
                "Relative sensitivity-analysis weights. "
                "They are not claimed real-world monetary values."
            ),
        },
        "validation_selection": best,
        "final_test_metrics": final_metrics,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print("=" * 70)
    print("MERCHANT MODEL FINALIZED")
    print("=" * 70)

    print("Saved:")
    print(f"  {MODEL_PATH}")
    print(f"  {IMPUTER_PATH}")
    print(f"  {METADATA_PATH}")
    print(f"  {OUTPUT_PATH}")


if __name__ == "__main__":
    main()