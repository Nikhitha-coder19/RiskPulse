import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path("Data/merchant/merchant_transactions.csv")
MODEL_PATH = Path("model/merchant_logistic_regression.pkl")
IMPUTER_PATH = Path("model/merchant_imputer.pkl")

OUTPUT_PATH = Path("model/cost_analysis_merchant.json")

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
    print("MERCHANT COST-SENSITIVE ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)

    # Chronological split — same held-out test set used previously
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    split_idx = int(len(df) * 0.80)

    test = df.iloc[split_idx:].copy()

    X_test = test.drop(columns=EXCLUDED)
    y_test = test[TARGET].values

    print(f"Test rows: {len(test):,}")
    print(f"Fraud rate: {y_test.mean():.4f}")

    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)

    X_test_imp = imputer.transform(X_test)

    probabilities = model.predict_proba(X_test_imp)[:, 1]

    # ------------------------------------------------------------
    # Cost assumptions
    #
    # These are RELATIVE cost ratios, not claimed real-world rupee
    # values. FN represents fraud/chargeback exposure.
    # FP represents unnecessary intervention / lost conversion.
    # ------------------------------------------------------------

    cost_ratios = {
        "1_to_1": {"fn_cost": 1, "fp_cost": 1},
        "2_to_1": {"fn_cost": 2, "fp_cost": 1},
        "5_to_1": {"fn_cost": 5, "fp_cost": 1},
        "10_to_1": {"fn_cost": 10, "fp_cost": 1},
        "20_to_1": {"fn_cost": 20, "fp_cost": 1},
    }

    thresholds = np.arange(0.10, 0.91, 0.01)

    all_results = {}

    for ratio_name, costs in cost_ratios.items():

        fn_cost = costs["fn_cost"]
        fp_cost = costs["fp_cost"]

        best = None
        threshold_results = []

        for threshold in thresholds:

            y_pred = (probabilities >= threshold).astype(int)

            fp = int(((y_test == 0) & (y_pred == 1)).sum())
            fn = int(((y_test == 1) & (y_pred == 0)).sum())

            tp = int(((y_test == 1) & (y_pred == 1)).sum())
            tn = int(((y_test == 0) & (y_pred == 0)).sum())

            total_cost = (
                fn * fn_cost
                + fp * fp_cost
            )

            result = {
                "threshold": round(float(threshold), 2),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "total_cost": int(total_cost),
            }

            threshold_results.append(result)

            if best is None or total_cost < best["total_cost"]:
                best = result

        all_results[ratio_name] = {
            "fn_cost": fn_cost,
            "fp_cost": fp_cost,
            "best_threshold": best["threshold"],
            "best_total_cost": best["total_cost"],
            "best_fp": best["fp"],
            "best_fn": best["fn"],
            "best_tp": best["tp"],
            "best_tn": best["tn"],
            "thresholds": threshold_results,
        }

        print()
        print(
            f"FN:FP = {fn_cost}:{fp_cost}"
        )
        print(
            f"  Best threshold : {best['threshold']:.2f}"
        )
        print(
            f"  Total cost     : {best['total_cost']:,}"
        )
        print(
            f"  FP             : {best['fp']:,}"
        )
        print(
            f"  FN             : {best['fn']:,}"
        )
        print(
            f"  TP             : {best['tp']:,}"
        )
        print(
            f"  TN             : {best['tn']:,}"
        )

    output = {
        "model": "merchant_logistic_regression",
        "test_rows": len(test),
        "fraud_rate": float(y_test.mean()),
        "cost_definition": {
            "FN": (
                "Fraudulent/risky activity incorrectly allowed through, "
                "representing potential fraud, chargeback, return, "
                "investigation or remediation exposure."
            ),
            "FP": (
                "Legitimate activity incorrectly flagged, representing "
                "potential lost conversion, merchant revenue opportunity "
                "and customer friction."
            ),
        },
        "important_note": (
            "Cost values are relative sensitivity-analysis weights, "
            "not claimed real-world monetary losses."
        ),
        "results": all_results,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print("=" * 70)
    print(f"Saved: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()