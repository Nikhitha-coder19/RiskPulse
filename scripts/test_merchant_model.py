import json
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "model" / "merchant_logistic_regression.pkl"
IMPUTER_PATH = ROOT / "model" / "merchant_imputer.pkl"
METADATA_PATH = ROOT / "model" / "merchant_preprocessing_metadata.json"
RESULTS_PATH = ROOT / "model" / "merchant_final_results.json"


print("=" * 70)
print("RISKPULSE — MERCHANT MODEL HEALTH CHECK")
print("=" * 70)


# ---------------------------------------------------------
# 1. Check artifacts
# ---------------------------------------------------------

for path in [
    MODEL_PATH,
    IMPUTER_PATH,
    METADATA_PATH,
    RESULTS_PATH,
]:
    print(f"\nChecking: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")

    print("  OK")


# ---------------------------------------------------------
# 2. Load model + imputer
# ---------------------------------------------------------

print("\nLoading Merchant Logistic Regression...")

model = joblib.load(MODEL_PATH)
imputer = joblib.load(IMPUTER_PATH)

print(f"  Model type: {type(model).__name__}")

print("\nLoading merchant imputer...")
print(f"  Imputer type: {type(imputer).__name__}")


# ---------------------------------------------------------
# 3. Load preprocessing metadata
# ---------------------------------------------------------

print("\nLoading merchant preprocessing metadata...")

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

metadata_features = metadata.get("features", [])

if not metadata_features:
    raise RuntimeError(
        "Merchant preprocessing metadata does not contain a feature list."
    )

print(f"  Metadata feature count: {len(metadata_features)}")


print("\n  Metadata features:")
for feature in metadata_features:
    print(f"    - {feature}")


# ---------------------------------------------------------
# 4. Check expected feature count
# ---------------------------------------------------------

if len(metadata_features) != 19:
    raise RuntimeError(
        f"Expected 19 merchant features, "
        f"found {len(metadata_features)}."
    )

print("\n  Expected 19-feature contract: PASS")


# ---------------------------------------------------------
# 5. Check imputer feature contract
# ---------------------------------------------------------

print("\nChecking imputer feature contract...")

if not hasattr(imputer, "feature_names_in_"):
    raise RuntimeError(
        "Merchant imputer does not contain feature_names_in_."
    )

imputer_features = list(imputer.feature_names_in_)

print(f"  Imputer feature count: {len(imputer_features)}")


if imputer_features != metadata_features:
    print("\nERROR: Imputer and metadata feature contracts differ!")

    print("\n  Imputer features:")
    for feature in imputer_features:
        print(f"    - {feature}")

    raise RuntimeError(
        "Merchant imputer feature contract does not match metadata."
    )

print("  Metadata/imputer feature contract: PASS")


# ---------------------------------------------------------
# 6. Inspect model dimensions
# ---------------------------------------------------------

print("\nChecking Logistic Regression dimensions...")

if not hasattr(model, "n_features_in_"):
    raise RuntimeError(
        "Merchant Logistic Regression does not contain n_features_in_."
    )

model_feature_count = int(model.n_features_in_)

print(f"  Model feature count: {model_feature_count}")

if model_feature_count != len(metadata_features):
    raise RuntimeError(
        f"Model expects {model_feature_count} features, "
        f"but metadata defines {len(metadata_features)}."
    )

print("  Model feature count: PASS")


# ---------------------------------------------------------
# 7. Load saved results
# ---------------------------------------------------------

print("\nLoading merchant final results...")

with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    results = json.load(f)

print("  Results JSON: OK")


# ---------------------------------------------------------
# 8. Verify saved threshold
# ---------------------------------------------------------

threshold_selection = metadata.get("threshold_selection", {})

selected_threshold = threshold_selection.get("threshold")

selected_fn_cost = threshold_selection.get("fn_cost")
selected_fp_cost = threshold_selection.get("fp_cost")
selection_set = threshold_selection.get("selection_set")

print(f"  Selected threshold: {selected_threshold}")
print(f"  FN cost:             {selected_fn_cost}")
print(f"  FP cost:             {selected_fp_cost}")
print(f"  Selection set:       {selection_set}")

if selected_threshold is None:
    raise RuntimeError(
        "No selected threshold found in merchant preprocessing metadata."
    )

if selected_fn_cost != 10:
    raise RuntimeError(
        f"Expected FN cost of 10, found: {selected_fn_cost}"
    )

if selected_fp_cost != 1:
    raise RuntimeError(
        f"Expected FP cost of 1, found: {selected_fp_cost}"
    )

if selection_set != "validation":
    raise RuntimeError(
        f"Expected threshold selection set to be validation, "
        f"found: {selection_set}"
    )

print("  Threshold metadata: PASS")

# ---------------------------------------------------------
# 9. Build controlled feature vector
# ---------------------------------------------------------

print("\nBuilding controlled merchant feature vector...")

test_values = {}

for feature in metadata_features:
    test_values[feature] = 0.0


if "transaction_amount" in test_values:
    test_values["transaction_amount"] = 2500.0

if "merchant_history_available" in test_values:
    test_values["merchant_history_available"] = 0.0

if "is_new_merchant" in test_values:
    test_values["is_new_merchant"] = 1.0


X = pd.DataFrame(
    [test_values],
    columns=metadata_features,
)

print(f"  Input shape: {X.shape}")


# ---------------------------------------------------------
# 10. Apply saved imputer
# ---------------------------------------------------------

print("\nApplying saved merchant imputer...")

X_imputed = imputer.transform(X)

print(f"  Imputed shape: {X_imputed.shape}")

if X_imputed.shape[1] != model_feature_count:
    raise RuntimeError(
        "Imputer output feature count does not match model."
    )

print("  Imputation: PASS")


# ---------------------------------------------------------
# 11. Run prediction
# ---------------------------------------------------------

print("\nRunning Merchant Logistic Regression prediction...")

probability = float(model.predict_proba(X_imputed)[0, 1])

threshold = float(selected_threshold)

prediction = int(probability >= threshold)

print(f"  Fraud probability: {probability:.6f}")
print(f"  Saved threshold: {threshold:.6f}")
print(f"  Thresholded prediction: {prediction}")


# ---------------------------------------------------------
# 12. Final health result
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("MERCHANT MODEL HEALTH CHECK RESULT")
print("=" * 70)

print("Model loading:             PASS")
print("Imputer loading:           PASS")
print("Metadata loading:          PASS")
print("19-feature contract:       PASS")
print("Model dimensions:          PASS")
print("Imputer contract:          PASS")
print("Threshold metadata:        PASS")
print("Imputation execution:      PASS")
print("Prediction execution:      PASS")
print(f"Fraud probability:         {probability:.6f}")
print(f"Threshold:                 {threshold:.6f}")
print(f"Predicted class:           {prediction}")

print("=" * 70)
print("MERCHANT MODEL: HEALTHY")
print("=" * 70)