import json
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "model" / "random_forest.pkl"
METADATA_PATH = ROOT / "model" / "preprocessing_metadata.json"
LOCATION_MAP_PATH = ROOT / "model" / "customer_location_freq_map.json"
FINAL_RESULTS_PATH = ROOT / "model" / "traditional_final_results.json"


print("=" * 70)
print("RISKPULSE — TRADITIONAL MODEL HEALTH CHECK")
print("=" * 70)


# ---------------------------------------------------------
# 1. Check required artifacts
# ---------------------------------------------------------

required_artifacts = [
    MODEL_PATH,
    METADATA_PATH,
    LOCATION_MAP_PATH,
    FINAL_RESULTS_PATH,
]

for path in required_artifacts:
    print(f"\nChecking: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    print("  OK")


# ---------------------------------------------------------
# 2. Load model
# ---------------------------------------------------------

print("\nLoading Random Forest model...")

model = joblib.load(MODEL_PATH)

print(f"  Model type: {type(model).__name__}")
print(f"  Estimators: {model.n_estimators}")


# ---------------------------------------------------------
# 3. Load preprocessing metadata
# ---------------------------------------------------------

print("\nLoading preprocessing metadata...")

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

metadata_features = metadata.get("feature_cols", [])

print(f"  Metadata feature count: {len(metadata_features)}")


# ---------------------------------------------------------
# 4. Load finalized model results
# ---------------------------------------------------------

print("\nLoading finalized traditional model results...")

with open(FINAL_RESULTS_PATH, "r", encoding="utf-8") as f:
    final_results = json.load(f)

selected_model = final_results.get("selected_model")
selected_threshold = final_results.get("selected_threshold")

if selected_model != "Random Forest":
    raise RuntimeError(
        f"Unexpected selected model in final results: {selected_model}"
    )

if selected_threshold is None:
    raise RuntimeError(
        "Finalized traditional model results do not contain selected_threshold."
    )

selected_threshold = float(selected_threshold)

if not 0.0 < selected_threshold < 1.0:
    raise RuntimeError(
        f"Invalid production threshold: {selected_threshold}"
    )

print(f"  Selected production model: {selected_model}")
print(f"  Selected threshold:        {selected_threshold:.6f}")


# ---------------------------------------------------------
# 5. Validate threshold-selection metadata
# ---------------------------------------------------------

print("\nChecking threshold-selection metadata...")

threshold_selection = final_results.get("threshold_selection", {})

selection_method = threshold_selection.get("method")
test_set_used = threshold_selection.get("test_set_used_for_selection")

if selection_method != "validation_cost_minimization":
    raise RuntimeError(
        f"Unexpected threshold-selection method: {selection_method}"
    )

if test_set_used is not False:
    raise RuntimeError(
        "Test set was used for threshold selection. "
        "Finalized model fails the production-selection contract."
    )

print(f"  Selection method:          {selection_method}")
print(f"  Test set used for selection: {test_set_used}")
print("  Threshold-selection contract: PASS")


# ---------------------------------------------------------
# 6. Load customer location frequency map
# ---------------------------------------------------------

print("\nLoading customer location frequency map...")

with open(LOCATION_MAP_PATH, "r", encoding="utf-8") as f:
    location_map = json.load(f)

print(f"  Locations in map: {len(location_map)}")

if not location_map:
    raise RuntimeError("Customer location frequency map is empty.")


# ---------------------------------------------------------
# 7. Inspect model feature contract
# ---------------------------------------------------------

print("\nChecking model feature contract...")

if not hasattr(model, "feature_names_in_"):
    raise RuntimeError(
        "Traditional Random Forest does not contain feature_names_in_."
    )

model_features = list(model.feature_names_in_)

print(f"  Model feature count: {len(model_features)}")

print("\n  Model features:")
for feature in model_features:
    print(f"    - {feature}")


# ---------------------------------------------------------
# 8. Compare metadata vs model
# ---------------------------------------------------------

print("\nComparing metadata features with model features...")

if not metadata_features:
    raise RuntimeError(
        "Preprocessing metadata does not contain a feature list."
    )

if metadata_features != model_features:
    print("\nWARNING: Feature order/content differs!")

    metadata_set = set(metadata_features)
    model_set = set(model_features)

    print("\n  Missing from model:")
    for feature in metadata_set - model_set:
        print(f"    - {feature}")

    print("\n  Missing from metadata:")
    for feature in model_set - metadata_set:
        print(f"    - {feature}")

    raise RuntimeError(
        "Traditional model feature contract does not match metadata."
    )

print("  Feature contract: PASS")


# ---------------------------------------------------------
# 9. Verify expected feature count
# ---------------------------------------------------------

print("\nChecking expected feature count...")

if len(model_features) != 21:
    raise RuntimeError(
        f"Expected 21 traditional model features, "
        f"found {len(model_features)}."
    )

print("  Expected 21 features: PASS")


# ---------------------------------------------------------
# 10. Build a controlled smoke-test transaction
# ---------------------------------------------------------

print("\nBuilding controlled smoke-test feature vector...")

test_values = {}

for feature in model_features:
    test_values[feature] = 0.0


# Give the model a few realistic static values.
if "Transaction Amount" in test_values:
    test_values["Transaction Amount"] = 2500.0

if "Quantity" in test_values:
    test_values["Quantity"] = 1.0

if "Customer Age" in test_values:
    test_values["Customer Age"] = 24.0

if "Account Age Days" in test_values:
    test_values["Account Age Days"] = 180.0

if "Transaction Hour" in test_values:
    test_values["Transaction Hour"] = 14.0

if "transaction_day_of_week" in test_values:
    test_values["transaction_day_of_week"] = 2.0

if "transaction_day_of_month" in test_values:
    test_values["transaction_day_of_month"] = 15.0

if "transaction_month" in test_values:
    test_values["transaction_month"] = 8.0


# Use a known location frequency if possible.
if "customer_location_freq" in test_values:
    first_location = next(iter(location_map))

    test_values["customer_location_freq"] = float(
        location_map[first_location]
    )

    print(
        f"  Using location-frequency value from map: "
        f"{first_location} -> {test_values['customer_location_freq']}"
    )


X = pd.DataFrame([test_values], columns=model_features)


# ---------------------------------------------------------
# 11. Run prediction
# ---------------------------------------------------------

print("\nRunning Random Forest prediction...")

probability = float(model.predict_proba(X)[0, 1])

# Use the finalized production threshold.
production_prediction = int(
    probability >= selected_threshold
)

print(f"  Fraud probability:      {probability:.6f}")
print(f"  Production threshold:   {selected_threshold:.6f}")
print(f"  Production prediction:  {production_prediction}")


# ---------------------------------------------------------
# 12. Verify probability output
# ---------------------------------------------------------

if not 0.0 <= probability <= 1.0:
    raise RuntimeError(
        f"Invalid fraud probability returned by model: {probability}"
    )

print("  Probability range: PASS")


# ---------------------------------------------------------
# 13. Final health result
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TRADITIONAL MODEL HEALTH CHECK RESULT")
print("=" * 70)

print("Model loading:                 PASS")
print("Preprocessing metadata:        PASS")
print("Final results metadata:        PASS")
print("Location frequency map:        PASS")
print("Feature contract:              PASS")
print("21-feature contract:           PASS")
print("Threshold metadata:            PASS")
print("Validation-only selection:     PASS")
print("Prediction execution:          PASS")
print("Probability range:             PASS")
print(f"Fraud probability:              {probability:.6f}")
print(f"Production threshold:           {selected_threshold:.6f}")
print(f"Production prediction:          {production_prediction}")

print("=" * 70)
print("TRADITIONAL MODEL: HEALTHY")
print("=" * 70)