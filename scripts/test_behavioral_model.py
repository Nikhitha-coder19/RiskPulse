import json
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "model" / "behavioral_random_forest.pkl"
IMPUTER_PATH = ROOT / "model" / "behavioral_imputer.pkl"
RESULTS_PATH = ROOT / "model" / "behavioral_final_results.json"


print("=" * 70)
print("RISKPULSE — BEHAVIORAL MODEL HEALTH CHECK")
print("=" * 70)


# ---------------------------------------------------------
# 1. Check artifacts
# ---------------------------------------------------------

for path in [MODEL_PATH, IMPUTER_PATH, RESULTS_PATH]:
    print(f"\nChecking: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    print("  OK")


# ---------------------------------------------------------
# 2. Load model + imputer
# ---------------------------------------------------------

print("\nLoading Behavioral Random Forest...")

model = joblib.load(MODEL_PATH)
imputer = joblib.load(IMPUTER_PATH)

print(f"  Model type: {type(model).__name__}")
print(f"  Estimators: {model.n_estimators}")

print("\nLoading behavioral imputer...")
print(f"  Imputer type: {type(imputer).__name__}")


# ---------------------------------------------------------
# 3. Inspect feature contracts
# ---------------------------------------------------------

print("\nChecking model feature contract...")

if not hasattr(model, "feature_names_in_"):
    raise RuntimeError(
        "Behavioral Random Forest does not contain feature_names_in_."
    )

model_features = list(model.feature_names_in_)

print(f"  Model feature count: {len(model_features)}")

print("\n  Model features:")
for feature in model_features:
    print(f"    - {feature}")


print("\nChecking imputer feature contract...")

if not hasattr(imputer, "feature_names_in_"):
    raise RuntimeError(
        "Behavioral imputer does not contain feature_names_in_."
    )

imputer_features = list(imputer.feature_names_in_)

print(f"  Imputer feature count: {len(imputer_features)}")


if model_features != imputer_features:
    print("\nERROR: Model and imputer feature contracts differ!")

    print("\n  Model-only features:")
    for feature in set(model_features) - set(imputer_features):
        print(f"    - {feature}")

    print("\n  Imputer-only features:")
    for feature in set(imputer_features) - set(model_features):
        print(f"    - {feature}")

    raise RuntimeError(
        "Behavioral model and imputer feature contracts do not match."
    )

print("  Model/imputer feature contract: PASS")


# ---------------------------------------------------------
# 4. Load saved results
# ---------------------------------------------------------

print("\nLoading behavioral final results...")

with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    results = json.load(f)

print("  Results JSON: OK")


# ---------------------------------------------------------
# 5. Verify selected model + threshold
# ---------------------------------------------------------

selected_model = results.get("selected_model")
selected_threshold = results.get("selected_threshold")

print(f"  Selected model: {selected_model}")
print(f"  Selected threshold: {selected_threshold}")

if selected_model != "Random Forest":
    raise RuntimeError(
        f"Expected selected model to be Random Forest, got: {selected_model}"
    )

if selected_threshold is None:
    raise RuntimeError("No selected threshold found in results JSON.")

print("  Model selection metadata: PASS")


# ---------------------------------------------------------
# 6. Build controlled feature vector
# ---------------------------------------------------------

print("\nBuilding controlled behavioral feature vector...")

test_values = {}

for feature in model_features:
    test_values[feature] = 0.0


# For the prior-transaction feature, -1 represents
# unavailable history in the finalized training pipeline.
if "hours_since_customer_previous_transaction" in test_values:
    test_values["hours_since_customer_previous_transaction"] = -1.0


# Give the transaction a realistic amount if present.
if "transaction_amount" in test_values:
    test_values["transaction_amount"] = 2500.0


X = pd.DataFrame(
    [test_values],
    columns=model_features,
)


# ---------------------------------------------------------
# 7. Apply saved imputer
# ---------------------------------------------------------

print("\nApplying saved behavioral imputer...")

X_imputed = imputer.transform(X)

print(f"  Input shape: {X.shape}")
print(f"  Imputed shape: {X_imputed.shape}")

if X_imputed.shape[1] != len(model_features):
    raise RuntimeError(
        "Imputer output feature count does not match model feature count."
    )

print("  Imputation: PASS")


# ---------------------------------------------------------
# 8. Run prediction
# ---------------------------------------------------------

print("\nRunning Behavioral Random Forest prediction...")

probability = float(model.predict_proba(X_imputed)[0, 1])

threshold = float(selected_threshold)

prediction = int(probability >= threshold)

print(f"  Fraud probability: {probability:.6f}")
print(f"  Saved threshold: {threshold:.6f}")
print(f"  Thresholded prediction: {prediction}")


# ---------------------------------------------------------
# 9. Final health result
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("BEHAVIORAL MODEL HEALTH CHECK RESULT")
print("=" * 70)

print("Model loading:             PASS")
print("Imputer loading:           PASS")
print("Feature contract:          PASS")
print("Model selection metadata:  PASS")
print("Imputation execution:      PASS")
print("Prediction execution:      PASS")
print(f"Fraud probability:         {probability:.6f}")
print(f"Threshold:                 {threshold:.6f}")
print(f"Predicted class:           {prediction}")

print("=" * 70)
print("BEHAVIORAL MODEL: HEALTHY")
print("=" * 70)