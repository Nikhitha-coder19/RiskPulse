import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# ============================================================
# RiskPulse - Traditional Model Finalization
# ============================================================
# Purpose:
#   1. Create a validation split ONLY from processed_train.csv
#   2. Select threshold using validation data
#   3. Use FN:FP cost ratios 1:1, 2:1, 5:1, 10:1, 20:1
#   4. Use 10:1 as the operational modeling assumption
#   5. Retrain the selected model on ALL processed training data
#   6. Evaluate the final model ONCE on untouched processed_test.csv
#
# Important:
#   The test set is NEVER used for threshold selection.
# ============================================================


root = Path(".").resolve()
data_dir = root / "data"
model_dir = root / "model"
model_dir.mkdir(parents=True, exist_ok=True)

train_path = data_dir / "processed_train.csv"
test_path = data_dir / "processed_test.csv"

if not train_path.exists():
    raise SystemExit(f"Missing: {train_path}")

if not test_path.exists():
    raise SystemExit(f"Missing: {test_path}")


LABEL = "Is Fraudulent"
RANDOM_STATE = 42
VALIDATION_SIZE = 0.20

# Operational modeling assumption.
# This means a missed fraud is modeled as 10x more costly
# than a false positive.
OPERATIONAL_FN_FP_RATIO = 10

THRESHOLDS = np.arange(0.05, 0.96, 0.01)
COST_RATIOS = [1, 2, 5, 10, 20]


print("=" * 70)
print("RiskPulse Traditional Model Finalization")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

print("\n[1/7] Loading processed datasets...")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

if LABEL not in train.columns:
    raise SystemExit(f'Target column "{LABEL}" missing from training data.')

if LABEL not in test.columns:
    raise SystemExit(f'Target column "{LABEL}" missing from test data.')


X = train.drop(columns=[LABEL])
y = train[LABEL].astype(int)

X_test = test.drop(columns=[LABEL])
y_test = test[LABEL].astype(int)


# ------------------------------------------------------------
# 2. Align test columns
# ------------------------------------------------------------

if set(X.columns) != set(X_test.columns):
    missing_in_test = set(X.columns) - set(X_test.columns)
    extra_in_test = set(X_test.columns) - set(X.columns)

    for col in missing_in_test:
        X_test[col] = 0

    if extra_in_test:
        X_test = X_test.drop(columns=list(extra_in_test))

X_test = X_test[X.columns]


print(f"Training rows: {len(X):,}")
print(f"Test rows:     {len(X_test):,}")
print(f"Features:      {X.shape[1]}")
print(f"Train fraud:   {y.mean():.4%}")
print(f"Test fraud:    {y_test.mean():.4%}")


# ------------------------------------------------------------
# 3. Train/validation split
# ------------------------------------------------------------

print("\n[2/7] Creating validation split from training data only...")

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=VALIDATION_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)

print(f"Training subset:   {len(X_train):,}")
print(f"Validation subset: {len(X_val):,}")
print(f"Validation fraud:  {y_val.mean():.4%}")


# ------------------------------------------------------------
# 4. Train temporary models for validation threshold selection
# ------------------------------------------------------------

print("\n[3/7] Training temporary models for threshold selection...")

validation_models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        n_jobs=-1,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
}

validation_probabilities = {}

for name, model in validation_models.items():
    print(f"  Training {name}...")
    model.fit(X_train, y_train)
    validation_probabilities[name] = model.predict_proba(X_val)[:, 1]


# ------------------------------------------------------------
# 5. Select threshold using validation cost
# ------------------------------------------------------------

print("\n[4/7] Selecting threshold using validation data...")

cost_results = {}

for model_name, probs in validation_probabilities.items():

    model_costs = {}

    for ratio in COST_RATIOS:

        best = None

        for threshold in THRESHOLDS:

            preds = (probs >= threshold).astype(int)

            tn, fp, fn, tp = confusion_matrix(
                y_val,
                preds,
                labels=[0, 1],
            ).ravel()

            # Relative cost:
            # FP = 1
            # FN = ratio
            total_cost = (fn * ratio) + fp

            if best is None or total_cost < best["cost"]:
                best = {
                    "threshold": float(threshold),
                    "cost": int(total_cost),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                    "tn": int(tn),
                }

        model_costs[str(ratio)] = best

    cost_results[model_name] = model_costs


print("\nValidation cost results:")
print("-" * 70)

for model_name in cost_results:
    print(f"\n{model_name}")

    for ratio in COST_RATIOS:
        r = cost_results[model_name][str(ratio)]

        print(
            f"  FN:FP {ratio}:1 -> "
            f"threshold={r['threshold']:.2f}, "
            f"cost={r['cost']:,}, "
            f"FP={r['fp']:,}, "
            f"FN={r['fn']:,}"
        )


# Operational selection: 10:1
selected_candidates = {}

for model_name in cost_results:
    selected_candidates[model_name] = cost_results[model_name][
        str(OPERATIONAL_FN_FP_RATIO)
    ]

selected_model_name = min(
    selected_candidates,
    key=lambda name: selected_candidates[name]["cost"],
)

selected_threshold = selected_candidates[selected_model_name]["threshold"]

print("\nOperational selection:")
print(f"  Cost ratio:       FN:FP = {OPERATIONAL_FN_FP_RATIO}:1")
print(f"  Selected model:   {selected_model_name}")
print(f"  Selected threshold: {selected_threshold:.2f}")


# ------------------------------------------------------------
# 6. Retrain selected production model on ALL training data
# ------------------------------------------------------------

print("\n[5/7] Training final production model on ALL processed training data...")

if selected_model_name == "Random Forest":
    final_model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
else:
    final_model = LogisticRegression(
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        n_jobs=-1,
    )

final_model.fit(X, y)

final_model_path = model_dir / "random_forest.pkl"

# RiskPulse currently uses Random Forest as the traditional model.
# If the operational selection changes to LR, stop rather than
# silently saving LR under the RF filename.
if selected_model_name != "Random Forest":
    raise SystemExit(
        "\nSTOP: Operational selection chose Logistic Regression. "
        "Do not overwrite random_forest.pkl automatically. "
        "Review the result before changing the production model."
    )

joblib.dump(final_model, final_model_path)

print(f"Saved final production model: {final_model_path}")


# ------------------------------------------------------------
# 7. Final untouched test evaluation
# ------------------------------------------------------------

print("\n[6/7] Evaluating final model on untouched test set...")

test_probs = final_model.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= selected_threshold).astype(int)

precision = precision_score(y_test, test_preds, zero_division=0)
recall = recall_score(y_test, test_preds, zero_division=0)
f1 = f1_score(y_test, test_preds, zero_division=0)
roc_auc = roc_auc_score(y_test, test_probs)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_preds,
    labels=[0, 1],
).ravel()

fpr = fp / (fp + tn) if (fp + tn) else 0.0


print("\nFINAL TEST RESULTS")
print("-" * 70)
print(f"Model:             {selected_model_name}")
print(f"Threshold:         {selected_threshold:.2f}")
print(f"Precision:         {precision:.4f}")
print(f"Recall:            {recall:.4f}")
print(f"F1:                {f1:.4f}")
print(f"ROC-AUC:           {roc_auc:.4f}")
print(f"True Positives:    {tp:,}")
print(f"False Positives:   {fp:,}")
print(f"True Negatives:    {tn:,}")
print(f"False Negatives:   {fn:,}")
print(f"False Positive Rate: {fpr:.6f}")


# ------------------------------------------------------------
# Save final metadata
# ------------------------------------------------------------

print("\n[7/7] Saving finalized metadata...")

threshold_metadata = {
    "selected_model": selected_model_name,
    "selected_threshold": selected_threshold,
    "threshold_selection": {
        "method": "validation_cost_minimization",
        "validation_size": VALIDATION_SIZE,
        "random_state": RANDOM_STATE,
        "operational_fn_fp_ratio": OPERATIONAL_FN_FP_RATIO,
        "tested_fn_fp_ratios": COST_RATIOS,
        "threshold_range": {
            "start": float(THRESHOLDS.min()),
            "end": float(THRESHOLDS.max()),
            "step": 0.01,
        },
        "test_set_used_for_selection": False,
    },
    "final_test_metrics": {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "fpr": fpr,
    },
    "feature_cols": list(X.columns),
    "feature_count": len(X.columns),
    "data_contract": {
        "train_file": str(train_path),
        "test_file": str(test_path),
    },
    "note": (
        "Cost ratios are modeling assumptions, not claimed real-world "
        "Razorpay monetary costs. Threshold was selected using validation "
        "data only; final metrics were measured once on the untouched test set."
    ),
}

with open(
    model_dir / "traditional_final_results.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(threshold_metadata, f, indent=2)


with open(
    model_dir / "traditional_cost_analysis.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        {
            "operational_ratio": OPERATIONAL_FN_FP_RATIO,
            "tested_ratios": COST_RATIOS,
            "validation_results": cost_results,
            "selected_model": selected_model_name,
            "selected_threshold": selected_threshold,
        },
        f,
        indent=2,
    )


print("\n" + "=" * 70)
print("TRADITIONAL MODEL FINALIZATION COMPLETE")
print("=" * 70)
print(f"Production model: {final_model_path}")
print(f"Threshold:        {selected_threshold:.2f}")
print("Threshold chosen from validation: YES")
print("Test set used for threshold selection: NO")
print("Final test evaluation: COMPLETE")