import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Configurable assumptions (modeling assumptions / illustrative estimates)
FALSE_POSITIVE_COST = 50  # ₹50 operational/review cost per false positive
FRAUD_LOSS_PER_TRANSACTION = 1000  # ₹1000 illustrative fraud loss per missed fraudulent transaction

root = Path('.').resolve()
model_dir = root / 'model'
data_dir = root / 'data'

rf_path = model_dir / 'random_forest.pkl'
if not rf_path.exists():
    raise SystemExit('Random Forest model not found at model/random_forest.pkl')

rf = joblib.load(rf_path)

test_path = data_dir / 'processed_test.csv'
if not test_path.exists():
    raise SystemExit('Processed test CSV not found at data/processed_test.csv')

test = pd.read_csv(test_path)
label_col = 'Is Fraudulent'
if label_col not in test.columns:
    raise SystemExit(f'Target column "{label_col}" missing in processed_test.csv')

X_test = test.drop(columns=[label_col])
y_test = test[label_col].astype(int)

# Align columns with training-order expectation if necessary
# (models trained with DataFrame columns order; ensure test has same columns)
# If RF was trained with sklearn, it stores feature_names_in_ (since sklearn 1.0+)
if hasattr(rf, 'feature_names_in_'):
    expected_cols = list(rf.feature_names_in_)
    missing_in_test = set(expected_cols) - set(X_test.columns)
    missing_in_train = set(X_test.columns) - set(expected_cols)
    for c in missing_in_test:
        X_test[c] = 0
    # Drop extra columns not in expected
    if missing_in_train:
        X_test = X_test.drop(columns=list(missing_in_train))
    X_test = X_test[expected_cols]

probs = rf.predict_proba(X_test)[:, 1]

thresholds = [0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50]

results = {
    'assumptions': {
        'FALSE_POSITIVE_COST': FALSE_POSITIVE_COST,
        'FRAUD_LOSS_PER_TRANSACTION': FRAUD_LOSS_PER_TRANSACTION,
        'note': 'These are MODELING ASSUMPTIONS / ILLUSTRATIVE ESTIMATES only.'
    },
    'total_legit': int((y_test==0).sum()),
    'total_fraud': int((y_test==1).sum()),
    'thresholds': {}
}

best_threshold = None
best_total_exposure = None

for thr in thresholds:
    preds = (probs >= thr).astype(int)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    fpr = fp / float(fp + tn) if (fp + tn) > 0 else float('nan')
    fp_cost = int(fp) * FALSE_POSITIVE_COST
    fn_exposure = int(fn) * FRAUD_LOSS_PER_TRANSACTION
    total_exposure = fp_cost + fn_exposure
    results['thresholds'][str(thr)] = {
        'precision': round(precision, 6),
        'recall': round(recall, 6),
        'f1': round(f1, 6),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'fpr': round(fpr, 9),
        'fp_cost': fp_cost,
        'fn_exposure': fn_exposure,
        'total_exposure': total_exposure
    }
    if best_total_exposure is None or total_exposure < best_total_exposure:
        best_total_exposure = total_exposure
        best_threshold = thr

results['best_threshold_by_total_exposure'] = {'threshold': best_threshold, 'total_exposure': best_total_exposure}

# Save results
model_dir.mkdir(parents=True, exist_ok=True)
with open(model_dir / 'cost_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print compact table
print('Threshold | Precision | Recall | F1 | FP | FN | FP Cost | Estimated FN Exposure | Estimated Total Exposure')
for thr in thresholds:
    r = results['thresholds'][str(thr)]
    print(f"{thr:.2f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['fp']} | {r['fn']} | {r['fp_cost']} | {r['fn_exposure']} | {r['total_exposure']}")

bt = results['best_threshold_by_total_exposure']
print('\nLowest estimated total exposure under the above assumptions:')
print(f"Threshold {bt['threshold']:.2f} with Estimated Total Exposure = {bt['total_exposure']}")
print('\nNOTE: These monetary numbers are MODELING ASSUMPTIONS / ILLUSTRATIVE ESTIMATES only.')
