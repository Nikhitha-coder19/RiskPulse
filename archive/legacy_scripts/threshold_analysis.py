import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

root = Path('.').resolve()
model_dir = root / 'model'
train_dir = root / 'data'

# Load models
lr_path = model_dir / 'logistic_regression.pkl'
rf_path = model_dir / 'random_forest.pkl'
if not lr_path.exists() or not rf_path.exists():
    raise SystemExit('Trained model files not found in model/')

lr = joblib.load(lr_path)
rf = joblib.load(rf_path)

# Load held-out test set (do NOT modify)
test_path = train_dir / 'processed_test.csv'
if not test_path.exists():
    raise SystemExit('processed_test.csv not found in data/')

test = pd.read_csv(test_path)
label_col = 'Is Fraudulent'
if label_col not in test.columns:
    raise SystemExit(f'Target column "{label_col}" missing in processed_test.csv')

X_test = test.drop(columns=[label_col])
y_test = test[label_col].astype(int)

# Ensure feature alignment (models expect same feature order as training)
# We'll reorder X_test columns to match model input if possible (assume models were trained with same columns order)
# If model was trained with a different order, scikit-learn estimators accept any DataFrame order matching columns used during training.

results = {}
thresholds = [0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90]

models = {'Logistic Regression': lr, 'Random Forest': rf}

total_legit = int((y_test==0).sum())
total_fraud = int((y_test==1).sum())

for name, model in models.items():
    # predict probabilities
    try:
        probs = model.predict_proba(X_test)[:,1]
    except Exception as e:
        raise SystemExit(f'Error predicting probabilities for {name}: {e}')
    model_results = {
        'total_legit': total_legit,
        'total_fraud': total_fraud,
        'thresholds': {}
    }
    best_f1 = -1.0
    best_thr = None
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        precision = precision_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        fpr = fp / float(fp + tn) if (fp + tn) > 0 else float('nan')
        model_results['thresholds'][str(thr)] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fp': int(fp),
            'fn': int(fn),
            'fpr': fpr
        }
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    model_results['best_f1'] = {'threshold': best_thr, 'f1': best_f1}
    results[name] = model_results

# Save results
out_path = model_dir / 'threshold_results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)

# Print compact tables
for name, res in results.items():
    print(f"Model: {name}")
    print('Thr | Precision | Recall | F1 | FP | FN | FPR')
    for thr in thresholds:
        r = res['thresholds'][str(thr)]
        print(f"{thr:.2f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['fp']} | {r['fn']} | {r['fpr']:.6f}")
    bf = res['best_f1']
    print(f"Best F1 threshold: {bf['threshold']:.2f} (F1={bf['f1']:.3f})")
    print()

print(f"Total legitimate (test): {total_legit}")
print(f"Total fraudulent (test): {total_fraud}")
