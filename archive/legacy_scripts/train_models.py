import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib

root = Path('.').resolve()
train_path = root / 'data' / 'processed_train.csv'
test_path = root / 'data' / 'processed_test.csv'
if not train_path.exists() or not test_path.exists():
    raise SystemExit('Processed train/test CSVs not found in data/')

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Target column
label_col = 'Is Fraudulent'
if label_col not in train.columns or label_col not in test.columns:
    raise SystemExit(f'Target column "{label_col}" not found in processed files')

# Separate X and y
X_train = train.drop(columns=[label_col])
y_train = train[label_col].astype(int)
X_test = test.drop(columns=[label_col])
y_test = test[label_col].astype(int)

# Ensure feature columns match between train and test
missing_in_test = set(X_train.columns) - set(X_test.columns)
missing_in_train = set(X_test.columns) - set(X_train.columns)
if missing_in_test or missing_in_train:
    # Align columns: add missing columns with zeros
    for c in missing_in_test:
        X_test[c] = 0
    for c in missing_in_train:
        X_train[c] = 0
# Reorder test columns to match train
X_test = X_test[X_train.columns]

# Models
models = {}
# Logistic Regression with class_weight balanced
lr = LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=1000, n_jobs=-1)
lr.fit(X_train, y_train)
models['Logistic Regression'] = lr

# Random Forest with class_weight balanced
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
models['Random Forest'] = rf

# Evaluate on held-out test set using 0.5 threshold
results = {}
for name, model in models.items():
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    try:
        roc = roc_auc_score(y_test, probs)
    except Exception:
        roc = float('nan')
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    fp_count = int(fp)
    fn_count = int(fn)
    legit_total = int((y_test==0).sum())
    fraud_total = int((y_test==1).sum())
    fpr = fp_count / float(fp_count + tn) if (fp_count + tn) > 0 else float('nan')
    results[name] = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc,
        'fp': fp_count,
        'fn': fn_count,
        'fpr': fpr,
        'legit_total': legit_total,
        'fraud_total': fraud_total,
    }

# Save models
model_dir = root / 'model'
model_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(lr, model_dir / 'logistic_regression.pkl')
joblib.dump(rf, model_dir / 'random_forest.pkl')

# Save compact evaluation report
report = {'results': results}
with open(model_dir / 'baseline_results.json', 'w') as f:
    json.dump(report, f, indent=2)

# Print ONLY a compact comparison table
# Model | Precision | Recall | F1 | ROC-AUC | FP | FN | FPR
header = 'Model | Precision | Recall | F1 | ROC-AUC | FP | FN | FPR'
print(header)
for name in ['Logistic Regression','Random Forest']:
    r = results[name]
    print(f"{name} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['roc_auc']:.3f} | {r['fp']} | {r['fn']} | {r['fpr']:.6f}")

# Brief explanation which model performed better (based ONLY on measured test results)
# Use F1 as the primary balanced metric, but mention precision/recall tradeoffs.
lr_f1 = results['Logistic Regression']['f1']
rf_f1 = results['Random Forest']['f1']
if rf_f1 > lr_f1:
    better = 'Random Forest performed better based on higher F1.'
elif lr_f1 > rf_f1:
    better = 'Logistic Regression performed better based on higher F1.'
else:
    better = 'Both models have similar F1.'
print(better)
