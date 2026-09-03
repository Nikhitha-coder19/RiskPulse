import json
from pathlib import Path
import pandas as pd
import joblib

root = Path('.').resolve()
model_dir = root / 'model'
rf_path = model_dir / 'random_forest.pkl'
if not rf_path.exists():
    raise SystemExit('Random Forest model not found at model/random_forest.pkl')

rf = joblib.load(rf_path)

# Determine feature names
if hasattr(rf, 'feature_names_in_'):
    feature_names = list(rf.feature_names_in_)
else:
    # fallback: try to read processed_train.csv columns (excluding label)
    proc_train = root / 'data' / 'processed_train.csv'
    if proc_train.exists():
        df = pd.read_csv(proc_train)
        if 'Is Fraudulent' in df.columns:
            feature_names = [c for c in df.columns if c != 'Is Fraudulent']
        else:
            feature_names = list(df.columns)
    else:
        raise SystemExit('Cannot determine feature names; processed_train.csv missing and model has no feature names stored')

importances = rf.feature_importances_
if len(importances) != len(feature_names):
    raise SystemExit('Feature importance length does not match number of feature names')

total = float(importances.sum())
items = []
for name, imp in zip(feature_names, importances):
    pct = (imp / total * 100.0) if total > 0 else 0.0
    items.append({'feature': name, 'importance': float(imp), 'percentage': float(pct)})

# Sort descending
items_sorted = sorted(items, key=lambda x: x['importance'], reverse=True)
# Assign ranks
for i, it in enumerate(items_sorted, start=1):
    it['rank'] = i

# Save JSON
out_path = model_dir / 'feature_importance.json'
with open(out_path, 'w') as f:
    json.dump({'feature_importances': items_sorted}, f, indent=2)

# Print ranked list compactly
print('Rank | Feature | Importance | % Contribution')
for it in items_sorted:
    print(f"{it['rank']} | {it['feature']} | {it['importance']:.6f} | {it['percentage']:.3f}%")

# Top 10
top10 = items_sorted[:10]
print('\nTop 10 features:')
for it in top10:
    print(f"{it['rank']}. {it['feature']} ({it['percentage']:.3f}%)")

# Short interpretation
print('\nInterpretation:')
print('These numbers are "model importance" measures from the trained Random Forest; they indicate which features the model relies on most for making predictions, not proven causes of fraud.')
print('High-importance features suggest areas to inspect or engineer further: e.g., if "Transaction Amount" ranks high, large amounts influence risk predictions; if "account_age_bins" or "customer_location_freq" rank high, they reflect behavioral or profile signals the model uses.')
print('Use these insights to design monitoring rules, feature engineering, or business policies, but validate with further analysis before taking operational actions.')
