from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import json

root = Path('.').resolve()
# find original CSV (same logic as prepare_features.py)
csvs = list(root.rglob('*.csv'))
if not csvs:
    raise SystemExit('No CSV files found')

candidates = [p for p in csvs if 'data' in str(p).lower()]
csv_path = candidates[0] if candidates else csvs[0]

df = pd.read_csv(csv_path, low_memory=False)

# Identify label column as in prepare_features.py
label_col = None
for c in df.columns:
    if c.lower() == 'is fraudulent' or c.lower() == 'is_fraudulent' or c.lower() == 'is_fraud' or 'fraud' in c.lower():
        if 'is' in c.lower() or c.lower().startswith('fraud'):
            label_col = c
            break
if label_col is None:
    raise SystemExit('Label column not found')

# drop missing label rows
df = df[~df[label_col].isnull()]

# reproducible split
RANDOM_SEED = 42
X_train, X_test = train_test_split(df, test_size=0.2, stratify=df[label_col], random_state=RANDOM_SEED)

mapping = {}
if 'Customer Location' in X_train.columns:
    counts = X_train['Customer Location'].value_counts()
    mapping = (counts / len(X_train)).to_dict()

model_dir = root / 'model'
model_dir.mkdir(exist_ok=True)
with open(model_dir / 'customer_location_freq_map.json', 'w', encoding='utf-8') as f:
    # convert keys to strings
    json.dump({str(k): float(v) for k,v in mapping.items()}, f, indent=2)

# update preprocessing_metadata.json if exists
meta_path = model_dir / 'preprocessing_metadata.json'
meta = {}
if meta_path.exists():
    meta = json.load(open(meta_path))
meta['customer_location_freq_map'] = 'model/customer_location_freq_map.json'
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2)

print('Persisted customer_location_freq mapping to model/customer_location_freq_map.json')
print('Updated preprocessing_metadata.json')
