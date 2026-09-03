from pathlib import Path
import pandas as pd
import json
from sklearn.model_selection import train_test_split

root = Path('.').resolve()
csv_path = root / 'Data' / 'archive' / 'Fraudulent_E-Commerce_Transaction_Data.csv'
if not csv_path.exists():
    # fallback to any csv under Data
    csvs = list(root.rglob('*.csv'))
    csv_path = next((p for p in csvs if 'Fraudulent_E-Commerce_Transaction_Data.csv' in p.name), None)
    if csv_path is None and csvs:
        csv_path = csvs[0]
    if csv_path is None:
        raise SystemExit('No CSV found')

print('Using:', csv_path)

cols = ['Customer Location']
# identify label column name by peeking header
peek = pd.read_csv(csv_path, nrows=1)
label_col = None
for c in peek.columns:
    lc = c.lower()
    if 'is fraud' in lc or ('fraud' in lc and 'is' in lc):
        label_col = c
        break
if label_col is None:
    for c in peek.columns:
        if 'fraud' in c.lower():
            label_col = c
            break
if label_col is None:
    raise SystemExit('Label column not found')

cols.append(label_col)

# Read necessary columns in chunks
data_rows = []
for chunk in pd.read_csv(csv_path, usecols=cols, chunksize=200000):
    data_rows.append(chunk)

df = pd.concat(data_rows, ignore_index=True)
df = df.dropna(subset=[label_col])

# Reproduce train_test_split used earlier
X_train, X_test = train_test_split(df, test_size=0.2, stratify=df[label_col], random_state=42)

mapping = {}
if 'Customer Location' in X_train.columns:
    counts = X_train['Customer Location'].value_counts()
    mapping = (counts / len(X_train)).to_dict()

model_dir = root / 'model'
model_dir.mkdir(exist_ok=True)
with open(model_dir / 'customer_location_freq_map.json', 'w', encoding='utf-8') as f:
    json.dump({str(k): float(v) for k, v in mapping.items()}, f, indent=2)

print('Persisted mapping entries:', len(mapping))
