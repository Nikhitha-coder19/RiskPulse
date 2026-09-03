import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

root = Path('.').resolve()
model_dir = root / 'model'
model_dir.mkdir(exist_ok=True)

# Load trained Random Forest
rf_path = model_dir / 'random_forest.pkl'
if not rf_path.exists():
    raise SystemExit('Random Forest model not found at model/random_forest.pkl')
rf = joblib.load(rf_path)

# Load processed_train to infer one-hot templates and numeric feature stats
proc_train = root / 'data' / 'processed_train.csv'
if not proc_train.exists():
    raise SystemExit('processed_train.csv not found')
train_df = pd.read_csv(proc_train)

# Determine expected feature columns
if hasattr(rf, 'feature_names_in_'):
    feature_cols = list(rf.feature_names_in_)
else:
    feature_cols = [c for c in train_df.columns if c != 'Is Fraudulent']

# Identify one-hot feature templates by prefixes
onehot_prefixes = {
    'Payment Method': [c for c in feature_cols if c.startswith('Payment Method_')],
    'Product Category': [c for c in feature_cols if c.startswith('Product Category_')],
    'Device Used': [c for c in feature_cols if c.startswith('Device Used_')]
}

# Numeric features expected
numeric_features = [f for f in ['Transaction Amount','Quantity','Customer Age','Account Age Days','Transaction Hour','transaction_day_of_week','transaction_day_of_month','transaction_month','customer_location_freq'] if f in feature_cols]

# Load persisted training-time customer_location_freq mapping if present
loc_map_path = model_dir / 'customer_location_freq_map.json'
loc_freq_map = {}
raw_df = None
if loc_map_path.exists():
    try:
        loc_freq_map = json.load(open(loc_map_path))
        # keys are strings; ensure values are floats
        loc_freq_map = {k: float(v) for k, v in loc_freq_map.items()}
        using_persisted_map = True
    except Exception:
        loc_freq_map = {}
        using_persisted_map = False
else:
    using_persisted_map = False

if not using_persisted_map:
    # Fallback: compute mapping from raw CSV (not ideal for reproducibility)
    raw_csvs = list(root.rglob('*.csv'))
    raw = None
    for p in raw_csvs:
        if 'Fraudulent_E-Commerce_Transaction_Data.csv' in str(p):
            raw = p
            break
    if raw is None and raw_csvs:
        raw = raw_csvs[0]
    if raw is not None:
        # read only the Customer Location column to save memory
        try:
            raw_df = pd.read_csv(raw, usecols=['Customer Location'], low_memory=False)
            if 'Customer Location' in raw_df.columns:
                loc_counts = raw_df['Customer Location'].value_counts()
                total = len(raw_df)
                loc_freq_map = (loc_counts / total).to_dict()
        except Exception:
            loc_freq_map = {}

# Save preprocessing metadata

# Save preprocessing metadata
preproc_meta = {
    'feature_cols': feature_cols,
    'onehot_prefixes': {k: v for k, v in onehot_prefixes.items()},
    'numeric_features': numeric_features,
    'customer_location_freq_map_sample_size': len(raw_df) if raw_df is not None else 0,
    'customer_location_freq_map_used_persisted': bool(using_persisted_map),
    'customer_location_freq_map_path': str(loc_map_path) if loc_map_path.exists() else None,
}
with open(model_dir / 'preprocessing_metadata.json', 'w') as f:
    json.dump(preproc_meta, f, indent=2)

# Load feature importance info
fi_path = model_dir / 'feature_importance.json'
if fi_path.exists():
    fi = json.load(open(fi_path))['feature_importances']
else:
    fi = []

# Compute training means for numeric features to compare
train_stats = {}
for f in numeric_features:
    if f in train_df.columns:
        train_stats[f] = {'mean': float(train_df[f].mean()), 'median': float(train_df[f].median())}

# Reusable prediction function
def preprocess_transaction(tx: dict):
    # tx expected to follow original dataset keys (e.g., 'Transaction Amount','Payment Method',...)
    x = {}
    # Parse Transaction Date
    if 'Transaction Date' in tx and pd.notnull(tx.get('Transaction Date')):
        try:
            dt = pd.to_datetime(tx['Transaction Date'])
        except Exception:
            dt = None
    elif 'Transaction Hour' in tx:
        dt = None
    else:
        dt = None
    if dt is not None:
        x['transaction_day_of_week'] = int(dt.dayofweek)
        x['transaction_day_of_month'] = int(dt.day)
        x['transaction_month'] = int(dt.month)
        x['Transaction Hour'] = int(dt.hour)
    else:
        # fallbacks
        if 'transaction_day_of_week' in tx:
            x['transaction_day_of_week'] = int(tx['transaction_day_of_week'])
        if 'transaction_day_of_month' in tx:
            x['transaction_day_of_month'] = int(tx['transaction_day_of_month'])
        if 'transaction_month' in tx:
            x['transaction_month'] = int(tx['transaction_month'])
        if 'Transaction Hour' in tx:
            x['Transaction Hour'] = int(tx['Transaction Hour'])

    # Numeric features
    for f in ['Transaction Amount','Quantity','Customer Age','Account Age Days','Transaction Hour']:
        if f in feature_cols:
            val = tx.get(f, None)
            x[f] = val if val is not None else 0
    # customer_location_freq from mapping
    if 'Customer Location' in tx and 'customer_location_freq' in feature_cols:
        x['customer_location_freq'] = loc_freq_map.get(tx.get('Customer Location'), 0.0)
    elif 'customer_location_freq' in feature_cols:
        x['customer_location_freq'] = 0.0

    # One-hot groups
    for orig_col, templates in onehot_prefixes.items():
        if not templates:
            continue
        # get category value from tx
        key = orig_col
        cat = tx.get(key, None)
        for col in templates:
            col_suffix = col.split(orig_col + '_')[-1] if (orig_col + '_') in col else col.split('_',1)[-1]
            # create 1 if matches
            if cat is not None and str(cat) == col_suffix:
                x[col] = 1
            else:
                x[col] = 0
    # Ensure all feature_cols present in x
    for c in feature_cols:
        if c not in x:
            x[c] = 0
    # Build DataFrame row with correct column order
    row = pd.DataFrame([x], columns=feature_cols)
    return row


def predict_transaction(tx: dict):
    x = preprocess_transaction(tx)
    prob = float(rf.predict_proba(x)[:,1][0])
    # risk score 0-100
    score = int(round(prob * 100))
    # Risk bucket
    if score >= 70:
        bucket = 'HIGH'
    elif score >= 33:
        bucket = 'MEDIUM'
    else:
        bucket = 'LOW'
    # risk factors: use top 5 important features and show transaction values vs train mean
    factors = []
    for it in fi[:10]:
        fname = it['feature']
        if fname in x.columns:
            val = x[fname].iloc[0]
            info = {'feature': fname, 'value': float(val), 'importance_pct': float(it['percentage'])}
            # add comparison to train mean if numeric
            if fname in train_stats:
                info['train_mean'] = train_stats[fname]['mean']
                info['delta_from_mean'] = float(val) - train_stats[fname]['mean']
            factors.append(info)
    return {'probability': prob, 'score': score, 'risk': bucket, 'factors': factors}

sample = None
# Create a test transaction sample. Prefer raw dataset sample if available; otherwise synthesize
sample = None
if raw_df is not None:
    # attempt to read a small slice of the raw CSV to build a realistic sample
    try:
        raw_sample = pd.read_csv(raw, nrows=5)
        row = raw_sample.iloc[0]
        sample = {
            'Transaction Amount': row.get('Transaction Amount'),
            'Payment Method': row.get('Payment Method'),
            'Product Category': row.get('Product Category'),
            'Quantity': int(row.get('Quantity')) if pd.notnull(row.get('Quantity')) else 1,
            'Customer Age': int(row.get('Customer Age')) if pd.notnull(row.get('Customer Age')) else 30,
            'Customer Location': row.get('Customer Location'),
            'Device Used': row.get('Device Used'),
            'Account Age Days': int(row.get('Account Age Days')) if pd.notnull(row.get('Account Age Days')) else 100,
            'Transaction Date': row.get('Transaction Date')
        }
    except Exception:
        sample = None

if sample is None:
    # fallback: construct a plausible transaction using training statistics and a location from the persisted map
    loc_key = None
    if loc_freq_map:
        loc_key = next(iter(loc_freq_map.keys()))
    sample = {
        'Transaction Amount': float(train_df['Transaction Amount'].median()) if 'Transaction Amount' in train_df.columns else 100.0,
        'Payment Method': 'debit card',
        'Product Category': 'clothing',
        'Quantity': int(train_df['Quantity'].median()) if 'Quantity' in train_df.columns else 1,
        'Customer Age': int(train_df['Customer Age'].median()) if 'Customer Age' in train_df.columns else 30,
        'Customer Location': loc_key,
        'Device Used': 'mobile',
        'Account Age Days': int(train_df['Account Age Days'].median()) if 'Account Age Days' in train_df.columns else 100,
        'Transaction Date': None
    }

# Run test prediction
result = predict_transaction(sample)
print('Test transaction prediction:')
print(f"Probability={result['probability']:.4f}, Score={result['score']}, Risk={result['risk']}")
print('Top factor snapshots:')
for f in result['factors']:
    print(f)

# Save engine artifacts (preprocessing metadata already saved)
with open(model_dir / 'engine_readme.txt', 'w') as f:
    f.write('RiskPulse engine artifacts:\n')
    f.write('- random_forest.pkl: trained model\n')
    f.write('- preprocessing_metadata.json: expected feature columns and templates\n')
    f.write('- feature_importance.json: model feature importance\n')

print('\nSaved preprocessing metadata to model/preprocessing_metadata.json and engine_readme.txt')
