from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Locate CSV in workspace (prefer data/ folder)
root = Path('.').resolve()
csvs = list(root.rglob('*.csv'))
if not csvs:
    raise SystemExit('No CSV files found')

candidates = [p for p in csvs if 'data' in str(p).lower()]
csv_path = candidates[0] if candidates else csvs[0]
# Load original CSV (do not modify it)
df = pd.read_csv(csv_path, low_memory=False)

# Use "Is Fraudulent" as target label
label_col = None
for c in df.columns:
    if c.lower() == 'is fraudulent' or c.lower() == 'is_fraudulent' or c.lower() == 'is_fraud' or 'fraud' in c.lower():
        if 'is' in c.lower() or c.lower().startswith('fraud'):
            label_col = c
            break
if label_col is None:
    raise SystemExit('Label column "Is Fraudulent" not found')

# Parse Transaction Date and derive temporal features
if 'Transaction Date' in df.columns:
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')
    df['transaction_day_of_week'] = df['Transaction Date'].dt.dayofweek
    df['transaction_day_of_month'] = df['Transaction Date'].dt.day
    df['transaction_month'] = df['Transaction Date'].dt.month
    # also ensure Transaction Hour exists
    if 'Transaction Hour' not in df.columns:
        df['Transaction Hour'] = df['Transaction Date'].dt.hour
else:
    raise SystemExit('Transaction Date column required but not found')

# Forbidden raw features (do not use as model features)
forbidden = [
    'Transaction ID',
    'Customer ID',
    label_col,
    'IP Address',
    'Shipping Address',
    'Billing Address',
]

# Candidate features to use
candidates = [
    'Transaction Amount',
    'Payment Method',
    'Product Category',
    'Quantity',
    'Customer Age',
    'Customer Location',
    'Device Used',
    'Account Age Days',
    'Transaction Hour',
    'transaction_day_of_week',
    'transaction_day_of_month',
    'transaction_month',
]

# Keep only columns that exist in df
features = [c for c in candidates if c in df.columns]

# Safety check: drop forbidden if present
for col in forbidden:
    if col in df.columns:
        # do not drop from original file; we will exclude from feature matrix later
        pass

# IMPORTANT: Avoid data leakage
# Comments:
# Data leakage happens when information that will not be available at prediction time (or that is derived from the target) is used to create features or during preprocessing in a way that leaks target information into training. To prevent leakage we:
# - Split the data into train/test before fitting any transformers that use target information (e.g., target encoding).
# - Compute encodings or statistics (e.g., categorical levels, frequency counts) on the training set only and apply them to the test set.
# - Do not use future data (e.g., future transactions) to construct features for past transactions.

# Prepare label y and drop rows with missing label
df = df.copy()
df = df[~df[label_col].isnull()]

y = df[label_col].astype(int)

# Now split into train/test with stratification to preserve fraud ratio
RANDOM_SEED = 42
X_all = df  # we'll build feature matrices from these
X_train_df, X_test_df, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
)

# Feature engineering and encoding: Fit encodings on train only
# We'll do:
# - Frequency encoding for high-cardinality 'Customer Location'
# - One-hot (get_dummies) for Payment Method, Product Category, Device Used
# - Keep numeric cols as-is and fill missing with sensible values

# Numeric features to keep
numeric_feats = [f for f in ['Transaction Amount','Quantity','Customer Age','Account Age Days','Transaction Hour','transaction_day_of_week','transaction_day_of_month','transaction_month'] if f in X_train_df.columns]

# Customer Location frequency encoding based on train
if 'Customer Location' in X_train_df.columns:
    location_counts = X_train_df['Customer Location'].value_counts()
    location_freq = (location_counts / len(X_train_df)).to_dict()
    X_train_df['customer_location_freq'] = X_train_df['Customer Location'].map(location_freq).fillna(0.0)
    X_test_df['customer_location_freq'] = X_test_df['Customer Location'].map(location_freq).fillna(0.0)
    # include the new column in numeric_feats
    numeric_feats.append('customer_location_freq')

# One-hot encode selected categorical features using train categories only
onehot_cols = [c for c in ['Payment Method','Product Category','Device Used'] if c in X_train_df.columns]

def one_hot_fit_transform(df_train, df_apply, cols):
    # create dummies on train
    dtrain = pd.get_dummies(df_train[cols].astype(str), prefix=cols)
    dapply = pd.get_dummies(df_apply[cols].astype(str), prefix=cols)
    # align columns: ensure dapply has same columns as dtrain
    for col in dtrain.columns:
        if col not in dapply.columns:
            dapply[col] = 0
    # drop any extra columns in dapply not in dtrain
    extra = [c for c in dapply.columns if c not in dtrain.columns]
    if extra:
        dapply = dapply.drop(columns=extra)
    # ensure same column order
    dapply = dapply[dtrain.columns]
    return dtrain, dapply, list(dtrain.columns)

onehot_train = pd.DataFrame(index=X_train_df.index)
onehot_test = pd.DataFrame(index=X_test_df.index)
onehot_feature_names = []
if onehot_cols:
    dtrain, dtest, onehot_feature_names = one_hot_fit_transform(X_train_df, X_test_df, onehot_cols)
    onehot_train = dtrain
    onehot_test = dtest

# Build final feature DataFrames
X_train_parts = []
X_test_parts = []

# numeric
if numeric_feats:
    X_train_parts.append(X_train_df[numeric_feats].fillna(0))
    X_test_parts.append(X_test_df[numeric_feats].fillna(0))

# one-hot
if not onehot_train.empty:
    X_train_parts.append(onehot_train)
    X_test_parts.append(onehot_test)

# concatenate
X_train = pd.concat(X_train_parts, axis=1)
X_test = pd.concat(X_test_parts, axis=1)

# Final feature names
feature_names = X_train.columns.tolist()

# Save processed datasets (include label column)
proc_dir = root / 'data'
proc_dir.mkdir(parents=True, exist_ok=True)
train_out = proc_dir / 'processed_train.csv'
test_out = proc_dir / 'processed_test.csv'

train_save = X_train.copy()
train_save[label_col] = y_train.values
train_save.to_csv(train_out, index=False)

test_save = X_test.copy()
test_save[label_col] = y_test.values
test_save.to_csv(test_out, index=False)

# Compact summary (ONLY these lines printed)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
train_fraud = int(y_train.sum())
train_total = len(y_train)
print(f"Train fraud: {train_fraud} ({train_fraud/train_total*100:.3f}%)")
test_fraud = int(y_test.sum())
test_total = len(y_test)
print(f"Test fraud: {test_fraud} ({test_fraud/test_total*100:.3f}%)")
print(f"Feature names: {feature_names}")
print(f"Num features: {len(feature_names)}")
