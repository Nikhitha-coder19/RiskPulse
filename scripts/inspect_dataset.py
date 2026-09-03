from pathlib import Path
import pandas as pd
import numpy as np
import sys

# Find CSV files under folders named 'data' (case-insensitive), fallback to first CSV
root = Path('.').resolve()
csv_paths = list(root.rglob('*.csv'))
if not csv_paths:
    print('No CSV files found in workspace.')
    sys.exit(1)

# Prefer CSVs whose path contains '/data/' (case-insensitive)
candidates = [p for p in csv_paths if any(part.lower() == 'data' for part in p.parts)]
if not candidates:
    # Also check for folders named 'data' anywhere in path (substring)
    candidates = [p for p in csv_paths if 'data' in str(p).lower()]

chosen = candidates[0] if candidates else csv_paths[0]
rel_chosen = chosen.relative_to(root)
print(f"Selected CSV: {rel_chosen}")

# Load safely with pandas (do not modify file)
try:
    df = pd.read_csv(chosen, low_memory=False)
except Exception as e:
    print('Failed to read CSV with pandas:', e)
    sys.exit(2)

n_rows, n_cols = df.shape
print(f"Rows: {n_rows}")
print(f"Columns: {n_cols}")

print('\nColumn names:')
for c in df.columns.tolist():
    print(' -', c)

print('\nColumn dtypes:')
print(df.dtypes)

print('\nMissing / null values per column:')
print(df.isnull().sum())

print('\nUnique values per column:')
print(df.nunique(dropna=False))

# Try to identify label column (fraudulent flag)
label_col = None
for c in df.columns:
    lower = c.lower()
    if 'fraud' in lower or lower in ('is_fraud', 'isfraud', 'label', 'class'):
        # check values
        vals = df[c].dropna().unique()
        if len(vals) > 0:
            label_col = c
            break

# If not found, look for binary columns
if label_col is None:
    for c in df.columns:
        vals = df[c].dropna().unique()
        if len(vals) == 2:
            label_col = c
            break

if label_col:
    print(f"\nDetected label column: {label_col}")
    counts = df[label_col].value_counts(dropna=False)
    print('Value counts:')
    print(counts)
    total = len(df)
    print('\nFraud / Legitimate breakdown (inferred):')
    for val, cnt in counts.items():
        pct = cnt/total*100
        print(f" - {val}: {cnt} rows ({pct:.2f}%)")
else:
    print('\nNo obvious fraud label column detected automatically.')

# Transaction amount statistics: find likely amount column
amt_col = None
amt_candidates = [c for c in df.columns if 'amount' in c.lower() or 'amt' in c.lower() or 'transactionamt' in c.lower()]
if amt_candidates:
    amt_col = amt_candidates[0]
else:
    # fallback to numeric columns with 'amount'-like meaning
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        amt_col = numeric_cols[0]

if amt_col:
    print(f"\nTransaction amount column (inferred): {amt_col}")
    # coerce to numeric
    amt = pd.to_numeric(df[amt_col], errors='coerce')
    print(' - min:', amt.min())
    print(' - max:', amt.max())
    print(' - mean:', amt.mean())
    print(' - median:', amt.median())
else:
    print('\nNo transaction amount column identified.')

# Transaction date: detect date-like columns
date_cols = [c for c in df.columns if any(k in c.lower() for k in ['date','time','timestamp'])]
parsed_dates = {}
for c in date_cols:
    try:
        parsed = pd.to_datetime(df[c], errors='coerce')
        if parsed.notnull().any():
            parsed_dates[c] = parsed
    except Exception:
        continue

if parsed_dates:
    # pick the column with the most non-null parsable dates
    best = max(parsed_dates.keys(), key=lambda k: parsed_dates[k].notnull().sum())
    dt = parsed_dates[best]
    print(f"\nTransaction date column (inferred): {best}")
    print(' - earliest:', dt.min())
    print(' - latest:', dt.max())
else:
    print('\nNo transaction date/time column identified automatically.')

print('\nFive sample rows:')
with pd.option_context('display.max_columns', None, 'display.width', 200):
    print(df.head(5).to_string(index=False))

# Identify columns likely useful for fraud detection
useful = []
not_use = []
id_like = []
for c in df.columns:
    lc = c.lower()
    nunique = df[c].nunique(dropna=False)
    # id-like
    if 'id' in lc or lc.endswith('id') or lc.startswith('id'):
        id_like.append(c)
        not_use.append((c, 'Identifier / high cardinality - not a predictive feature'))
        continue
    if nunique == n_rows:
        not_use.append((c, 'Unique per row (likely identifier)'))
        continue
    if nunique <= 1:
        not_use.append((c, 'Constant column (no predictive value)'))
        continue
    # PII-like
    if any(keyword in lc for keyword in ['name','email','phone','address']):
        not_use.append((c, 'Contains PII; avoid using raw values for privacy'))
        continue
    # label column exclude
    if label_col and c == label_col:
        not_use.append((c, 'Label column - do not use as feature'))
        continue
    # otherwise consider useful
    useful.append((c, nunique))

print('\nColumns likely useful for fraud detection (candidates):')
# sort useful by nunique desc
for c, nunique in sorted(useful, key=lambda x: -x[1])[:50]:
    print(f" - {c}: {nunique} unique values")

print('\nColumns suggested to NOT use and reasons:')
for c, reason in not_use:
    print(f" - {c}: {reason}")

# Check Transaction ID and Customer ID repeats
tran_id_cols = [c for c in df.columns if 'transaction' in c.lower() and 'id' in c.lower()]
cust_id_cols = [c for c in df.columns if 'customer' in c.lower() and 'id' in c.lower()]
# generic id columns
generic_id_cols = [c for c in df.columns if (c.lower()=='id' or c.lower().endswith('_id') or c.lower().startswith('id_'))]

print('\nTransaction ID columns found:', tran_id_cols if tran_id_cols else 'None')
print('Customer ID columns found:', cust_id_cols if cust_id_cols else 'None')
print('Generic ID-like columns:', generic_id_cols if generic_id_cols else 'None')

def check_repeats(cols):
    for c in cols:
        nunique = df[c].nunique(dropna=False)
        print(f" - {c}: {nunique} unique values out of {n_rows} rows; repeats: {n_rows - nunique}")

if tran_id_cols:
    check_repeats(tran_id_cols)
if cust_id_cols:
    check_repeats(cust_id_cols)
if generic_id_cols and not (tran_id_cols or cust_id_cols):
    check_repeats(generic_id_cols)

# Check for duplicate rows
dup_count = df.duplicated().sum()
print(f"\nDuplicate rows detected: {dup_count}")

print('\nAnalysis complete.')
