from pathlib import Path
import pandas as pd
import numpy as np
import sys

root = Path('.').resolve()
csv_paths = list(root.rglob('*.csv'))
if not csv_paths:
    print('No CSV files found.')
    sys.exit(1)

# prefer files under 'data' folder
candidates = [p for p in csv_paths if 'data' in str(p).lower()]
chosen = candidates[0] if candidates else csv_paths[0]
print(f"Using: {chosen.relative_to(root)}\n")

df = pd.read_csv(chosen, low_memory=False)

# detect label column
label_col = None
for c in df.columns:
    lc = c.lower()
    if 'fraud' in lc or lc in ('is_fraud','isfraud','label','class'):
        label_col = c
        break
if label_col is None:
    for c in df.columns:
        vals = df[c].dropna().unique()
        if len(vals) == 2:
            label_col = c
            break
if label_col is None:
    print('No binary label column detected. Exiting.')
    sys.exit(2)

# ensure label is numeric 0/1
df[label_col] = pd.to_numeric(df[label_col], errors='coerce').fillna(0).astype(int)

# helper
def breakdown(col, top_n=None, bins=None, labels=None):
    s = df.copy()
    if bins is not None:
        s['_group'] = pd.cut(s[col], bins=bins, labels=labels, include_lowest=True)
    else:
        s['_group'] = s[col]
    grp = s.groupby('_group', dropna=False)[label_col].agg(['count','sum']).rename(columns={'sum':'fraud'})
    grp['fraud_pct'] = (grp['fraud'] / grp['count'] * 100).round(3)
    grp = grp.reset_index().rename(columns={'_group': 'group'})
    if top_n:
        # for location: choose top N by count
        top_groups = grp.sort_values('count', ascending=False).head(top_n)['group']
        grp = grp[grp['group'].isin(top_groups)].sort_values('count', ascending=False)
    return grp

results = {}

# 1 Payment Method
results['payment_method'] = breakdown('Payment Method')
# 2 Product Category
results['product_category'] = breakdown('Product Category')
# 3 Device Used
results['device_used'] = breakdown('Device Used')
# 4 Transaction Hour
if 'Transaction Hour' in df.columns:
    results['transaction_hour'] = breakdown('Transaction Hour')
else:
    # try to parse hour from Transaction Date
    if 'Transaction Date' in df.columns:
        dt = pd.to_datetime(df['Transaction Date'], errors='coerce')
        df['Transaction Hour'] = dt.dt.hour
        results['transaction_hour'] = breakdown('Transaction Hour')

# 5 Customer Location top 10
results['customer_location_top10'] = breakdown('Customer Location', top_n=10)

# 6 Account Age Days bins
if 'Account Age Days' in df.columns:
    bins = [-1,7,30,90,180,365, np.inf]
    labels = ['0-7d','8-30d','31-90d','91-180d','181-365d','>365d']
    results['account_age_bins'] = breakdown('Account Age Days', bins=bins, labels=labels)

# 7 Transaction Amount bins
if 'Transaction Amount' in df.columns:
    bins = [0,50,100,200,500,1000,5000, np.inf]
    labels = ['0-50','51-100','101-200','201-500','501-1000','1001-5000','5001+']
    results['amount_bins'] = breakdown('Transaction Amount', bins=bins, labels=labels)

# 8 Quantity
if 'Quantity' in df.columns:
    results['quantity'] = breakdown('Quantity')

# overall fraud pct
total = len(df)
total_fraud = int(df[label_col].sum())
overall_pct = total_fraud / total * 100

# Print compact results
pd.set_option('display.max_rows', 200)
print(f"Overall: total={total}, fraud={total_fraud}, fraud_pct={overall_pct:.3f}%\n")

for k in ['payment_method','product_category','device_used','transaction_hour','customer_location_top10','account_age_bins','amount_bins','quantity']:
    if k not in results:
        continue
    print(k.replace('_',' ').title())
    dfk = results[k]
    print(dfk.to_string(index=False))
    print()

# Analyze dramatic differences
dramatic = []
for name, dfk in results.items():
    if dfk.empty:
        continue
    max_pct = dfk['fraud_pct'].max()
    min_pct = dfk['fraud_pct'].min()
    range_pct = max_pct - min_pct
    ratio = (max_pct / min_pct) if min_pct>0 else np.inf
    if range_pct >= 10 or ratio >= 3:
        dramatic.append((name, min_pct, max_pct, range_pct, ratio))

if dramatic:
    print('Features with dramatic fraud-rate differences (range_pct>=10% or max/min>=3):')
    for name, mn, mx, rng, ratio in dramatic:
        print(f" - {name}: min={mn:.3f}%, max={mx:.3f}%, range={rng:.3f}%, ratio={ratio:.2f}")
else:
    print('No features with dramatic fraud-rate differences by the chosen thresholds.')

# Temporal support check
if 'Transaction Date' in df.columns:
    dt = pd.to_datetime(df['Transaction Date'], errors='coerce')
    nonnull_dates = dt.dropna()
    if not nonnull_dates.empty:
        first = nonnull_dates.min()
        last = nonnull_dates.max()
        days = (last - first).days + 1
        tx_per_day = total / days
        print(f"\nTransaction Date range: {first.date()} to {last.date()} ({days} days), avg tx/day={tx_per_day:.1f}")
        if days >= 30 and tx_per_day >= 100:
            print('Conclusion: enough temporal variation and volume to support a temporal fraud-spike detector.')
        else:
            print('Conclusion: not enough temporal span or volume for a reliable temporal spike detector.')

# Done
print('\nDone.')
