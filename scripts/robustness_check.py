import json
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

root = Path('.').resolve()
train_path = root / 'data' / 'processed_train.csv'
test_path = root / 'data' / 'processed_test.csv'
model_dir = root / 'model'
model_dir.mkdir(exist_ok=True)

if not train_path.exists() or not test_path.exists():
    raise SystemExit('processed_train.csv or processed_test.csv missing')

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
label_col = 'Is Fraudulent'
if label_col not in train.columns or label_col not in test.columns:
    raise SystemExit('Label column missing in processed files')

# 1. Class distribution
train_counts = train[label_col].value_counts().to_dict()
test_counts = test[label_col].value_counts().to_dict()
train_total = int(len(train))
test_total = int(len(test))

# 2. Numeric feature separation stats
# Identify model features (exclude label)
features = [c for c in train.columns if c != label_col]

# numeric features
numeric_feats = train[features].select_dtypes(include=[np.number]).columns.tolist()

num_stats = {}
for f in numeric_feats:
    stats_legit = {}
    stats_fraud = {}
    legit = test[test[label_col]==0][f].dropna()
    fraud = test[test[label_col]==1][f].dropna()
    stats_legit['mean'] = float(legit.mean()) if len(legit)>0 else None
    stats_legit['median'] = float(legit.median()) if len(legit)>0 else None
    stats_legit['min'] = float(legit.min()) if len(legit)>0 else None
    stats_legit['max'] = float(legit.max()) if len(legit)>0 else None
    stats_fraud['mean'] = float(fraud.mean()) if len(fraud)>0 else None
    stats_fraud['median'] = float(fraud.median()) if len(fraud)>0 else None
    stats_fraud['min'] = float(fraud.min()) if len(fraud)>0 else None
    stats_fraud['max'] = float(fraud.max()) if len(fraud)>0 else None
    # t-test
    try:
        tstat, pval = stats.ttest_ind(legit, fraud, equal_var=False, nan_policy='omit')
    except Exception:
        tstat, pval = None, None
    num_stats[f] = {'legit': stats_legit, 'fraud': stats_fraud, 't_stat': tstat, 'p_value': pval}

# 3. Fraud rate across bins for specific features
bin_results = {}
# Account Age Days
if 'Account Age Days' in test.columns:
    bins = [-1,7,30,90,180,365,np.inf]
    labels = ['0-7d','8-30d','31-90d','91-180d','181-365d','>365d']
    test['account_age_bin'] = pd.cut(test['Account Age Days'], bins=bins, labels=labels, include_lowest=True)
    gp = test.groupby('account_age_bin')[label_col].agg(['count','sum'])
    gp['fraud_rate'] = (gp['sum']/gp['count']).fillna(0).round(6)
    bin_results['Account Age Days'] = gp.reset_index().to_dict(orient='records')

# Transaction Amount
if 'Transaction Amount' in test.columns:
    bins_amt = [0,50,100,200,500,1000,5000,np.inf]
    labels_amt = ['0-50','51-100','101-200','201-500','501-1000','1001-5000','5001+']
    test['amount_bin'] = pd.cut(test['Transaction Amount'], bins=bins_amt, labels=labels_amt, include_lowest=True)
    gp = test.groupby('amount_bin')[label_col].agg(['count','sum'])
    gp['fraud_rate'] = (gp['sum']/gp['count']).fillna(0).round(6)
    bin_results['Transaction Amount'] = gp.reset_index().to_dict(orient='records')

# Transaction Hour
if 'Transaction Hour' in test.columns:
    # hourly
    gp = test.groupby('Transaction Hour')[label_col].agg(['count','sum'])
    gp['fraud_rate'] = (gp['sum']/gp['count']).fillna(0).round(6)
    hourly = gp.reset_index().to_dict(orient='records')
    # aggregated periods
    bins_hr = [-1,5,11,17,23]
    labels_hr = ['night(0-5)','morning(6-11)','afternoon(12-17)','evening(18-23)']
    test['hour_period'] = pd.cut(test['Transaction Hour'], bins=bins_hr, labels=labels_hr, include_lowest=True)
    gp2 = test.groupby('hour_period')[label_col].agg(['count','sum'])
    gp2['fraud_rate'] = (gp2['sum']/gp2['count']).fillna(0).round(6)
    bin_results['Transaction Hour'] = {'hourly': hourly, 'periods': gp2.reset_index().to_dict(orient='records')}

# 4. Check if any feature nearly determines label
deterministic_flags = []
# For categorical-like features (including one-hot), check categories with near-0 or near-1 fraud rate and significant support
cat_feats = [c for c in train.columns if c not in numeric_feats and c != label_col]
for c in cat_feats:
    vals = test[c].fillna('<<NA>>')
    gp = vals.to_frame().join(test[label_col]).groupby(c)[label_col].agg(['count','sum'])
    gp['fraud_rate'] = (gp['sum']/gp['count']).fillna(0)
    # find categories with fraud_rate==1.0 or 0.0 with count > threshold (0.001 of test)
    thresh = max(5, int(0.001 * len(test)))
    for idx, row in gp.iterrows():
        if row['count'] >= thresh and (row['fraud_rate'] >= 0.999 or row['fraud_rate'] <= 0.001):
            deterministic_flags.append({'feature': c, 'value': idx, 'count': int(row['count']), 'fraud_rate': float(row['fraud_rate'])})

# For numeric features, check unique values with extreme fraud rates
for f in numeric_feats:
    vals = test[[f, label_col]].dropna()
    if vals.empty:
        continue
    grouped = vals.groupby(f)[label_col].agg(['count','sum'])
    grouped['fraud_rate'] = (grouped['sum']/grouped['count'])
    thresh = max(5, int(0.001 * len(test)))
    extreme = grouped[(grouped['count'] >= thresh) & ((grouped['fraud_rate'] >= 0.999) | (grouped['fraud_rate'] <= 0.001))]
    for idx, row in extreme.iterrows():
        deterministic_flags.append({'feature': f, 'value': float(idx), 'count': int(row['count']), 'fraud_rate': float(row['fraud_rate'])})

# 5. Train/test feature distribution differences
dist_diffs = {'numeric':{}, 'categorical':{}}
for f in numeric_feats:
    a = train[f].dropna()
    b = test[f].dropna()
    if len(a)==0 or len(b)==0:
        continue
    pooled_std = np.sqrt(((a.std()**2*(len(a)-1) + b.std()**2*(len(b)-1)) / (len(a)+len(b)-2))) if (len(a)+len(b)-2)>0 else 0.0
    smd = (a.mean() - b.mean()) / pooled_std if pooled_std>0 else 0.0
    # ks test
    try:
        ks_stat, ks_p = stats.ks_2samp(a, b)
    except Exception:
        ks_stat, ks_p = None, None
    dist_diffs['numeric'][f] = {'mean_train': float(a.mean()), 'mean_test': float(b.mean()), 'smd': float(smd), 'ks_stat': ks_stat, 'ks_pvalue': ks_p}

# categorical: compare top categories proportions
for f in cat_feats:
    a = train[f].fillna('<<NA>>')
    b = test[f].fillna('<<NA>>')
    pa = a.value_counts(normalize=True)
    pb = b.value_counts(normalize=True)
    cats = set(pa.index).union(pb.index)
    diffs = []
    for cat in cats:
        dif = abs(pa.get(cat,0.0) - pb.get(cat,0.0))
        if dif >= 0.10:  # major difference threshold
            diffs.append({'category': cat, 'train_pct': float(pa.get(cat,0.0)), 'test_pct': float(pb.get(cat,0.0)), 'abs_diff': float(dif)})
    if diffs:
        dist_diffs['categorical'][f] = diffs

# 6. Target leakage checks
leakage_findings = []
# Check feature names
for f in features:
    if 'fraud' in f.lower():
        leakage_findings.append({'feature': f, 'issue': 'Feature name contains "fraud"'})
    if f.lower() in ('transaction id','customer id','id') or f.lower().endswith('_id'):
        leakage_findings.append({'feature': f, 'issue': 'Identifier-like feature present'})
# Check features with extremely high importance could be leakage proxies (we flag but do not conclude)

# 7. LIMITATIONS
limitations = [
    'Dataset is synthetic; patterns may not reflect real-world behavior.',
    'Feature importances and separations indicate model reliance, not causation.',
    'Binning choices are heuristic and for exploratory checks only.'
]

report = {
    'class_distribution': {
        'train': {'total': train_total, 'counts': train_counts},
        'test': {'total': test_total, 'counts': test_counts}
    },
    'numeric_feature_stats_by_label': num_stats,
    'binned_rates': bin_results,
    'deterministic_flags': deterministic_flags,
    'distribution_differences': dist_diffs,
    'leakage_checks': leakage_findings,
    'limitations': limitations
}

# Save report
with open(model_dir / 'robustness_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=(lambda o: None))

# Print concise summary
print('Class distribution (train vs test):')
print(f" train: {train_counts} (n={train_total})")
print(f" test:  {test_counts} (n={test_total})\n")

print('Top numeric separation highlights (selected features):')
for f in ['Account Age Days','Transaction Amount','Transaction Hour']:
    if f in num_stats:
        ld = num_stats[f]
        print(f"{f}: legit_mean={ld['legit']['mean']:.3f}, fraud_mean={ld['fraud']['mean']:.3f}, legit_median={ld['legit']['median']:.3f}, fraud_median={ld['fraud']['median']:.3f}")

print('\nDeterministic / near-deterministic flags:')
if deterministic_flags:
    for d in deterministic_flags[:10]:
        print(d)
else:
    print(' None found')

print('\nMajor train/test distribution differences (categorical):')
if dist_diffs['categorical']:
    for f, diffs in dist_diffs['categorical'].items():
        print(f"{f}: {len(diffs)} categories with abs diff >= 0.10")
else:
    print(' None found')

print('\nLIMITATIONS:')
for L in limitations:
    print('-', L)

print('\nSaved model/robustness_report.json')
