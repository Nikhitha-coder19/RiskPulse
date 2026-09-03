import os
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

SEED = 42
rng = np.random.default_rng(SEED)

OUTPUT_DIR = "Data/merchant"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "merchant_transactions.csv")

N_MERCHANTS = 2500
N_TRANSACTIONS = 300_000

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# MERCHANT PROFILES
# ============================================================

merchant_ids = [f"merchant_{i:06d}" for i in range(N_MERCHANTS)]

# Merchant scenario distribution
scenario_names = [
    "normal_merchant",
    "new_merchant",
    "suspicious_merchant",
    "merchant_fraud",
    "merchant_velocity_anomaly",
    "merchant_network_anomaly",
    "legitimate_high_volume",
]

scenario_probs = [
    0.72,
    0.05,
    0.05,
    0.05,
    0.04,
    0.04,
    0.05,
]

merchant_scenarios = rng.choice(
    scenario_names,
    size=N_MERCHANTS,
    p=scenario_probs,
)

merchant_profile = pd.DataFrame({
    "merchant_id": merchant_ids,
    "merchant_scenario": merchant_scenarios,
})

# Merchant-specific baseline behaviour
merchant_profile["base_amount"] = np.clip(
    rng.normal(250, 70, N_MERCHANTS),
    30,
    600,
)

merchant_profile["base_rate"] = rng.uniform(
    0.5,
    3.0,
    N_MERCHANTS
)

# Risk multipliers
merchant_profile["risk_multiplier"] = merchant_profile[
    "merchant_scenario"
].map({
    "normal_merchant": 1.0,
    "new_merchant": 1.0,
    "suspicious_merchant": 2.5,
    "merchant_fraud": 5.0,
    "merchant_velocity_anomaly": 2.0,
    "merchant_network_anomaly": 2.0,
    "legitimate_high_volume": 1.0,
})

# ============================================================
# ASSIGN TRANSACTIONS TO MERCHANTS
# ============================================================

merchant_weights = rng.lognormal(
    mean=0,
    sigma=0.45,
    size=N_MERCHANTS,
)

merchant_weights /= merchant_weights.sum()

merchant_index = rng.choice(
    N_MERCHANTS,
    size=N_TRANSACTIONS,
    p=merchant_weights,
)

df = pd.DataFrame({
    "merchant_id": np.array(merchant_ids)[merchant_index],
})

# ============================================================
# TIMESTAMPS
# ============================================================

start = pd.Timestamp("2024-01-01")
end = pd.Timestamp("2025-01-01")

seconds_range = int((end - start).total_seconds())

df["timestamp"] = start + pd.to_timedelta(
    rng.integers(
        0,
        seconds_range,
        size=N_TRANSACTIONS,
    ),
    unit="s",
)

df = df.sort_values("timestamp").reset_index(drop=True)

# ============================================================
# MERCHANT PROFILE JOIN
# ============================================================

df = df.merge(
    merchant_profile,
    on="merchant_id",
    how="left",
)

# ============================================================
# CUSTOMER / IP / DEVICE IDENTITIES
# ============================================================

N_CUSTOMERS = 100_000
N_IPS = 60_000
N_DEVICES = 80_000

df["customer_id"] = rng.integers(
    0,
    N_CUSTOMERS,
    size=len(df),
)

df["ip_id"] = rng.integers(
    0,
    N_IPS,
    size=len(df),
)

df["device_id"] = rng.integers(
    0,
    N_DEVICES,
    size=len(df),
)

df["customer_id"] = df["customer_id"].map(
    lambda x: f"customer_{x:06d}"
)

df["ip_id"] = df["ip_id"].map(
    lambda x: f"ip_{x:06d}"
)

df["device_id"] = df["device_id"].map(
    lambda x: f"device_{x:06d}"
)

# ============================================================
# TRANSACTION AMOUNT
# ============================================================

df["transaction_amount"] = np.maximum(
    5,
    df["base_amount"] *
    rng.lognormal(
        mean=0,
        sigma=0.35,
        size=len(df),
    ),
)

# Suspicious / fraudulent merchants tend to show amount anomalies
amount_multiplier = np.ones(len(df))

amount_multiplier[
    df["merchant_scenario"].isin([
        "suspicious_merchant",
        "merchant_fraud",
    ])
] = rng.uniform(
    1.2,
    2.8,
    (
        df["merchant_scenario"].isin([
            "suspicious_merchant",
            "merchant_fraud",
        ])
    ).sum(),
)

df["transaction_amount"] *= amount_multiplier

# ============================================================
# TRANSACTION ID
# ============================================================

df["transaction_id"] = [
    f"txn_{i:09d}"
    for i in range(len(df))
]

# ============================================================
# HISTORICAL MERCHANT FEATURES
#
# IMPORTANT:
# These are calculated BEFORE the current transaction.
# ============================================================

df = df.sort_values(
    ["merchant_id", "timestamp"]
).reset_index(drop=True)

g = df.groupby("merchant_id", sort=False)

# Number of previous merchant transactions
df["merchant_transaction_count_before"] = (
    g.cumcount()
)

# Previous transaction amount
df["merchant_previous_amount"] = (
    g["transaction_amount"].shift(1)
)

# Historical average amount
df["merchant_amount_sum_before"] = (
    g["transaction_amount"]
    .cumsum()
    .shift(1)
)

df["merchant_avg_amount_before"] = np.where(
    df["merchant_transaction_count_before"] > 0,
    df["merchant_amount_sum_before"]
    / df["merchant_transaction_count_before"],
    0,
)

# ============================================================
# CUSTOMER / IP / DEVICE HISTORY
# ============================================================

# Faster deterministic calculation using cumulative sets
def cumulative_unique(series):
    seen = set()
    result = []

    for value in series:
        result.append(len(seen))
        seen.add(value)

    return result


df["merchant_unique_customers_before"] = (
    df.groupby("merchant_id")["customer_id"]
    .transform(cumulative_unique)
)

df["merchant_unique_ips_before"] = (
    df.groupby("merchant_id")["ip_id"]
    .transform(cumulative_unique)
)

df["merchant_unique_devices_before"] = (
    df.groupby("merchant_id")["device_id"]
    .transform(cumulative_unique)
)

# ============================================================
# NEW MERCHANT SIGNAL
# ============================================================

df["is_new_merchant"] = (
    df["merchant_transaction_count_before"] == 0
).astype(int)

df["merchant_history_available"] = (
    df["merchant_transaction_count_before"] > 0
).astype(int)

# ============================================================
# MERCHANT VELOCITY
# ============================================================

# Previous transaction timestamp
df["merchant_previous_timestamp"] = (
    g["timestamp"].shift(1)
)

df["seconds_since_merchant_previous"] = (
    df["timestamp"]
    - df["merchant_previous_timestamp"]
).dt.total_seconds()

df["seconds_since_merchant_previous"] = (
    df["seconds_since_merchant_previous"]
    .fillna(-1)
)

# ============================================================
# PREVIOUS 24H TRANSACTION COUNT
# ============================================================

def rolling_previous_24h(group):
    timestamps = group["timestamp"].values
    result = np.zeros(len(group), dtype=int)

    left = 0

    for i in range(len(group)):
        while (
            left < i
            and timestamps[i] - timestamps[left]
            > np.timedelta64(24, "h")
        ):
            left += 1

        result[i] = i - left

    return pd.Series(
        result,
        index=group.index,
    )


df["merchant_transactions_last_24h"] = (
    df.groupby("merchant_id", group_keys=False)
    .apply(rolling_previous_24h)
    .reset_index(level=0, drop=True)
)

# ============================================================
# MERCHANT NETWORK FEATURES
# ============================================================

df["ip_unique_merchants_before"] = (
    df.groupby("ip_id")["merchant_id"]
    .transform(cumulative_unique)
)

df["ip_unique_customers_before"] = (
    df.groupby("ip_id")["customer_id"]
    .transform(cumulative_unique)
)

# ============================================================
# HISTORICAL FRAUD COUNT
# ============================================================

# Temporary fraud probability based on merchant scenario
scenario = df["merchant_scenario"]

fraud_probability = np.select(
    [
        scenario == "normal_merchant",
        scenario == "new_merchant",
        scenario == "suspicious_merchant",
        scenario == "merchant_fraud",
        scenario == "merchant_velocity_anomaly",
        scenario == "merchant_network_anomaly",
        scenario == "legitimate_high_volume",
    ],
    [
        0.01,
        0.08,
        0.30,
        0.45,
        0.22,
        0.20,
        0.015,
    ],
    default=0.02,
)

# Amount anomaly increases risk
amount_ratio = np.where(
    df["merchant_avg_amount_before"] > 0,
    df["transaction_amount"]
    / df["merchant_avg_amount_before"],
    1.0,
)

fraud_probability += np.where(
    amount_ratio > 2.0,
    0.08,
    0,
)

# New merchant gets a small uncertainty/risk uplift,
# but is NOT automatically fraud.
fraud_probability += np.where(
    df["is_new_merchant"] == 1,
    0.03,
    0,
)

# Velocity anomaly
fraud_probability += np.where(
    df["merchant_transactions_last_24h"] > 10,
    0.08,
    0,
)

fraud_probability = np.clip(
    fraud_probability,
    0,
    0.90,
)

df["is_fraud"] = (
    rng.random(len(df)) < fraud_probability
).astype(int)

# ============================================================
# HISTORICAL FRAUD FEATURES
#
# Recalculate using ONLY previous transactions.
# ============================================================

df["merchant_fraud_count_before"] = (
    df.groupby("merchant_id")["is_fraud"]
    .transform(
        lambda s: s.cumsum().shift(1).fillna(0)
    )
)

df["merchant_fraud_rate_before"] = np.where(
    df["merchant_transaction_count_before"] > 0,
    df["merchant_fraud_count_before"]
    / df["merchant_transaction_count_before"],
    0,
)

# ============================================================
# RISK SIGNALS
# ============================================================

df["amount_deviation_ratio"] = np.where(
    df["merchant_avg_amount_before"] > 0,
    df["transaction_amount"]
    / df["merchant_avg_amount_before"],
    1.0,
)

df["high_amount_deviation"] = (
    df["amount_deviation_ratio"] >= 2.0
).astype(int)

df["high_velocity"] = (
    df["merchant_transactions_last_24h"] >= 10
).astype(int)

df["merchant_high_historical_fraud"] = (
    df["merchant_fraud_rate_before"] >= 0.15
).astype(int)

# ============================================================
# CLEANUP
# ============================================================

df = df.sort_values("timestamp").reset_index(drop=True)

df = df[
    [
        "transaction_id",
        "timestamp",
        "merchant_id",
        "customer_id",
        "ip_id",
        "device_id",
        "transaction_amount",

        "merchant_scenario",

        "merchant_transaction_count_before",
        "merchant_avg_amount_before",
        "merchant_previous_amount",

        "merchant_unique_customers_before",
        "merchant_unique_ips_before",
        "merchant_unique_devices_before",

        "merchant_transactions_last_24h",
        "seconds_since_merchant_previous",

        "ip_unique_merchants_before",
        "ip_unique_customers_before",

        "merchant_fraud_count_before",
        "merchant_fraud_rate_before",

        "is_new_merchant",
        "merchant_history_available",

        "amount_deviation_ratio",
        "high_amount_deviation",
        "high_velocity",
        "merchant_high_historical_fraud",

        "is_fraud",
    ]
]

# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False,
)

print("=" * 70)
print("MERCHANT DATASET GENERATED")
print("=" * 70)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Merchants: {df['merchant_id'].nunique():,}")

print("\nScenario counts:")
print(
    df["merchant_scenario"]
    .value_counts()
    .to_string()
)

print("\nFraud rate:")
print(
    round(df["is_fraud"].mean(), 4)
)

print("\nNew merchant transactions:")
print(
    df["is_new_merchant"].sum()
)

print("\nFraud rate among new merchant transactions:")
print(
    round(
        df.loc[
            df["is_new_merchant"] == 1,
            "is_fraud"
        ].mean(),
        4,
    )
)

print("\nScenario fraud rates:")
print(
    df.groupby("merchant_scenario")["is_fraud"]
    .agg(["count", "sum", "mean"])
    .round(4)
    .to_string()
)

print("\nSaved to:")
print(OUTPUT_PATH)