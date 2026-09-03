from pathlib import Path
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "Data" / "archive" / "Fraudulent_E-Commerce_Transaction_Data.csv"
OUTPUT_PATH = ROOT / "model" / "dataset_blueprint.json"

print("Loading original dataset...")
df = pd.read_csv(DATA_PATH)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

blueprint = {
    "source_dataset": DATA_PATH.name,
    "rows": int(len(df)),
    "columns": list(df.columns),
}

# -----------------------------
# Numerical distributions
# -----------------------------

numeric_columns = [
    "Transaction Amount",
    "Quantity",
    "Customer Age",
    "Account Age Days",
    "Transaction Hour",
]

blueprint["numeric"] = {}

for column in numeric_columns:
    series = pd.to_numeric(df[column], errors="coerce")

    blueprint["numeric"][column] = {
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()),
        "p01": float(series.quantile(0.01)),
        "p05": float(series.quantile(0.05)),
        "p25": float(series.quantile(0.25)),
        "p75": float(series.quantile(0.75)),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
    }

# -----------------------------
# Categorical distributions
# -----------------------------

categorical_columns = [
    "Payment Method",
    "Product Category",
    "Device Used",
]

blueprint["categorical"] = {}

for column in categorical_columns:
    counts = df[column].value_counts(normalize=True)

    blueprint["categorical"][column] = {
        str(value): float(percent)
        for value, percent in counts.items()
    }

# -----------------------------
# Location distribution
# -----------------------------

location_counts = df["Customer Location"].value_counts(normalize=True)

blueprint["location"] = {
    "unique_locations": int(df["Customer Location"].nunique()),
    "top_locations": {
        str(value): float(percent)
        for value, percent in location_counts.head(100).items()
    }
}

# -----------------------------
# Time distribution
# -----------------------------

dates = pd.to_datetime(df["Transaction Date"], errors="coerce")

blueprint["time"] = {
    "date_min": str(dates.min()),
    "date_max": str(dates.max()),
    "hour_distribution": {
        str(int(hour)): float(percent)
        for hour, percent in df["Transaction Hour"]
        .value_counts(normalize=True)
        .sort_index()
        .items()
    },
    "day_of_week_distribution": {
        str(int(day)): float(percent)
        for day, percent in df["Transaction Date"]
        .pipe(lambda x: pd.to_datetime(x).dt.dayofweek)
        .value_counts(normalize=True)
        .sort_index()
        .items()
    }
}

# -----------------------------
# Fraud distribution
# -----------------------------

fraud_rate = df["Is Fraudulent"].mean()

blueprint["fraud"] = {
    "fraud_rate": float(fraud_rate),
    "legitimate_rate": float(1 - fraud_rate),
    "fraud_count": int(df["Is Fraudulent"].sum()),
    "legitimate_count": int((df["Is Fraudulent"] == 0).sum()),
}

# -----------------------------
# Repetition / entity analysis
# -----------------------------

entity_columns = [
    "Customer ID",
    "Transaction ID",
    "IP Address",
    "Shipping Address",
    "Billing Address",
    "Customer Location",
    "Device Used",
]

blueprint["entities"] = {}

for column in entity_columns:
    blueprint["entities"][column] = {
        "unique_values": int(df[column].nunique()),
        "unique_ratio": float(df[column].nunique() / len(df)),
    }

# -----------------------------
# Amount by fraud status
# -----------------------------

amount_by_fraud = (
    df.groupby("Is Fraudulent")["Transaction Amount"]
    .agg(["mean", "median", "std"])
)

blueprint["amount_by_fraud"] = {
    str(int(index)): {
        "mean": float(row["mean"]),
        "median": float(row["median"]),
        "std": float(row["std"]),
    }
    for index, row in amount_by_fraud.iterrows()
}

# -----------------------------
# Save blueprint
# -----------------------------

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(blueprint, f, indent=2)

print()
print("===================================")
print("DATASET BLUEPRINT CREATED")
print("===================================")
print(f"Saved to: {OUTPUT_PATH}")
print()
print("Fraud rate:", f"{fraud_rate * 100:.2f}%")
print("Unique customers:", f"{df['Customer ID'].nunique():,}")
print("Unique transactions:", f"{df['Transaction ID'].nunique():,}")
print("Unique IPs:", f"{df['IP Address'].nunique():,}")
print("Unique locations:", f"{df['Customer Location'].nunique():,}")
print("Unique devices:", f"{df['Device Used'].nunique():,}")