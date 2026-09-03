import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Data" / "merchant" / "merchant_transactions.csv"
MODEL_DIR = ROOT / "model"

TARGET_COL = "is_fraud"

EXCLUDED_COLUMNS = {
    "transaction_id",
    "timestamp",
    "merchant_id",
    "customer_id",
    "ip_id",
    "device_id",
    "merchant_scenario",
}


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Merchant dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = {
        "transaction_amount",
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
        TARGET_COL,
    }

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(f"Missing required merchant columns: {missing}")

    return df


def build_time_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values(
        "timestamp",
        kind="mergesort",
    )

    split_index = max(
        1,
        int(len(df) * (1.0 - test_fraction)),
    )

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Chronological split produced an empty partition."
        )

    return train_df, test_df


def prepare_features(df):
    X = df.drop(
        columns=[TARGET_COL],
        errors="ignore",
    ).copy()

    X = X[
        [
            col
            for col in X.columns
            if col not in EXCLUDED_COLUMNS
        ]
    ]

    X = X.select_dtypes(
        include=[np.number]
    ).copy()

    y = df[TARGET_COL].astype(int)

    if X.empty:
        raise ValueError(
            "No numeric merchant features remain."
        )

    return X, y


def evaluate_model(
    name,
    model,
    X_test,
    y_test,
):
    y_proba = model.predict_proba(X_test)[:, 1]

    y_pred = (
        y_proba >= 0.5
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        y_proba,
    )

    fpr = (
        fp / float(fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    return {
        "model": name,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "fp": int(fp),
        "fn": int(fn),
        "fpr": float(fpr),
        "tp": int(tp),
        "tn": int(tn),
        "confusion_matrix": [
            [int(tn), int(fp)],
            [int(fn), int(tp)],
        ],
    }


def main():
    print("=" * 70)
    print("TRAINING MERCHANT RISK MODELS")
    print("=" * 70)

    df = load_data()

    print(f"Rows: {len(df):,}")
    print(
        f"Fraud rate: {df[TARGET_COL].mean():.4f}"
    )

    train_df, test_df = build_time_split(
        df,
        test_fraction=0.2,
    )

    X_train, y_train = prepare_features(
        train_df
    )

    X_test, y_test = prepare_features(
        test_df
    )

    if list(X_train.columns) != list(
        X_test.columns
    ):
        X_test = X_test.reindex(
            columns=X_train.columns
        )

    print(
        f"Train rows: {len(train_df):,}"
    )

    print(
        f"Test rows: {len(test_df):,}"
    )

    print(
        f"Merchant features: {len(X_train.columns)}"
    )

    print("\nFeatures:")
    for col in X_train.columns:
        print(f"  - {col}")

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
    )

    X_test_imp = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
    )

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}

    for model_name, model in models.items():
        print(
            f"\nTraining {model_name}..."
        )

        model.fit(
            X_train_imp,
            y_train,
        )

        results[model_name] = evaluate_model(
            model_name,
            model,
            X_test_imp,
            y_test,
        )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        models["Logistic Regression"],
        MODEL_DIR
        / "merchant_logistic_regression.pkl",
    )

    joblib.dump(
        imputer,
        MODEL_DIR
        / "merchant_imputer.pkl",
    )

    metadata = {
        "feature_columns": list(
            X_train.columns
        ),
        "excluded_columns": sorted(
            EXCLUDED_COLUMNS
        ),
        "target_column": TARGET_COL,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }

    with open(
        MODEL_DIR
        / "merchant_preprocessing_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    report = {
        "data_path": str(DATA_PATH),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "target_column": TARGET_COL,
        "feature_columns": list(
            X_train.columns
        ),
        "excluded_columns": sorted(
            EXCLUDED_COLUMNS
        ),
        "metrics": results,
    }

    with open(
        MODEL_DIR
        / "merchant_results.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("MERCHANT MODEL COMPARISON")
    print("=" * 70)

    print(
        "Model | Precision | Recall | F1 | ROC-AUC | FP | FN | FPR"
    )

    for model_name in [
        "Logistic Regression",
        "Random Forest",
    ]:
        r = results[model_name]

        print(
            f"{model_name} | "
            f"{r['precision']:.3f} | "
            f"{r['recall']:.3f} | "
            f"{r['f1']:.3f} | "
            f"{r['roc_auc']:.3f} | "
            f"{r['fp']} | "
            f"{r['fn']} | "
            f"{r['fpr']:.6f}"
        )

    print("\nSaved:")
    print(
        "  model/merchant_logistic_regression.pkl"
    )
    print(
        "  model/merchant_imputer.pkl"
    )
    print(
        "  model/merchant_preprocessing_metadata.json"
    )
    print(
        "  model/merchant_results.json"
    )


if __name__ == "__main__":
    main()