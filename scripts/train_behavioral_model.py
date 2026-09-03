import json
from pathlib import Path

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
from sklearn.model_selection import TimeSeriesSplit


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Data" / "behavioral" / "behavioral_transactions.csv"
MODEL_DIR = ROOT / "model"

TARGET_COL = "is_fraud"
EXCLUDED_COLUMNS = {
    "transaction_id",
    "customer_id",
    "merchant_id",
    "device_id",
    "ip_id",
    "shipping_address_id",
    "billing_address_id",
    "scenario_type",
}


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Behavioral dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = {
        "timestamp",
        TARGET_COL,
        "transaction_amount",
        "quantity",
        "customer_age",
        "account_age_days",
        "customer_transaction_count_before",
        "customer_avg_amount_before",
        "customer_max_amount_before",
        "customer_amount_deviation",
        "customer_transactions_last_24h",
        "customer_transactions_last_7d",
        "merchant_transaction_count_before",
        "merchant_avg_amount_before",
        "merchant_transactions_last_24h",
        "device_transaction_count_before",
        "ip_transaction_count_before",
        "ip_unique_customers_before",
        "ip_unique_merchants_before",
        "shipping_address_transaction_count_before",
        "billing_address_transaction_count_before",
        "shipping_address_unique_customers_before",
        "billing_address_unique_customers_before",
        "shipping_address_unique_merchants_before",
        "billing_address_unique_merchants_before",
        "shipping_address_transactions_last_24h",
        "billing_address_transactions_last_24h",
        "hours_since_customer_previous_transaction",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required behavioral columns: {missing}")
    return df


def build_time_split(
    df: pd.DataFrame,
    train_fraction: float = 0.64,
    validation_fraction: float = 0.16,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1.")

    if not (0.0 < validation_fraction < 1.0):
        raise ValueError("validation_fraction must be between 0 and 1.")

    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be less than 1.")

    df = df.sort_values("timestamp", kind="mergesort").copy()

    train_end = int(len(df) * train_fraction)
    validation_end = int(len(df) * (train_fraction + validation_fraction))

    train_df = df.iloc[:train_end].copy()
    validation_df = df.iloc[train_end:validation_end].copy()
    test_df = df.iloc[validation_end:].copy()

    if train_df.empty or validation_df.empty or test_df.empty:
        raise ValueError("Chronological split produced an empty partition.")

    return train_df, validation_df, test_df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COL], errors="ignore").copy()
    X = X[[col for col in X.columns if col not in EXCLUDED_COLUMNS]]
    X = X.select_dtypes(include=[np.number]).copy()

    if "timestamp" in X.columns:
        X = X.drop(columns=["timestamp"])

    y = df[TARGET_COL].astype(int)

    if X.empty:
        raise ValueError("No numeric model features remain after feature filtering.")

    X = X.copy()
    X["hours_since_customer_previous_transaction"] = pd.to_numeric(
        X["hours_since_customer_previous_transaction"], errors="coerce"
    )
    X["hours_since_customer_previous_transaction"] = X["hours_since_customer_previous_transaction"].fillna(
        -1.0
    )
    return X, y


def evaluate_model(
    name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    fpr = fp / float(fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "model": name,
        "threshold": float(threshold),
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
def select_threshold(
    model,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> dict:
    y_proba = model.predict_proba(X_validation)[:, 1]

    thresholds = np.arange(0.05, 0.96, 0.01)

    cost_ratios = [1, 2, 5, 10, 20]
    cost_results = {}

    for ratio in cost_ratios:
        best = None

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            tn, fp, fn, tp = confusion_matrix(
                y_validation,
                y_pred,
                labels=[0, 1],
            ).ravel()

            cost = (fn * ratio) + fp

            if best is None or cost < best["cost"]:
                best = {
                    "threshold": float(threshold),
                    "cost": int(cost),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                    "tn": int(tn),
                }

        cost_results[f"fn_fp_{ratio}_to_1"] = best

    # Operational choice:
    # Missing fraud is treated as 10x more costly than
    # unnecessarily flagging a legitimate transaction.
    selected_ratio = 10
    selected = cost_results[f"fn_fp_{selected_ratio}_to_1"]

    return {
        "selected_cost_ratio": f"{selected_ratio}:1",
        "selected_threshold": selected["threshold"],
        "selected_cost": selected["cost"],
        "selected_fp": selected["fp"],
        "selected_fn": selected["fn"],
        "cost_analysis": cost_results,
    }


def main() -> None:
    df = load_data()

    train_df, validation_df, test_df = build_time_split(df)

    X_train, y_train = prepare_features(train_df)
    X_validation, y_validation = prepare_features(validation_df)
    X_test, y_test = prepare_features(test_df)

    if list(X_train.columns) != list(X_validation.columns):
        X_validation = X_validation.reindex(columns=X_train.columns)

    if list(X_train.columns) != list(X_test.columns):
        X_test = X_test.reindex(columns=X_train.columns)

    imputer = SimpleImputer(strategy="median")

    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
    )

    X_validation_imp = pd.DataFrame(
        imputer.transform(X_validation),
        columns=X_validation.columns,
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

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    import joblib

    model_results = {}
    threshold_results = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        model.fit(X_train_imp, y_train)

        threshold_info = select_threshold(
            model,
            X_validation_imp,
            y_validation,
        )

        threshold_results[model_name] = threshold_info

        final_threshold = threshold_info["selected_threshold"]

        final_metrics = evaluate_model(
            model_name,
            model,
            X_test_imp,
            y_test,
            final_threshold,
        )

        model_results[model_name] = final_metrics

        print(
            f"{model_name} | "
            f"threshold={final_threshold:.2f} | "
            f"precision={final_metrics['precision']:.3f} | "
            f"recall={final_metrics['recall']:.3f} | "
            f"F1={final_metrics['f1']:.3f} | "
            f"ROC-AUC={final_metrics['roc_auc']:.3f}"
        )

        filename = {
            "Logistic Regression": "behavioral_logistic_regression.pkl",
            "Random Forest": "behavioral_random_forest.pkl",
        }[model_name]

        joblib.dump(model, MODEL_DIR / filename)

    # Select the model using validation-set performance/cost,
    # before touching the final test set.
    selected_model_name = min(
        threshold_results,
        key=lambda name: threshold_results[name]["selected_cost"],
    )

    selected_model = models[selected_model_name]
    selected_threshold = threshold_results[selected_model_name]["selected_threshold"]
    joblib.dump(
        imputer,
        MODEL_DIR / "behavioral_imputer.pkl",
    )

    report = {
        "data_path": str(DATA_PATH),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "final_test_rows": int(len(test_df)),
        "target_column": TARGET_COL,
        "excluded_columns": sorted(EXCLUDED_COLUMNS),
        "feature_count": int(len(X_train.columns)),
        "features": list(X_train.columns),
        "split": {
            "train_fraction": 0.64,
            "validation_fraction": 0.16,
            "final_test_fraction": 0.20,
            "method": "chronological",
        },
        "selected_model": selected_model_name,
        "selected_threshold": float(selected_threshold),
        "metrics": model_results,
        "threshold_selection": threshold_results,
    }

    with open(
        MODEL_DIR / "behavioral_final_results.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL BEHAVIORAL MODEL COMPARISON")
    print("=" * 70)

    print(
        "Model | Threshold | Precision | Recall | F1 | ROC-AUC | FP | FN | FPR"
    )

    for model_name in [
        "Logistic Regression",
        "Random Forest",
    ]:
        r = model_results[model_name]

        print(
            f"{model_name} | "
            f"{r['threshold']:.2f} | "
            f"{r['precision']:.3f} | "
            f"{r['recall']:.3f} | "
            f"{r['f1']:.3f} | "
            f"{r['roc_auc']:.3f} | "
            f"{r['fp']} | "
            f"{r['fn']} | "
            f"{r['fpr']:.6f}"
        )

    print("\nSelected model:", selected_model_name)
    print("Selected threshold:", selected_threshold)
    print("Feature count:", len(X_train.columns))


if __name__ == "__main__":
    main()
