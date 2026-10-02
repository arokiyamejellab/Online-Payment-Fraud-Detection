"""
Online Payment Fraud Detection — Model Training
---------------------------------------------------
Trains Logistic Regression and Random Forest classifiers on online
payment transaction data to flag fraud, evaluates both, and saves the
best-performing model (plus encoders/scaler) for the Streamlit app.

Dataset: data/fraud.csv
Columns: Transaction_ID, Customer_ID, Age, Gender, Payment_Type,
         Transaction_Amount, Transaction_Hour, Location, Device_Type,
         Network_Type, Previous_Transactions, Fraud

Note: this dataset is heavily imbalanced (~1% fraud), which is typical
for real-world fraud data. Both models use class_weight="balanced" to
compensate, and Recall / F1-Score matter more here than raw Accuracy.

Usage:
    python train_model.py
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

DATA_PATH = "data/fraud.csv"
TARGET_COL = "Fraud"
DROP_COLS = ["Transaction_ID", "Customer_ID"]
RANDOM_STATE = 42

CATEGORICAL_COLS = ["Gender", "Payment_Type", "Location", "Device_Type", "Network_Type"]
NUMERIC_COLS = ["Age", "Transaction_Amount", "Transaction_Hour", "Previous_Transactions"]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded dataset with shape: {df.shape}")
    print(f"Class balance:\n{df[TARGET_COL].value_counts()}")
    print(f"Fraud rate: {(df[TARGET_COL] == 'Yes').mean():.2%}")
    return df


def clean_and_encode(df: pd.DataFrame):
    """Drop ID columns, encode categoricals + target, return df + encoders."""
    df = df.copy()
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.dropna()

    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le.classes_.tolist()

    target_le = LabelEncoder()
    df[TARGET_COL] = target_le.fit_transform(df[TARGET_COL])  # No=0, Yes=1
    encoders[TARGET_COL] = target_le.classes_.tolist()

    return df, encoders


def train_and_evaluate(df: pd.DataFrame):
    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS
    X = df[feature_cols]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = {}
    fitted = {}
    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        results[name] = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds, zero_division=0), 4),
            "recall": round(recall_score(y_test, preds, zero_division=0), 4),
            "f1": round(f1_score(y_test, preds, zero_division=0), 4),
        }
        fitted[name] = model

        print(f"\n=== {name} ===")
        for k, v in results[name].items():
            print(f"{k.capitalize():10}: {v}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, preds))
        print(classification_report(y_test, preds, zero_division=0))

    best_name = max(results, key=lambda k: results[k]["f1"])
    print(f"\nBest performing model (by F1-Score): {best_name}")

    return fitted[best_name], best_name, scaler, results, feature_cols


def main():
    df = load_data(DATA_PATH)
    df, encoders = clean_and_encode(df)
    best_model, best_name, scaler, results, feature_cols = train_and_evaluate(df)

    joblib.dump(best_model, "model/fraud_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")

    metadata = {
        "best_model": best_name,
        "feature_cols": feature_cols,
        "categorical_cols": CATEGORICAL_COLS,
        "numeric_cols": NUMERIC_COLS,
        "encoders": encoders,
        "uses_scaled_input": best_name == "Logistic Regression",
        "results": results,
        "amount_range": [
            float(df["Transaction_Amount"].min()),
            float(df["Transaction_Amount"].max()),
        ],
    }
    with open("model/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved model, scaler, and metadata to ./model/")


if __name__ == "__main__":
    main()
