"""
PaySense AI — Model Training Script
Run: python model/train_model.py
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

SEED = 42
np.random.seed(SEED)

BANKS = ["GTBank", "Access Bank", "Zenith Bank", "UBA", "First Bank",
         "Fidelity", "Sterling", "Wema", "Polaris", "Union Bank"]
METHODS = ["Card", "Transfer", "USSD", "QR Code", "Bank Debit"]
NETWORKS = ["MTN", "Airtel", "Glo", "9mobile"]
CATEGORIES = ["Airtime", "Bills", "Transfer", "Shopping", "Food", "Transport", "Savings"]


def generate_data(n=5000):
    np.random.seed(SEED)

    data = {
        "amount":              np.random.lognormal(10, 1.8, n),
        "bank":                np.random.choice(BANKS, n),
        "payment_method":      np.random.choice(METHODS, n),
        "network":             np.random.choice(NETWORKS, n),
        "category":            np.random.choice(CATEGORIES, n),
        "hour":                np.random.randint(0, 24, n),
        "day_of_week":         np.random.randint(0, 7, n),
        "account_age_days":    np.random.randint(1, 1825, n),
        "prev_failures_24h":   np.random.randint(0, 5, n),
        "prev_success_7d":     np.random.randint(0, 30, n),
        "balance_ratio":       np.random.uniform(0, 3, n),
        "is_international":    np.random.randint(0, 2, n),
        "device_changed":      np.random.randint(0, 2, n),
        "retry_attempt":       np.random.randint(0, 4, n),
        "daily_txn_count":     np.random.randint(1, 20, n),
    }

    df = pd.DataFrame(data)

    # Failure probability model
    fail_prob = (
        0.25 * (df["prev_failures_24h"] / 5) +
        0.20 * (1 - np.clip(df["balance_ratio"], 0, 1)) +
        0.15 * (df["retry_attempt"] / 4) +
        0.12 * (df["is_international"]) +
        0.10 * (df["hour"].isin([0,1,2,3,4,5]).astype(int)) +
        0.08 * (df["device_changed"]) +
        0.10 * np.random.uniform(0, 0.5, n)
    )

    df["failed"] = (np.random.uniform(0, 1, n) < fail_prob).astype(int)
    return df


def train():
    df = generate_data()

    # Encode categoricals
    encoders = {}
    for col in ["bank", "payment_method", "network", "category"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    features = [c for c in df.columns if c != "failed"]
    X = df[features]
    y = df["failed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # ── XGBoost ──────────────────────────────────────────────────────────────
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=SEED,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
        "f1_score":  round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "failure_rate": round(y.mean(), 4),
        "total_trained": len(df),
    }

    print("Training complete!")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    os.makedirs("model", exist_ok=True)
    pickle.dump(model,    open("model/paysense_model.pkl", "wb"))
    pickle.dump(encoders, open("model/encoders.pkl", "wb"))
    pickle.dump(features, open("model/features.pkl", "wb"))

    with open("model/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save test data for insights
    X_test_df = X_test.copy()
    X_test_df["failed"]      = y_test.values
    X_test_df["fail_prob"]   = y_proba
    X_test_df.to_csv("model/test_data.csv", index=False)

    # Save label maps for display
    label_maps = {
        "bank":           BANKS,
        "payment_method": METHODS,
        "network":        NETWORKS,
        "category":       CATEGORIES,
    }
    with open("model/label_maps.json", "w") as f:
        json.dump(label_maps, f, indent=2)

    print("All model files saved!")


if __name__ == "__main__":
    train()