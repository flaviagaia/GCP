from __future__ import annotations

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from projects.classic_ml.common import evaluate_binary_classifier, round_metrics


def build_dataset(seed: int = 44) -> pd.DataFrame:
    features, target = make_classification(
        n_samples=1800,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        weights=[0.94, 0.06],
        class_sep=1.15,
        flip_y=0.01,
        random_state=seed,
    )
    columns = [
        "transaction_velocity",
        "merchant_risk",
        "device_change",
        "geo_distance",
        "amount_deviation",
        "chargeback_signal",
        "identity_conflict",
        "behavior_shift",
        "night_activity",
        "account_age",
        "card_testing_signal",
        "basket_irregularity",
    ]
    df = pd.DataFrame(features, columns=columns)
    df["amount_zscore"] = df["amount_deviation"].abs() * 1.7
    df["cross_border_flag"] = (df["geo_distance"] > df["geo_distance"].median()).astype(int)
    df["target_fraud"] = target
    return df


def run() -> dict:
    df = build_dataset()
    X = df.drop(columns=["target_fraud"])
    y = df["target_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    supervised = Pipeline(
        [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2500, class_weight="balanced"))]
    )
    supervised.fit(X_train, y_train)
    supervised_scores = supervised.predict_proba(X_test)[:, 1]
    supervised_metrics = evaluate_binary_classifier(
        "logistic_regression_balanced", y_test.to_numpy(), supervised_scores, threshold=0.45
    )

    contamination = max(float(y_train.mean()), 0.02)
    anomaly = IsolationForest(n_estimators=240, contamination=contamination, random_state=42)
    anomaly.fit(X_train)
    anomaly_scores = -anomaly.decision_function(X_test)
    anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (
        anomaly_scores.max() - anomaly_scores.min() + 1e-9
    )
    anomaly_metrics = evaluate_binary_classifier(
        "isolation_forest", y_test.to_numpy(), anomaly_scores, threshold=0.55
    )

    selected = supervised_metrics if supervised_metrics["average_precision"] >= anomaly_metrics["average_precision"] else anomaly_metrics
    return {
        "project_name": "fraud_detection_baseline",
        "target_name": "target_fraud",
        "sample_size": len(df),
        "positive_rate": round(float(y.mean()), 4),
        "selected_model": selected["selected_model"],
        "selected_metrics": round_metrics(selected),
        "comparison": {
            "logistic_regression_balanced": round_metrics(supervised_metrics),
            "isolation_forest": round_metrics(anomaly_metrics),
        },
    }

