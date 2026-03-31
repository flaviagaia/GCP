from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from projects.classic_ml.common import ProjectReport, evaluate_binary_classifier, round_metrics


def build_dataset(seed: int = 43) -> pd.DataFrame:
    features, target = make_classification(
        n_samples=1400,
        n_features=11,
        n_informative=7,
        n_redundant=1,
        weights=[0.68, 0.32],
        class_sep=0.9,
        random_state=seed,
    )
    columns = [
        "engagement_score",
        "support_load",
        "price_pressure",
        "contract_flexibility",
        "usage_depth",
        "nps_proxy",
        "feature_adoption",
        "billing_friction",
        "tenure_signal",
        "expansion_signal",
        "health_score",
    ]
    df = pd.DataFrame(features, columns=columns)
    df["tenure_months"] = (np.clip(df["tenure_signal"] * 8 + 18, 1, None)).round(0)
    df["support_to_engagement"] = (df["support_load"].abs() + 1.0) / (df["engagement_score"].abs() + 1.2)
    df["target_churn"] = target
    return df


def run() -> ProjectReport:
    df = build_dataset()
    X = df.drop(columns=["target_churn"])
    y = df["target_churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    candidates: list[tuple[str, object]] = [
        (
            "logistic_regression",
            Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]),
        ),
        ("random_forest", RandomForestClassifier(n_estimators=220, random_state=42)),
    ]

    best_metrics = None
    best_score = -1.0
    for name, estimator in candidates:
        estimator.fit(X_train, y_train)
        y_score = estimator.predict_proba(X_test)[:, 1]
        metrics = evaluate_binary_classifier(name, y_test.to_numpy(), y_score)
        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_metrics = metrics

    return ProjectReport(
        project_name="customer_churn_prediction",
        target_name="target_churn",
        sample_size=len(df),
        positive_rate=float(y.mean()),
        **round_metrics(best_metrics),
    )

