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


def build_dataset(seed: int = 42) -> pd.DataFrame:
    features, target = make_classification(
        n_samples=1200,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        weights=[0.73, 0.27],
        class_sep=1.05,
        random_state=seed,
    )
    columns = [
        "income_stability",
        "debt_ratio",
        "credit_utilization",
        "recent_delinquency",
        "installment_burden",
        "credit_history_length",
        "application_velocity",
        "hard_inquiries",
        "asset_buffer",
        "behavior_score",
    ]
    df = pd.DataFrame(features, columns=columns)
    df["monthly_income"] = 6000 + df["income_stability"] * 1200 - df["debt_ratio"] * 400
    df["loan_to_income"] = (df["installment_burden"] + 2.5) / (np.abs(df["monthly_income"]) / 3000)
    df["target_default"] = target
    return df


def run() -> ProjectReport:
    df = build_dataset()
    X = df.drop(columns=["target_default"])
    y = df["target_default"]
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
        project_name="credit_default_prediction",
        target_name="target_default",
        sample_size=len(df),
        positive_rate=float(y.mean()),
        **round_metrics(best_metrics),
    )

