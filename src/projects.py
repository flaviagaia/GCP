from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_factory import (
    build_credit_default_dataset,
    build_customer_churn_dataset,
    build_fraud_detection_dataset,
)


@dataclass
class ProjectReport:
    project_name: str
    target_name: str
    sample_size: int
    positive_rate: float
    selected_model: str
    roc_auc: float
    average_precision: float
    precision: float
    recall: float
    f1: float


def _evaluate_binary_classifier(name: str, y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "selected_model": name,
        "roc_auc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def run_credit_default_prediction() -> ProjectReport:
    df = build_credit_default_dataset()
    X = df.drop(columns=["target_default"])
    y = df["target_default"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    candidates: list[tuple[str, object]] = [
        (
            "logistic_regression",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=2000)),
                ]
            ),
        ),
        ("random_forest", RandomForestClassifier(n_estimators=220, random_state=42)),
    ]

    best_metrics = None
    best_score = -1.0
    for name, estimator in candidates:
        estimator.fit(X_train, y_train)
        y_score = estimator.predict_proba(X_test)[:, 1]
        metrics = _evaluate_binary_classifier(name, y_test.to_numpy(), y_score)
        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_metrics = metrics

    return ProjectReport(
        project_name="credit_default_prediction",
        target_name="target_default",
        sample_size=len(df),
        positive_rate=float(y.mean()),
        **{key: round(value, 4) if isinstance(value, float) else value for key, value in best_metrics.items()},
    )


def run_customer_churn_prediction() -> ProjectReport:
    df = build_customer_churn_dataset()
    X = df.drop(columns=["target_churn"])
    y = df["target_churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    candidates: list[tuple[str, object]] = [
        (
            "logistic_regression",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=2000)),
                ]
            ),
        ),
        ("random_forest", RandomForestClassifier(n_estimators=220, random_state=42)),
    ]

    best_metrics = None
    best_score = -1.0
    for name, estimator in candidates:
        estimator.fit(X_train, y_train)
        y_score = estimator.predict_proba(X_test)[:, 1]
        metrics = _evaluate_binary_classifier(name, y_test.to_numpy(), y_score)
        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_metrics = metrics

    return ProjectReport(
        project_name="customer_churn_prediction",
        target_name="target_churn",
        sample_size=len(df),
        positive_rate=float(y.mean()),
        **{key: round(value, 4) if isinstance(value, float) else value for key, value in best_metrics.items()},
    )


def run_fraud_detection_baseline() -> dict:
    df = build_fraud_detection_dataset()
    X = df.drop(columns=["target_fraud"])
    y = df["target_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    supervised = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2500, class_weight="balanced")),
        ]
    )
    supervised.fit(X_train, y_train)
    supervised_scores = supervised.predict_proba(X_test)[:, 1]
    supervised_metrics = _evaluate_binary_classifier(
        "logistic_regression_balanced",
        y_test.to_numpy(),
        supervised_scores,
        threshold=0.45,
    )

    contamination = max(float(y_train.mean()), 0.02)
    anomaly = IsolationForest(
        n_estimators=240,
        contamination=contamination,
        random_state=42,
    )
    anomaly.fit(X_train)
    anomaly_scores = -anomaly.decision_function(X_test)
    anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (
        anomaly_scores.max() - anomaly_scores.min() + 1e-9
    )
    anomaly_metrics = _evaluate_binary_classifier(
        "isolation_forest",
        y_test.to_numpy(),
        anomaly_scores,
        threshold=0.55,
    )

    selected = supervised_metrics if supervised_metrics["average_precision"] >= anomaly_metrics["average_precision"] else anomaly_metrics

    return {
        "project_name": "fraud_detection_baseline",
        "target_name": "target_fraud",
        "sample_size": len(df),
        "positive_rate": round(float(y.mean()), 4),
        "selected_model": selected["selected_model"],
        "selected_metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in selected.items()},
        "comparison": {
            "logistic_regression_balanced": {k: round(v, 4) if isinstance(v, float) else v for k, v in supervised_metrics.items()},
            "isolation_forest": {k: round(v, 4) if isinstance(v, float) else v for k, v in anomaly_metrics.items()},
        },
    }


def bundle_reports() -> dict:
    credit_report = asdict(run_credit_default_prediction())
    churn_report = asdict(run_customer_churn_prediction())
    fraud_report = run_fraud_detection_baseline()
    return {
        "projects": [credit_report, churn_report, fraud_report],
    }

