from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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


def evaluate_binary_classifier(
    name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "selected_model": name,
        "roc_auc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def round_metrics(payload: dict) -> dict:
    return {key: round(value, 4) if isinstance(value, float) else value for key, value in payload.items()}


def report_to_dict(report: ProjectReport) -> dict:
    return asdict(report)

