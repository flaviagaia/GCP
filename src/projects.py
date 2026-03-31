from __future__ import annotations

from projects.classic_ml.credit_default_prediction.project import run as run_credit_default_prediction
from projects.classic_ml.customer_churn_prediction.project import run as run_customer_churn_prediction
from projects.classic_ml.fraud_detection_baseline.project import run as run_fraud_detection_baseline
from projects.classic_ml.common import report_to_dict


def bundle_reports() -> dict:
    credit_report = report_to_dict(run_credit_default_prediction())
    churn_report = report_to_dict(run_customer_churn_prediction())
    fraud_report = run_fraud_detection_baseline()
    return {"projects": [credit_report, churn_report, fraud_report]}

