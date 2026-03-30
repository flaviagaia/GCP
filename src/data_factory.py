from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def build_credit_default_dataset(seed: int = 42) -> pd.DataFrame:
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


def build_customer_churn_dataset(seed: int = 42) -> pd.DataFrame:
    features, target = make_classification(
        n_samples=1400,
        n_features=11,
        n_informative=7,
        n_redundant=1,
        weights=[0.68, 0.32],
        class_sep=0.9,
        random_state=seed + 1,
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


def build_fraud_detection_dataset(seed: int = 42) -> pd.DataFrame:
    features, target = make_classification(
        n_samples=1800,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        weights=[0.94, 0.06],
        class_sep=1.15,
        flip_y=0.01,
        random_state=seed + 2,
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

