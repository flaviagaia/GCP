from __future__ import annotations

import unittest

from src.projects import bundle_reports


class ClassicMLPortfolioTestCase(unittest.TestCase):
    def test_bundle_reports_has_three_projects(self) -> None:
        report = bundle_reports()
        self.assertEqual(len(report["projects"]), 3)
        names = {project["project_name"] for project in report["projects"]}
        self.assertEqual(
            names,
            {
                "credit_default_prediction",
                "customer_churn_prediction",
                "fraud_detection_baseline",
            },
        )

    def test_core_metrics_are_reasonable(self) -> None:
        report = bundle_reports()
        projects = {project["project_name"]: project for project in report["projects"]}
        self.assertGreater(projects["credit_default_prediction"]["roc_auc"], 0.7)
        self.assertGreater(projects["customer_churn_prediction"]["roc_auc"], 0.7)
        self.assertGreater(
            projects["fraud_detection_baseline"]["selected_metrics"]["average_precision"],
            0.15,
        )


if __name__ == "__main__":
    unittest.main()
