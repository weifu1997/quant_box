from __future__ import annotations

import unittest

import pandas as pd

from scripts.run_ml_strategy import _markdown_report, _resolve_training_start_date


class MLStrategyReportTests(unittest.TestCase):
    def test_markdown_report_documents_factor_source_regime_and_coverage(self) -> None:
        diagnostics = pd.DataFrame(
            [
                {
                    "signal_date": "2024-01-31",
                    "train_rows_used": 10,
                    "no_lookahead": True,
                    "model_used": "ridge_numpy",
                    "skip_reason": "",
                    "ensemble_size": 1,
                    "ensemble_models": "ridge_numpy",
                    "feature_count": 64,
                    "feature_ic_evolved": False,
                },
                {
                    "signal_date": "2024-02-29",
                    "train_rows_used": 0,
                    "no_lookahead": False,
                    "model_used": "skipped",
                    "skip_reason": "insufficient_train_rows",
                    "ensemble_size": 0,
                    "ensemble_models": "",
                    "feature_count": 0,
                    "feature_ic_evolved": False,
                },
            ]
        )
        skipped = diagnostics[diagnostics["skip_reason"] != ""]
        report = _markdown_report(
            {
                "total_return": 0.1,
                "annual_return": 0.1,
                "max_drawdown": -0.05,
                "sharpe": 1.0,
                "calmar": 2.0,
                "annual_turnover": 3.0,
                "annual_trade_cost_ratio": 0.01,
            },
            diagnostics,
            pd.DataFrame({"year": [2024], "total_return": [0.1], "annual_return": [0.1], "max_drawdown": [-0.05]}),
            pd.DataFrame({"regime": ["bull"], "segments": [1], "days": [10]}),
            {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "actual_start": "2024-01-02",
                "actual_end": "2024-12-31",
                "price_dates": 10,
                "symbols": 2,
                "gap_dates": 1,
                "mean_coverage": 0.95,
                "min_coverage": 0.5,
            },
            pd.DataFrame(),
            skipped,
            pd.DataFrame({"year": [2024], "days": [10], "has_equity": [True], "passes_min_days": [True]}),
            {
                "enabled": True,
                "dates_neutralized": 1,
                "industry_dates": 1,
                "market_cap_dates": 0,
                "market_cap_field": "circ_mv",
            },
            {
                "enabled": True,
                "features_added": 2,
                "dates_matched": 1,
                "lag_days": 1,
                "fields": ["circ_mv", "pb"],
            },
            {
                "enabled": True,
                "dates_blended": 1,
                "average_components": 3.0,
                "bull_defensive_weight": 0.0,
                "sideways_defensive_weight": 0.5,
                "bear_defensive_weight": 1.0,
            },
            {
                "ml_strategy": {
                    "feature_limit": 64,
                    "label_horizon_sessions": 20,
                    "model_objective": "classification",
                    "train_years": 3,
                    "training_neutralization": {"enabled": True, "industry": True, "market_cap": False},
                },
                "market_regime": {"lag_days": 1},
                "reporting_regime": {"lag_days": 0},
            },
            "equity.svg",
        )

        self.assertIn("Factor source: Alpha158 price-volume features", report)
        self.assertIn("Label mode: raw_return", report)
        self.assertIn("Label return adjustment: raw", report)
        self.assertIn("Label volatility window: 20", report)
        self.assertIn("Training neutralization enabled: true", report)
        self.assertIn("Model objective: classification", report)
        self.assertIn("Status: FAIL", report)
        self.assertIn("annual_return_below_target", report)
        self.assertIn("Min yearly annual return target: 20.00%", report)
        self.assertIn("Years below return target: [2024]", report)
        self.assertIn("yearly_annual_return_below_target", report)
        self.assertIn("## Training Warmup", report)
        self.assertIn("Required warmup start date: 2021-01-01", report)
        self.assertIn("Training data load start date: 2021-01-01", report)
        self.assertIn("Warmup data starts early enough for first requested year: false", report)
        self.assertIn("First completed ML signal date: 2024-01-31", report)
        self.assertIn("## Model Ensemble", report)
        self.assertIn("Ensemble window: 3", report)
        self.assertIn("Models in ensemble: {'ridge_numpy': 1}", report)
        self.assertIn("## Feature Evolution", report)
        self.assertIn("## Feature Extensions", report)
        self.assertIn("Daily basic features added: 2", report)
        self.assertIn("Price-derived features added: 0", report)
        self.assertIn("## Regime Score Blend", report)
        self.assertIn("Bear defensive weight: 1.00", report)
        self.assertIn("## Neutralization", report)
        self.assertIn("Dates neutralized: 1", report)
        self.assertIn("Market-cap field: circ_mv", report)
        self.assertIn("Fundamental factors used: false", report)
        self.assertIn("Fundamental lag applied: not applicable", report)
        self.assertIn("Timing regime: realtime lagged", report)
        self.assertIn("Reporting regime: objective reporting only", report)
        self.assertIn("Skipped monthly model fits: 1", report)
        self.assertIn("Mean price coverage: 95.00%", report)
        self.assertNotIn("future fundamental factors", report)

    def test_resolve_training_start_date_supports_auto_and_override(self) -> None:
        self.assertEqual(
            _resolve_training_start_date("2015-01-01", {"train_years": 3, "training_start_date": "auto"}),
            "2012-01-01",
        )
        self.assertEqual(
            _resolve_training_start_date("2015-01-01", {"train_years": 3, "training_start_date": "2010-01-01"}),
            "2010-01-01",
        )


if __name__ == "__main__":
    unittest.main()
