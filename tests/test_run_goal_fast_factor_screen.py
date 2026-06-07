from __future__ import annotations

import unittest

import pandas as pd

from scripts.run_goal_fast_factor_screen import (
    _screen_quality_fields,
    _select_screen_columns,
    _single_factor_scores,
    _slice_rebalance_factor_dates,
)


class RunGoalFastFactorScreenTests(unittest.TestCase):
    def test_screen_quality_fields_use_configured_thresholds(self) -> None:
        config = {
            "quality": {
                "min_backtest_annual_return": 0.25,
                "min_yearly_annual_return": 0.10,
                "max_backtest_drawdown_limit": -0.15,
                "max_yearly_drawdown_limit": -0.20,
            }
        }

        fields = _screen_quality_fields(
            {
                "annual_return": 0.24,
                "max_drawdown": -0.16,
                "min_year_annual_return": 0.11,
                "worst_year_drawdown": -0.18,
            },
            config,
        )
        passing = _screen_quality_fields(
            {
                "annual_return": 0.26,
                "max_drawdown": -0.14,
                "min_year_annual_return": 0.11,
                "worst_year_drawdown": -0.18,
            },
            config,
        )

        self.assertFalse(fields["meets_full_target"])
        self.assertAlmostEqual(fields["target_gap"], 0.02)
        self.assertTrue(passing["meets_full_target"])
        self.assertEqual(passing["target_gap"], 0.0)

    def test_screen_quality_fields_reject_weak_yearly_metrics(self) -> None:
        config = {
            "quality": {
                "min_backtest_annual_return": 0.20,
                "min_yearly_annual_return": 0.10,
                "max_backtest_drawdown_limit": -0.20,
                "max_yearly_drawdown_limit": -0.20,
            }
        }

        fields = _screen_quality_fields(
            {
                "annual_return": 0.30,
                "max_drawdown": -0.10,
                "min_year_annual_return": -0.05,
                "worst_year_drawdown": -0.25,
            },
            config,
        )

        self.assertFalse(fields["meets_full_target"])
        self.assertFalse(fields["yearly_annual_return_pass"])
        self.assertFalse(fields["yearly_drawdown_pass"])
        self.assertAlmostEqual(fields["target_gap"], 0.20)

    def test_single_factor_scores_reverse_long_low_direction_and_deduplicate(self) -> None:
        index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-02 09:30"), "a"),
                (pd.Timestamp("2024-01-02 15:00"), "a"),
                (pd.Timestamp("2024-01-02 15:00"), "b"),
            ],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame({"F1": [1.0, 2.0, 3.0]}, index=index)

        scores = _single_factor_scores(factors, "F1", "long_low")

        daily = scores.xs(pd.Timestamp("2024-01-02"), level=0)
        self.assertEqual(daily.index.tolist(), ["A", "B"])
        self.assertEqual(float(daily.loc["A"]), -2.0)
        self.assertEqual(float(daily.loc["B"]), -3.0)

    def test_slice_rebalance_factor_dates_keeps_month_end_signal_dates(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-31", "2024-02-01", "2024-02-29"])
        index = pd.MultiIndex.from_product([dates, ["A"]], names=["datetime", "instrument"])
        factors = pd.DataFrame({"F1": range(len(index))}, index=index)

        sliced = _slice_rebalance_factor_dates(factors, "monthly")

        self.assertEqual(
            sliced.index.get_level_values("datetime").unique().tolist(),
            [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")],
        )

    def test_select_screen_columns_supports_start_index_and_limit(self) -> None:
        columns = ["A", "B", "C", "D"]

        self.assertEqual(_select_screen_columns(columns, 2, 2), ["B", "C"])
        self.assertEqual(_select_screen_columns(columns, 0, 0), columns)


if __name__ == "__main__":
    unittest.main()
