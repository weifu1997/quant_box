"""模块说明：覆盖 test_run_goal_fast_factor_screen 相关行为的测试用例。"""

from __future__ import annotations

import unittest

import pandas as pd

from scripts.run_goal_fast_factor_screen import _fast_yearly_stats, _requested_screen_columns, _screen_quality_fields


class RunGoalFastFactorScreenTests(unittest.TestCase):
    """类说明：组织 RunGoalFastFactorScreenTests 测试用例。"""
    def test_screen_quality_fields_use_configured_thresholds(self) -> None:
        """函数说明：验证 test_screen_quality_fields_use_configured_thresholds 覆盖的行为场景。"""
        config = {
            "quality": {
                "min_backtest_annual_return": 0.25,
                "max_backtest_drawdown_limit": -0.15,
            }
        }

        fields = _screen_quality_fields({"annual_return": 0.24, "max_drawdown": -0.16}, config)
        passing = _screen_quality_fields({"annual_return": 0.26, "max_drawdown": -0.14}, config)

        self.assertFalse(fields["meets_full_target"])
        self.assertAlmostEqual(fields["target_gap"], 0.02)
        self.assertTrue(passing["meets_full_target"])
        self.assertEqual(passing["target_gap"], 0.0)
        self.assertTrue(passing["formal_confirmation_required"])
        self.assertIn("fast_screen_ignores_formal", passing["approximation_notes"])

    def test_screen_quality_fields_apply_turnover_gate(self) -> None:
        config = {
            "quality": {
                "min_backtest_annual_return": 0.20,
                "max_backtest_drawdown_limit": -0.20,
                "max_annual_turnover": 2.0,
            }
        }

        fields = _screen_quality_fields(
            {"annual_return": 0.30, "max_drawdown": -0.10, "annual_weight_turnover": 3.0},
            config,
        )

        self.assertFalse(fields["turnover_pass"])
        self.assertFalse(fields["meets_full_target"])
        self.assertGreater(fields["target_gap"], 0.0)

    def test_screen_quality_fields_require_yearly_targets_when_available(self) -> None:
        config = {
            "quality": {
                "min_backtest_annual_return": 0.20,
                "max_backtest_drawdown_limit": -0.20,
            }
        }
        yearly = pd.DataFrame(
            [
                {"year": 2023, "annual_return": 0.10, "max_drawdown": -0.10},
                {"year": 2024, "annual_return": 0.25, "max_drawdown": -0.25},
            ]
        )

        fields = _screen_quality_fields({"annual_return": 0.30, "max_drawdown": -0.10}, config, yearly=yearly)

        self.assertFalse(fields["meets_full_target"])
        self.assertEqual(fields["year_count"], 2)
        self.assertEqual(fields["year_ann_pass"], 1)
        self.assertEqual(fields["year_dd_pass"], 1)

    def test_requested_screen_columns_preserve_available_column_names(self) -> None:
        available = ["ROC60", "DB_circ_mv", "DB_turnover_rate_f"]

        selected = _requested_screen_columns("db_circ_mv,ROC60", available)

        self.assertEqual(selected, ["DB_circ_mv", "ROC60"])
        with self.assertRaisesRegex(ValueError, "MISSING"):
            _requested_screen_columns("MISSING", available)

    def test_fast_yearly_stats_annualizes_sparse_equity_by_calendar_time(self) -> None:
        equity = pd.Series(
            [100.0, 120.0],
            index=pd.to_datetime(["2024-01-02", "2024-12-31"]),
            name="equity",
        )

        yearly = _fast_yearly_stats(equity)

        self.assertEqual(yearly["year"].tolist(), [2024])
        self.assertAlmostEqual(float(yearly.iloc[0]["total_return"]), 0.20)
        self.assertLess(float(yearly.iloc[0]["annual_return"]), 0.21)


if __name__ == "__main__":
    unittest.main()
