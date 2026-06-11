"""模块说明：覆盖 test_run_goal_fast_factor_screen 相关行为的测试用例。"""

from __future__ import annotations

import unittest

import pandas as pd

from scripts._shared import yearly_stats
from scripts.run_goal_klen_risk_sweep import _float_list, _optional_float_list
from scripts.run_goal_fast_factor_screen import (
    _fast_yearly_stats,
    _filter_scores_by_selection_risk,
    _liquidity_row,
    _requested_screen_columns,
    _screen_quality_fields,
    _screen_yearly_quality_fields,
    _selected_columns,
    _selected_directions,
    _selection_risk_price_fields,
)


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

    def test_screen_yearly_quality_fields_report_failures_and_gap(self) -> None:
        import pandas as pd

        yearly = pd.DataFrame(
            [
                {"year": 2022, "annual_return": 0.30, "max_drawdown": -0.10},
                {"year": 2023, "annual_return": 0.10, "max_drawdown": -0.25},
                {"year": 2024, "annual_return": 0.22, "max_drawdown": -0.18},
            ]
        )
        config = {
            "quality": {
                "min_backtest_annual_return": 0.20,
                "max_backtest_drawdown_limit": -0.20,
            }
        }

        fields = _screen_yearly_quality_fields(yearly, config)

        self.assertEqual(fields["year_count"], 3)
        self.assertEqual(fields["year_ann_pass"], 2)
        self.assertEqual(fields["year_dd_pass"], 2)
        self.assertFalse(fields["yearly_all_pass"])
        self.assertEqual(fields["years_below_return_target"], "2023")
        self.assertEqual(fields["years_breaching_drawdown_limit"], "2023")
        self.assertAlmostEqual(fields["yearly_target_gap"], 0.15)

    def test_selected_columns_preserves_requested_valid_columns(self) -> None:
        selected = _selected_columns(["KMID", "KLEN", "LOW0"], "klen, LOW0, klen")

        self.assertEqual(selected, ["KLEN", "LOW0"])

    def test_selected_columns_rejects_missing_columns(self) -> None:
        with self.assertRaises(ValueError):
            _selected_columns(["KMID"], "KLEN")

    def test_selected_directions_rejects_unknown_values(self) -> None:
        self.assertEqual(_selected_directions("long_low"), {"long_low"})
        with self.assertRaises(ValueError):
            _selected_directions("long_low,sideways")

    def test_liquidity_row_reports_rejected_and_kept_side(self) -> None:
        row = _liquidity_row({"enabled": True, "side": "high", "quantile": 0.8})

        self.assertEqual(row["liquidity_rejected_side"], "high")
        self.assertEqual(row["liquidity_kept_side"], "lower_liquidity")

    def test_shared_yearly_stats_uses_calendar_span_for_sparse_equity(self) -> None:
        import pandas as pd

        equity = pd.Series(
            [100.0, 110.0, 120.0],
            index=pd.to_datetime(["2024-01-31", "2024-06-30", "2024-12-31"]),
            name="equity",
        )

        stats = yearly_stats(equity, {"annual_trading_days": 252})

        observed = float(stats.iloc[0]["annual_return"])
        self.assertLess(observed, 0.25)
        self.assertGreater(observed, 0.20)

    def test_klen_sweep_parses_numeric_grid_values(self) -> None:
        self.assertEqual(_float_list("7, 15", value_type=int), [7, 15])
        self.assertEqual(_optional_float_list("none, 0.12"), [None, 0.12])

    def test_selection_risk_price_fields_include_required_data(self) -> None:
        fields = _selection_risk_price_fields(
            {
                "selection_risk_filter": {
                    "enabled": True,
                    "required_price_fields": ["open", "close"],
                    "require_positive_volume": True,
                    "max_limit_down_days": 0,
                }
            }
        )

        self.assertEqual(fields, {"open", "close", "volume", "low"})

    def test_filter_scores_by_selection_risk_applies_monthly_signal_filter(self) -> None:
        import pandas as pd

        date = pd.Timestamp("2024-01-03")
        index = pd.MultiIndex.from_product([[date], ["A", "B"]], names=["datetime", "instrument"])
        scores = pd.Series([1.0, 2.0], index=index, name="score")
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        open_prices = pd.DataFrame({"A": [10.0, 10.0], "B": [10.0, 10.0]}, index=dates)
        close_prices = pd.DataFrame({"A": [10.0, 10.0], "B": [10.0, 10.0]}, index=dates)
        volume = pd.DataFrame({"A": [100.0, 100.0], "B": [0.0, 0.0]}, index=dates)
        low = pd.DataFrame({"A": [10.0, 10.0], "B": [10.0, 10.0]}, index=dates)
        prices = pd.concat({"open": open_prices, "close": close_prices, "volume": volume, "low": low}, axis=1)
        prices.columns = pd.MultiIndex.from_tuples(prices.columns, names=["field", "instrument"])
        config = {
            "selection_risk_filter": {
                "enabled": True,
                "lookback_sessions": 2,
                "required_price_fields": ["open", "close"],
                "max_missing_price_sessions": 0,
                "max_limit_down_days": 0,
                "require_positive_volume": True,
            }
        }

        filtered = _filter_scores_by_selection_risk(scores, prices, config)
        daily = filtered.xs(date, level="datetime")

        self.assertFalse(pd.isna(daily.loc["A"]))
        self.assertTrue(pd.isna(daily.loc["B"]))


if __name__ == "__main__":
    unittest.main()
