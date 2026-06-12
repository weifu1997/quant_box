"""模块说明：覆盖 test_run_factor_columns 相关行为的测试用例。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts.run_backtest import _requested_factor_columns as backtest_factor_columns
from scripts.run_optimize import _requested_factor_columns as optimize_factor_columns


class RunFactorColumnTests(unittest.TestCase):
    """类说明：组织 RunFactorColumnTests 测试用例。"""
    def test_backtest_requested_columns_include_regime_score_blend_components(self) -> None:
        """函数说明：验证 test_backtest_requested_columns_include_regime_score_blend_components 覆盖的行为场景。"""
        with patch("scripts._shared.factor_cache_columns", return_value=["LOW0", "STD20", "BETA20", "ROC20"]):
            columns = backtest_factor_columns(
                "unused.parquet",
                {"factor_group": "dynamic_ic_selector"},
                {"candidates": ["factor:LOW0"]},
                {},
                {
                    "enabled": True,
                    "defensive_components": [
                        {"column": "STD20", "direction": -1.0},
                        {"column": "BETA20", "direction": -1.0},
                    ],
                },
                {},
            )

        self.assertEqual(columns, ["BETA20", "LOW0", "STD20"])

    def test_optimize_requested_columns_include_regime_score_blend_components(self) -> None:
        """函数说明：验证 test_optimize_requested_columns_include_regime_score_blend_components 覆盖的行为场景。"""
        with patch("scripts._shared.factor_cache_columns", return_value=["LOW0", "STD20", "BETA20", "ROC20"]):
            columns = optimize_factor_columns(
                "unused.parquet",
                ["dynamic_ic_selector"],
                {"candidates": ["factor:LOW0"]},
                {
                    "enabled": True,
                    "defensive_components": [
                        {"column": "STD20", "direction": -1.0},
                        {"column": "BETA20", "direction": -1.0},
                    ],
                },
                {},
            )

        self.assertEqual(columns, ["BETA20", "LOW0", "STD20"])

    def test_backtest_requested_columns_include_regime_score_filter_components(self) -> None:
        """函数说明：验证 test_backtest_requested_columns_include_regime_score_filter_components 覆盖的行为场景。"""
        with patch("scripts._shared.factor_cache_columns", return_value=["LOW0", "ROC20", "STD20"]):
            columns = backtest_factor_columns(
                "unused.parquet",
                {"factor_group": "dynamic_ic_selector"},
                {"candidates": ["factor:LOW0"]},
                {},
                {"enabled": False},
                {
                    "enabled": True,
                    "rules": [{"regime": "bear", "components": [{"column": "ROC20", "direction": 1.0}], "min_score": 0.0}],
                },
            )

        self.assertEqual(columns, ["LOW0", "ROC20"])

    def test_backtest_requested_columns_include_static_factor_blend_weights(self) -> None:
        with patch("scripts._shared.factor_cache_columns", return_value=["KLEN", "LOW0", "STD20"]):
            columns = backtest_factor_columns(
                "unused.parquet",
                {"factor_group": "factor_blend", "factor_weights": {"KLEN": -1.0, "LOW0": 0.3}},
                {},
                {},
                {"enabled": False},
                {},
            )

        self.assertEqual(columns, ["KLEN", "LOW0"])

    def test_backtest_requested_columns_reject_missing_exact_factor(self) -> None:
        with patch("scripts._shared.factor_cache_columns", return_value=["LOW0", "ROC20"]):
            with self.assertRaisesRegex(ValueError, "factor_group 'factor:DB_circ_mv' did not match"):
                backtest_factor_columns(
                    "unused.parquet",
                    {"factor_group": "factor:DB_circ_mv"},
                    {},
                    {},
                    {"enabled": False},
                    {},
                )

    def test_backtest_requested_columns_reject_ambiguous_dynamic_candidate(self) -> None:
        with patch("scripts._shared.factor_cache_columns", return_value=["ROC5", "ROC10", "LOW0"]):
            with self.assertRaisesRegex(ValueError, "dynamic_ic_selector candidate 'momentum' matched 2 columns"):
                backtest_factor_columns(
                    "unused.parquet",
                    {"factor_group": "dynamic_ic_selector"},
                    {"candidates": ["momentum"]},
                    {},
                    {"enabled": False},
                    {},
                )

    def test_backtest_requested_columns_reject_missing_static_blend_weight(self) -> None:
        with patch("scripts._shared.factor_cache_columns", return_value=["KLEN", "LOW0"]):
            with self.assertRaisesRegex(ValueError, "factor_blend references columns missing"):
                backtest_factor_columns(
                    "unused.parquet",
                    {"factor_group": "factor_blend", "factor_weights": {"KLEN": -1.0, "DB_pb": 0.5}},
                    {},
                    {},
                    {"enabled": False},
                    {},
                )


if __name__ == "__main__":
    unittest.main()
