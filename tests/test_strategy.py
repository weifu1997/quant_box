"""模块说明：覆盖 test_strategy 相关行为的测试用例。"""

from __future__ import annotations

import unittest

import pandas as pd

from src.strategy import (
    _NORMALIZED_FACTOR_FRAME_CACHE,
    _normalize_factor_frame_for_scoring,
    _required_row_mean_factor_count,
    composite_factor,
    factor_columns_for_method,
    generate_holdings_by_day,
    resample_signals,
    ROW_MEAN_REQUIRED_FACTOR_FRACTION,
    select_stocks,
)


class StrategyTests(unittest.TestCase):
    """类说明：组织 StrategyTests 测试用例。"""
    def test_select_stocks_limits_turnover(self) -> None:
        """函数说明：验证 test_select_stocks_limits_turnover 覆盖的行为场景。"""
        scores = pd.Series(
            [10, 9, 8, 7, 6, 5],
            index=["D", "E", "A", "B", "C", "F"],
        )
        previous = ["A", "B", "C"]

        selected = select_stocks(scores, top_n=3, previous_holdings=previous, max_turnover=1)

        self.assertEqual(len(selected), 3)
        self.assertLessEqual(len(set(selected) - set(previous)), 1)
        self.assertIn("D", selected)

    def test_select_stocks_uses_rank_buffer(self) -> None:
        """函数说明：验证 test_select_stocks_uses_rank_buffer 覆盖的行为场景。"""
        scores = pd.Series(
            [10, 9, 8, 7, 6, 5],
            index=["A", "B", "C", "D", "E", "F"],
        )
        previous = ["A", "B", "F"]

        selected = select_stocks(scores, top_n=3, previous_holdings=previous, max_turnover=1, rank_buffer=3)

        self.assertIn("F", selected)
        self.assertLessEqual(len(set(selected) - set(previous)), 1)

    def test_select_stocks_deduplicates_previous_holdings(self) -> None:
        """函数说明：验证 test_select_stocks_deduplicates_previous_holdings 覆盖的行为场景。"""
        scores = pd.Series([10, 9, 8, 7], index=["A", "B", "C", "D"])
        selected = select_stocks(scores, top_n=3, previous_holdings=["A", "A", "B"], max_turnover=1)

        self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(len(selected), 3)

    def test_select_stocks_normalizes_codes_before_turnover_check(self) -> None:
        """函数说明：验证 test_select_stocks_normalizes_codes_before_turnover_check 覆盖的行为场景。"""
        scores = pd.Series([10, 9, 8, 7], index=[" a ", "b", "C", "D"])
        previous = ["A", " B ", "c"]

        selected = select_stocks(scores, top_n=3, previous_holdings=previous, max_turnover=1)

        self.assertEqual(selected, ["A", "B", "C"])
        self.assertLessEqual(len(set(selected) - {"A", "B", "C"}), 1)

    def test_select_stocks_keeps_highest_score_when_normalized_codes_duplicate(self) -> None:
        """函数说明：验证 test_select_stocks_keeps_highest_score_when_normalized_codes_duplicate 覆盖的行为场景。"""
        scores = pd.Series([100, 1, 99], index=[" a ", "A", "B"])

        selected = select_stocks(scores, top_n=1)

        self.assertEqual(selected, ["A"])

    def test_select_stocks_caps_group_concentration_without_previous_holdings(self) -> None:
        """函数说明：验证 test_select_stocks_caps_group_concentration_without_previous_holdings 覆盖的行为场景。"""
        scores = pd.Series([10, 9, 8, 7, 6], index=["A", "B", "C", "D", "E"])
        groups = {"A": "bank", "B": "bank", "C": "bank", "D": "tech", "E": "health"}

        selected = select_stocks(scores, top_n=4, group_map=groups, max_group_weight=0.5)

        self.assertEqual(selected, ["A", "B", "D", "E"])
        self.assertLessEqual(sum(1 for code in selected if groups[code] == "bank"), 2)

    def test_select_stocks_applies_group_cap_with_previous_holdings_and_turnover_limit(self) -> None:
        """函数说明：验证 test_select_stocks_applies_group_cap_with_previous_holdings_and_turnover_limit 覆盖的行为场景。"""
        scores = pd.Series([10, 9, 8, 7, 6, 5], index=["A", "B", "C", "D", "E", "F"])
        groups = {"A": "bank", "B": "bank", "C": "bank", "D": "tech", "E": "health", "F": "energy"}
        previous = ["A", "B", "C", "D"]

        selected = select_stocks(
            scores,
            top_n=4,
            previous_holdings=previous,
            max_turnover=1,
            group_map=groups,
            max_group_weight=0.5,
        )

        self.assertEqual(len(selected), 4)
        self.assertLessEqual(len(set(selected) - set(previous)), 1)
        self.assertLessEqual(sum(1 for code in selected if groups[code] == "bank"), 2)

    def test_select_stocks_does_not_overfill_group_when_group_cap_cannot_be_satisfied(self) -> None:
        """函数说明：验证 test_select_stocks_does_not_overfill_group_when_group_cap_cannot_be_satisfied 覆盖的行为场景。"""
        scores = pd.Series([10, 9, 8], index=["A", "B", "C"])
        groups = {"A": "bank", "B": "bank", "C": "bank"}

        selected = select_stocks(scores, top_n=3, group_map=groups, max_group_weight=0.34)

        self.assertEqual(selected, ["A"])
        self.assertLessEqual(sum(1 for code in selected if groups[code] == "bank"), 1)

    def test_select_stocks_does_not_overfill_group_from_previous_holdings_fallback(self) -> None:
        """函数说明：验证 test_select_stocks_does_not_overfill_group_from_previous_holdings_fallback 覆盖的行为场景。"""
        scores = pd.Series([10, 9, 8, 7], index=["A", "B", "C", "D"])
        groups = {"A": "bank", "B": "bank", "C": "bank", "D": "tech"}

        selected = select_stocks(
            scores,
            top_n=3,
            previous_holdings=["A", "B", "C"],
            max_turnover=0,
            group_map=groups,
            max_group_weight=0.34,
        )

        self.assertEqual(selected, ["A"])
        self.assertLessEqual(sum(1 for code in selected if groups[code] == "bank"), 1)

    def test_generate_holdings_by_day_returns_empty_frame_when_no_scores_are_selectable(self) -> None:
        """函数说明：验证 test_generate_holdings_by_day_returns_empty_frame_when_no_scores_are_selectable 覆盖的行为场景。"""
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")], ["A", "B"]],
            names=["datetime", "instrument"],
        )
        scores = pd.Series([None, None, None, None], index=index, name="score", dtype="float64")

        holdings = generate_holdings_by_day(scores, top_n=2, max_turnover=1)

        self.assertTrue(holdings.empty)
        self.assertEqual(holdings.columns.tolist(), ["date", "instrument", "weight"])

    def test_composite_factor_returns_score_series(self) -> None:
        """函数说明：验证 test_composite_factor_returns_score_series 覆盖的行为场景。"""
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")], ["A", "B", "C", "D", "E"]],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame(
            {
                "ROC5": range(1, 11),
                "MOM10": range(2, 12),
                "OTHER": [100.0] * 10,
            },
            index=index,
        )

        scores = composite_factor(factors, method="momentum")

        self.assertEqual(scores.name, "score")
        self.assertEqual(len(scores), len(index))
        self.assertTrue(scores.notna().all())

    def test_composite_factor_supports_ic_weighted(self) -> None:
        """函数说明：验证 test_composite_factor_supports_ic_weighted 覆盖的行为场景。"""
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")], ["A", "B", "C", "D", "E"]],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame({"F1": range(1, 11), "F2": range(10, 0, -1)}, index=index)

        scores = composite_factor(factors, method="ic_weighted", factor_weights={"F1": 1.0, "F2": -0.5})

        self.assertEqual(scores.name, "score")
        self.assertEqual(len(scores), len(index))

    def test_composite_factor_tolerates_partial_missing_values(self) -> None:
        """函数说明：验证 test_composite_factor_tolerates_partial_missing_values 覆盖的行为场景。"""
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-02")], ["A", "B", "C", "D", "E"]],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame({"ROC5": [1, 2, 3, 4, 5], "MOM10": [2, 3, None, 5, 6]}, index=index)

        scores = composite_factor(factors, method="momentum")

        self.assertTrue(pd.notna(scores.loc[(pd.Timestamp("2024-01-02"), "C")]))

    def test_composite_factor_requires_named_fraction_of_valid_factors(self) -> None:
        """函数说明：验证行均值评分使用命名的有效因子比例阈值。"""
        factors = pd.DataFrame(
            {
                "ROC5": [1.0, 2.0, 3.0, 4.0],
                "MOM10": [None, 3.0, 4.0, 5.0],
                "BIAS20": [None, 4.0, 5.0, 6.0],
            },
            index=["A", "B", "C", "D"],
        )

        scores = composite_factor(factors, method="momentum", min_obs=1)

        self.assertEqual(ROW_MEAN_REQUIRED_FACTOR_FRACTION, 0.5)
        self.assertEqual(_required_row_mean_factor_count(3), 2)
        self.assertTrue(pd.isna(scores.loc["A"]))
        self.assertTrue(pd.notna(scores.loc["B"]))

    def test_composite_factor_groups_intraday_rows_by_trade_date(self) -> None:
        """函数说明：验证 test_composite_factor_groups_intraday_rows_by_trade_date 覆盖的行为场景。"""
        index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-02 15:00"), "A"),
                (pd.Timestamp("2024-01-02 09:30"), "B"),
                (pd.Timestamp("2024-01-02 09:30"), "C"),
            ],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame({"ROC5": [3.0, 2.0, 1.0]}, index=index)

        scores = composite_factor(factors, method="momentum", min_obs=3)

        self.assertEqual(set(scores.index.get_level_values(0)), {pd.Timestamp("2024-01-02")})
        self.assertFalse(scores.isna().any())
        daily = scores.xs(pd.Timestamp("2024-01-02"), level=0)
        self.assertGreater(float(daily.loc["A"]), float(daily.loc["B"]))
        self.assertGreater(float(daily.loc["B"]), float(daily.loc["C"]))

    def test_normalize_factor_frame_for_scoring_caches_same_source_frame(self) -> None:
        """函数说明：验证重复规范化同一个因子框架会复用缓存结果。"""
        _NORMALIZED_FACTOR_FRAME_CACHE.clear()
        index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-02 09:30"), "B"),
                (pd.Timestamp("2024-01-02 15:00"), "A"),
                (pd.Timestamp("2024-01-02 09:30"), "A"),
            ],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame({"ROC5": [2.0, 3.0, 1.0]}, index=index)

        try:
            first = _normalize_factor_frame_for_scoring(factors)
            second = _normalize_factor_frame_for_scoring(factors)

            self.assertIs(first, second)
            self.assertEqual(len(_NORMALIZED_FACTOR_FRAME_CACHE), 1)
            self.assertEqual(
                first.index.tolist(),
                [(pd.Timestamp("2024-01-02"), "A"), (pd.Timestamp("2024-01-02"), "B")],
            )
            self.assertAlmostEqual(float(first.loc[(pd.Timestamp("2024-01-02"), "A"), "ROC5"]), 3.0)
        finally:
            _NORMALIZED_FACTOR_FRAME_CACHE.clear()

    def test_composite_factor_selects_group_columns_before_scoring(self) -> None:
        """函数说明：验证 test_composite_factor_selects_group_columns_before_scoring 覆盖的行为场景。"""
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-02")], ["A", "B", "C", "D", "E"]],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame(
            {
                "STD5": [1, 2, 3, 4, 5],
                "VOLUME0": [10, 9, 8, 7, 6],
                "ROC5": [5, 4, 3, 2, 1],
            },
            index=index,
        )

        scores = composite_factor(factors, method="volatility")
        expected = composite_factor(factors[["STD5"]], method="volatility")

        pd.testing.assert_series_equal(scores, expected)

    def test_factor_columns_for_volume_group_includes_alpha158_volume_families(self) -> None:
        columns = ["VWAP0", "VMA20", "VSTD20", "WVMA60", "VSUMP20", "VSUMN30", "VSUMD60", "ROC20"]

        selected = factor_columns_for_method(columns, "volume")

        self.assertEqual(selected, ["VWAP0", "VMA20", "VSTD20", "WVMA60", "VSUMP20", "VSUMN30", "VSUMD60"])

    def test_composite_factor_supports_inverse_factor_group(self) -> None:
        """函数说明：验证 test_composite_factor_supports_inverse_factor_group 覆盖的行为场景。"""
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-02")], ["A", "B", "C", "D", "E"]],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame(
            {
                "STD5": [1, 2, 3, 4, 5],
                "ROC5": [5, 4, 3, 2, 1],
            },
            index=index,
        )

        high_scores = composite_factor(factors, method="volatility")
        low_scores = composite_factor(factors, method="low_volatility")

        pd.testing.assert_series_equal(low_scores, (high_scores * -1).rename("score"))
        self.assertEqual(factor_columns_for_method(factors.columns, "low_volatility"), ["STD5"])

    def test_composite_factor_supports_exact_single_factor_group(self) -> None:
        """函数说明：验证 test_composite_factor_supports_exact_single_factor_group 覆盖的行为场景。"""
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-02")], ["A", "B", "C", "D", "E"]],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame(
            {
                "STD5": [1, 2, 3, 4, 5],
                "VSTD5": [5, 4, 3, 2, 1],
                "ROC5": [1, 1, 1, 1, 1],
            },
            index=index,
        )

        scores = composite_factor(factors, method="factor:STD5")
        expected = composite_factor(factors[["STD5"]], method="volatility")
        inverse = composite_factor(factors, method="low_factor:STD5")

        pd.testing.assert_series_equal(scores, expected)
        pd.testing.assert_series_equal(inverse, (expected * -1).rename("score"))
        self.assertEqual(factor_columns_for_method(factors.columns, "factor:STD5"), ["STD5"])

    def test_resample_signals_supports_monthly_with_pandas_me_alias(self) -> None:
        """函数说明：验证 test_resample_signals_supports_monthly_with_pandas_me_alias 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-31", "2024-02-01", "2024-02-29"])
        index = pd.MultiIndex.from_product([dates, ["A", "B", "C", "D", "E"]], names=["datetime", "instrument"])
        scores = pd.Series(range(len(index)), index=index, name="score")

        sampled = resample_signals(scores, "monthly")

        self.assertEqual(
            sorted(sampled.index.get_level_values(0).unique().strftime("%Y-%m-%d").tolist()),
            ["2024-01-31", "2024-02-29"],
        )

    def test_resample_signals_accepts_string_date_index(self) -> None:
        """函数说明：验证 test_resample_signals_accepts_string_date_index 覆盖的行为场景。"""
        dates = ["2024-01-02", "2024-01-31", "2024-02-01", "2024-02-29"]
        index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["datetime", "instrument"])
        scores = pd.Series(range(len(index)), index=index, name="score")

        sampled = resample_signals(scores, "monthly")

        self.assertEqual(
            sorted(sampled.index.get_level_values(0).unique().strftime("%Y-%m-%d").tolist()),
            ["2024-01-31", "2024-02-29"],
        )
        self.assertEqual(sampled.index.get_level_values(1).tolist(), ["A", "B", "A", "B"])


if __name__ == "__main__":
    unittest.main()
