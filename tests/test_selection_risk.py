"""模块说明：覆盖 test_selection_risk 相关行为的测试用例。"""

from __future__ import annotations

import unittest

import pandas as pd

from src.selection_risk import _NORMALIZED_PRICE_CACHE, _normalize_price_frame, filter_scores_by_selection_risk


class SelectionRiskTests(unittest.TestCase):
    """类说明：组织 SelectionRiskTests 测试用例。"""
    def test_filter_masks_recent_limit_down_and_missing_required_price(self) -> None:
        """函数说明：验证 test_filter_masks_recent_limit_down_and_missing_required_price 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        prices = _price_panel(
            dates,
            open_values={"A": [10.0, 9.0], "B": [10.0, None], "C": [10.0, 10.2]},
            close_values={"A": [10.0, 9.0], "B": [10.0, 10.1], "C": [10.0, 10.2]},
            low_values={"A": [10.0, 9.0], "B": [10.0, 10.0], "C": [10.0, 10.1]},
        )
        scores = pd.Series([3.0, 2.0, 1.0], index=["A", "B", "C"], name="score")

        filtered = filter_scores_by_selection_risk(scores, prices, "2024-01-03", _config())

        self.assertTrue(pd.isna(filtered.loc["A"]))
        self.assertTrue(pd.isna(filtered.loc["B"]))
        self.assertEqual(float(filtered.loc["C"]), 1.0)

    def test_filter_uses_only_signal_date_and_prior_history(self) -> None:
        """函数说明：验证 test_filter_uses_only_signal_date_and_prior_history 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        prices = _price_panel(
            dates,
            open_values={"A": [10.0, 10.2, 9.0], "B": [10.0, 10.1, 10.2]},
            close_values={"A": [10.0, 10.2, 9.0], "B": [10.0, 10.1, 10.2]},
            low_values={"A": [10.0, 10.1, 9.0], "B": [10.0, 10.0, 10.1]},
        )
        scores = pd.Series([2.0, 1.0], index=["A", "B"], name="score")

        filtered = filter_scores_by_selection_risk(scores, prices, "2024-01-03", _config())

        self.assertEqual(float(filtered.loc["A"]), 2.0)
        self.assertEqual(float(filtered.loc["B"]), 1.0)

    def test_filter_matches_price_columns_case_insensitively(self) -> None:
        """函数说明：验证 test_filter_matches_price_columns_case_insensitively 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        prices = pd.concat(
            {
                " open ": pd.DataFrame({" 000001.sz ": [10.0, 10.1]}, index=dates),
                " close ": pd.DataFrame({" 000001.sz ": [10.0, 10.1]}, index=dates),
                " low ": pd.DataFrame({" 000001.sz ": [10.0, 10.0]}, index=dates),
                " volume ": pd.DataFrame({" 000001.sz ": [1000.0, 1000.0]}, index=dates),
            },
            axis=1,
        )
        scores = pd.Series([1.0], index=["000001.SZ"], name="score")

        filtered = filter_scores_by_selection_risk(scores, prices, "2024-01-03", _config())

        self.assertEqual(float(filtered.loc["000001.SZ"]), 1.0)

    def test_filter_uses_last_intraday_price_per_trade_date(self) -> None:
        """函数说明：验证 test_filter_uses_last_intraday_price_per_trade_date 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02 15:00", "2024-01-02 09:30", "2024-01-03 15:00"])
        prices = pd.concat(
            {
                "open": pd.DataFrame({"A": [10.0, None, 10.1]}, index=dates),
                "close": pd.DataFrame({"A": [10.0, None, 10.1]}, index=dates),
                "low": pd.DataFrame({"A": [10.0, None, 10.0]}, index=dates),
                "volume": pd.DataFrame({"A": [1000.0, None, 1000.0]}, index=dates),
            },
            axis=1,
        )
        scores = pd.Series([1.0], index=["A"], name="score")

        filtered = filter_scores_by_selection_risk(scores, prices, "2024-01-03", _config())

        self.assertEqual(float(filtered.loc["A"]), 1.0)

    def test_filter_rejects_flat_ohlcv_price_frame(self) -> None:
        """函数说明：验证 test_filter_rejects_flat_ohlcv_price_frame 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        prices = pd.DataFrame(
            {
                "open": [10.0, 10.1],
                "close": [10.0, 10.1],
                "volume": [1000.0, 1000.0],
            },
            index=dates,
        )
        scores = pd.Series([1.0], index=["000001.SZ"], name="score")

        with self.assertRaisesRegex(ValueError, "close-price panel"):
            filter_scores_by_selection_risk(scores, prices, "2024-01-03", _config())

    def test_growth_board_limit_down_uses_growth_threshold_when_star_threshold_differs(self) -> None:
        """函数说明：验证 test_growth_board_limit_down_uses_growth_threshold_when_star_threshold_differs 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        stock = "300001.SZ"
        prices = _price_panel(
            dates,
            open_values={stock: [10.0, 8.8]},
            close_values={stock: [10.0, 8.8]},
            low_values={stock: [10.0, 8.8]},
        )
        scores = pd.Series([1.0], index=[stock], name="score")
        config = _config()
        config["backtest"] = {"star_limit_down_threshold": 0.099, "growth_limit_down_threshold": 0.199}

        filtered = filter_scores_by_selection_risk(scores, prices, "2024-01-03", config)

        self.assertEqual(float(filtered.loc[stock]), 1.0)

    def test_missing_price_allowance_counts_sessions_not_fields(self) -> None:
        """函数说明：验证 test_missing_price_allowance_counts_sessions_not_fields 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        prices = pd.concat(
            {
                "open": pd.DataFrame({"A": [10.0, None], "B": [None, None], "C": [10.0, 10.1]}, index=dates),
                "close": pd.DataFrame({"A": [10.0, None], "B": [None, None], "C": [10.0, 10.1]}, index=dates),
                "low": pd.DataFrame({"A": [10.0, None], "B": [None, None], "C": [10.0, 10.0]}, index=dates),
                "volume": pd.DataFrame({"A": [1000.0, None], "B": [None, None], "C": [1000.0, 1000.0]}, index=dates),
            },
            axis=1,
        )
        scores = pd.Series([3.0, 2.0, 1.0], index=["A", "B", "C"], name="score")
        config = _config()
        config["selection_risk_filter"]["max_missing_price_sessions"] = 1
        config["selection_risk_filter"]["max_limit_down_days"] = None

        filtered = filter_scores_by_selection_risk(scores, prices, "2024-01-03", config)

        self.assertEqual(float(filtered.loc["A"]), 3.0)
        self.assertTrue(pd.isna(filtered.loc["B"]))
        self.assertEqual(float(filtered.loc["C"]), 1.0)

    def test_normalized_price_frame_cache_reuses_same_source_panel(self) -> None:
        """函数说明：验证同一价格面板重复规范化会复用缓存。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        prices = _price_panel(
            dates,
            open_values={"A": [10.0, 10.1]},
            close_values={"A": [10.0, 10.1]},
            low_values={"A": [10.0, 10.0]},
        )
        _NORMALIZED_PRICE_CACHE.clear()

        first = _normalize_price_frame(prices)
        second = _normalize_price_frame(prices)

        self.assertIs(first, second)
        self.assertEqual(len(_NORMALIZED_PRICE_CACHE), 1)


def _config() -> dict:
    """函数说明：处理 config 的内部辅助逻辑。"""
    return {
        "selection_risk_filter": {
            "enabled": True,
            "lookback_sessions": 2,
            "required_price_fields": ["open", "close"],
            "max_missing_price_sessions": 0,
            "max_limit_down_days": 0,
            "require_positive_volume": True,
        },
        "backtest": {"limit_down_threshold": 0.099},
    }


def _price_panel(
    dates: pd.DatetimeIndex,
    open_values: dict[str, list[float | None]],
    close_values: dict[str, list[float | None]],
    low_values: dict[str, list[float | None]],
) -> pd.DataFrame:
    """函数说明：处理 price_panel 的内部辅助逻辑。"""
    columns = sorted(close_values)
    volume = {code: [1000.0] * len(dates) for code in columns}
    return pd.concat(
        {
            "open": pd.DataFrame(open_values, index=dates),
            "close": pd.DataFrame(close_values, index=dates),
            "low": pd.DataFrame(low_values, index=dates),
            "volume": pd.DataFrame(volume, index=dates),
        },
        axis=1,
    )


if __name__ == "__main__":
    unittest.main()
