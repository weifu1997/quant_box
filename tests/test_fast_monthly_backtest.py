"""模块说明：覆盖 test_fast_monthly_backtest 相关行为的测试用例。"""

from __future__ import annotations

import unittest

import pandas as pd

from src.fast_monthly_backtest import _period_returns, run_fast_period_backtest
from src.strategy import resample_signals
from tests.fixtures.real_data import require_real_market_data


class FastMonthlyBacktestTests(unittest.TestCase):
    """类说明：组织 FastMonthlyBacktestTests 测试用例。"""
    def test_fast_period_backtest_selects_top_scores(self) -> None:
        """函数说明：验证 test_fast_period_backtest_selects_top_scores 覆盖的行为场景。"""
        market = require_real_market_data(start="2024-01-02", end="2024-04-30")
        scores = resample_signals(market.factors["LOW0"].rename("score"), "monthly")

        result = run_fast_period_backtest(
            scores,
            market.prices,
            str(market.start.date()),
            str(market.end.date()),
            {"initial_capital": 100.0, "top_n": 2, "max_turnover": 2},
        )

        self.assertFalse(result.equity_curve.empty)
        self.assertIn("total_return", result.metrics)
        self.assertFalse(result.weights.empty)
        self.assertTrue(set(result.weights["instrument"]).issubset(set(market.instruments)))

    def test_fast_period_backtest_matches_score_and_price_instruments_case_insensitively(self) -> None:
        """函数说明：验证 test_fast_period_backtest_matches_score_and_price_instruments_case_insensitively 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-31", "2024-02-01", "2024-02-29"])
        prices = pd.DataFrame(
            {
                ("close", "000001.sz"): [10.0, 10.0, 12.0],
            },
            index=dates,
        )
        prices.columns = pd.MultiIndex.from_tuples(prices.columns, names=["field", "instrument"])
        scores = pd.Series(
            [1.0],
            index=pd.MultiIndex.from_tuples([(dates[0], "000001.SZ")], names=["datetime", "instrument"]),
            name="score",
        )

        result = run_fast_period_backtest(
            scores,
            prices,
            "2024-01-01",
            "2024-02-29",
            {"initial_capital": 100.0, "top_n": 1, "max_turnover": 1},
        )

        self.assertFalse(result.weights.empty)
        self.assertEqual(result.weights["instrument"].tolist(), ["000001.SZ"])

    def test_fast_period_backtest_keeps_highest_score_when_normalized_codes_duplicate(self) -> None:
        """函数说明：验证 test_fast_period_backtest_keeps_highest_score_when_normalized_codes_duplicate 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-31", "2024-02-01", "2024-02-29"])
        prices = pd.DataFrame(
            {
                ("close", "A"): [10.0, 10.0, 12.0],
                ("close", "B"): [10.0, 10.0, 9.0],
            },
            index=dates,
        )
        prices.columns = pd.MultiIndex.from_tuples(prices.columns, names=["field", "instrument"])
        scores = pd.Series(
            [10.0, 1.0, 5.0],
            index=pd.MultiIndex.from_tuples(
                [(dates[0], " a "), (dates[0], "A"), (dates[0], "B")],
                names=["datetime", "instrument"],
            ),
            name="score",
        )

        result = run_fast_period_backtest(
            scores,
            prices,
            "2024-01-01",
            "2024-02-29",
            {"initial_capital": 100.0, "top_n": 1, "max_turnover": 1},
        )

        self.assertEqual(result.weights["instrument"].tolist(), ["A"])

    def test_fast_period_backtest_uses_latest_intraday_score_per_signal_date(self) -> None:
        """函数说明：验证 test_fast_period_backtest_uses_latest_intraday_score_per_signal_date 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-31", "2024-02-01", "2024-02-29"])
        prices = pd.DataFrame(
            {
                ("close", "A"): [10.0, 10.0, 12.0],
                ("close", "B"): [10.0, 10.0, 9.0],
            },
            index=dates,
        )
        prices.columns = pd.MultiIndex.from_tuples(prices.columns, names=["field", "instrument"])
        scores = pd.Series(
            [100.0, 1.0, 1.0, 50.0],
            index=pd.MultiIndex.from_tuples(
                [
                    (pd.Timestamp("2024-01-31 09:30"), "A"),
                    (pd.Timestamp("2024-01-31 09:30"), "B"),
                    (pd.Timestamp("2024-01-31 15:00"), "A"),
                    (pd.Timestamp("2024-01-31 15:00"), "B"),
                ],
                names=["datetime", "instrument"],
            ),
            name="score",
        )

        result = run_fast_period_backtest(
            scores,
            prices,
            "2024-01-01",
            "2024-02-29",
            {"initial_capital": 100.0, "top_n": 1, "max_turnover": 1},
        )

        self.assertEqual(result.weights["instrument"].tolist(), ["B"])

    def test_fast_period_backtest_uses_last_intraday_price_per_trade_date(self) -> None:
        """函数说明：验证 test_fast_period_backtest_uses_last_intraday_price_per_trade_date 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-31 15:00", "2024-02-01 09:30", "2024-02-01 15:00", "2024-02-29 15:00"])
        prices = pd.DataFrame(
            {
                ("close", "A"): [10.0, 10.0, 20.0, 40.0],
            },
            index=dates,
        )
        prices.columns = pd.MultiIndex.from_tuples(prices.columns, names=["field", "instrument"])
        scores = pd.Series(
            [1.0],
            index=pd.MultiIndex.from_tuples([(pd.Timestamp("2024-01-31"), "A")], names=["datetime", "instrument"]),
            name="score",
        )

        result = run_fast_period_backtest(
            scores,
            prices,
            "2024-01-01",
            "2024-02-29",
            {"initial_capital": 100.0, "top_n": 1, "max_turnover": 1},
        )

        self.assertAlmostEqual(result.equity_curve.iloc[-1], 200.0)

    def test_fast_period_backtest_marks_intraperiod_drawdown_to_market(self) -> None:
        dates = pd.to_datetime(["2024-02-01", "2024-02-15", "2024-02-29"])
        prices = pd.DataFrame(
            {
                ("close", "A"): [10.0, 5.0, 10.0],
            },
            index=dates,
        )
        prices.columns = pd.MultiIndex.from_tuples(prices.columns, names=["field", "instrument"])
        scores = pd.Series(
            [1.0],
            index=pd.MultiIndex.from_tuples([(pd.Timestamp("2024-01-31"), "A")], names=["datetime", "instrument"]),
            name="score",
        )

        result = run_fast_period_backtest(
            scores,
            prices,
            "2024-01-01",
            "2024-02-29",
            {"initial_capital": 100.0, "top_n": 1, "max_turnover": 1},
        )

        self.assertIn(pd.Timestamp("2024-02-15"), result.equity_curve.index)
        self.assertAlmostEqual(float(result.equity_curve.loc[pd.Timestamp("2024-02-15")]), 50.0)
        self.assertAlmostEqual(result.equity_curve.iloc[-1], 100.0)
        self.assertAlmostEqual(result.metrics["max_drawdown"], -0.5)

    def test_fast_period_backtest_rejects_flat_ohlcv_price_frame(self) -> None:
        """函数说明：验证 test_fast_period_backtest_rejects_flat_ohlcv_price_frame 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-31", "2024-02-01", "2024-02-29"])
        prices = pd.DataFrame(
            {
                "open": [10.0, 10.5, 11.0],
                "close": [10.2, 10.8, 11.2],
                "volume": [1000.0, 1200.0, 1300.0],
            },
            index=dates,
        )
        scores = pd.Series(
            [1.0],
            index=pd.MultiIndex.from_tuples([(dates[0], "000001.SZ")], names=["datetime", "instrument"]),
            name="score",
        )

        with self.assertRaisesRegex(ValueError, "close-price panel"):
            run_fast_period_backtest(
                scores,
                prices,
                "2024-01-01",
                "2024-02-28",
                {"initial_capital": 100.0, "top_n": 1, "max_turnover": 1},
            )

    def test_fast_period_backtest_respects_zero_exposure(self) -> None:
        """函数说明：验证 test_fast_period_backtest_respects_zero_exposure 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-31", "2024-02-01", "2024-02-29"])
        prices = pd.DataFrame(
            {
                ("close", "A"): [10.0, 10.0, 20.0],
            },
            index=dates,
        )
        prices.columns = pd.MultiIndex.from_tuples(prices.columns, names=["field", "instrument"])
        scores = pd.Series(
            [1.0],
            index=pd.MultiIndex.from_tuples([(dates[0], "A")], names=["datetime", "instrument"]),
            name="score",
        )
        exposure = pd.Series([0.0, 0.0, 0.0], index=dates)

        result = run_fast_period_backtest(
            scores,
            prices,
            "2024-01-01",
            "2024-02-29",
            {"initial_capital": 100.0, "top_n": 1, "max_turnover": 1, "exposure_schedule": exposure},
        )

        self.assertAlmostEqual(result.equity_curve.iloc[-1], 100.0)
        self.assertTrue(result.weights.empty or result.weights["weight"].sum() == 0.0)

    def test_fast_period_backtest_uses_latest_intraday_exposure_per_trade_date(self) -> None:
        """函数说明：验证 test_fast_period_backtest_uses_latest_intraday_exposure_per_trade_date 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-31", "2024-02-01", "2024-02-29"])
        prices = pd.DataFrame(
            {
                ("close", "A"): [10.0, 10.0, 20.0],
            },
            index=dates,
        )
        prices.columns = pd.MultiIndex.from_tuples(prices.columns, names=["field", "instrument"])
        scores = pd.Series(
            [1.0],
            index=pd.MultiIndex.from_tuples([(dates[0], "A")], names=["datetime", "instrument"]),
            name="score",
        )
        exposure = pd.Series(
            [0.5, 1.0],
            index=pd.to_datetime(["2024-02-01 15:00", "2024-02-01 09:30"]),
        )

        result = run_fast_period_backtest(
            scores,
            prices,
            "2024-01-01",
            "2024-02-29",
            {"initial_capital": 100.0, "top_n": 1, "max_turnover": 1, "exposure_schedule": exposure},
        )

        self.assertAlmostEqual(float(result.weights.iloc[0]["weight"]), 0.5)
        self.assertAlmostEqual(result.equity_curve.iloc[-1], 150.0)

    def test_fast_period_backtest_applies_max_industry_weight(self) -> None:
        """函数说明：验证 test_fast_period_backtest_applies_max_industry_weight 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-31", "2024-02-01", "2024-02-29", "2024-03-01"])
        signal_dates = dates[[0, 2]]
        instruments = ["A", "B", "C", "D"]
        prices = pd.DataFrame(
            {("close", code): [10.0, 10.0, 10.0, 10.0] for code in instruments},
            index=dates,
        )
        prices.columns = pd.MultiIndex.from_tuples(prices.columns, names=["field", "instrument"])
        scores = pd.Series(
            [10.0, 9.0, 8.0, 7.0, 10.0, 9.0, 8.0, 7.0],
            index=pd.MultiIndex.from_product([signal_dates, instruments], names=["datetime", "instrument"]),
            name="score",
        )

        result = run_fast_period_backtest(
            scores,
            prices,
            "2024-01-01",
            "2024-03-31",
            {
                "initial_capital": 100.0,
                "top_n": 3,
                "max_turnover": 3,
                "industry_map": pd.Series({"A": "bank", "B": "bank", "C": "tech", "D": "health"}),
                "max_industry_weight": 0.5,
            },
        )

        self.assertEqual(result.weights["instrument"].tolist(), ["A", "C", "D"])

    def test_fast_period_backtest_applies_rebalance_drift_threshold_to_weight_trims(self) -> None:
        """函数说明：验证 test_fast_period_backtest_applies_rebalance_drift_threshold_to_weight_trims 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-31", "2024-02-01", "2024-02-29", "2024-03-01", "2024-03-29", "2024-04-01"])
        signal_dates = dates[[0, 2, 4]]
        instruments = ["A", "B", "C"]
        prices = pd.DataFrame(
            {("close", code): [10.0, 10.0, 10.0, 10.0, 10.0, 10.0] for code in instruments},
            index=dates,
        )
        prices.columns = pd.MultiIndex.from_tuples(prices.columns, names=["field", "instrument"])
        scores = pd.Series(
            [3.0, 2.0, 1.0, 3.1, 2.0, 1.0, 3.1, 2.0, 1.0],
            index=pd.MultiIndex.from_product([signal_dates, instruments], names=["datetime", "instrument"]),
            name="score",
        )

        baseline = run_fast_period_backtest(
            scores,
            prices,
            "2024-01-01",
            "2024-03-31",
            {"initial_capital": 100.0, "top_n": 3, "max_turnover": 3, "score_weighted": True},
        )
        drift = run_fast_period_backtest(
            scores,
            prices,
            "2024-01-01",
            "2024-03-31",
            {
                "initial_capital": 100.0,
                "top_n": 3,
                "max_turnover": 3,
                "score_weighted": True,
                "rebalance_drift_threshold": 0.02,
            },
        )

        self.assertGreater(baseline.metrics["total_weight_turnover"], drift.metrics["total_weight_turnover"])
        self.assertAlmostEqual(drift.metrics["total_weight_turnover"], 1.0)

    def test_fast_period_backtest_applies_rebalance_drift_threshold_to_price_drift(self) -> None:
        """函数说明：验证 test_fast_period_backtest_applies_rebalance_drift_threshold_to_price_drift 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-31", "2024-02-01", "2024-02-29", "2024-03-01", "2024-03-29", "2024-04-01"])
        signal_dates = dates[[0, 2, 4]]
        prices = pd.DataFrame(
            {
                ("close", "A"): [10.0, 10.0, 11.0, 11.0, 11.0, 11.0],
                ("close", "B"): [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            },
            index=dates,
        )
        prices.columns = pd.MultiIndex.from_tuples(prices.columns, names=["field", "instrument"])
        scores = pd.Series(
            [2.0, 1.0, 2.0, 1.0, 2.0, 1.0],
            index=pd.MultiIndex.from_product([signal_dates, ["A", "B"]], names=["datetime", "instrument"]),
            name="score",
        )

        baseline = run_fast_period_backtest(
            scores,
            prices,
            "2024-01-01",
            "2024-03-31",
            {"initial_capital": 100.0, "top_n": 2, "max_turnover": 2},
        )
        drift = run_fast_period_backtest(
            scores,
            prices,
            "2024-01-01",
            "2024-03-31",
            {"initial_capital": 100.0, "top_n": 2, "max_turnover": 2, "rebalance_drift_threshold": 0.03},
        )

        self.assertGreater(baseline.metrics["total_weight_turnover"], drift.metrics["total_weight_turnover"])
        self.assertAlmostEqual(drift.metrics["total_weight_turnover"], 1.0)

    def test_fast_period_backtest_drift_threshold_does_not_keep_dropped_holding(self) -> None:
        """函数说明：验证 test_fast_period_backtest_drift_threshold_does_not_keep_dropped_holding 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-31", "2024-02-01", "2024-02-29", "2024-03-01", "2024-03-29", "2024-04-01"])
        signal_dates = dates[[0, 2, 4]]
        instruments = ["A", "B", "C"]
        prices = pd.DataFrame(
            {("close", code): [10.0, 10.0, 10.0, 10.0, 10.0, 10.0] for code in instruments},
            index=dates,
        )
        prices.columns = pd.MultiIndex.from_tuples(prices.columns, names=["field", "instrument"])
        scores = pd.Series(
            [3.0, 2.0, 1.0, 1.0, 3.0, 2.0, 1.0, 3.0, 2.0],
            index=pd.MultiIndex.from_product([signal_dates, instruments], names=["datetime", "instrument"]),
            name="score",
        )

        result = run_fast_period_backtest(
            scores,
            prices,
            "2024-01-01",
            "2024-03-31",
            {"initial_capital": 100.0, "top_n": 2, "max_turnover": 2, "rebalance_drift_threshold": 1.0},
        )

        latest = result.weights[result.weights["date"] == dates[3]]
        self.assertEqual(latest["instrument"].tolist(), ["B", "C"])

    def test_period_returns_uses_available_prices_when_boundaries_are_missing(self) -> None:
        """函数说明：验证 test_period_returns_uses_available_prices_when_boundaries_are_missing 覆盖的行为场景。"""
        close = pd.DataFrame(
            {"A": [10.0, 12.0]},
            index=pd.to_datetime(["2024-01-03", "2024-01-05"]),
        )

        returns = _period_returns(
            close,
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-06"),
            pd.Index(["A"]),
        )

        self.assertAlmostEqual(float(returns.loc["A"]), 0.2)

    def test_period_returns_empty_when_no_valid_price_window_exists(self) -> None:
        """函数说明：验证 test_period_returns_empty_when_no_valid_price_window_exists 覆盖的行为场景。"""
        close = pd.DataFrame(
            {"A": [10.0]},
            index=pd.to_datetime(["2024-01-03"]),
        )

        returns = _period_returns(
            close,
            pd.Timestamp("2024-01-04"),
            pd.Timestamp("2024-01-05"),
            pd.Index(["A"]),
        )

        self.assertTrue(returns.empty)


if __name__ == "__main__":
    unittest.main()
