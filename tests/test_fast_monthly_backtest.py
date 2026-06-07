from __future__ import annotations

import unittest

import pandas as pd

from src.fast_monthly_backtest import run_fast_period_backtest


class FastMonthlyBacktestTests(unittest.TestCase):
    def test_fast_period_backtest_selects_top_scores(self) -> None:
        dates = pd.to_datetime(["2024-01-31", "2024-02-01", "2024-02-29", "2024-03-01"])
        signal_dates = dates[[0, 2]]
        prices = pd.DataFrame(
            {
                ("close", "A"): [10.0, 10.0, 12.0, 12.0],
                ("close", "B"): [10.0, 10.0, 9.0, 9.0],
            },
            index=dates,
        )
        prices.columns = pd.MultiIndex.from_tuples(prices.columns, names=["field", "instrument"])
        scores = pd.Series(
            [1.0, 0.0, 1.0, 0.0],
            index=pd.MultiIndex.from_tuples(
                [
                    (signal_dates[0], "A"),
                    (signal_dates[0], "B"),
                    (signal_dates[1], "A"),
                    (signal_dates[1], "B"),
                ],
                names=["datetime", "instrument"],
            ),
            name="score",
        )

        result = run_fast_period_backtest(
            scores,
            prices,
            "2024-01-01",
            "2024-03-31",
            {"initial_capital": 100.0, "top_n": 1, "max_turnover": 1},
        )

        self.assertGreater(result.equity_curve.iloc[-1], 119.0)
        self.assertEqual(result.weights["instrument"].tolist(), ["A"])

    def test_fast_period_backtest_matches_score_and_price_instruments_case_insensitively(self) -> None:
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

    def test_fast_period_backtest_uses_last_intraday_price_per_trade_date(self) -> None:
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

    def test_fast_period_backtest_rejects_flat_ohlcv_price_frame(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
