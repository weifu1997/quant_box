from __future__ import annotations

import unittest

import pandas as pd

from src.fast_monthly_backtest import run_fast_period_backtest


class FastMonthlyBacktestTests(unittest.TestCase):
    def test_fast_period_backtest_selects_top_scores(self) -> None:
        dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"])
        prices = pd.DataFrame(
            {
                ("close", "A"): [10.0, 12.0, 12.0],
                ("close", "B"): [10.0, 9.0, 9.0],
            },
            index=dates,
        )
        prices.columns = pd.MultiIndex.from_tuples(prices.columns, names=["field", "instrument"])
        scores = pd.Series(
            [1.0, 0.0, 1.0, 0.0],
            index=pd.MultiIndex.from_tuples(
                [
                    (dates[0], "A"),
                    (dates[0], "B"),
                    (dates[1], "A"),
                    (dates[1], "B"),
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
        self.assertEqual(result.weights["instrument"].tolist(), ["A", "A"])

    def test_fast_period_backtest_respects_zero_exposure(self) -> None:
        dates = pd.to_datetime(["2024-01-31", "2024-02-29"])
        prices = pd.DataFrame(
            {
                ("close", "A"): [10.0, 20.0],
            },
            index=dates,
        )
        prices.columns = pd.MultiIndex.from_tuples(prices.columns, names=["field", "instrument"])
        scores = pd.Series(
            [1.0],
            index=pd.MultiIndex.from_tuples([(dates[0], "A")], names=["datetime", "instrument"]),
            name="score",
        )
        exposure = pd.Series([0.0, 0.0], index=dates)

        result = run_fast_period_backtest(
            scores,
            prices,
            "2024-01-01",
            "2024-02-29",
            {"initial_capital": 100.0, "top_n": 1, "max_turnover": 1, "exposure_schedule": exposure},
        )

        self.assertAlmostEqual(result.equity_curve.iloc[-1], 100.0)
        self.assertTrue(result.weights.empty or result.weights["weight"].sum() == 0.0)


if __name__ == "__main__":
    unittest.main()
