from __future__ import annotations

import unittest

import pandas as pd

from src.backtest import run_backtest


class BacktestTests(unittest.TestCase):
    def test_run_backtest_produces_equity_curve(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["datetime", "instrument"])
        scores = pd.Series([2, 1, 2, 1, 1, 2], index=index, name="score")
        prices = pd.DataFrame({"A": [10.0, 11.0, 12.0], "B": [20.0, 20.5, 21.0]}, index=dates)

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-04",
            {
                "initial_capital": 100000,
                "commission": 0.0003,
                "stamp_tax": 0.001,
                "top_n": 1,
                "max_turnover": 1,
                "annual_trading_days": 252,
            },
        )

        self.assertEqual(len(result.equity_curve), 2)
        self.assertIn("total_return", result.metrics)
        self.assertFalse(result.trades.empty)
        self.assertEqual(pd.Timestamp(result.trades.iloc[0]["date"]), pd.Timestamp("2024-01-03"))


if __name__ == "__main__":
    unittest.main()
