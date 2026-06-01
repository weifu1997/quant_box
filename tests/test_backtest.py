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

        self.assertEqual(len(result.equity_curve), 3)
        self.assertIn("total_return", result.metrics)
        self.assertFalse(result.trades.empty)
        self.assertEqual(pd.Timestamp(result.trades.iloc[0]["date"]), pd.Timestamp("2024-01-03"))

    def test_limit_up_blocks_buy(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        index = pd.MultiIndex.from_product([dates, ["A"]], names=["datetime", "instrument"])
        scores = pd.Series([10, 10], index=index, name="score")
        prices = pd.concat(
            {
                "close": pd.DataFrame({"A": [10.0, 11.0]}, index=dates),
                "volume": pd.DataFrame({"A": [1000.0, 1000.0]}, index=dates),
            },
            axis=1,
        )

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-03",
            {
                "initial_capital": 100000,
                "top_n": 1,
                "max_turnover": 1,
                "limit_up_threshold": 0.099,
            },
        )

        self.assertTrue((result.trades["status"] == "blocked").any())
        self.assertEqual(result.equity_curve.iloc[-1], 100000)

    def test_stop_loss_forces_exit(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        index = pd.MultiIndex.from_product([[dates[0]], ["A"]], names=["datetime", "instrument"])
        scores = pd.Series([10], index=index, name="score")
        prices = pd.concat(
            {
                "close": pd.DataFrame({"A": [10.0, 10.0, 9.4]}, index=dates),
                "volume": pd.DataFrame({"A": [1000.0, 1000.0, 1000.0]}, index=dates),
                "amount": pd.DataFrame({"A": [1000.0, 1000.0, 1000.0]}, index=dates),
            },
            axis=1,
        )

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-04",
            {"initial_capital": 100000, "top_n": 1, "max_turnover": 1, "stop_loss_pct": 0.05},
        )

        risk_trades = result.trades[result.trades["status"] == "risk_exit"]
        self.assertFalse(risk_trades.empty)
        self.assertEqual(risk_trades.iloc[0]["reason"], "stop_loss")

    def test_capacity_warning_is_recorded(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        index = pd.MultiIndex.from_product([[dates[0]], ["A"]], names=["datetime", "instrument"])
        scores = pd.Series([10], index=index, name="score")
        prices = pd.concat(
            {
                "close": pd.DataFrame({"A": [10.0, 10.0]}, index=dates),
                "volume": pd.DataFrame({"A": [1000.0, 1000.0]}, index=dates),
                "amount": pd.DataFrame({"A": [1.0, 1.0]}, index=dates),
            },
            axis=1,
        )

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-03",
            {
                "initial_capital": 100000,
                "top_n": 1,
                "max_turnover": 1,
                "capacity_warning_threshold": 0.05,
                "amount_unit": 1000.0,
            },
        )

        filled = result.trades[result.trades["status"] == "filled"]
        self.assertTrue(bool(filled.iloc[0]["capacity_warning"]))


if __name__ == "__main__":
    unittest.main()
