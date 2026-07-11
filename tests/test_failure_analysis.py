from __future__ import annotations

import unittest

import pandas as pd

from src.backtest import BacktestResult
from src.failure_analysis import build_yearly_breakdown


class FailureAnalysisTests(unittest.TestCase):
    def test_yearly_cost_ratio_uses_segment_starting_equity(self) -> None:
        dates = pd.bdate_range("2025-01-01", periods=253)
        equity = pd.Series(10_000_000.0, index=dates, name="equity")
        equity.iloc[-1] = 11_000_000.0
        trades = pd.DataFrame(
            [
                {
                    "date": dates[-1],
                    "side": "SELL",
                    "status": "filled",
                    "commission_cost": 100_000.0,
                    "tax_cost": 0.0,
                    "transfer_fee_cost": 0.0,
                    "slippage_cost": 0.0,
                }
            ]
        )
        result = BacktestResult(equity_curve=equity, holdings=pd.DataFrame(), trades=trades, metrics={})

        yearly = build_yearly_breakdown(result, {"initial_capital": 1_000_000.0, "annual_trading_days": 252})

        self.assertAlmostEqual(float(yearly.iloc[0]["annual_trade_cost_ratio"]), 0.01)


if __name__ == "__main__":
    unittest.main()
