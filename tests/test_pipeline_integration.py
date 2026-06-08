from __future__ import annotations

import unittest

import pandas as pd

from src.backtest import run_backtest
from src.scoring import build_strategy_scores
from src.strategy import resample_signals


class PipelineIntegrationTests(unittest.TestCase):
    def test_scores_resample_and_backtest_minimal_pipeline(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        instruments = ["000001.SZ", "000002.SZ"]
        factor_index = pd.MultiIndex.from_product([dates[:2], instruments], names=["datetime", "instrument"])
        factors = pd.DataFrame(
            {
                "LOW0": [2.0, 1.0, 2.0, 1.0],
                "ROC20": [0.1, -0.1, 0.1, -0.1],
            },
            index=factor_index,
        )
        prices = pd.concat(
            {
                "open": pd.DataFrame({"000001.SZ": [10.0, 10.0, 11.0], "000002.SZ": [10.0, 10.0, 9.0]}, index=dates),
                "close": pd.DataFrame({"000001.SZ": [10.0, 10.0, 12.0], "000002.SZ": [10.0, 10.0, 9.0]}, index=dates),
                "volume": pd.DataFrame({"000001.SZ": [1000.0] * 3, "000002.SZ": [1000.0] * 3}, index=dates),
                "amount": pd.DataFrame({"000001.SZ": [1000.0] * 3, "000002.SZ": [1000.0] * 3}, index=dates),
            },
            axis=1,
        )

        scores = build_strategy_scores(
            factors,
            {
                "strategy": {"factor_group": "factor:LOW0", "min_cross_section_obs": 1},
                "liquidity_filter": {"enabled": False},
                "regime_score_blend": {"enabled": False},
                "regime_score_filter": {"enabled": False},
            },
            price_df=prices,
        )
        monthly_scores = resample_signals(scores, "daily")
        result = run_backtest(
            monthly_scores,
            prices,
            "2024-01-02",
            "2024-01-04",
            {
                "initial_capital": 100000,
                "top_n": 1,
                "max_turnover": 1,
                "commission": 0.0,
                "stamp_tax": 0.0,
                "slippage": 0.0,
            },
        )

        self.assertFalse(result.equity_curve.empty)
        self.assertIn("total_return", result.metrics)
        self.assertIn("000001.SZ", result.trades["instrument"].tolist())


if __name__ == "__main__":
    unittest.main()
