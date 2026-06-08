from __future__ import annotations

import unittest

from src.backtest import run_backtest
from src.scoring import build_strategy_scores
from src.strategy import resample_signals
from tests.fixtures.real_data import require_real_market_data


class PipelineIntegrationTests(unittest.TestCase):
    def test_scores_resample_and_backtest_real_data_pipeline(self) -> None:
        market = require_real_market_data(start="2024-01-02", end="2024-03-29")

        scores = build_strategy_scores(
            market.factors,
            {
                "strategy": {"factor_group": "factor:LOW0", "min_cross_section_obs": 1},
                "liquidity_filter": {"enabled": False},
                "regime_score_blend": {"enabled": False},
                "regime_score_filter": {"enabled": False},
            },
            price_df=market.prices,
        )
        daily_scores = resample_signals(scores, "daily")
        result = run_backtest(
            daily_scores,
            market.prices,
            str(market.start.date()),
            str(market.end.date()),
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
        self.assertFalse(result.trades.empty)
        self.assertTrue(set(result.trades["instrument"]).issubset({code.upper() for code in market.instruments}))


if __name__ == "__main__":
    unittest.main()
