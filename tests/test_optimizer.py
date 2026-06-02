from __future__ import annotations

import unittest

from src.optimizer import _optimization_score


class OptimizerTests(unittest.TestCase):
    def test_optimization_score_penalizes_turnover_and_trade_cost(self) -> None:
        cheap = {"sharpe": 1.0, "annual_turnover": 1.0, "annual_trade_cost_ratio": 0.02}
        expensive = {"sharpe": 1.0, "annual_turnover": 10.0, "annual_trade_cost_ratio": 0.20}

        self.assertGreater(_optimization_score(cheap), _optimization_score(expensive))

    def test_optimization_score_treats_missing_metrics_as_zero(self) -> None:
        self.assertEqual(_optimization_score({"sharpe": float("nan")}), 0.0)


if __name__ == "__main__":
    unittest.main()
