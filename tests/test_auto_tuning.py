from __future__ import annotations

import unittest

import pandas as pd

from src.auto_tuning import apply_strategy_params, select_stable_params, summarize_parameter_validation


class AutoTuningTests(unittest.TestCase):
    def test_select_stable_params_prefers_consistent_out_of_sample_results(self) -> None:
        validation = pd.DataFrame(
            [
                {
                    "factor_group": "momentum",
                    "top_n": 5,
                    "max_turnover": 1,
                    "rank_buffer": 10,
                    "rebalance_freq": "weekly",
                    "optimization_score": 2.5,
                    "annual_return": 0.4,
                    "sharpe": 2.0,
                    "max_drawdown": -0.05,
                    "annual_turnover": 1.0,
                    "annual_trade_cost_ratio": 0.01,
                },
                {
                    "factor_group": "momentum",
                    "top_n": 5,
                    "max_turnover": 1,
                    "rank_buffer": 10,
                    "rebalance_freq": "weekly",
                    "optimization_score": -2.0,
                    "annual_return": -0.3,
                    "sharpe": -1.0,
                    "max_drawdown": -0.4,
                    "annual_turnover": 1.0,
                    "annual_trade_cost_ratio": 0.01,
                },
                {
                    "factor_group": "ic_weighted",
                    "top_n": 7,
                    "max_turnover": 1,
                    "rank_buffer": 20,
                    "rebalance_freq": "weekly",
                    "optimization_score": 0.8,
                    "annual_return": 0.08,
                    "sharpe": 0.7,
                    "max_drawdown": -0.08,
                    "annual_turnover": 1.0,
                    "annual_trade_cost_ratio": 0.01,
                },
                {
                    "factor_group": "ic_weighted",
                    "top_n": 7,
                    "max_turnover": 1,
                    "rank_buffer": 20,
                    "rebalance_freq": "weekly",
                    "optimization_score": 0.7,
                    "annual_return": 0.07,
                    "sharpe": 0.6,
                    "max_drawdown": -0.09,
                    "annual_turnover": 1.0,
                    "annual_trade_cost_ratio": 0.01,
                },
            ]
        )

        summary = summarize_parameter_validation(validation)
        selected = select_stable_params(summary)

        self.assertEqual(selected["factor_group"], "ic_weighted")
        self.assertEqual(selected["top_n"], 7)
        self.assertEqual(selected["rank_buffer"], 20)

    def test_apply_strategy_params_does_not_mutate_source_config(self) -> None:
        config = {"strategy": {"top_n": 5, "factor_group": "momentum"}, "backtest": {"initial_capital": 1000}}

        selected = apply_strategy_params(config, {"top_n": 7, "rebalance_freq": "weekly"})

        self.assertEqual(config["strategy"]["top_n"], 5)
        self.assertEqual(selected["strategy"]["top_n"], 7)
        self.assertEqual(selected["strategy"]["rebalance_freq"], "weekly")


if __name__ == "__main__":
    unittest.main()
