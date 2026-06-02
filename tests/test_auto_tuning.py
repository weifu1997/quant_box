from __future__ import annotations

import unittest

import pandas as pd

from src.auto_tuning import apply_strategy_params, assess_parameter_quality, select_stable_params, summarize_parameter_validation


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

    def test_assess_parameter_quality_blocks_unstable_selected_params(self) -> None:
        summary = pd.DataFrame(
            [
                {
                    "factor_group": "momentum",
                    "top_n": 5,
                    "max_turnover": 1,
                    "rank_buffer": 10,
                    "rebalance_freq": "weekly",
                    "windows": 2,
                    "positive_return_rate": 0.25,
                    "sharpe_mean": -0.1,
                    "max_drawdown_worst": -0.5,
                    "annual_turnover_mean": 2.0,
                    "annual_trade_cost_ratio_mean": 0.01,
                    "auto_score": 1.0,
                }
            ]
        )

        quality = assess_parameter_quality(summary)

        self.assertFalse(quality.is_acceptable)
        self.assertIn("validation_windows_below_threshold:2<3", quality.issues)
        self.assertTrue(any(issue.startswith("positive_return_rate_below_threshold") for issue in quality.issues))


if __name__ == "__main__":
    unittest.main()
