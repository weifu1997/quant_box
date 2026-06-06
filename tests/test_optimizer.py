from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.optimizer import (
    BASELINE_GRID,
    DEFAULT_GRID,
    _optimization_score,
    run_walk_forward_grid_validation,
    run_walk_forward_optimization,
    with_current_risk_defaults,
)


class OptimizerTests(unittest.TestCase):
    def test_optimization_score_penalizes_turnover_and_trade_cost(self) -> None:
        cheap = {"sharpe": 1.0, "annual_turnover": 1.0, "annual_trade_cost_ratio": 0.02}
        expensive = {"sharpe": 1.0, "annual_turnover": 10.0, "annual_trade_cost_ratio": 0.20}

        self.assertGreater(_optimization_score(cheap), _optimization_score(expensive))

    def test_optimization_score_penalizes_deep_drawdown(self) -> None:
        stable = {
            "annual_return": 0.2,
            "max_drawdown": -0.25,
            "sharpe": 1.0,
            "calmar": 0.8,
            "annual_turnover": 1.0,
            "annual_trade_cost_ratio": 0.01,
        }
        fragile = {**stable, "max_drawdown": -0.55, "calmar": 0.35}

        self.assertGreater(_optimization_score(stable, drawdown_limit=-0.4), _optimization_score(fragile, drawdown_limit=-0.4))

    def test_optimization_score_treats_missing_metrics_as_zero(self) -> None:
        self.assertEqual(_optimization_score({"sharpe": float("nan")}), 0.0)

    def test_baseline_grid_is_smaller_than_full_grid(self) -> None:
        self.assertLess(_grid_size(BASELINE_GRID), _grid_size(DEFAULT_GRID))
        self.assertIn("dynamic_ic_selector", BASELINE_GRID["factor_group"])
        self.assertIn("dynamic_ic_selector", DEFAULT_GRID["factor_group"])
        self.assertEqual(BASELINE_GRID["top_n"], [7, 10, 15])

    def test_current_risk_defaults_are_added_as_grid_columns(self) -> None:
        grid = with_current_risk_defaults(
            BASELINE_GRID,
            {
                "stop_loss_pct": 0.08,
                "circuit_breaker_drawdown": 0.08,
                "circuit_breaker_cooldown_days": 5,
                "circuit_breaker_target_exposure": 0.30,
                "target_vol": None,
            },
        )

        self.assertEqual(grid["stop_loss_pct"], [0.08])
        self.assertEqual(grid["circuit_breaker_drawdown"], [0.08])
        self.assertEqual(grid["circuit_breaker_cooldown_days"], [5])
        self.assertEqual(grid["circuit_breaker_target_exposure"], [0.30])
        self.assertEqual(grid["target_vol"], [None])

    def test_run_walk_forward_optimization_returns_out_of_sample_window(self) -> None:
        factors, prices = _walk_forward_data()
        grid = _small_grid()

        result = run_walk_forward_optimization(
            factors,
            prices,
            base_config=_base_backtest_config(),
            start_date="2023-01-02",
            end_date="2024-02-15",
            grid=grid,
            train_years=1,
            test_months=1,
            step_months=12,
            use_rolling_ic=False,
        )

        self.assertFalse(result.empty)
        self.assertIn("optimization_score", result.columns)
        self.assertIn("test_start", result.columns)

    def test_run_walk_forward_grid_validation_evaluates_all_grid_combinations(self) -> None:
        factors, prices = _walk_forward_data()
        grid = _small_grid()

        result = run_walk_forward_grid_validation(
            factors,
            prices,
            base_config=_base_backtest_config(),
            start_date="2023-01-02",
            end_date="2024-02-15",
            grid=grid,
            train_years=1,
            test_months=1,
            step_months=12,
            use_rolling_ic=False,
        )

        self.assertFalse(result.empty)
        self.assertEqual(len(result), 4)
        self.assertEqual(set(result["top_n"]), {1, 2})
        self.assertEqual(set(result["rebalance_freq"]), {"daily", "weekly"})

    def test_grid_validation_passes_full_scoring_config(self) -> None:
        factors, prices = _walk_forward_data()
        grid = {
            "factor_group": ["dynamic_ic_selector"],
            "top_n": [1],
            "max_turnover": [1],
            "rank_buffer": [0],
            "rebalance_freq": ["monthly"],
            "stop_loss_pct": [0.08],
        }
        captured: list[dict] = []

        def fake_build_scores(factor_df: pd.DataFrame, config: dict, price_df: pd.DataFrame | None = None) -> pd.Series:
            captured.append(config)
            return factor_df["ROC5"].rename("score")

        with patch("src.optimizer.build_strategy_scores", side_effect=fake_build_scores):
            result = run_walk_forward_grid_validation(
                factors,
                prices,
                base_config=_base_backtest_config(),
                start_date="2023-01-02",
                end_date="2024-02-15",
                grid=grid,
                train_years=1,
                test_months=1,
                step_months=12,
                use_rolling_ic=False,
                scoring_config={"strategy": {"factor_group": "old"}, "liquidity_filter": {"enabled": True}},
            )

        self.assertFalse(result.empty)
        self.assertEqual(captured[0]["strategy"]["factor_group"], "dynamic_ic_selector")
        self.assertEqual(captured[0]["strategy"]["stop_loss_pct"], 0.08)
        self.assertTrue(captured[0]["liquidity_filter"]["enabled"])

    def test_ic_weighted_uses_full_scoring_config_when_available(self) -> None:
        factors, prices = _walk_forward_data()
        grid = {
            "factor_group": ["ic_weighted"],
            "top_n": [1],
            "max_turnover": [1],
            "rank_buffer": [0],
            "rebalance_freq": ["monthly"],
        }
        captured: list[dict] = []

        def fake_build_scores(factor_df: pd.DataFrame, config: dict, price_df: pd.DataFrame | None = None) -> pd.Series:
            captured.append(config)
            return factor_df["ROC5"].rename("score")

        with patch("src.optimizer.build_strategy_scores", side_effect=fake_build_scores):
            result = run_walk_forward_grid_validation(
                factors,
                prices,
                base_config=_base_backtest_config(),
                start_date="2023-01-02",
                end_date="2024-02-15",
                grid=grid,
                train_years=1,
                test_months=1,
                step_months=12,
                use_rolling_ic=False,
                scoring_config={"strategy": {"factor_group": "old"}, "liquidity_filter": {"enabled": True}},
            )

        self.assertFalse(result.empty)
        self.assertTrue(captured)
        self.assertEqual(captured[0]["strategy"]["factor_group"], "ic_weighted")
        self.assertTrue(captured[0]["liquidity_filter"]["enabled"])


def _walk_forward_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2023-01-02", periods=300, freq="B")
    instruments = ["A", "B", "C", "D", "E"]
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    values = []
    for day_idx, _date in enumerate(dates):
        for inst_idx, _instrument in enumerate(instruments):
            values.append(day_idx * 0.01 + inst_idx)
    factors = pd.DataFrame({"ROC5": values}, index=index)
    prices = pd.DataFrame(
        {
            instrument: [10.0 + inst_idx + day_idx * 0.01 for day_idx in range(len(dates))]
            for inst_idx, instrument in enumerate(instruments)
        },
        index=dates,
    )
    return factors, prices


def _small_grid() -> dict[str, list]:
    return {
        "factor_group": ["momentum"],
        "top_n": [1, 2],
        "max_turnover": [1],
        "rank_buffer": [0],
        "rebalance_freq": ["daily", "weekly"],
    }


def _grid_size(grid: dict[str, list]) -> int:
    size = 1
    for values in grid.values():
        size *= len(values)
    return size


def _base_backtest_config() -> dict:
    return {
        "initial_capital": 100000,
        "commission": 0.0,
        "stamp_tax": 0.0,
        "top_n": 1,
        "max_turnover": 1,
        "rank_buffer": 0,
        "annual_trading_days": 252,
    }


if __name__ == "__main__":
    unittest.main()
