from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.market_regime import (
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_SIDEWAYS,
    aggregate_regime_performance,
    apply_defensive_timing_to_backtest_config,
    defensive_exposure_for_date,
    defensive_exposure_schedule,
    detect_market_regime,
    detect_reporting_regime,
    summarize_regime_performance,
)


class MarketRegimeTests(unittest.TestCase):
    def test_detect_market_regime_uses_realtime_lagged_indicators(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=12)
        close = pd.DataFrame({"A": [10, 10.2, 10.4, 10.7, 11.0, 11.2, 11.5, 11.8, 12.1, 12.3, 12.6, 12.9]}, index=dates)
        prices = pd.concat({"close": close}, axis=1)
        config = {
            "market_regime": {
                "enabled": True,
                "ma_window": 3,
                "momentum_window": 2,
                "volatility_window": 2,
                "min_periods": 2,
                "high_volatility_threshold": 10.0,
                "lag_days": 1,
            }
        }

        regimes = detect_market_regime(prices, config)

        self.assertEqual(regimes.iloc[0], REGIME_SIDEWAYS)
        self.assertIn(REGIME_BULL, set(regimes))

    def test_detect_market_regime_normalizes_benchmark_file_columns(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=6)
        with TemporaryDirectory() as tmp:
            benchmark_file = Path(tmp) / "benchmark.csv"
            pd.DataFrame(
                {
                    " date ": dates.strftime("%Y-%m-%d"),
                    " close ": [10.0, 10.2, 10.5, 10.8, 11.1, 11.4],
                }
            ).to_csv(benchmark_file, index=False)
            config = {
                "market_regime": {
                    "enabled": True,
                    "benchmark_file": str(benchmark_file),
                    "ma_window": 2,
                    "momentum_window": 1,
                    "volatility_window": 2,
                    "min_periods": 1,
                    "high_volatility_threshold": 10.0,
                    "lag_days": 0,
                }
            }

            regimes = detect_market_regime(pd.DataFrame(), config)

        self.assertFalse(regimes.empty)
        self.assertIn(REGIME_BULL, set(regimes))

    def test_reporting_regime_uses_objective_unlagged_labels(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=12)
        close = pd.DataFrame({"A": [10, 10.2, 10.4, 10.7, 11.0, 11.2, 11.5, 11.8, 12.1, 12.3, 12.6, 12.9]}, index=dates)
        prices = pd.concat({"close": close}, axis=1)
        config = {
            "market_regime": {
                "enabled": True,
                "ma_window": 3,
                "momentum_window": 2,
                "volatility_window": 2,
                "min_periods": 2,
                "high_volatility_threshold": 10.0,
                "lag_days": 2,
            },
            "reporting_regime": {"enabled": True, "lag_days": 0},
        }

        timing = detect_market_regime(prices, config)
        reporting = detect_reporting_regime(prices, config)

        self.assertNotEqual(reporting.to_list(), timing.to_list())
        self.assertEqual(reporting.name, "reporting_regime")
        self.assertIn(REGIME_BULL, set(reporting.iloc[:4]))

    def test_defensive_exposure_schedule_maps_regimes_to_total_exposure(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=3)
        regimes = pd.Series([REGIME_BULL, REGIME_SIDEWAYS, REGIME_BEAR], index=dates)
        config = {
            "defensive_timing": {
                "enabled": True,
                "bull_exposure": 1.0,
                "sideways_exposure": 0.8,
                "bear_exposure": 0.4,
            }
        }

        exposure = defensive_exposure_schedule(regimes, config, dates)

        self.assertEqual(exposure.to_list(), [1.0, 0.8, 0.4])

    def test_detect_market_regime_can_mark_benchmark_drawdown_as_bear(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=6)
        close = pd.DataFrame({"A": [10.0, 11.0, 12.0, 11.5, 10.7, 10.6]}, index=dates)
        prices = pd.concat({"close": close}, axis=1)
        config = {
            "market_regime": {
                "enabled": True,
                "ma_window": 10,
                "momentum_window": 5,
                "volatility_window": 5,
                "min_periods": 1,
                "high_volatility_threshold": 10.0,
                "bear_drawdown_threshold": 0.10,
                "drawdown_window": 6,
                "lag_days": 0,
            }
        }

        regimes = detect_market_regime(prices, config)

        self.assertEqual(regimes.iloc[-1], REGIME_BEAR)

    def test_apply_defensive_timing_adds_backtest_exposure_schedule(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=6)
        close = pd.DataFrame({"A": [10.0, 9.8, 9.5, 9.1, 8.9, 8.6]}, index=dates)
        prices = pd.concat({"close": close}, axis=1)
        config = {
            "market_regime": {
                "enabled": True,
                "ma_window": 2,
                "momentum_window": 1,
                "volatility_window": 2,
                "min_periods": 1,
                "high_volatility_threshold": 10.0,
                "lag_days": 0,
            },
            "defensive_timing": {"enabled": True, "bear_exposure": 0.4, "sideways_exposure": 0.8, "bull_exposure": 1.0},
        }

        bt_config = apply_defensive_timing_to_backtest_config({"initial_capital": 1000}, prices, config)

        self.assertIn("exposure_schedule", bt_config)
        self.assertLess(float(bt_config["exposure_schedule"].iloc[-1]), 1.0)
        self.assertEqual(defensive_exposure_for_date(prices, config, dates[-1]), float(bt_config["exposure_schedule"].iloc[-1]))

    def test_detect_market_regime_reads_hs300_con_code_constituents(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=6)
        close = pd.DataFrame(
            {
                "000001.SZ": [10.0, 10.4, 10.8, 11.2, 11.6, 12.0],
                "000002.SZ": [20.0, 19.0, 18.0, 17.0, 16.0, 15.0],
                "000003.SZ": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0],
            },
            index=dates,
        )
        prices = pd.concat({" close ": close}, axis=1)
        with TemporaryDirectory() as tmp:
            constituents = Path(tmp) / "hs300_constituents.csv"
            pd.DataFrame({"con_code": ["000001.SZ"], "trade_date": ["2024-01-02"]}).to_csv(constituents, index=False)
            config = {
                "data": {"hs300_constituents_file": str(constituents)},
                "market_regime": {
                    "enabled": True,
                    "ma_window": 2,
                    "momentum_window": 1,
                    "volatility_window": 2,
                    "min_periods": 1,
                    "high_volatility_threshold": 10.0,
                    "lag_days": 0,
                },
            }

            regimes = detect_market_regime(prices, config)

        self.assertEqual(regimes.iloc[-1], REGIME_BULL)

    def test_detect_market_regime_uses_equal_weight_mean_proxy(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=2)
        close = pd.DataFrame(
            {
                "A": [100.0, 200.0],
                "B": [100.0, 90.0],
                "C": [100.0, 90.0],
            },
            index=dates,
        )
        prices = pd.concat({"close": close}, axis=1)
        config = {
            "market_regime": {
                "enabled": True,
                "ma_window": 2,
                "momentum_window": 1,
                "volatility_window": 2,
                "min_periods": 1,
                "high_volatility_threshold": 10.0,
                "lag_days": 0,
            }
        }

        regimes = detect_market_regime(prices, config)

        self.assertEqual(regimes.iloc[-1], REGIME_BULL)

    def test_regime_performance_summarizes_segments_and_aggregates_by_state(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=5)
        equity = pd.Series([100.0, 110.0, 105.0, 120.0, 114.0], index=dates)
        regimes = pd.Series([REGIME_BULL, REGIME_BULL, REGIME_BEAR, REGIME_BEAR, REGIME_BULL], index=dates)

        stats = summarize_regime_performance(equity, regimes, {"annual_trading_days": 252})
        summary = aggregate_regime_performance(stats)

        self.assertEqual(len(stats), 3)
        self.assertEqual(set(summary["regime"]), {REGIME_BULL, REGIME_BEAR})
        self.assertIn("worst_drawdown", summary.columns)

    def test_regime_performance_includes_switch_day_return(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=2)
        equity = pd.Series([100.0, 90.0], index=dates)
        regimes = pd.Series([REGIME_BULL, REGIME_BEAR], index=dates)

        stats = summarize_regime_performance(equity, regimes, {"annual_trading_days": 252})

        bear = stats[stats["regime"] == REGIME_BEAR].iloc[0]
        self.assertAlmostEqual(float(bear["total_return"]), -0.10)
        self.assertAlmostEqual(float(bear["max_drawdown"]), -0.10)


if __name__ == "__main__":
    unittest.main()
