from __future__ import annotations

import unittest

import pandas as pd

from src.market_regime import (
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_SIDEWAYS,
    aggregate_regime_performance,
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

    def test_regime_performance_summarizes_segments_and_aggregates_by_state(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=5)
        equity = pd.Series([100.0, 110.0, 105.0, 120.0, 114.0], index=dates)
        regimes = pd.Series([REGIME_BULL, REGIME_BULL, REGIME_BEAR, REGIME_BEAR, REGIME_BULL], index=dates)

        stats = summarize_regime_performance(equity, regimes, {"annual_trading_days": 252})
        summary = aggregate_regime_performance(stats)

        self.assertEqual(len(stats), 3)
        self.assertEqual(set(summary["regime"]), {REGIME_BULL, REGIME_BEAR})
        self.assertIn("worst_drawdown", summary.columns)


if __name__ == "__main__":
    unittest.main()
