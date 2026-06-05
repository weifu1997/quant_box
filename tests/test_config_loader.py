from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.config_loader import DEFAULT_CONFIG, _expand_env_values


class ConfigLoaderTests(unittest.TestCase):
    def test_expand_env_values_supports_whole_value_and_embedded_values(self) -> None:
        value = {
            "token": "${TUSHARE_TOKEN}",
            "url": "http://${TUSHARE_HOST}:${TUSHARE_PORT}/api",
            "items": ["${TUSHARE_HOST}", "plain"],
        }

        with patch.dict(
            os.environ,
            {"TUSHARE_TOKEN": "secret", "TUSHARE_HOST": "127.0.0.1", "TUSHARE_PORT": "8020"},
            clear=False,
        ):
            expanded = _expand_env_values(value)

        self.assertEqual(expanded["token"], "secret")
        self.assertEqual(expanded["url"], "http://127.0.0.1:8020/api")
        self.assertEqual(expanded["items"], ["127.0.0.1", "plain"])

    def test_expand_env_values_missing_variable_becomes_empty_string(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_expand_env_values("http://${MISSING_HOST}:8020/"), "http://:8020/")

    def test_default_strategy_includes_portfolio_circuit_breaker(self) -> None:
        strategy = DEFAULT_CONFIG["strategy"]

        self.assertEqual(strategy["circuit_breaker_drawdown"], 0.12)
        self.assertEqual(strategy["circuit_breaker_cooldown_days"], 20)
        self.assertEqual(strategy["rank_buffer"], 0)

    def test_default_data_config_includes_daily_basic_cache(self) -> None:
        self.assertEqual(DEFAULT_CONFIG["data"]["history_start_date"], "2012-01-01")
        self.assertEqual(DEFAULT_CONFIG["data"]["daily_basic_file"], "data/factors/daily_basic.parquet")

    def test_default_scoring_uses_low_liquidity_bucket_and_winner_take_all_dynamic_ic(self) -> None:
        self.assertEqual(DEFAULT_CONFIG["liquidity_filter"]["side"], "low")
        self.assertEqual(DEFAULT_CONFIG["dynamic_ic_selector"]["top_k"], 1)

    def test_default_quality_includes_full_backtest_return_and_drawdown_gates(self) -> None:
        quality = DEFAULT_CONFIG["quality"]

        self.assertEqual(quality["target_annual_return"], 0.20)
        self.assertEqual(quality["min_backtest_annual_return"], 0.20)
        self.assertEqual(quality["max_backtest_drawdown_limit"], -0.20)

    def test_default_ml_strategy_is_configured_but_disabled_for_existing_pipeline(self) -> None:
        ml = DEFAULT_CONFIG["ml_strategy"]

        self.assertFalse(ml["enabled"])
        self.assertEqual(ml["model_type"], "ridge_numpy")
        self.assertEqual(ml["model_objective"], "regression")
        self.assertEqual(ml["class_weight"], "balanced")
        self.assertEqual(ml["training_start_date"], "auto")
        self.assertEqual(ml["label_horizon_sessions"], 20)
        self.assertEqual(ml["label_mode"], "cross_sectional_top_quantile")
        self.assertEqual(ml["label_return_adjustment"], "raw")
        self.assertEqual(ml["label_volatility_window"], 20)
        self.assertEqual(ml["label_volatility_floor"], 0.01)
        self.assertEqual(ml["label_min_cross_section_obs"], 20)
        self.assertEqual(ml["label_top_quantile"], 0.20)
        self.assertEqual(ml["label_bottom_quantile"], 0.20)
        self.assertEqual(ml["ensemble_window"], 3)
        self.assertFalse(ml["feature_ic_evolution"])
        self.assertEqual(ml["feature_limit"], 158)
        self.assertEqual(ml["feature_ic_top_k"], 30)
        self.assertEqual(ml["min_price_history_sessions"], 240)
        self.assertEqual(ml["target_annual_return"], 0.20)
        self.assertEqual(ml["min_yearly_annual_return"], 0.20)
        self.assertEqual(ml["max_drawdown_limit"], -0.20)
        self.assertTrue(ml["training_neutralization"]["enabled"])
        self.assertTrue(ml["training_neutralization"]["industry"])
        self.assertFalse(ml["fundamental_factors_enabled"])
        self.assertEqual(ml["fundamental_lag_days"], 90)
        self.assertEqual(DEFAULT_CONFIG["reporting_regime"]["lag_days"], 0)
        self.assertTrue(DEFAULT_CONFIG["defensive_timing"]["enabled"])
        self.assertEqual(DEFAULT_CONFIG["defensive_timing"]["sideways_exposure"], 1.0)
        self.assertFalse(DEFAULT_CONFIG["neutralization"]["enabled"])
        self.assertTrue(DEFAULT_CONFIG["neutralization"]["industry"])
        self.assertTrue(DEFAULT_CONFIG["neutralization"]["market_cap"])
        self.assertFalse(DEFAULT_CONFIG["feature_extensions"]["enabled"])
        self.assertTrue(DEFAULT_CONFIG["feature_extensions"]["daily_basic"])
        self.assertTrue(DEFAULT_CONFIG["feature_extensions"]["price_derived"])
        self.assertEqual(DEFAULT_CONFIG["feature_extensions"]["price_feature_lag_sessions"], 1)
        self.assertIn("low_amount_20", DEFAULT_CONFIG["feature_extensions"]["price_features"])
        self.assertTrue(DEFAULT_CONFIG["regime_score_blend"]["enabled"])
        self.assertEqual(DEFAULT_CONFIG["regime_score_blend"]["bear_defensive_weight"], 1.0)


if __name__ == "__main__":
    unittest.main()
