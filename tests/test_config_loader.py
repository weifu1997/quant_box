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

    def test_default_scoring_uses_low_liquidity_bucket_and_winner_take_all_dynamic_ic(self) -> None:
        self.assertEqual(DEFAULT_CONFIG["liquidity_filter"]["side"], "low")
        self.assertEqual(DEFAULT_CONFIG["dynamic_ic_selector"]["top_k"], 1)

    def test_default_quality_includes_full_backtest_return_and_drawdown_gates(self) -> None:
        quality = DEFAULT_CONFIG["quality"]

        self.assertEqual(quality["target_annual_return"], 0.20)
        self.assertEqual(quality["min_backtest_annual_return"], 0.20)
        self.assertEqual(quality["max_backtest_drawdown_limit"], -0.40)

    def test_default_ml_strategy_is_configured_but_disabled_for_existing_pipeline(self) -> None:
        ml = DEFAULT_CONFIG["ml_strategy"]

        self.assertFalse(ml["enabled"])
        self.assertEqual(ml["model_type"], "auto")
        self.assertEqual(ml["label_horizon_sessions"], 20)
        self.assertEqual(ml["fundamental_lag_days"], 90)
        self.assertTrue(DEFAULT_CONFIG["defensive_timing"]["enabled"])


if __name__ == "__main__":
    unittest.main()
