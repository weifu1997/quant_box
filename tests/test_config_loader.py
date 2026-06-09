from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.config_loader import DEFAULT_CONFIG, DEFAULT_CONFIG_PATH, _expand_env_values, load_config


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

    def test_expand_env_values_missing_variable_raises(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "MISSING_HOST"):
                _expand_env_values("http://${MISSING_HOST}:8020/")

    def test_default_strategy_includes_portfolio_circuit_breaker(self) -> None:
        strategy = DEFAULT_CONFIG["strategy"]

        self.assertEqual(strategy["circuit_breaker_drawdown"], 0.08)
        self.assertEqual(strategy["circuit_breaker_cooldown_days"], 5)
        self.assertEqual(strategy["circuit_breaker_target_exposure"], 0.30)
        self.assertEqual(strategy["rank_buffer"], 30)
        self.assertEqual(strategy["stop_loss_pct"], 0.08)
        self.assertIsNone(strategy["take_profit_pct"])
        self.assertIsNone(strategy["max_industry_weight"])
        self.assertEqual(strategy["rebalance_drift_threshold"], 0.0)

    def test_settings_yaml_loads_current_strategy_overrides(self) -> None:
        with self.assertNoLogs("src.config_loader", level="WARNING"):
            config = load_config(DEFAULT_CONFIG_PATH)

        self.assertEqual(config["tushare"]["http_url"], "http://your-proxy-server:8020/")
        self.assertEqual(config["tushare"]["token"], "your_token")
        self.assertEqual(config["strategy"]["rank_buffer"], 20)
        self.assertEqual(config["strategy"]["factor_group"], "momentum")
        self.assertIsNone(config["strategy"]["circuit_breaker_drawdown"])
        self.assertEqual(config["strategy"]["take_profit_pct"], 0.35)
        self.assertEqual(config["defensive_timing"]["bear_exposure"], 0.60)
        self.assertEqual(config["liquidity_filter"]["quantile"], 0.65)
        self.assertFalse(config["regime_score_blend"]["enabled"])
        self.assertTrue(config["selection_risk_filter"]["enabled"])
        self.assertEqual(config["selection_risk_filter"]["lookback_sessions"], 3)
        self.assertTrue(config["validated_strategy"]["enabled"])
        self.assertTrue(config["validated_strategy"]["require_is_acceptable"])
        self.assertTrue(config["backtest"]["exposure_schedule_rebalance_on_signal_only"])
        self.assertTrue(config["backtest"]["equity_overlay"]["enabled"])
        self.assertTrue(config["backtest"]["equity_overlay"]["rebalance_on_signal_only"])

    def test_load_config_warns_for_unknown_keys_without_rejecting_them(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.yaml"
            path.write_text(
                """
strategy:
  typo_top_n: 3
unknown_section:
  enabled: true
""",
                encoding="utf-8",
            )

            with self.assertLogs("src.config_loader", level="WARNING") as logs:
                config = load_config(path)

        self.assertEqual(config["strategy"]["typo_top_n"], 3)
        self.assertTrue(config["unknown_section"]["enabled"])
        output = "\n".join(logs.output)
        self.assertIn("strategy.typo_top_n", output)
        self.assertIn("unknown_section", output)

    def test_default_data_config_includes_daily_basic_cache(self) -> None:
        self.assertEqual(DEFAULT_CONFIG["data"]["history_start_date"], "2012-01-01")
        self.assertEqual(DEFAULT_CONFIG["data"]["daily_basic_file"], "data/factors/daily_basic.parquet")
        self.assertEqual(DEFAULT_CONFIG["data"]["st_calendar_file"], "data/raw/st_calendar.csv")

    def test_default_scoring_excludes_low_liquidity_bucket_and_uses_stable_dynamic_ic(self) -> None:
        self.assertEqual(DEFAULT_CONFIG["liquidity_filter"]["side"], "low")
        self.assertEqual(DEFAULT_CONFIG["liquidity_filter"]["quantile"], 0.20)
        self.assertEqual(DEFAULT_CONFIG["ic"]["correlation_rebalance_sessions"], 1)
        self.assertEqual(DEFAULT_CONFIG["dynamic_ic_selector"]["top_k"], 3)
        self.assertEqual(DEFAULT_CONFIG["dynamic_ic_selector"]["metric"], "ic_ir")
        self.assertIn("factor:VMA60", DEFAULT_CONFIG["dynamic_ic_selector"]["candidates"])
        self.assertIn("factor:VSUMN30", DEFAULT_CONFIG["dynamic_ic_selector"]["candidates"])
        self.assertNotIn("factor:RSV5", DEFAULT_CONFIG["dynamic_ic_selector"]["candidates"])

    def test_default_selection_risk_filter_is_configured_but_disabled(self) -> None:
        risk_filter = DEFAULT_CONFIG["selection_risk_filter"]

        self.assertFalse(risk_filter["enabled"])
        self.assertEqual(risk_filter["lookback_sessions"], 5)
        self.assertEqual(risk_filter["required_price_fields"], ["open", "close"])
        self.assertEqual(risk_filter["max_limit_down_days"], 0)

    def test_default_backtest_uses_conservative_execution_assumptions(self) -> None:
        backtest = DEFAULT_CONFIG["backtest"]

        self.assertEqual(backtest["slippage"], 0.0005)
        self.assertTrue(backtest["dynamic_slippage_enabled"])
        self.assertEqual(backtest["stale_price_exit_policy"], "haircut_exit")
        self.assertEqual(backtest["stop_fill_policy"], "conservative")
        self.assertEqual(backtest["star_limit_up_threshold"], 0.199)
        self.assertEqual(backtest["st_limit_down_threshold"], 0.049)
        self.assertFalse(backtest["exposure_schedule_rebalance_on_signal_only"])
        self.assertFalse(backtest["equity_overlay"]["enabled"])
        self.assertEqual(backtest["equity_overlay"]["ma_window"], 90)
        self.assertEqual(backtest["equity_overlay"]["rebalance_threshold"], 0.05)

    def test_default_manual_orders_include_execution_buffers(self) -> None:
        manual_orders = DEFAULT_CONFIG["manual_orders"]

        self.assertEqual(manual_orders["limit_price_buffer"], 0.002)
        self.assertEqual(manual_orders["cash_redistribution_overweight_tolerance"], 0.10)
        self.assertEqual(manual_orders["confirmation_dir"], "outputs/order_confirmations")
        self.assertEqual(manual_orders["fill_feedback_dir"], "outputs/fill_feedback")

    def test_default_research_and_data_governance_outputs_are_configured(self) -> None:
        research = DEFAULT_CONFIG["research"]
        governance = DEFAULT_CONFIG["data_governance"]

        self.assertEqual(research["benchmark"]["method"], "equal_weight_universe")
        self.assertEqual(research["exposure"]["industry_file"], "data/raw/mainboard_a_stocks.csv")
        self.assertEqual(research["exposure"]["market_cap_field"], "circ_mv")
        self.assertEqual(governance["st_calendar_file"], "data/raw/st_calendar.csv")
        self.assertEqual(governance["index_constituents_file"], "data/raw/hs300_constituents.csv")
        self.assertEqual(governance["index_fallback_codes"], ["399300.SZ"])
        self.assertIn("trade_date", governance["required_index_columns"])
        self.assertEqual(governance["min_daily_basic_date_coverage"], 1.0)
        self.assertEqual(governance["min_index_constituents_month_coverage"], 1.0)
        self.assertEqual(governance["adj_factor_meta_file"], "data/factors/adj_factor_meta.json")

    def test_default_quality_includes_full_backtest_return_and_drawdown_gates(self) -> None:
        quality = DEFAULT_CONFIG["quality"]

        self.assertEqual(quality["target_annual_return"], 0.20)
        self.assertEqual(quality["min_optimizer_annual_return"], 0.20)
        self.assertEqual(quality["min_backtest_annual_return"], 0.20)
        self.assertEqual(quality["max_backtest_drawdown_limit"], -0.20)
        self.assertEqual(quality["max_drawdown_limit"], -0.20)

    def test_default_ml_strategy_is_configured_but_disabled_for_existing_pipeline(self) -> None:
        ml = DEFAULT_CONFIG["ml_strategy"]

        self.assertFalse(ml["enabled"])
        self.assertEqual(ml["model_type"], "ridge_numpy")
        self.assertEqual(ml["model_objective"], "regression")
        self.assertEqual(ml["class_weight"], "balanced")
        self.assertEqual(ml["training_start_date"], "auto")
        self.assertEqual(ml["feature_ic_rebalance_sessions"], 1)
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
        self.assertIsNone(DEFAULT_CONFIG["market_regime"]["bear_drawdown_threshold"])
        self.assertEqual(DEFAULT_CONFIG["market_regime"]["drawdown_window"], 252)
        self.assertTrue(DEFAULT_CONFIG["defensive_timing"]["enabled"])
        self.assertEqual(DEFAULT_CONFIG["defensive_timing"]["sideways_exposure"], 0.60)
        self.assertEqual(DEFAULT_CONFIG["defensive_timing"]["bear_exposure"], 0.30)
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
        self.assertFalse(DEFAULT_CONFIG["regime_score_filter"]["enabled"])
        self.assertEqual(DEFAULT_CONFIG["regime_score_filter"]["rules"], [])


if __name__ == "__main__":
    unittest.main()
