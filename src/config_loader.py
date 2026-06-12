"""模块说明：加载默认配置、本地配置和环境变量覆盖项。"""

from __future__ import annotations

from copy import deepcopy
import logging
import os
from pathlib import Path
import re
from typing import Any, Callable

import yaml


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"
LOCAL_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.local.yaml"
DEFAULT_CONFIG: dict[str, Any] = {
    "tushare": {"http_url": "http://your-proxy-server:8020/", "token": "your_token", "timeout": 30},
    "data": {
        "start_date": "2015-01-01",
        "history_start_date": "2012-01-01",
        "end_date": "auto",
        "target_date_cutoff_time": "20:00",
        "timezone": "Asia/Shanghai",
        "trade_calendar_lookback_days": 90,
        "universe": "mainboard_a",
        "freq": "daily",
        "raw_dir": "data/raw",
        "constituents_file": "data/raw/mainboard_a_stocks.csv",
        "hs300_constituents_file": "data/raw/hs300_constituents.csv",
        "daily_basic_file": "data/factors/daily_basic.parquet",
        "daily_batch_size": 20,
        "daily_window_days": 500,
        "max_new_symbols_per_run": None,
        "update_chunk_size": 300,
        "update_sleep_seconds": 0,
        "update_progress_file": "outputs/data_update_progress.json",
        "retries": 2,
        "retry_max_wait": 1,
        "exclude_st": True,
        "st_calendar_file": "data/raw/st_calendar.csv",
    },
    "qlib": {"provider_uri": "data/qlib_data", "region": "cn", "instruments": "mainboard_a", "missing_value": None},
    "factors": {"cache_file": "data/factors/alpha158.parquet"},
    "fundamentals": {
        "fina_indicator_file": "data/fundamentals/fina_indicator.parquet",
        "dividend_file": "data/fundamentals/dividend.parquet",
        "fallback_lag_days": 120,
        "update_sleep_seconds": 0.0,
    },
    "fundamental_screen": {
        "include_in_auto_report": False,
        "auto_report_top_n": 10,
        "top_n": 30,
        "min_roe": 0.08,
        "max_debt_to_assets": 0.60,
        "min_dividend_yield": 0.015,
        "min_positive_dividend_years": 2,
        "dividend_lookback_years": 5,
        "min_ocf_to_opincome": 0.80,
        "min_fcf_yield": 0.0,
        "max_pe_ttm": 30.0,
        "max_pb": 5.0,
        "watch_min_score": 4,
        "market_cap_unit": 10000.0,
        "statement_amount_unit": 1.0,
        "prefer_annual_fina": True,
        "max_annual_report_age_days": 550,
    },
    "validated_strategy": {
        "enabled": False,
        "candidate": "",
        "summary_file": None,
        "require_is_acceptable": True,
    },
    "ml_strategy": {
        "enabled": False,
        "model_type": "ridge_numpy",
        "model_objective": "regression",
        "class_weight": "balanced",
        "fallback_on_missing_model": False,
        "train_years": 3,
        "training_start_date": "auto",
        "label_horizon_sessions": 20,
        "label_mode": "cross_sectional_top_quantile",
        "label_return_adjustment": "raw",
        "label_volatility_window": 20,
        "label_volatility_floor": 0.01,
        "label_min_cross_section_obs": 20,
        "label_top_quantile": 0.20,
        "label_bottom_quantile": 0.20,
        "rebalance_freq": "monthly",
        "ensemble_window": 3,
        "feature_columns": None,
        "feature_limit": 158,
        "feature_ic_evolution": False,
        "feature_ic_window": 252,
        "feature_ic_top_k": 30,
        "feature_ic_min_periods": 60,
        "feature_ic_min_obs": 20,
        "feature_ic_min_abs_ic": 0.02,
        "feature_ic_method": "spearman",
        "feature_ic_rebalance_sessions": 1,
        "min_price_history_sessions": 240,
        "min_feature_fraction": 0.5,
        "min_train_rows": 20_000,
        "max_train_rows": 80_000,
        "random_state": 42,
        "label_clip": 0.30,
        "ridge_alpha": 10.0,
        "top_n": 15,
        "score_weighted": False,
        "target_annual_return": 0.20,
        "min_yearly_annual_return": 0.20,
        "max_drawdown_limit": -0.20,
        "training_neutralization": {
            "enabled": True,
            "industry": True,
            "market_cap": True,
            "market_cap_field": "circ_mv",
            "min_obs": 20,
        },
        "fundamental_factors_enabled": False,
        "fundamental_lag_days": 90,
    },
    "market_regime": {
        "enabled": True,
        "benchmark_file": None,
        "benchmark_symbol": None,
        "ma_window": 120,
        "momentum_window": 20,
        "volatility_window": 20,
        "min_periods": 60,
        "bull_momentum_min": 0.0,
        "bear_momentum_max": 0.0,
        "bear_drawdown_threshold": None,
        "drawdown_window": 252,
        "high_volatility_threshold": None,
        "volatility_quantile_window": 252,
        "high_volatility_quantile": 0.75,
        "lag_days": 1,
    },
    "reporting_regime": {"enabled": True, "lag_days": 0},
    "defensive_timing": {
        "enabled": True,
        "bull_exposure": 1.0,
        "sideways_exposure": 0.60,
        "bear_exposure": 0.30,
        "exposure_rebalance_threshold": 0.05,
    },
    "neutralization": {
        "enabled": False,
        "industry": True,
        "market_cap": True,
        "market_cap_field": "circ_mv",
        "min_obs": 20,
    },
    "feature_extensions": {
        "enabled": False,
        "daily_basic": True,
        "price_derived": True,
        "daily_basic_lag_days": 1,
        "price_feature_lag_sessions": 1,
        "daily_basic_fields": [
            "turnover_rate",
            "turnover_rate_f",
            "volume_ratio",
            "pe_ttm",
            "pb",
            "ps_ttm",
            "total_mv",
            "circ_mv",
        ],
        "price_features": [
            "low_amount_5",
            "low_amount_20",
            "low_amount_60",
            "amount_log_20",
            "return_20",
            "return_60",
            "volatility_20",
            "illiquidity_20",
        ],
        "min_coverage": 0.0,
    },
    "regime_score_blend": {
        "enabled": True,
        "bull_defensive_weight": 0.0,
        "sideways_defensive_weight": 0.5,
        "bear_defensive_weight": 1.0,
        "defensive_components": [
            {"column": "STD20", "direction": -1.0},
            {"column": "BETA20", "direction": -1.0},
            {"column": "ROC20", "direction": 1.0},
        ],
    },
    "regime_score_filter": {
        "enabled": False,
        "rules": [],
    },
    "ic": {
        "window": 252,
        "min_periods": 60,
        "min_abs_ic": 0.02,
        "corr_threshold": 0.7,
        "correlation_rebalance_sessions": 1,
        "top_k": 30,
        "weight_smoothing": 0.6,
        "max_weight_turnover": 0.5,
        "price_file": "data/prices/ohlcv_adjusted.parquet",
        "weights_cache_enabled": True,
        "weights_cache_file": "data/factors/rolling_ic_weights.parquet",
    },
    "dynamic_ic_selector": {
        "candidates": [
            "factor:LOW0",
            "factor:VMA60",
            "factor:VSUMN30",
            "factor:VSUMN60",
            "factor:VSUMN20",
            "factor:VMA30",
            "factor:VMA20",
            "factor:MIN5",
            "inverse_factor:KUP",
            "inverse_factor:KLEN",
            "inverse_factor:KLOW",
            "inverse_factor:CORR5",
        ],
        "horizon": 20,
        "window": 504,
        "min_periods": 120,
        "min_obs": 20,
        "method": "spearman",
        "metric": "ic_ir",
        "top_k": 3,
        "fallback_candidate": "factor:LOW0",
    },
    "liquidity_filter": {
        "enabled": True,
        "field": "amount",
        "window": 10,
        "min_periods": 5,
        "quantile": 0.20,
        "side": "low",
    },
    "selection_risk_filter": {
        "enabled": False,
        "lookback_sessions": 5,
        "required_price_fields": ["open", "close"],
        "max_missing_price_sessions": 0,
        "max_limit_down_days": 0,
        "limit_down_buffer": 0.0,
        "require_positive_volume": True,
    },
    "strategy": {
        "top_n": 15,
        "max_turnover": 1,
        "rank_buffer": 30,
        "factor_group": "dynamic_ic_selector",
        "rebalance_freq": "monthly",
        "stop_loss_pct": 0.08,
        "take_profit_pct": None,
        "circuit_breaker_drawdown": 0.08,
        "circuit_breaker_cooldown_days": 5,
        "circuit_breaker_target_exposure": 0.30,
        "max_industry_weight": None,
        "rebalance_drift_threshold": 0.0,
        "min_cross_section_obs": 5,
    },
    "backtest": {
        "initial_capital": 1_000_000,
        "commission": 0.0003,
        "min_commission_per_order": 5.0,
        "stamp_tax": 0.001,
        "transfer_fee": 0.00001,
        "annual_trading_days": 252,
        "trade_price_field": "open",
        "valuation_price_field": "close",
        "slippage": 0.0005,
        "dynamic_slippage_enabled": True,
        "dynamic_slippage_threshold": 0.02,
        "dynamic_slippage_multiplier": 2.0,
        "max_slippage": 0.03,
        "max_participation_rate": 0.05,
        "capacity_window": 20,
        "capacity_warning_threshold": 0.05,
        "amount_unit": 1000.0,
        "stale_price_exit_days": 20,
        "stale_price_exit_policy": "haircut_exit",
        "stale_price_haircut": 0.5,
        "stop_fill_policy": "conservative",
        "stop_fill_buffer": 0.005,
        "limit_up_threshold": 0.099,
        "limit_down_threshold": 0.099,
        "star_limit_up_threshold": 0.199,
        "star_limit_down_threshold": 0.199,
        "growth_limit_up_threshold": 0.199,
        "growth_limit_down_threshold": 0.199,
        "bj_limit_up_threshold": 0.299,
        "bj_limit_down_threshold": 0.299,
        "st_limit_up_threshold": 0.049,
        "st_limit_down_threshold": 0.049,
        "exposure_schedule_rebalance_on_signal_only": False,
        "equity_overlay": {
            "enabled": False,
            "min_periods": 5,
            "ma_window": 90,
            "momentum_window": 5,
            "drawdown_window": 60,
            "drawdown_cut": 0.20,
            "bull_exposure": 1.0,
            "sideways_exposure": 0.2,
            "bear_exposure": 0.0,
            "max_exposure": 1.0,
            "rebalance_threshold": 0.05,
            "rebalance_on_signal_only": False,
        },
    },
    "quality": {
        "min_raw_coverage": 0.95,
        "min_price_coverage": 0.95,
        "min_factor_coverage": 0.95,
        "require_latest_end_date": True,
        "min_validation_windows": 3,
        "min_positive_return_rate": 0.5,
        "min_sharpe_mean": 0.0,
        "max_drawdown_limit": -0.20,
        "target_annual_return": 0.20,
        "min_optimizer_annual_return": 0.20,
        "min_backtest_annual_return": 0.20,
        "max_backtest_drawdown_limit": -0.20,
        "max_annual_turnover": 20.0,
        "max_annual_trade_cost_ratio": 0.2,
        "optimizer_timeout_seconds": None,
        "max_optimizer_combinations": None,
    },
    "account": {
        "file": "config/account.yaml",
        "current_holdings_file": "config/current_holdings.csv",
        "total_asset": 1_000_000,
        "cash": 0.0,
        "max_position_pct": None,
        "lot_size": 100,
        "star_market_lot_size": 200,
    },
    "manual_orders": {
        "limit_price_buffer": 0.002,
        "cash_redistribution_overweight_tolerance": 0.10,
        "confirmation_dir": "outputs/order_confirmations",
        "fill_feedback_dir": "outputs/fill_feedback",
        "confirmed_orders_file": "config/current_holdings.csv",
    },
    "research": {
        "benchmark": {
            "method": "equal_weight_universe",
            "symbol": None,
            "file": None,
        },
        "exposure": {
            "industry_file": "data/raw/mainboard_a_stocks.csv",
            "daily_basic_file": "data/factors/daily_basic.parquet",
            "market_cap_field": "circ_mv",
        },
    },
    "data_governance": {
        "st_calendar_file": "data/raw/st_calendar.csv",
        "index_constituents_file": "data/raw/hs300_constituents.csv",
        "index_fallback_codes": ["399300.SZ"],
        "required_index_columns": ["index_code", "con_code", "trade_date", "weight"],
        "min_daily_basic_date_coverage": 1.0,
        "min_index_constituents_month_coverage": 1.0,
        "adj_factor_meta_file": "data/factors/adj_factor_meta.json",
    },
    "reports": {"history_dir": "outputs/history"},
    "outputs": {"dir": "outputs", "holdings_file": "outputs/latest_holdings.csv"},
}


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """函数说明：加载 load_config 主要逻辑。"""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            file_config = yaml.safe_load(f) or {}
        _warn_unknown_config_keys(file_config)
        config = _deep_merge(DEFAULT_CONFIG, file_config)
    elif config_path is None:
        config = deepcopy(DEFAULT_CONFIG)
    else:
        raise FileNotFoundError(f"Config file not found: {path}")
    if config_path is None and LOCAL_CONFIG_PATH.exists():
        with LOCAL_CONFIG_PATH.open("r", encoding="utf-8") as f:
            local_config = yaml.safe_load(f) or {}
        _warn_unknown_config_keys(local_config)
        config = _deep_merge(config, local_config)
    config = _expand_env_values(config)
    errors = validate_config(config)
    if errors:
        formatted = "\n- ".join(errors)
        raise ValueError(f"Invalid config:\n- {formatted}")
    return config


def validate_config(config: dict[str, Any]) -> list[str]:
    """函数说明：校验已合并配置的结构、类型和关键取值范围。"""
    errors: list[str] = []
    if not isinstance(config, dict):
        return ["config must be a mapping"]

    for key, default_value in DEFAULT_CONFIG.items():
        if isinstance(default_value, dict):
            value = config.get(key)
            if not isinstance(value, dict):
                errors.append(f"{key} must be a mapping")

    for key_path, validator in _CONFIG_VALIDATORS.items():
        value = _get_nested_config_value(config, key_path)
        if value is _MISSING:
            errors.append(f"{key_path} is required")
            continue
        message = validator(value)
        if message:
            errors.append(f"{key_path} {message}")

    return errors


def resolve_path(value: str | Path) -> Path:
    """函数说明：解析 resolve_path 主要逻辑。"""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """函数说明：处理 deep_merge 的内部辅助逻辑。"""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _warn_unknown_config_keys(override: dict[str, Any], schema: dict[str, Any] | None = None, prefix: str = "") -> None:
    """函数说明：处理 warn_unknown_config_keys 的内部辅助逻辑。"""
    schema = DEFAULT_CONFIG if schema is None else schema
    for key, value in override.items():
        key_path = f"{prefix}.{key}" if prefix else str(key)
        if key not in schema:
            logger.warning("Unknown config key: %s", key_path)
            continue
        schema_value = schema[key]
        if isinstance(value, dict) and isinstance(schema_value, dict):
            _warn_unknown_config_keys(value, schema_value, key_path)


_MISSING = object()
_Validator = Callable[[Any], str | None]


def _get_nested_config_value(config: dict[str, Any], key_path: str) -> Any:
    value: Any = config
    for key in key_path.split("."):
        if not isinstance(value, dict) or key not in value:
            return _MISSING
        value = value[key]
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _optional(validator: _Validator) -> _Validator:
    def _validate(value: Any) -> str | None:
        if value is None:
            return None
        return validator(value)

    return _validate


def _bool_value(value: Any) -> str | None:
    return None if isinstance(value, bool) else "must be a boolean"


def _string_value(value: Any) -> str | None:
    return None if isinstance(value, (str, Path)) else "must be a string"


def _int_at_least(min_value: int) -> _Validator:
    def _validate(value: Any) -> str | None:
        if not isinstance(value, int) or isinstance(value, bool):
            return "must be an integer"
        if value < min_value:
            return f"must be >= {min_value}"
        return None

    return _validate


def _number_at_least(min_value: float) -> _Validator:
    def _validate(value: Any) -> str | None:
        if not _is_number(value):
            return "must be a number"
        if float(value) < min_value:
            return f"must be >= {min_value:g}"
        return None

    return _validate


def _number_between(min_value: float, max_value: float) -> _Validator:
    def _validate(value: Any) -> str | None:
        if not _is_number(value):
            return "must be a number"
        number = float(value)
        if number < min_value or number > max_value:
            return f"must be between {min_value:g} and {max_value:g}"
        return None

    return _validate


def _list_of_strings(value: Any) -> str | None:
    if not isinstance(value, list):
        return "must be a list"
    if any(not isinstance(item, str) for item in value):
        return "must contain only strings"
    return None


_CONFIG_VALIDATORS: dict[str, _Validator] = {
    "tushare.http_url": _string_value,
    "tushare.token": _string_value,
    "tushare.timeout": _number_at_least(0),
    "data.start_date": _string_value,
    "data.history_start_date": _string_value,
    "data.end_date": _string_value,
    "data.target_date_cutoff_time": _string_value,
    "data.timezone": _string_value,
    "data.raw_dir": _string_value,
    "data.constituents_file": _string_value,
    "data.daily_basic_file": _string_value,
    "data.trade_calendar_lookback_days": _int_at_least(1),
    "data.daily_batch_size": _int_at_least(1),
    "data.daily_window_days": _int_at_least(1),
    "data.max_new_symbols_per_run": _optional(_int_at_least(1)),
    "data.update_chunk_size": _int_at_least(1),
    "data.update_sleep_seconds": _number_at_least(0),
    "data.retries": _int_at_least(0),
    "data.retry_max_wait": _number_at_least(0),
    "data.exclude_st": _bool_value,
    "qlib.provider_uri": _string_value,
    "factors.cache_file": _string_value,
    "fundamentals.fallback_lag_days": _int_at_least(0),
    "fundamentals.update_sleep_seconds": _number_at_least(0),
    "fundamental_screen.include_in_auto_report": _bool_value,
    "fundamental_screen.auto_report_top_n": _int_at_least(1),
    "fundamental_screen.top_n": _int_at_least(1),
    "fundamental_screen.min_positive_dividend_years": _int_at_least(0),
    "fundamental_screen.dividend_lookback_years": _int_at_least(1),
    "validated_strategy.enabled": _bool_value,
    "validated_strategy.candidate": _string_value,
    "validated_strategy.summary_file": _optional(_string_value),
    "validated_strategy.require_is_acceptable": _bool_value,
    "ml_strategy.enabled": _bool_value,
    "ml_strategy.train_years": _int_at_least(1),
    "ml_strategy.label_horizon_sessions": _int_at_least(1),
    "ml_strategy.label_volatility_floor": _number_at_least(0),
    "ml_strategy.label_min_cross_section_obs": _int_at_least(1),
    "ml_strategy.label_top_quantile": _number_between(0, 1),
    "ml_strategy.label_bottom_quantile": _number_between(0, 1),
    "ml_strategy.ensemble_window": _int_at_least(1),
    "ml_strategy.feature_columns": _optional(_list_of_strings),
    "ml_strategy.feature_limit": _int_at_least(1),
    "ml_strategy.min_price_history_sessions": _int_at_least(1),
    "ml_strategy.min_feature_fraction": _number_between(0, 1),
    "ml_strategy.min_train_rows": _int_at_least(1),
    "ml_strategy.max_train_rows": _int_at_least(1),
    "market_regime.enabled": _bool_value,
    "market_regime.ma_window": _int_at_least(1),
    "market_regime.momentum_window": _int_at_least(1),
    "market_regime.volatility_window": _int_at_least(1),
    "market_regime.min_periods": _int_at_least(1),
    "market_regime.high_volatility_quantile": _number_between(0, 1),
    "market_regime.lag_days": _int_at_least(0),
    "reporting_regime.enabled": _bool_value,
    "reporting_regime.lag_days": _int_at_least(0),
    "defensive_timing.enabled": _bool_value,
    "defensive_timing.bull_exposure": _number_between(0, 1),
    "defensive_timing.sideways_exposure": _number_between(0, 1),
    "defensive_timing.bear_exposure": _number_between(0, 1),
    "defensive_timing.exposure_rebalance_threshold": _number_between(0, 1),
    "neutralization.enabled": _bool_value,
    "neutralization.min_obs": _int_at_least(1),
    "feature_extensions.enabled": _bool_value,
    "feature_extensions.daily_basic": _bool_value,
    "feature_extensions.price_derived": _bool_value,
    "feature_extensions.daily_basic_lag_days": _int_at_least(0),
    "feature_extensions.price_feature_lag_sessions": _int_at_least(0),
    "feature_extensions.daily_basic_fields": _list_of_strings,
    "feature_extensions.price_features": _list_of_strings,
    "feature_extensions.min_coverage": _number_between(0, 1),
    "regime_score_blend.enabled": _bool_value,
    "regime_score_blend.bull_defensive_weight": _number_between(0, 1),
    "regime_score_blend.sideways_defensive_weight": _number_between(0, 1),
    "regime_score_blend.bear_defensive_weight": _number_between(0, 1),
    "regime_score_filter.enabled": _bool_value,
    "ic.window": _int_at_least(1),
    "ic.min_periods": _int_at_least(1),
    "ic.min_abs_ic": _number_at_least(0),
    "ic.corr_threshold": _number_between(0, 1),
    "ic.correlation_rebalance_sessions": _int_at_least(1),
    "ic.top_k": _int_at_least(1),
    "ic.weight_smoothing": _number_between(0, 1),
    "ic.max_weight_turnover": _number_between(0, 1),
    "ic.price_file": _string_value,
    "ic.weights_cache_enabled": _bool_value,
    "ic.weights_cache_file": _string_value,
    "dynamic_ic_selector.candidates": _list_of_strings,
    "dynamic_ic_selector.horizon": _int_at_least(1),
    "dynamic_ic_selector.window": _int_at_least(1),
    "dynamic_ic_selector.min_periods": _int_at_least(1),
    "dynamic_ic_selector.min_obs": _int_at_least(1),
    "dynamic_ic_selector.top_k": _int_at_least(1),
    "liquidity_filter.enabled": _bool_value,
    "liquidity_filter.field": _string_value,
    "liquidity_filter.window": _int_at_least(1),
    "liquidity_filter.min_periods": _int_at_least(1),
    "liquidity_filter.quantile": _number_between(0, 1),
    "liquidity_filter.side": _string_value,
    "selection_risk_filter.enabled": _bool_value,
    "selection_risk_filter.lookback_sessions": _int_at_least(1),
    "selection_risk_filter.required_price_fields": _list_of_strings,
    "selection_risk_filter.max_missing_price_sessions": _int_at_least(0),
    "selection_risk_filter.max_limit_down_days": _int_at_least(0),
    "selection_risk_filter.limit_down_buffer": _number_at_least(0),
    "selection_risk_filter.require_positive_volume": _bool_value,
    "strategy.top_n": _int_at_least(1),
    "strategy.max_turnover": _int_at_least(0),
    "strategy.rank_buffer": _int_at_least(0),
    "strategy.factor_group": _string_value,
    "strategy.rebalance_freq": _string_value,
    "strategy.stop_loss_pct": _optional(_number_between(0, 1)),
    "strategy.take_profit_pct": _optional(_number_at_least(0)),
    "strategy.circuit_breaker_drawdown": _optional(_number_between(0, 1)),
    "strategy.circuit_breaker_cooldown_days": _int_at_least(0),
    "strategy.circuit_breaker_target_exposure": _number_between(0, 1),
    "strategy.max_industry_weight": _optional(_number_between(0, 1)),
    "strategy.rebalance_drift_threshold": _number_between(0, 1),
    "strategy.min_cross_section_obs": _int_at_least(1),
    "backtest.initial_capital": _number_at_least(0),
    "backtest.commission": _number_between(0, 1),
    "backtest.min_commission_per_order": _number_at_least(0),
    "backtest.stamp_tax": _number_between(0, 1),
    "backtest.transfer_fee": _number_between(0, 1),
    "backtest.annual_trading_days": _int_at_least(1),
    "backtest.trade_price_field": _string_value,
    "backtest.valuation_price_field": _string_value,
    "backtest.slippage": _number_between(0, 1),
    "backtest.dynamic_slippage_enabled": _bool_value,
    "backtest.dynamic_slippage_threshold": _number_between(0, 1),
    "backtest.dynamic_slippage_multiplier": _number_at_least(0),
    "backtest.max_slippage": _number_between(0, 1),
    "backtest.max_participation_rate": _number_between(0, 1),
    "backtest.capacity_window": _int_at_least(1),
    "backtest.capacity_warning_threshold": _number_between(0, 1),
    "backtest.amount_unit": _number_at_least(0),
    "backtest.stale_price_exit_days": _int_at_least(0),
    "backtest.stale_price_exit_policy": _string_value,
    "backtest.stale_price_haircut": _number_between(0, 1),
    "backtest.stop_fill_policy": _string_value,
    "backtest.stop_fill_buffer": _number_between(0, 1),
    "backtest.exposure_schedule_rebalance_on_signal_only": _bool_value,
    "backtest.equity_overlay.enabled": _bool_value,
    "backtest.equity_overlay.min_periods": _int_at_least(1),
    "backtest.equity_overlay.ma_window": _int_at_least(1),
    "backtest.equity_overlay.momentum_window": _int_at_least(1),
    "backtest.equity_overlay.drawdown_window": _int_at_least(1),
    "backtest.equity_overlay.drawdown_cut": _number_between(0, 1),
    "backtest.equity_overlay.bull_exposure": _number_between(0, 1),
    "backtest.equity_overlay.sideways_exposure": _number_between(0, 1),
    "backtest.equity_overlay.bear_exposure": _number_between(0, 1),
    "backtest.equity_overlay.max_exposure": _number_between(0, 1),
    "backtest.equity_overlay.rebalance_threshold": _number_between(0, 1),
    "backtest.equity_overlay.rebalance_on_signal_only": _bool_value,
    "quality.min_raw_coverage": _number_between(0, 1),
    "quality.min_price_coverage": _number_between(0, 1),
    "quality.min_factor_coverage": _number_between(0, 1),
    "quality.require_latest_end_date": _bool_value,
    "quality.min_validation_windows": _int_at_least(1),
    "quality.min_positive_return_rate": _number_between(0, 1),
    "quality.max_annual_turnover": _number_at_least(0),
    "quality.max_annual_trade_cost_ratio": _number_between(0, 1),
    "quality.optimizer_timeout_seconds": _optional(_number_at_least(0)),
    "quality.max_optimizer_combinations": _optional(_int_at_least(1)),
    "account.total_asset": _number_at_least(0),
    "account.cash": _number_at_least(0),
    "account.max_position_pct": _optional(_number_between(0, 1)),
    "account.lot_size": _int_at_least(1),
    "account.star_market_lot_size": _int_at_least(1),
    "manual_orders.limit_price_buffer": _number_between(0, 1),
    "manual_orders.cash_redistribution_overweight_tolerance": _number_between(0, 1),
    "outputs.dir": _string_value,
    "outputs.holdings_file": _string_value,
}


def _expand_env_values(value: Any) -> Any:
    """函数说明：处理 expand_env_values 的内部辅助逻辑。"""
    if isinstance(value, dict):
        return {key: _expand_env_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_values(item) for item in value]
    if isinstance(value, str):
        return re.sub(r"\$\{([^}]+)\}", _env_replacement, value)
    return value


def _env_replacement(match: re.Match[str]) -> str:
    """函数说明：处理 env_replacement 的内部辅助逻辑。"""
    name = match.group(1)
    value = os.getenv(name)
    if value is None:
        raise ValueError(f"Environment variable {name} is required by config but is not set.")
    return value
