from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import re
from typing import Any

import yaml


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
        "st_calendar_file": None,
    },
    "qlib": {"provider_uri": "data/qlib_data", "region": "cn", "instruments": "mainboard_a", "missing_value": None},
    "factors": {"cache_file": "data/factors/alpha158.parquet"},
    "ml_strategy": {
        "enabled": False,
        "model_type": "ridge_numpy",
        "model_objective": "regression",
        "class_weight": "balanced",
        "fallback_on_missing_model": True,
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
    "ic": {
        "window": 252,
        "min_periods": 60,
        "min_abs_ic": 0.02,
        "corr_threshold": 0.7,
        "top_k": 30,
        "weight_smoothing": 0.6,
        "max_weight_turnover": 0.5,
        "price_file": "data/prices/ohlcv_adjusted.parquet",
        "weights_cache_enabled": True,
        "weights_cache_file": "data/factors/rolling_ic_weights.pkl",
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
        "side": "high",
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
    },
    "reports": {"history_dir": "outputs/history"},
    "outputs": {"dir": "outputs", "holdings_file": "outputs/latest_holdings.csv"},
}


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            config = _deep_merge(DEFAULT_CONFIG, yaml.safe_load(f) or {})
    elif config_path is None:
        config = deepcopy(DEFAULT_CONFIG)
    else:
        raise FileNotFoundError(f"Config file not found: {path}")
    if config_path is None and LOCAL_CONFIG_PATH.exists():
        with LOCAL_CONFIG_PATH.open("r", encoding="utf-8") as f:
            local_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, local_config)
    return _expand_env_values(config)


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def merged_section(*sections: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for section in sections:
        if section:
            merged.update(deepcopy(section))
    return merged


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _expand_env_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_values(item) for item in value]
    if isinstance(value, str):
        return re.sub(r"\$\{([^}]+)\}", lambda match: os.getenv(match.group(1), ""), value)
    return value
