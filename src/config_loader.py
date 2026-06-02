from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"
LOCAL_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.local.yaml"
DEFAULT_CONFIG: dict[str, Any] = {
    "tushare": {"http_url": "http://your-proxy-server:8020/", "token": "your_token", "timeout": 30},
    "data": {
        "start_date": "2015-01-01",
        "end_date": "2025-12-31",
        "universe": "mainboard_a",
        "freq": "daily",
        "raw_dir": "data/raw",
        "constituents_file": "data/raw/mainboard_a_stocks.csv",
        "daily_batch_size": 20,
        "daily_window_days": 250,
        "max_new_symbols_per_run": None,
        "update_chunk_size": 20,
        "update_sleep_seconds": 90,
        "update_progress_file": "outputs/data_update_progress.json",
        "retries": 3,
        "exclude_st": True,
        "st_calendar_file": None,
    },
    "qlib": {"provider_uri": "data/qlib_data", "region": "cn", "instruments": "mainboard_a", "missing_value": -1.0},
    "factors": {"cache_file": "data/factors/alpha158.parquet"},
    "ic": {
        "window": 252,
        "min_periods": 60,
        "min_abs_ic": 0.02,
        "corr_threshold": 0.7,
        "top_k": 30,
        "weight_smoothing": 0.6,
        "max_weight_turnover": 0.5,
        "price_file": "data/prices/ohlcv.parquet",
        "weights_cache_enabled": True,
        "weights_cache_file": "data/factors/rolling_ic_weights.pkl",
    },
    "strategy": {"top_n": 7, "max_turnover": 1, "rank_buffer": 10, "factor_group": "ic_weighted", "rebalance_freq": "weekly"},
    "backtest": {
        "initial_capital": 1_000_000,
        "commission": 0.0003,
        "min_commission_per_order": 5.0,
        "stamp_tax": 0.001,
        "transfer_fee": 0.00001,
        "annual_trading_days": 252,
        "trade_price_field": "open",
        "valuation_price_field": "close",
        "slippage": 0.001,
        "max_participation_rate": 0.05,
        "capacity_window": 20,
        "capacity_warning_threshold": 0.05,
        "amount_unit": 1000.0,
        "stale_price_exit_days": 20,
        "stale_price_haircut": 0.5,
    },
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
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return os.getenv(value[2:-1], "")
    return value
