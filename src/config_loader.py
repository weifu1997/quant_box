from __future__ import annotations

from copy import deepcopy
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
        "universe": "hs300",
        "freq": "daily",
        "raw_dir": "data/raw",
        "constituents_file": "data/raw/hs300_constituents.csv",
    },
    "qlib": {"provider_uri": "data/qlib_data", "region": "cn", "instruments": "csi300", "missing_value": -1.0},
    "factors": {"cache_file": "data/factors/alpha158.parquet"},
    "strategy": {"top_n": 7, "max_turnover": 1, "rank_buffer": 5, "factor_group": "momentum", "rebalance_freq": "daily"},
    "backtest": {"initial_capital": 1_000_000, "commission": 0.0003, "stamp_tax": 0.001, "annual_trading_days": 252},
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
    return config


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
