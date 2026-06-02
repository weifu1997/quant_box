from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config_loader import resolve_path
from src.factor_ic import calculate_rolling_ic, make_rolling_ic_weights
from src.strategy import composite_factor


def build_strategy_scores(
    factors: pd.DataFrame,
    config: dict,
    price_df: pd.DataFrame | None = None,
    price_file: str | Path | None = None,
) -> pd.Series:
    strategy_cfg = config.get("strategy", {})
    factor_group = strategy_cfg.get("factor_group", "momentum")
    if factor_group != "ic_weighted":
        return composite_factor(factors, method=factor_group)

    ic_cfg = config.get("ic", {})
    prices = price_df if price_df is not None else _load_ic_price_frame(price_file or ic_cfg.get("price_file", "data/prices/ohlcv.parquet"))
    rolling_ic = calculate_rolling_ic(
        factors,
        prices,
        window=int(ic_cfg.get("window", 252)),
        min_periods=int(ic_cfg.get("min_periods", 60)),
    )
    dynamic_weights = make_rolling_ic_weights(
        rolling_ic,
        top_k=int(ic_cfg.get("top_k", 30)),
        min_abs_ic=float(ic_cfg.get("min_abs_ic", 0.02)),
        min_periods=int(ic_cfg.get("min_periods", 60)),
        correlation_threshold=float(ic_cfg.get("corr_threshold", 0.7)),
        weight_smoothing=float(ic_cfg.get("weight_smoothing", 0.0)),
        max_weight_turnover=_optional_float(ic_cfg.get("max_weight_turnover")),
    )
    return composite_factor(factors, method=factor_group, factor_weights_dynamic=dynamic_weights)


def _load_ic_price_frame(path_value: str | Path) -> pd.DataFrame:
    price_path = resolve_path(path_value)
    if not price_path.exists() and price_path.name == "ohlcv.parquet":
        fallback_price_file = price_path.with_name("close.parquet")
        if fallback_price_file.exists():
            price_path = fallback_price_file
    if not price_path.exists():
        raise FileNotFoundError(f"Price file not found for rolling IC weights: {price_path}")
    return pd.read_parquet(price_path)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
