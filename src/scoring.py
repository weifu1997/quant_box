from __future__ import annotations

import json
import pickle
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
    dynamic_weights = _load_or_compute_dynamic_weights(factors, prices, ic_cfg)
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


def _load_or_compute_dynamic_weights(factors: pd.DataFrame, prices: pd.DataFrame, ic_cfg: dict) -> dict[pd.Timestamp, pd.Series]:
    cache_file = ic_cfg.get("weights_cache_file")
    expected_meta = _weights_cache_meta(factors, prices, ic_cfg)
    if cache_file and bool(ic_cfg.get("weights_cache_enabled", True)):
        cache_path = resolve_path(cache_file)
        cached = _read_weights_cache(cache_path, expected_meta)
        if cached is not None:
            return cached

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
    if cache_file and bool(ic_cfg.get("weights_cache_enabled", True)):
        _write_weights_cache(resolve_path(cache_file), dynamic_weights, expected_meta)
    return dynamic_weights


def _read_weights_cache(cache_path: Path, expected_meta: dict[str, object]) -> dict[pd.Timestamp, pd.Series] | None:
    meta_path = _weights_cache_meta_path(cache_path)
    if not cache_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta != expected_meta:
            return None
        with cache_path.open("rb") as f:
            weights = pickle.load(f)
    except (OSError, json.JSONDecodeError, pickle.PickleError, EOFError, AttributeError, ValueError):
        return None
    if not isinstance(weights, dict):
        return None
    return {pd.Timestamp(date).normalize(): series for date, series in weights.items() if isinstance(series, pd.Series)}


def _write_weights_cache(cache_path: Path, weights: dict[pd.Timestamp, pd.Series], meta: dict[str, object]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(weights, f, protocol=pickle.HIGHEST_PROTOCOL)
    _weights_cache_meta_path(cache_path).write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def _weights_cache_meta_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.name}.meta.json")


def _weights_cache_meta(factors: pd.DataFrame, prices: pd.DataFrame, ic_cfg: dict) -> dict[str, object]:
    return {
        "params": {
            "window": int(ic_cfg.get("window", 252)),
            "min_periods": int(ic_cfg.get("min_periods", 60)),
            "top_k": int(ic_cfg.get("top_k", 30)),
            "min_abs_ic": float(ic_cfg.get("min_abs_ic", 0.02)),
            "corr_threshold": float(ic_cfg.get("corr_threshold", 0.7)),
            "weight_smoothing": float(ic_cfg.get("weight_smoothing", 0.0)),
            "max_weight_turnover": _optional_float(ic_cfg.get("max_weight_turnover")),
        },
        "factors": _frame_signature(factors),
        "prices": _frame_signature(prices),
    }


def _frame_signature(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {"rows": 0, "columns": [], "start": "", "end": "", "symbols": []}
    if isinstance(frame.index, pd.MultiIndex):
        dates = pd.to_datetime(frame.index.get_level_values(0)).normalize()
        symbols = sorted(set(frame.index.get_level_values(1).astype(str).str.upper()))
    else:
        dates = pd.to_datetime(frame.index).normalize()
        if isinstance(frame.columns, pd.MultiIndex):
            symbols = sorted(set(frame.columns.get_level_values(-1).astype(str).str.upper()))
        else:
            symbols = sorted(set(frame.columns.astype(str).str.upper()))
    return {
        "rows": int(len(frame)),
        "columns": [str(col) for col in frame.columns],
        "start": str(dates.min().date()),
        "end": str(dates.max().date()),
        "symbols": symbols,
    }
