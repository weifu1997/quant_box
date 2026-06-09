"""模块说明：汇总股票池与本地价格数据之间的覆盖情况。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common import PRICE_FIELD_COLUMNS, is_stock_csv as _is_stock_csv, looks_like_field_table as _looks_like_field_table
from src.config_loader import load_config, resolve_path
from src.data_fetcher import filter_universe_frame
from src.trading_calendar import resolve_target_date_value


def summarize_universe_coverage(
    config: dict | None = None,
    price_df: pd.DataFrame | None = None,
    price_file: str | Path | None = None,
) -> dict[str, float | int | str]:
    """函数说明：汇总 summarize_universe_coverage 主要逻辑。"""
    cfg = config or load_config()
    data_cfg = cfg.get("data", {})
    raw_dir = resolve_path(data_cfg.get("raw_dir", "data/raw"))
    universe_file = resolve_path(data_cfg.get("constituents_file", "data/raw/mainboard_a_stocks.csv"))

    raw_symbols = {
        path.stem.upper()
        for path in raw_dir.glob("*.csv")
        if _is_stock_csv(path)
    }
    target_symbols = _load_target_symbols(cfg, universe_file)
    price_symbols = _price_symbols(price_df, price_file)

    target_count = len(target_symbols)
    local_count = len(raw_symbols)
    price_count = len(price_symbols)
    local_target_count = len(raw_symbols & target_symbols) if target_symbols else local_count
    price_target_count = len(price_symbols & target_symbols) if target_symbols else price_count

    return {
        "universe": str(data_cfg.get("universe", "mainboard_a")),
        "target_symbols": target_count,
        "raw_stock_files": local_count,
        "raw_target_symbols": local_target_count,
        "price_panel_symbols": price_count,
        "price_target_symbols": price_target_count,
        "raw_target_coverage": _ratio(local_target_count, target_count),
        "price_target_coverage": _ratio(price_target_count, target_count),
    }


def _load_target_symbols(config: dict, universe_file: Path) -> set[str]:
    """函数说明：加载 load_target_symbols 的内部辅助逻辑。"""
    data_cfg = config.get("data", {})
    if not universe_file.exists():
        return set()
    df = pd.read_csv(universe_file)
    filtered = filter_universe_frame(
        df,
        universe=str(data_cfg.get("universe", "mainboard_a")),
        as_of_date=resolve_target_date_value(data_cfg.get("end_date"), config=config),
        exclude_st=bool(data_cfg.get("exclude_st", True)),
    )
    for col in ["ts_code", "con_code", "instrument", "code"]:
        if col in filtered.columns:
            return _normalize_symbols(filtered[col].dropna())
    return set()


def _price_symbols(price_df: pd.DataFrame | None, price_file: str | Path | None) -> set[str]:
    """函数说明：处理 price_symbols 的内部辅助逻辑。"""
    prices = price_df
    if prices is None and price_file is not None:
        path = resolve_path(price_file)
        if path.exists():
            prices = pd.read_parquet(path)
    if prices is None:
        return set()
    if isinstance(prices.columns, pd.MultiIndex):
        return _normalize_symbols(prices.columns.get_level_values(-1))
    if _looks_like_field_table(prices.columns):
        raise ValueError("Non-MultiIndex price_df must be a close-price panel with instrument columns.")
    return _normalize_symbols(prices.columns)


def _ratio(part: int, whole: int) -> float:
    """函数说明：处理 ratio 的内部辅助逻辑。"""
    return float(part / whole) if whole else 0.0


def _normalize_symbols(values: object) -> set[str]:
    """函数说明：规范化 normalize_symbols 的内部辅助逻辑。"""
    symbols = pd.Index(values).dropna().astype(str).str.strip().str.upper()
    return set(symbol for symbol in symbols if symbol)
