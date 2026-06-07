from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from src.config_loader import load_config, resolve_path
from src.data_fetcher import filter_universe_frame
from src.trading_calendar import resolve_target_date_value


def summarize_universe_coverage(
    config: dict | None = None,
    price_df: pd.DataFrame | None = None,
    price_file: str | Path | None = None,
) -> dict[str, float | int | str]:
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
    prices = price_df
    if prices is None and price_file is not None:
        path = resolve_path(price_file)
        if path.exists():
            prices = pd.read_parquet(path)
    if prices is None:
        return set()
    if isinstance(prices.columns, pd.MultiIndex):
        return _normalize_symbols(prices.columns.get_level_values(-1))
    return _normalize_symbols(prices.columns)


def _is_stock_csv(path: Path) -> bool:
    return bool(re.match(r"^\d{6}\.(SZ|SH)\.CSV$", path.name.upper()))


def _ratio(part: int, whole: int) -> float:
    return float(part / whole) if whole else 0.0


def _normalize_symbols(values: object) -> set[str]:
    symbols = pd.Index(values).dropna().astype(str).str.strip().str.upper()
    return set(symbol for symbol in symbols if symbol)
