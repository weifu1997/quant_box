"""模块说明：提供跨模块复用的数据规范化工具。"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


PRICE_FIELD_COLUMNS = frozenset({"open", "high", "low", "close", "volume", "vol", "amount", "vwap", "adj_factor", "is_st"})


def normalize_instrument(value: object) -> str:
    """函数说明：规范化 normalize_instrument 主要逻辑。"""
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def normalize_instrument_index(values: Iterable[object], name: object | None = None) -> pd.Index:
    """Normalize instrument labels while preserving input length and order."""
    return pd.Index([normalize_instrument(value) for value in values], name=name)


def normalize_instruments(values: Iterable[object]) -> list[str]:
    """Normalize instrument labels, dropping blanks and duplicates."""
    result: list[str] = []
    seen: set[str] = set()
    for instrument in normalize_instrument_index(values):
        if not instrument or instrument in seen:
            continue
        result.append(str(instrument))
        seen.add(str(instrument))
    return result


def normalize_datetime_index(
    values: object,
    *,
    normalize: bool = True,
    dropna: bool = False,
    unique: bool = False,
    sort: bool = False,
) -> pd.DatetimeIndex:
    """Parse datetime-like values into a DatetimeIndex with optional cleanup."""
    dates = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce"))
    if dropna:
        dates = dates[~dates.isna()]
    if normalize:
        dates = dates.normalize()
    if unique:
        dates = dates.unique()
    if sort:
        dates = dates.sort_values()
    return pd.DatetimeIndex(dates)


def normalize_multiindex_date_instrument(
    index: pd.MultiIndex,
    *,
    date_level: int | str = 0,
    instrument_level: int | str = 1,
    names: list[str | None] | None = None,
    drop_invalid: bool = True,
) -> pd.MultiIndex:
    """Normalize the date/instrument levels of a two-level MultiIndex."""
    dates = normalize_datetime_index(index.get_level_values(date_level), normalize=True)
    instruments = normalize_instrument_index(index.get_level_values(instrument_level))
    if drop_invalid:
        keep = (~dates.isna()) & (instruments != "")
        dates = dates[keep]
        instruments = instruments[keep]
    output_names = names or [index.names[0], index.names[1]]
    return pd.MultiIndex.from_arrays([dates, instruments], names=output_names)


def looks_like_field_table(columns: pd.Index, price_fields: set[str] | frozenset[str] = PRICE_FIELD_COLUMNS) -> bool:
    """函数说明：处理 looks_like_field_table 主要逻辑。"""
    labels = {str(column).strip().lower() for column in columns}
    return len(labels) > 1 and bool(labels & price_fields)


def is_stock_csv(path: Path) -> bool:
    """函数说明：判断 is_stock_csv 是否成立。"""
    name = path.name.upper()
    return len(name) == len("000001.SZ.CSV") and name[:6].isdigit() and name[6:] in {".SZ.CSV", ".SH.CSV"}


def parse_datetime_values(values: object) -> pd.Series:
    """函数说明：解析 parse_datetime_values 主要逻辑。"""
    parsed = pd.to_datetime(values, errors="coerce")
    parsed_series = pd.Series(parsed)
    if parsed_series.isna().any():
        mixed = pd.to_datetime(values, errors="coerce", format="mixed")
        parsed_series = parsed_series.where(parsed_series.notna(), pd.Series(mixed))
    return parsed_series


def coverage_ratio(part: int, whole: int) -> float:
    """函数说明：处理 coverage_ratio 主要逻辑。"""
    return float(part / whole) if whole else 0.0


def close_price_frame(price_df: pd.DataFrame, normalize_symbols: bool = True) -> pd.DataFrame:
    """函数说明：处理 close_price_frame 主要逻辑。"""
    if price_df.empty:
        return pd.DataFrame()
    if isinstance(price_df.columns, pd.MultiIndex):
        fields = price_df.columns.get_level_values(0).astype(str).str.strip().str.lower()
        if "close" not in set(fields):
            return pd.DataFrame(index=price_df.index)
        close = price_df.loc[:, fields == "close"].copy()
        raw_columns = close.columns.get_level_values(-1)
    else:
        if looks_like_field_table(price_df.columns):
            raise ValueError("Non-MultiIndex price_df must be a close-price panel with instrument columns.")
        close = price_df.copy()
        raw_columns = close.columns
    close.columns = [normalize_instrument(value) if normalize_symbols else str(value) for value in raw_columns]

    raw_dates = pd.DatetimeIndex(pd.to_datetime(close.index, errors="coerce"))
    valid_dates = ~raw_dates.isna()
    close = close.loc[valid_dates].copy()
    raw_dates = raw_dates[valid_dates]
    if not close.empty:
        order = np.argsort(raw_dates.to_numpy(), kind="mergesort")
        close = close.iloc[order].copy()
        raw_dates = raw_dates[order]
    close.index = raw_dates.normalize()
    if normalize_symbols:
        close = close.loc[:, close.columns != ""]
        if close.columns.has_duplicates:
            close = close.loc[:, ~close.columns.duplicated(keep="last")]
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
