from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PRICE_FIELD_COLUMNS = frozenset({"open", "high", "low", "close", "volume", "vol", "amount", "vwap", "adj_factor", "is_st"})


def normalize_instrument(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def looks_like_field_table(columns: pd.Index, price_fields: set[str] | frozenset[str] = PRICE_FIELD_COLUMNS) -> bool:
    labels = {str(column).strip().lower() for column in columns}
    return len(labels) > 1 and bool(labels & price_fields)


def is_stock_csv(path: Path) -> bool:
    name = path.name.upper()
    return len(name) == len("000001.SZ.CSV") and name[:6].isdigit() and name[6:] in {".SZ.CSV", ".SH.CSV"}


def parse_datetime_values(values: object) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    parsed_series = pd.Series(parsed)
    if parsed_series.isna().any():
        mixed = pd.to_datetime(values, errors="coerce", format="mixed")
        parsed_series = parsed_series.where(parsed_series.notna(), pd.Series(mixed))
    return parsed_series


def coverage_ratio(part: int, whole: int) -> float:
    return float(part / whole) if whole else 0.0


def close_price_frame(price_df: pd.DataFrame, normalize_symbols: bool = True) -> pd.DataFrame:
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
