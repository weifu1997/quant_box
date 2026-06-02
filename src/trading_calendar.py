from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config_loader import resolve_path


def price_calendar(price_df: pd.DataFrame | None = None, price_file: str | Path | None = None) -> pd.DatetimeIndex:
    prices = price_df
    if prices is None:
        if price_file is None:
            return pd.DatetimeIndex([])
        path = resolve_path(price_file)
        if not path.exists():
            return pd.DatetimeIndex([])
        prices = pd.read_parquet(path)
    if prices.empty:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(pd.to_datetime(prices.index).normalize()).unique().sort_values()


def latest_trade_date(price_df: pd.DataFrame | None = None, price_file: str | Path | None = None) -> pd.Timestamp | None:
    calendar = price_calendar(price_df=price_df, price_file=price_file)
    if calendar.empty:
        return None
    return pd.Timestamp(calendar.max()).normalize()


def next_trade_date(
    signal_date: str | pd.Timestamp,
    price_df: pd.DataFrame | None = None,
    price_file: str | Path | None = None,
) -> pd.Timestamp | None:
    calendar = price_calendar(price_df=price_df, price_file=price_file)
    if calendar.empty:
        return None
    signal_ts = pd.Timestamp(signal_date).normalize()
    pos = calendar.searchsorted(signal_ts, side="right")
    if pos >= len(calendar):
        return None
    return pd.Timestamp(calendar[pos]).normalize()


def next_business_day(date: str | pd.Timestamp) -> pd.Timestamp:
    current = pd.Timestamp(date).normalize() + pd.Timedelta(days=1)
    while current.weekday() >= 5:
        current += pd.Timedelta(days=1)
    return current
