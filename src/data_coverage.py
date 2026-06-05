from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PRICE_GAP_COLUMNS = ["date", "total_symbols", "priced_symbols", "missing_symbols", "coverage", "missing_instruments"]
YEAR_COVERAGE_COLUMNS = ["year", "start", "end", "days", "has_equity", "passes_min_days"]


def price_coverage_summary(price_df: pd.DataFrame, start_date: str, end_date: str) -> dict[str, Any]:
    close = _close_frame(price_df)
    close = _slice_dates(close, start_date, end_date)
    if close.empty:
        return {
            "start_date": str(pd.Timestamp(start_date).date()),
            "end_date": str(pd.Timestamp(end_date).date()),
            "price_dates": 0,
            "symbols": 0,
            "complete_dates": 0,
            "gap_dates": 0,
            "mean_coverage": 0.0,
            "min_coverage": 0.0,
        }

    coverage = close.notna().sum(axis=1).div(max(len(close.columns), 1))
    return {
        "start_date": str(pd.Timestamp(start_date).date()),
        "end_date": str(pd.Timestamp(end_date).date()),
        "actual_start": str(close.index.min().date()),
        "actual_end": str(close.index.max().date()),
        "price_dates": int(len(close)),
        "symbols": int(len(close.columns)),
        "complete_dates": int((coverage >= 1.0).sum()),
        "gap_dates": int((coverage < 1.0).sum()),
        "mean_coverage": float(coverage.mean()) if len(coverage) else 0.0,
        "min_coverage": float(coverage.min()) if len(coverage) else 0.0,
    }


def build_price_data_gaps(price_df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    close = _close_frame(price_df)
    close = _slice_dates(close, start_date, end_date)
    if close.empty:
        return pd.DataFrame(columns=PRICE_GAP_COLUMNS)

    rows: list[dict[str, object]] = []
    total_symbols = len(close.columns)
    for date, row in close.iterrows():
        missing = [str(column) for column, value in row.items() if pd.isna(value)]
        if not missing:
            continue
        priced = total_symbols - len(missing)
        rows.append(
            {
                "date": pd.Timestamp(date).date().isoformat(),
                "total_symbols": int(total_symbols),
                "priced_symbols": int(priced),
                "missing_symbols": int(len(missing)),
                "coverage": float(priced / total_symbols) if total_symbols else 0.0,
                "missing_instruments": ",".join(missing),
            }
        )
    return pd.DataFrame(rows, columns=PRICE_GAP_COLUMNS)


def build_skipped_months(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty or "skip_reason" not in diagnostics.columns:
        return pd.DataFrame(columns=list(diagnostics.columns) if not diagnostics.empty else ["signal_date", "skip_reason"])
    reasons = diagnostics["skip_reason"].fillna("").astype(str).str.strip()
    return diagnostics.loc[reasons != ""].copy()


def build_yearly_equity_coverage(
    equity_curve: pd.Series,
    start_date: str,
    end_date: str,
    min_trading_days: int = 1,
) -> pd.DataFrame:
    start_year = int(pd.Timestamp(start_date).year)
    end_year = int(pd.Timestamp(end_date).year)
    if equity_curve.empty:
        rows = [
            {"year": year, "start": "", "end": "", "days": 0, "has_equity": False, "passes_min_days": False}
            for year in range(start_year, end_year + 1)
        ]
        return pd.DataFrame(rows, columns=YEAR_COVERAGE_COLUMNS)

    equity = equity_curve.dropna().sort_index()
    equity.index = pd.to_datetime(equity.index).normalize()
    rows: list[dict[str, object]] = []
    for year in range(start_year, end_year + 1):
        segment = equity[equity.index.year == year]
        days = int(len(segment))
        rows.append(
            {
                "year": int(year),
                "start": segment.index.min().date().isoformat() if days else "",
                "end": segment.index.max().date().isoformat() if days else "",
                "days": days,
                "has_equity": bool(days > 0),
                "passes_min_days": bool(days >= max(1, int(min_trading_days))),
            }
        )
    return pd.DataFrame(rows, columns=YEAR_COVERAGE_COLUMNS)


def _close_frame(price_df: pd.DataFrame) -> pd.DataFrame:
    if price_df.empty:
        return pd.DataFrame()
    if isinstance(price_df.columns, pd.MultiIndex):
        fields = price_df.columns.get_level_values(0).astype(str).str.lower()
        if "close" not in set(fields):
            return pd.DataFrame(index=price_df.index)
        close = price_df.loc[:, fields == "close"].copy()
        close.columns = close.columns.get_level_values(-1).astype(str)
    elif "close" in price_df.columns:
        close = price_df[["close"]].copy()
    else:
        close = price_df.copy()
    close.index = pd.to_datetime(close.index).normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _slice_dates(frame: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    return frame.loc[(frame.index >= start) & (frame.index <= end)]
