"""模块说明：规范化 Tushare 返回的数据帧结构。"""

from __future__ import annotations

import pandas as pd

from src.tushare_client import (
    DAILY_BASIC_FIELDS,
    DAILY_FIELDS,
    INDEX_WEIGHT_FIELDS,
    ST_CALENDAR_FIELDS,
    _normalize_symbol_series,
    _parse_tushare_dates,
)


def normalize_daily_frame(df: pd.DataFrame, default_ts_code: str | None = None) -> pd.DataFrame:
    """函数说明：规范化 normalize_daily_frame 主要逻辑。"""
    renamed = df.rename(columns={"volume": "vol", "date": "trade_date"}).copy()
    if "ts_code" not in renamed.columns and default_ts_code:
        renamed["ts_code"] = default_ts_code
    missing = [col for col in DAILY_FIELDS if col not in renamed.columns]
    if missing:
        if renamed.empty:
            return pd.DataFrame(columns=DAILY_FIELDS)
        raise ValueError(f"Daily data is missing columns: {missing}")

    renamed = renamed[DAILY_FIELDS]
    renamed["trade_date"] = _parse_tushare_dates(renamed["trade_date"])
    for col in ["open", "high", "low", "close", "vol", "amount"]:
        renamed[col] = pd.to_numeric(renamed[col], errors="coerce")
    if "adj_factor" in df.columns:
        renamed["adj_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce")
    renamed["ts_code"] = _normalize_symbol_series(renamed["ts_code"])
    renamed = renamed.dropna(subset=["ts_code", "trade_date", "close"])
    renamed = renamed[renamed["ts_code"] != ""].sort_values(["ts_code", "trade_date"])
    renamed = renamed.drop_duplicates(["ts_code", "trade_date"], keep="last")
    _fill_zero_ohlc_suspended_rows(renamed)
    _validate_daily_ohlcv(renamed)
    return renamed.reset_index(drop=True)


def _fill_zero_ohlc_suspended_rows(frame: pd.DataFrame) -> None:
    """Normalize Tushare zero-OHLC suspended rows using the valid close price."""
    if frame.empty:
        return
    zero_ohlc = frame[["open", "high", "low"]].eq(0).all(axis=1)
    no_turnover = frame[["vol", "amount"]].eq(0).all(axis=1)
    valid_close = frame["close"].gt(0)
    suspended = zero_ohlc & no_turnover & valid_close
    if not bool(suspended.any()):
        return
    close_values = frame.loc[suspended, "close"]
    frame.loc[suspended, "open"] = close_values
    frame.loc[suspended, "high"] = close_values
    frame.loc[suspended, "low"] = close_values


def _validate_daily_ohlcv(frame: pd.DataFrame) -> None:
    """Reject final normalized daily rows with impossible OHLCV values."""
    if frame.empty:
        return
    price_columns = ["open", "high", "low", "close"]
    price_values = frame[price_columns]
    missing_prices = price_values.isna().any(axis=1)
    missing_flows = frame[["vol", "amount"]].isna().any(axis=1)
    non_positive_prices = price_values.le(0).any(axis=1)
    negative_flows = frame[["vol", "amount"]].lt(0).any(axis=1)
    high_below_range = frame["high"] < frame[["open", "low", "close"]].max(axis=1)
    low_above_range = frame["low"] > frame[["open", "high", "close"]].min(axis=1)
    invalid = missing_prices | missing_flows | non_positive_prices | negative_flows | high_below_range | low_above_range
    if not bool(invalid.any()):
        return
    examples = ", ".join(_daily_row_labels(frame.loc[invalid].head(5)))
    raise ValueError(f"Daily data has invalid OHLCV values in {int(invalid.sum())} rows: {examples}")


def _daily_row_labels(frame: pd.DataFrame) -> list[str]:
    """Return compact row labels for validation errors."""
    labels: list[str] = []
    for row in frame.itertuples(index=False):
        date_text = "" if pd.isna(row.trade_date) else pd.Timestamp(row.trade_date).date().isoformat()
        labels.append(f"{row.ts_code}@{date_text}")
    return labels


def normalize_daily_basic_frame(df: pd.DataFrame) -> pd.DataFrame:
    """函数说明：规范化 normalize_daily_basic_frame 主要逻辑。"""
    if df.empty:
        return pd.DataFrame(columns=DAILY_BASIC_FIELDS)
    renamed = df.rename(columns={"date": "trade_date", "code": "ts_code"}).copy()
    for column in DAILY_BASIC_FIELDS:
        if column not in renamed.columns:
            renamed[column] = pd.NA
    renamed = renamed[DAILY_BASIC_FIELDS]
    renamed["ts_code"] = _normalize_symbol_series(renamed["ts_code"])
    renamed["trade_date"] = _parse_tushare_dates(renamed["trade_date"])
    for column in DAILY_BASIC_FIELDS:
        if column not in {"ts_code", "trade_date"}:
            renamed[column] = pd.to_numeric(renamed[column], errors="coerce")
    renamed = renamed.dropna(subset=["ts_code", "trade_date"])
    renamed = renamed[renamed["ts_code"] != ""]
    return renamed.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def normalize_index_constituents_frame(df: pd.DataFrame, default_index_code: str | None = None) -> pd.DataFrame:
    """函数说明：规范化 normalize_index_constituents_frame 主要逻辑。"""
    if df.empty:
        return pd.DataFrame(columns=INDEX_WEIGHT_FIELDS)
    renamed = df.rename(columns={"ts_code": "con_code", "code": "con_code", "date": "trade_date"}).copy()
    if "index_code" not in renamed.columns and default_index_code:
        renamed["index_code"] = default_index_code
    missing = [column for column in INDEX_WEIGHT_FIELDS if column not in renamed.columns]
    if missing:
        raise ValueError(f"Index constituent data is missing columns: {missing}")
    renamed = renamed[INDEX_WEIGHT_FIELDS]
    renamed["index_code"] = _normalize_symbol_series(renamed["index_code"])
    renamed["con_code"] = _normalize_symbol_series(renamed["con_code"])
    renamed["trade_date"] = _parse_tushare_dates(renamed["trade_date"]).dt.normalize()
    renamed["weight"] = pd.to_numeric(renamed["weight"], errors="coerce")
    renamed = renamed.dropna(subset=["index_code", "con_code", "trade_date"])
    renamed = renamed[(renamed["index_code"] != "") & (renamed["con_code"] != "")]
    return renamed.sort_values(["trade_date", "index_code", "con_code"]).reset_index(drop=True)


def normalize_st_calendar_frame(df: pd.DataFrame) -> pd.DataFrame:
    """函数说明：规范化 normalize_st_calendar_frame 主要逻辑。"""
    if df.empty:
        return pd.DataFrame(columns=ST_CALENDAR_FIELDS)
    renamed = df.rename(
        columns={
            "code": "ts_code",
            "end": "end_date",
            "reason": "change_reason",
        }
    ).copy()
    if "ts_code" not in renamed.columns:
        raise ValueError("ST calendar source is missing ts_code column.")
    renamed["start_date"] = _coalesce_st_start_date(renamed)
    for column in ["name", "start_date", "end_date", "ann_date", "change_reason"]:
        if column not in renamed.columns:
            renamed[column] = pd.NA
    renamed["ts_code"] = _normalize_symbol_series(renamed["ts_code"])
    renamed["name"] = renamed["name"].astype(str)
    renamed["change_reason"] = renamed["change_reason"].astype(str)
    is_st = renamed["name"].str.contains(r"\*?ST", case=False, regex=True) | renamed["change_reason"].str.contains(
        r"\bST\b|\*ST|鐗瑰埆澶勭悊", case=False, regex=True
    )
    st_reason_pattern = r"\bST\b|\*ST|特别处理|退市风险警示|其他风险警示|实施ST|实施\*ST"
    is_st = is_st | renamed["change_reason"].str.contains(st_reason_pattern, case=False, regex=True)
    renamed = renamed[is_st].copy()
    if renamed.empty:
        return pd.DataFrame(columns=ST_CALENDAR_FIELDS)
    renamed["st_start_date"] = _parse_tushare_dates(renamed["start_date"])
    renamed["st_end_date"] = _parse_tushare_dates(renamed["end_date"])
    renamed["ann_date"] = _parse_tushare_dates(renamed["ann_date"])
    renamed = renamed.dropna(subset=["ts_code", "st_start_date"])
    renamed = renamed[renamed["ts_code"] != ""]
    result = renamed[ST_CALENDAR_FIELDS].copy()
    return result.sort_values(["ts_code", "st_start_date"]).reset_index(drop=True)


def _coalesce_st_start_date(frame: pd.DataFrame) -> pd.Series:
    """函数说明：处理 coalesce_st_start_date 的内部辅助逻辑。"""
    result = pd.Series(pd.NA, index=frame.index, dtype="object")
    for column in ["start_date", "start", "date", "ann_date"]:
        if column not in frame.columns:
            continue
        values = frame[column].replace(r"^\s*$", pd.NA, regex=True)
        result = result.where(result.notna(), values)
    return result
