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
    return renamed.reset_index(drop=True)


def normalize_daily_basic_frame(df: pd.DataFrame) -> pd.DataFrame:
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
    if df.empty:
        return pd.DataFrame(columns=ST_CALENDAR_FIELDS)
    renamed = df.rename(
        columns={
            "code": "ts_code",
            "start": "start_date",
            "end": "end_date",
            "date": "start_date",
            "reason": "change_reason",
        }
    ).copy()
    if "ts_code" not in renamed.columns:
        raise ValueError("ST calendar source is missing ts_code column.")
    for column in ["name", "start_date", "end_date", "ann_date", "change_reason"]:
        if column not in renamed.columns:
            renamed[column] = pd.NA
    renamed["ts_code"] = _normalize_symbol_series(renamed["ts_code"])
    renamed["name"] = renamed["name"].astype(str)
    renamed["change_reason"] = renamed["change_reason"].astype(str)
    is_st = renamed["name"].str.contains(r"\*?ST", case=False, regex=True) | renamed["change_reason"].str.contains(
        r"\bST\b|\*ST|鐗瑰埆澶勭悊", case=False, regex=True
    )
    renamed = renamed[is_st].copy()
    if renamed.empty:
        return pd.DataFrame(columns=ST_CALENDAR_FIELDS)
    renamed["st_start_date"] = _parse_tushare_dates(renamed["start_date"].where(renamed["start_date"].notna(), renamed["ann_date"]))
    renamed["st_end_date"] = _parse_tushare_dates(renamed["end_date"])
    renamed["ann_date"] = _parse_tushare_dates(renamed["ann_date"])
    renamed = renamed.dropna(subset=["ts_code", "st_start_date"])
    renamed = renamed[renamed["ts_code"] != ""]
    result = renamed[ST_CALENDAR_FIELDS].copy()
    return result.sort_values(["ts_code", "st_start_date"]).reset_index(drop=True)
