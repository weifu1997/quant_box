"""模块说明：通过 Tushare 拉取、更新和补齐市场数据。"""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import io
import json
import logging
from pathlib import Path
import random
import time
from typing import Iterable

import pandas as pd

from src.common import coverage_ratio as _coverage_ratio
from src.config_loader import load_config, resolve_path
from src.data_fetcher_frames import (
    normalize_daily_basic_frame,
    normalize_daily_frame,
    normalize_index_constituents_frame,
    normalize_st_calendar_frame,
)
from src.trading_calendar import resolve_target_date_value
from src.tushare_client import (
    ADJ_FACTOR_FIELDS,
    DAILY_BASIC_FIELDS,
    DAILY_FIELDS,
    INDEX_WEIGHT_FIELDS,
    MAINBOARD_PREFIXES,
    NAMECHANGE_FIELDS,
    ST_CALENDAR_FIELDS,
    STOCK_BASIC_FIELDS,
    TushareHttpClient,
    _date_windows,
    _default_port,
    _format_tushare_date,
    _is_tushare_connection_error,
    _normalize_symbol,
    _normalize_symbol_series,
    _parse_tushare_dates,
    _parse_tushare_frame,
    _retry_wait_seconds,
    _unique_normalized_symbols,
    describe_endpoint,
)


logger = logging.getLogger(__name__)


class ResumableUpdateResult(dict):
    """类说明：封装 ResumableUpdateResult 相关数据和行为。"""
    def __init__(
        self,
        written: dict[str, Path] | None = None,
        *,
        status: str = "",
        progress_path: str | Path | None = None,
        failed_symbols: int = 0,
        remaining_symbols: int = 0,
        latest_symbols: int = 0,
        fresh_or_confirmed_symbols: int = 0,
        last_error: str = "",
    ) -> None:
        """函数说明：初始化实例状态。"""
        super().__init__(written or {})
        self.status = status
        self.progress_path = Path(progress_path) if progress_path is not None else None
        self.failed_symbols = int(failed_symbols)
        self.remaining_symbols = int(remaining_symbols)
        self.latest_symbols = int(latest_symbols)
        self.fresh_or_confirmed_symbols = int(fresh_or_confirmed_symbols)
        self.last_error = str(last_error or "")

    def to_status_dict(self) -> dict[str, object]:
        """函数说明：处理 to_status_dict 主要逻辑。"""
        return {
            "status": self.status,
            "progress_path": str(self.progress_path) if self.progress_path is not None else "",
            "failed_symbols": self.failed_symbols,
            "remaining_symbols": self.remaining_symbols,
            "latest_symbols": self.latest_symbols,
            "fresh_or_confirmed_symbols": self.fresh_or_confirmed_symbols,
            "last_error": self.last_error,
            "written_symbols": len(self),
        }


def _resumable_result_from_progress(written: dict[str, Path], progress_path: Path, progress: dict[str, object]) -> ResumableUpdateResult:
    """函数说明：处理 resumable_result_from_progress 的内部辅助逻辑。"""
    return ResumableUpdateResult(
        written,
        status=str(progress.get("status", "")),
        progress_path=progress_path,
        failed_symbols=_safe_int(progress.get("failed_symbols"), 0),
        remaining_symbols=_safe_int(progress.get("remaining_symbols"), 0),
        latest_symbols=_safe_int(progress.get("latest_symbols"), 0),
        fresh_or_confirmed_symbols=_safe_int(progress.get("fresh_or_confirmed_symbols"), 0),
        last_error=str(progress.get("last_error", "")),
    )

def fetch_daily_stock(
    ts_code: str,
    start_date: str | datetime,
    end_date: str | datetime,
    client: TushareHttpClient | None = None,
    retries: int = 5,
    retry_max_wait: float | None = None,
) -> pd.DataFrame:
    """函数说明：拉取 fetch_daily_stock 主要逻辑。"""
    ts_code = _normalize_symbol(ts_code)
    client = client or TushareHttpClient.from_config()
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            params = {
                "ts_code": ts_code,
                "start_date": _format_tushare_date(start_date),
                "end_date": _format_tushare_date(end_date),
            }
            df = client.call("daily", params=params, fields=DAILY_FIELDS)
            daily = normalize_daily_frame(df, default_ts_code=ts_code)
            adj = client.call("adj_factor", params=params, fields=ADJ_FACTOR_FIELDS)
            return merge_adj_factor(daily, adj, default_ts_code=ts_code)
        except (RuntimeError, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                wait_seconds = _retry_wait_seconds(attempt, retry_max_wait)
                logger.warning("Retrying %s daily data after error: %s", ts_code, exc)
                time.sleep(wait_seconds)
    raise ValueError(f"{ts_code} daily data response is invalid after {retries} attempts: {last_error}") from last_error


def fetch_daily_stocks(
    ts_codes: Iterable[str],
    start_date: str | datetime,
    end_date: str | datetime,
    client: TushareHttpClient | None = None,
    retries: int = 5,
    retry_max_wait: float | None = None,
    batch_size: int = 100,
    window_days: int | None = None,
    skip_failed: bool = True,
) -> pd.DataFrame:
    """函数说明：拉取 fetch_daily_stocks 主要逻辑。"""
    codes = _unique_normalized_symbols(ts_codes)
    if not codes:
        return pd.DataFrame(columns=[*DAILY_FIELDS, "adj_factor"])
    if len(codes) == 1 or batch_size <= 1:
        frames = []
        failed_codes: list[str] = []
        for code in codes:
            try:
                frames.append(
                    fetch_daily_stock(
                        code,
                        start_date,
                        end_date,
                        client=client,
                        retries=retries,
                        retry_max_wait=retry_max_wait,
                    )
                )
            except (RuntimeError, ValueError) as exc:
                if not skip_failed:
                    raise
                failed_codes.append(code)
                logger.error("Skipping %s after daily fetch failure: %s", code, exc)
        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[*DAILY_FIELDS, "adj_factor"])
        result.attrs["failed_codes"] = failed_codes
        return result

    client = client or TushareHttpClient.from_config()
    frames: list[pd.DataFrame] = []
    all_failed_codes: set[str] = set()
    for batch in _batched(codes, batch_size):
        batch_frames: list[pd.DataFrame] = []
        failed_codes: set[str] = set()
        for window_start, window_end in _date_windows(start_date, end_date, window_days):
            window_df = _fetch_daily_stock_batch(
                batch,
                window_start,
                window_end,
                client=client,
                retries=retries,
                retry_max_wait=retry_max_wait,
                skip_failed=skip_failed,
            )
            failed_codes.update(window_df.attrs.get("failed_codes", []))
            batch_frames.append(window_df)
        if batch_frames:
            batch_df = pd.concat(batch_frames, ignore_index=True)
            if failed_codes:
                batch_df = batch_df[~batch_df["ts_code"].isin(failed_codes)].copy() if not batch_df.empty else batch_df
            all_failed_codes.update(failed_codes)
            frames.append(batch_df)
    if not frames:
        result = pd.DataFrame(columns=[*DAILY_FIELDS, "adj_factor"])
        result.attrs["failed_codes"] = sorted(all_failed_codes)
        return result
    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(["ts_code", "trade_date"], keep="last")
    result.attrs["failed_codes"] = sorted(all_failed_codes)
    return result


def fetch_daily_basic(
    trade_date: str | datetime,
    client: TushareHttpClient | None = None,
    fields: Iterable[str] | str | None = None,
    retries: int = 5,
    retry_max_wait: float | None = None,
) -> pd.DataFrame:
    """函数说明：拉取 fetch_daily_basic 主要逻辑。"""
    client = client or TushareHttpClient.from_config()
    params = {"trade_date": _format_tushare_date(trade_date)}
    requested_fields = fields or DAILY_BASIC_FIELDS
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            df = client.call("daily_basic", params=params, fields=requested_fields)
            return normalize_daily_basic_frame(df)
        except (RuntimeError, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                wait_seconds = _retry_wait_seconds(attempt, retry_max_wait)
                logger.warning("Retrying daily_basic %s after error: %s", params["trade_date"], exc)
                time.sleep(wait_seconds)
    raise ValueError(f"daily_basic response is invalid for {params['trade_date']} after {retries} attempts: {last_error}") from last_error


def fetch_index_constituents(
    index_code: str = "000300.SH",
    start_date: str | datetime | None = None,
    end_date: str | datetime | None = None,
    client: TushareHttpClient | None = None,
    fields: Iterable[str] | str | None = None,
    retries: int = 5,
    retry_max_wait: float | None = None,
) -> pd.DataFrame:
    """函数说明：拉取 fetch_index_constituents 主要逻辑。"""
    client = client or TushareHttpClient.from_config()
    params = {"index_code": index_code}
    if start_date is not None:
        params["start_date"] = _format_tushare_date(start_date)
    if end_date is not None:
        params["end_date"] = _format_tushare_date(end_date)
    requested_fields = fields or INDEX_WEIGHT_FIELDS
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            frame = client.call("index_weight", params=params, fields=requested_fields)
            return normalize_index_constituents_frame(frame, default_index_code=index_code)
        except (RuntimeError, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                wait_seconds = _retry_wait_seconds(attempt, retry_max_wait)
                logger.warning("Retrying index_weight %s after error: %s", index_code, exc)
                time.sleep(wait_seconds)
    raise ValueError(f"index_weight response is invalid for {index_code} after {retries} attempts: {last_error}") from last_error


def update_index_constituents_data(
    index_code: str = "000300.SH",
    start_date: str | datetime | None = None,
    end_date: str | datetime | None = None,
    out_file: str | Path | None = None,
    client: TushareHttpClient | None = None,
    sleep_seconds: float = 0.0,
    retries: int | None = None,
    retry_max_wait: float | None = None,
    window_days: int = 31,
    max_windows: int | None = None,
    skip_failed: bool = True,
    fallback_index_codes: Iterable[str] | None = None,
) -> Path:
    """函数说明：更新 update_index_constituents_data 主要逻辑。"""
    config = load_config()
    data_cfg = config.get("data", {})
    gov_cfg = config.get("data_governance", {})
    start = pd.Timestamp(start_date or data_cfg.get("history_start_date") or data_cfg["start_date"]).normalize()
    end = pd.Timestamp(resolve_target_date_value(end_date or data_cfg["end_date"], config=config)).normalize()
    path = resolve_path(out_file or data_cfg.get("hs300_constituents_file", "data/raw/hs300_constituents.csv"))
    path.parent.mkdir(parents=True, exist_ok=True)

    client = client or TushareHttpClient.from_config(config)
    retries = int(retries if retries is not None else data_cfg.get("retries", 5))
    retry_max_wait = retry_max_wait if retry_max_wait is not None else data_cfg.get("retry_max_wait", 30)
    retry_max_wait = float(retry_max_wait) if retry_max_wait is not None else None
    frames: list[pd.DataFrame] = []
    windows = list(_date_windows(start, end, window_days))
    if max_windows is not None:
        windows = windows[: max(0, int(max_windows))]
    fallback_values = fallback_index_codes if fallback_index_codes is not None else gov_cfg.get("index_fallback_codes", [])
    candidate_codes = _index_candidate_codes(index_code, fallback_values)
    for pos, (window_start, window_end) in enumerate(windows, start=1):
        frame, error = _fetch_index_window_with_fallback(
            candidate_codes,
            window_start,
            window_end,
            client=client,
            retries=retries,
            retry_max_wait=retry_max_wait,
        )
        if error is not None:
            if not skip_failed:
                raise error
            logger.error("Skipping index_weight window %s..%s after fetch failure: %s", window_start, window_end, error)
            continue
        if not frame.empty:
            frames.append(frame)
        if sleep_seconds > 0 and pos < len(windows):
            time.sleep(float(sleep_seconds))

    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if frames:
        new_data = pd.concat(frames, ignore_index=True)
        combined = pd.concat([existing, new_data], ignore_index=True) if not existing.empty else new_data
    else:
        combined = existing
    combined = normalize_index_constituents_frame(combined, default_index_code=index_code)
    combined = combined.drop_duplicates(["index_code", "con_code", "trade_date"], keep="last").sort_values(
        ["trade_date", "index_code", "con_code"]
    )
    combined.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def fetch_st_calendar(
    client: TushareHttpClient | None = None,
    fields: Iterable[str] | str | None = None,
    retries: int = 5,
    retry_max_wait: float | None = None,
) -> pd.DataFrame:
    """函数说明：拉取 fetch_st_calendar 主要逻辑。"""
    client = client or TushareHttpClient.from_config()
    requested_fields = fields or NAMECHANGE_FIELDS
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            frame = client.call("namechange", params={}, fields=requested_fields)
            return normalize_st_calendar_frame(frame)
        except (RuntimeError, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                wait_seconds = _retry_wait_seconds(attempt, retry_max_wait)
                logger.warning("Retrying namechange after error: %s", exc)
                time.sleep(wait_seconds)
    raise ValueError(f"namechange response is invalid after {retries} attempts: {last_error}") from last_error


def update_st_calendar_data(
    out_file: str | Path | None = None,
    client: TushareHttpClient | None = None,
    coverage_end_date: str | None = None,
    retries: int | None = None,
    retry_max_wait: float | None = None,
) -> Path:
    """函数说明：更新 update_st_calendar_data 主要逻辑。"""
    config = load_config()
    data_cfg = config.get("data", {})
    gov_cfg = config.get("data_governance", {})
    path = resolve_path(out_file or gov_cfg.get("st_calendar_file") or data_cfg.get("st_calendar_file") or "data/raw/st_calendar.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    retries = int(retries if retries is not None else data_cfg.get("retries", 5))
    retry_max_wait = retry_max_wait if retry_max_wait is not None else data_cfg.get("retry_max_wait", 30)
    retry_max_wait = float(retry_max_wait) if retry_max_wait is not None else None
    frame = fetch_st_calendar(client=client, retries=retries, retry_max_wait=retry_max_wait)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    _write_st_calendar_metadata(path, frame=frame, coverage_end_date=coverage_end_date or _resolve_st_calendar_coverage_end_date(config))
    return path


def _write_st_calendar_metadata(path: Path, frame: pd.DataFrame, coverage_end_date: str) -> None:
    meta_path = path.with_name(f"{path.name}.meta.json")
    event_start, event_end = _st_calendar_event_range(frame)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "tushare_namechange",
        "coverage_end_date": coverage_end_date,
        "event_start_date": event_start,
        "event_end_date": event_end,
        "rows": int(len(frame)),
    }
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_st_calendar_coverage_end_date(config: dict) -> str:
    try:
        return resolve_target_date_value(config.get("data", {}).get("end_date"), config=config)
    except (ValueError, TypeError):
        return ""


def _st_calendar_event_range(frame: pd.DataFrame) -> tuple[str, str]:
    date_parts: list[pd.Series] = []
    for column in ["st_start_date", "start_date", "begin_date", "date", "ann_date", "st_end_date", "end_date"]:
        if column not in frame.columns:
            continue
        values = pd.to_datetime(
            frame[column].astype(str).str.replace("-", "", regex=False),
            format="%Y%m%d",
            errors="coerce",
        ).dropna()
        if not values.empty:
            date_parts.append(values)
    if not date_parts:
        return "", ""
    dates = pd.concat(date_parts)
    return str(dates.min().date()), str(dates.max().date())


def _index_candidate_codes(index_code: str, fallback_index_codes: Iterable[str] | str | None) -> list[str]:
    """函数说明：处理 index_candidate_codes 的内部辅助逻辑。"""
    values: list[str] = [str(index_code).strip().upper()]
    if isinstance(fallback_index_codes, str):
        fallback_values = [item.strip() for item in fallback_index_codes.split(",")]
    else:
        fallback_values = list(fallback_index_codes or [])
    for value in fallback_values:
        code = str(value).strip().upper()
        if code and code not in values:
            values.append(code)
    return values


def _fetch_index_window_with_fallback(
    index_codes: list[str],
    window_start: str,
    window_end: str,
    client: TushareHttpClient,
    retries: int,
    retry_max_wait: float | None,
) -> tuple[pd.DataFrame, Exception | None]:
    """函数说明：拉取 fetch_index_window_with_fallback 的内部辅助逻辑。"""
    last_error: Exception | None = None
    for pos, code in enumerate(index_codes):
        try:
            frame = fetch_index_constituents(
                index_code=code,
                start_date=window_start,
                end_date=window_end,
                client=client,
                retries=retries,
                retry_max_wait=retry_max_wait,
            )
        except (RuntimeError, ValueError) as exc:
            last_error = exc
            continue
        if not frame.empty:
            if pos > 0:
                logger.info("Filled index_weight window %s..%s with fallback index_code %s.", window_start, window_end, code)
            return frame, None
    return pd.DataFrame(columns=INDEX_WEIGHT_FIELDS), last_error


def update_daily_basic_data(
    start_date: str | datetime | None = None,
    end_date: str | datetime | None = None,
    out_file: str | Path | None = None,
    trade_dates: Iterable[str | datetime] | None = None,
    client: TushareHttpClient | None = None,
    sleep_seconds: float = 0.0,
    retries: int | None = None,
    retry_max_wait: float | None = None,
    max_dates: int | None = None,
    skip_failed: bool = True,
) -> Path:
    """函数说明：更新 update_daily_basic_data 主要逻辑。"""
    config = load_config()
    data_cfg = config.get("data", {})
    start = pd.Timestamp(start_date or data_cfg.get("history_start_date") or data_cfg["start_date"]).normalize()
    end = pd.Timestamp(resolve_target_date_value(end_date or data_cfg["end_date"], config=config)).normalize()
    path = resolve_path(out_file or data_cfg.get("daily_basic_file", "data/factors/daily_basic.parquet"))
    path.parent.mkdir(parents=True, exist_ok=True)

    dates = _daily_basic_trade_dates(start, end, trade_dates)
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    existing = normalize_daily_basic_frame(existing) if not existing.empty else existing
    existing_dates = set(pd.to_datetime(existing["trade_date"]).dt.normalize()) if not existing.empty else set()
    pending_dates = [date for date in dates if pd.Timestamp(date).normalize() not in existing_dates]
    if max_dates is not None:
        pending_dates = pending_dates[: max(0, int(max_dates))]
    if not pending_dates:
        return path

    client = client or TushareHttpClient.from_config(config)
    retries = int(retries if retries is not None else data_cfg.get("retries", 5))
    retry_max_wait = retry_max_wait if retry_max_wait is not None else data_cfg.get("retry_max_wait", 30)
    retry_max_wait = float(retry_max_wait) if retry_max_wait is not None else None
    frames: list[pd.DataFrame] = []
    for pos, date in enumerate(pending_dates, start=1):
        try:
            frame = fetch_daily_basic(date, client=client, retries=retries, retry_max_wait=retry_max_wait)
        except (RuntimeError, ValueError) as exc:
            if not skip_failed:
                raise
            logger.error("Skipping daily_basic date %s after fetch failure: %s", date.date(), exc)
            continue
        if not frame.empty:
            frames.append(frame)
        if sleep_seconds > 0 and pos < len(pending_dates):
            time.sleep(float(sleep_seconds))

    if frames:
        new_data = pd.concat(frames, ignore_index=True)
        combined = pd.concat([existing, new_data], ignore_index=True) if not existing.empty else new_data
    else:
        combined = existing
    combined = normalize_daily_basic_frame(combined)
    combined = combined.drop_duplicates(["ts_code", "trade_date"], keep="last").sort_values(["trade_date", "ts_code"])
    combined.to_parquet(path, index=False)
    return path


def _fetch_daily_stock_batch(
    ts_codes: list[str],
    start_date: str | datetime,
    end_date: str | datetime,
    client: TushareHttpClient,
    retries: int = 5,
    retry_max_wait: float | None = None,
    skip_failed: bool = True,
) -> pd.DataFrame:
    """函数说明：拉取 fetch_daily_stock_batch 的内部辅助逻辑。"""
    params = {
        "ts_code": ",".join(ts_codes),
        "start_date": _format_tushare_date(start_date),
        "end_date": _format_tushare_date(end_date),
    }
    last_error: Exception | None = None
    failed_codes: list[str] = []
    for attempt in range(1, retries + 1):
        try:
            df = client.call("daily", params=params, fields=DAILY_FIELDS)
            daily = normalize_daily_frame(df)
            adj = _fetch_adj_factor_batch(ts_codes, start_date, end_date, client=client)
            adj = _complete_missing_adj_factors(daily, adj, start_date, end_date, client=client, retries=retries, skip_failed=skip_failed)
            if skip_failed:
                daily, adj, failed_codes = _drop_incomplete_adj_symbols(daily, adj)
                if failed_codes:
                    logger.error(
                        "Skipping %d symbols with incomplete adj_factor coverage in batch: %s",
                        len(failed_codes),
                        ",".join(failed_codes[:10]),
                    )
                if daily.empty:
                    result = pd.DataFrame(columns=[*DAILY_FIELDS, "adj_factor"])
                    result.attrs["failed_codes"] = failed_codes
                    return result
            result = merge_adj_factor(daily, adj)
            result.attrs["failed_codes"] = failed_codes if skip_failed else []
            return result
        except (RuntimeError, ValueError) as exc:
            last_error = exc
            if skip_failed and _is_tushare_connection_error(exc):
                break
            if attempt < retries:
                wait_seconds = _retry_wait_seconds(attempt, retry_max_wait)
                logger.warning("Retrying %d-stock daily batch after error: %s", len(ts_codes), exc)
                time.sleep(wait_seconds)
    logger.warning("Falling back to per-stock daily fetch for %d symbols after batch error: %s", len(ts_codes), last_error)
    frames = []
    fallback_retries = 1 if last_error is not None and _is_tushare_connection_error(last_error) else retries
    for code in ts_codes:
        try:
            frames.append(
                fetch_daily_stock(
                    code,
                    start_date,
                    end_date,
                    client=client,
                    retries=fallback_retries,
                    retry_max_wait=retry_max_wait,
                )
            )
        except (RuntimeError, ValueError) as exc:
            if not skip_failed:
                raise
            failed_codes.append(code)
            logger.error("Skipping %s after per-stock fallback failure: %s", code, exc)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[*DAILY_FIELDS, "adj_factor"])
    result.attrs["failed_codes"] = failed_codes
    return result


def _fetch_adj_factor_batch(
    ts_codes: list[str],
    start_date: str | datetime,
    end_date: str | datetime,
    client: TushareHttpClient,
) -> pd.DataFrame:
    """函数说明：拉取 fetch_adj_factor_batch 的内部辅助逻辑。"""
    params = {
        "ts_code": ",".join(ts_codes),
        "start_date": _format_tushare_date(start_date),
        "end_date": _format_tushare_date(end_date),
    }
    try:
        adj = client.call("adj_factor", params=params, fields=ADJ_FACTOR_FIELDS)
        if not adj.empty:
            return adj
    except (RuntimeError, ValueError) as exc:
        logger.warning("Falling back to per-stock adj_factor fetch for %d symbols: %s", len(ts_codes), exc)

    frames = []
    for code in ts_codes:
        per_stock_params = {**params, "ts_code": code}
        adj = client.call("adj_factor", params=per_stock_params, fields=ADJ_FACTOR_FIELDS)
        if not adj.empty:
            frames.append(adj)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ADJ_FACTOR_FIELDS)


def _complete_missing_adj_factors(
    daily: pd.DataFrame,
    adj: pd.DataFrame,
    start_date: str | datetime,
    end_date: str | datetime,
    client: TushareHttpClient,
    retries: int,
    skip_failed: bool,
) -> pd.DataFrame:
    """函数说明：处理 complete_missing_adj_factors 的内部辅助逻辑。"""
    if daily.empty:
        return adj
    daily_norm = normalize_daily_frame(daily)
    adj_norm = _normalize_adj_factor_frame(adj)
    expected = daily_norm[["ts_code", "trade_date"]].drop_duplicates()
    available = adj_norm[["ts_code", "trade_date", "adj_factor"]].drop_duplicates(["ts_code", "trade_date"])
    coverage = expected.merge(available, on=["ts_code", "trade_date"], how="left")
    missing_codes = sorted(coverage.loc[coverage["adj_factor"].isna(), "ts_code"].dropna().astype(str).unique())
    if not missing_codes:
        return adj_norm

    logger.warning(
        "Fetching incomplete adj_factor coverage for %d/%d symbols.",
        len(missing_codes),
        daily_norm["ts_code"].nunique(),
    )
    frames = [adj_norm] if not adj_norm.empty else []
    for code in missing_codes:
        try:
            params = {
                "ts_code": code,
                "start_date": _format_tushare_date(start_date),
                "end_date": _format_tushare_date(end_date),
            }
            piece = client.call("adj_factor", params=params, fields=ADJ_FACTOR_FIELDS)
            if not piece.empty:
                frames.append(_normalize_adj_factor_frame(piece, default_ts_code=code))
        except (RuntimeError, ValueError) as exc:
            if not skip_failed:
                raise
            logger.error("Skipping %s adj_factor completion after failure: %s", code, exc)
    if not frames:
        return pd.DataFrame(columns=ADJ_FACTOR_FIELDS)
    return pd.concat(frames, ignore_index=True).drop_duplicates(["ts_code", "trade_date"], keep="last")


def _drop_incomplete_adj_symbols(daily: pd.DataFrame, adj_factor: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """函数说明：处理 drop_incomplete_adj_symbols 的内部辅助逻辑。"""
    daily_norm = normalize_daily_frame(daily)
    adj_norm = _normalize_adj_factor_frame(adj_factor)
    if daily_norm.empty:
        return daily_norm, adj_norm, []

    expected = daily_norm[["ts_code", "trade_date"]].drop_duplicates()
    available = adj_norm[["ts_code", "trade_date", "adj_factor"]].drop_duplicates(["ts_code", "trade_date"])
    coverage = expected.merge(available, on=["ts_code", "trade_date"], how="left")
    failed_codes = sorted(coverage.loc[coverage["adj_factor"].isna(), "ts_code"].dropna().astype(str).unique())
    if not failed_codes:
        return daily_norm, adj_norm, []

    failed_set = set(failed_codes)
    daily_clean = daily_norm[~daily_norm["ts_code"].isin(failed_set)].copy()
    adj_clean = adj_norm[~adj_norm["ts_code"].isin(failed_set)].copy()
    return daily_clean, adj_clean, failed_codes


def _normalize_adj_factor_frame(adj_factor: pd.DataFrame, default_ts_code: str | None = None) -> pd.DataFrame:
    """函数说明：规范化 normalize_adj_factor_frame 的内部辅助逻辑。"""
    adj = adj_factor.rename(columns={"date": "trade_date"}).copy()
    if "ts_code" not in adj.columns and default_ts_code:
        adj["ts_code"] = default_ts_code
    required = set(ADJ_FACTOR_FIELDS)
    if not required.issubset(adj.columns):
        if adj.empty:
            return pd.DataFrame(columns=ADJ_FACTOR_FIELDS)
        missing = sorted(required - set(adj.columns))
        raise ValueError(f"Adj factor data is missing columns: {missing}")
    adj = adj[ADJ_FACTOR_FIELDS]
    adj["ts_code"] = _normalize_symbol_series(adj["ts_code"])
    adj["trade_date"] = _parse_tushare_dates(adj["trade_date"])
    adj["adj_factor"] = pd.to_numeric(adj["adj_factor"], errors="coerce")
    adj = adj.dropna(subset=["ts_code", "trade_date", "adj_factor"])
    adj = adj[adj["ts_code"] != ""]
    return adj.drop_duplicates(["ts_code", "trade_date"], keep="last").reset_index(drop=True)


def merge_adj_factor(daily: pd.DataFrame, adj_factor: pd.DataFrame, default_ts_code: str | None = None) -> pd.DataFrame:
    """函数说明：处理 merge_adj_factor 主要逻辑。"""
    daily = normalize_daily_frame(daily, default_ts_code=default_ts_code)
    if daily.empty:
        result = daily.copy()
        result["adj_factor"] = pd.Series(dtype="float64")
        return result
    adj = _normalize_adj_factor_frame(adj_factor, default_ts_code=default_ts_code)
    merged = daily.merge(adj, on=["ts_code", "trade_date"], how="left")
    if merged["adj_factor"].isna().any():
        missing = merged["adj_factor"].isna()
        missing_count = int(missing.sum())
        missing_codes = sorted(merged.loc[missing, "ts_code"].dropna().astype(str).unique())
        logger.warning(
            "Dropping %d daily rows with missing adj_factor for symbols: %s",
            missing_count,
            ",".join(missing_codes[:10]),
        )
        merged = merged.loc[~missing].copy()
    return merged


def fetch_hs300_stocks(
    date: str | datetime | None = None,
    client: TushareHttpClient | None = None,
    local_file: str | Path | None = None,
) -> list[str]:
    """函数说明：拉取 fetch_hs300_stocks 主要逻辑。"""
    config = load_config()
    data_cfg = config.get("data", {})
    local_path = resolve_path(local_file or data_cfg.get("hs300_constituents_file", "data/raw/hs300_constituents.csv"))
    if local_path.exists():
        df = pd.read_csv(local_path)
        code_col = "ts_code" if "ts_code" in df.columns else "con_code"
        if code_col not in df.columns:
            raise ValueError(f"{local_path} must contain ts_code or con_code column.")
        if date is not None and "trade_date" in df.columns:
            frame = normalize_index_constituents_frame(df)
            as_of = pd.Timestamp(date).normalize()
            frame = frame[frame["trade_date"] <= as_of]
            if not frame.empty:
                latest = frame["trade_date"].max()
                frame = frame[frame["trade_date"] == latest]
            df = frame
            code_col = "con_code"
        return sorted(_unique_normalized_symbols(df[code_col].dropna()))

    client = client or TushareHttpClient.from_config(config)
    df = fetch_index_constituents("000300.SH", start_date=date, end_date=date, client=client)
    if df.empty:
        raise RuntimeError("No HS300 constituents returned. Add data/raw/hs300_constituents.csv or check proxy params.")
    return sorted(_unique_normalized_symbols(df["con_code"].dropna()))


def fetch_stock_universe(
    universe: str | None = None,
    date: str | datetime | None = None,
    client: TushareHttpClient | None = None,
    local_file: str | Path | None = None,
    save_metadata: bool = True,
) -> list[str]:
    """函数说明：拉取 fetch_stock_universe 主要逻辑。"""
    config = load_config()
    data_cfg = config.get("data", {})
    universe = (universe or data_cfg.get("universe", "mainboard_a")).lower()
    as_of_date = date or resolve_target_date_value(data_cfg.get("end_date"), config=config)

    if universe in {"hs300", "csi300"}:
        return fetch_hs300_stocks(date=as_of_date, client=client, local_file=local_file)

    local_path = resolve_path(local_file or data_cfg.get("constituents_file", "data/raw/mainboard_a_stocks.csv"))
    if local_path.exists():
        df = pd.read_csv(local_path)
    else:
        client = client or TushareHttpClient.from_config(config)
        df = _fetch_stock_basic_history(client)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if save_metadata:
            df.to_csv(local_path, index=False, encoding="utf-8-sig")

    filtered = filter_universe_frame(
        df,
        universe=universe,
        as_of_date=as_of_date,
        exclude_st=bool(data_cfg.get("exclude_st", True)),
        st_calendar=_load_st_calendar(data_cfg.get("st_calendar_file")),
    )
    code_col = _code_column(filtered)
    return sorted(filtered[code_col].dropna().astype(str).unique().tolist())


def _fetch_stock_basic_history(client: TushareHttpClient) -> pd.DataFrame:
    """函数说明：拉取 fetch_stock_basic_history 的内部辅助逻辑。"""
    frames: list[pd.DataFrame] = []
    for status in ["L", "D"]:
        df = client.call(
            "stock_basic",
            params={"list_status": status},
            fields=STOCK_BASIC_FIELDS,
        )
        if not df.empty:
            frames.append(df)
    if not frames:
        raise RuntimeError("No stock_basic rows returned for A-share universe.")
    return pd.concat(frames, ignore_index=True).drop_duplicates("ts_code", keep="first")


def filter_universe_frame(
    df: pd.DataFrame,
    universe: str,
    as_of_date: str | datetime | None = None,
    exclude_st: bool = True,
    st_calendar: pd.DataFrame | str | Path | None = None,
) -> pd.DataFrame:
    """函数说明：过滤 filter_universe_frame 主要逻辑。"""
    if df.empty:
        return df
    result = df.copy()
    code_col = _code_column(result)
    result[code_col] = _normalize_symbol_series(result[code_col])
    result = result[result[code_col].notna() & (result[code_col] != "")].copy()

    if universe in {"mainboard_a", "a_mainboard", "mainboard"}:
        result = result[result[code_col].map(_is_mainboard_code)]
    elif universe in {"all_a", "a_share", "ashare"}:
        result = result[result[code_col].str.endswith((".SH", ".SZ"))]
    elif universe in {"hs300", "csi300"}:
        return result
    else:
        raise ValueError(f"Unsupported universe: {universe}")

    if exclude_st and st_calendar is not None and as_of_date is not None:
        result = _exclude_point_in_time_st(result, code_col, as_of_date, st_calendar)
    elif exclude_st and "name" in result.columns:
        names = result["name"].fillna("").astype(str).str.strip().str.upper()
        result = result[~names.str.contains("ST", regex=False)]

    if as_of_date is not None:
        as_of = pd.Timestamp(as_of_date)
        if "list_date" in result.columns:
            listed = pd.to_datetime(result["list_date"].astype(str).str.strip(), format="%Y%m%d", errors="coerce")
            result = result[listed.isna() | (listed <= as_of)]
        if "delist_date" in result.columns:
            delisted = pd.to_datetime(result["delist_date"].astype(str).str.strip(), format="%Y%m%d", errors="coerce")
            result = result[delisted.isna() | (delisted > as_of)]
        if "list_status" in result.columns:
            status = result["list_status"].fillna("L").astype(str).str.strip().str.upper()
            if "delist_date" in result.columns:
                delisted = pd.to_datetime(result["delist_date"].astype(str).str.strip(), format="%Y%m%d", errors="coerce")
                result = result[(status == "L") | (delisted.notna() & (delisted > as_of))]
            else:
                result = result[status == "L"]
    return result


def _load_st_calendar(path_value: str | Path | None) -> pd.DataFrame | None:
    """函数说明：加载 load_st_calendar 的内部辅助逻辑。"""
    if not path_value:
        return None
    path = resolve_path(path_value)
    if not path.exists():
        logger.warning("Configured ST calendar file does not exist: %s", path)
        return None
    return pd.read_csv(path)


def _exclude_point_in_time_st(
    df: pd.DataFrame,
    code_col: str,
    as_of_date: str | datetime,
    st_calendar: pd.DataFrame | str | Path,
) -> pd.DataFrame:
    """函数说明：处理 exclude_point_in_time_st 的内部辅助逻辑。"""
    calendar = pd.read_csv(resolve_path(st_calendar)) if isinstance(st_calendar, (str, Path)) else st_calendar.copy()
    if calendar.empty:
        return df
    st_code_col = _code_column(calendar)
    start_col = next((col for col in ["st_start_date", "start_date", "begin_date", "date"] if col in calendar.columns), None)
    end_col = next((col for col in ["st_end_date", "end_date", "remove_date"] if col in calendar.columns), None)
    if start_col is None:
        raise ValueError("ST calendar must contain one of: st_start_date, start_date, begin_date, date.")

    as_of = pd.Timestamp(as_of_date)
    calendar[st_code_col] = _normalize_symbol_series(calendar[st_code_col])
    calendar = calendar[calendar[st_code_col].notna() & (calendar[st_code_col] != "")].copy()
    starts = _parse_calendar_dates(calendar[start_col])
    ends = _parse_calendar_dates(calendar[end_col]) if end_col is not None else pd.Series(pd.NaT, index=calendar.index)
    active = (starts <= as_of) & (ends.isna() | (ends >= as_of))
    st_codes = set(calendar.loc[active, st_code_col].dropna())
    codes = _normalize_symbol_series(df[code_col])
    return df[~codes.isin(st_codes)]


def _parse_calendar_dates(series: pd.Series) -> pd.Series:
    """函数说明：解析 parse_calendar_dates 的内部辅助逻辑。"""
    text = series.astype("string").str.strip().str.replace("-", "", regex=False).str.replace("/", "", regex=False)
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce")


def _code_column(df: pd.DataFrame) -> str:
    """函数说明：处理 code_column 的内部辅助逻辑。"""
    for col in ["ts_code", "con_code", "instrument", "code"]:
        if col in df.columns:
            return col
    raise ValueError("Universe file must contain one of: ts_code, con_code, instrument, code.")


def _is_mainboard_code(code: str) -> bool:
    """函数说明：判断 is_mainboard_code 是否成立。"""
    symbol = code.split(".", 1)[0]
    exchange_ok = code.endswith((".SH", ".SZ"))
    return exchange_ok and symbol.startswith(MAINBOARD_PREFIXES)


def update_daily_data(
    stock_codes: Iterable[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    raw_dir: str | Path | None = None,
    force_full: bool = False,
) -> dict[str, Path]:
    """函数说明：更新 update_daily_data 主要逻辑。"""
    config = load_config()
    data_cfg = config.get("data", {})
    duplicate_keep = str(data_cfg.get("duplicate_keep", "first"))
    target_dir = resolve_path(raw_dir or data_cfg.get("raw_dir", "data/raw"))
    target_dir.mkdir(parents=True, exist_ok=True)

    start = start_date or data_cfg.get("history_start_date") or data_cfg["start_date"]
    end = resolve_target_date_value(end_date or data_cfg["end_date"], config=config)
    codes = list(stock_codes) if stock_codes is not None else fetch_stock_universe(date=end)
    client = TushareHttpClient.from_config(config)
    batch_size = int(data_cfg.get("daily_batch_size", 100))
    window_days = int(data_cfg.get("daily_window_days", 500))
    max_new_symbols = data_cfg.get("max_new_symbols_per_run")
    max_new_symbols = int(max_new_symbols) if max_new_symbols is not None else None
    retries = int(data_cfg.get("retries", 5))
    retry_max_wait = data_cfg.get("retry_max_wait", 30)
    retry_max_wait = float(retry_max_wait) if retry_max_wait is not None else None

    written: dict[str, Path] = {}
    failed: dict[str, str] = {}
    pending: dict[str, tuple[Path, str, bool]] = {}
    for code in codes:
        path = target_dir / f"{code}.csv"
        needs_adj_backfill = _needs_adj_factor_backfill(path)
        actual_start = start if force_full or needs_adj_backfill else _incremental_start(path, start)
        if pd.Timestamp(actual_start) > pd.Timestamp(end):
            written[code] = path
            continue
        pending[code] = (path, actual_start, needs_adj_backfill)
    pending = _limit_new_symbols_per_run(pending, max_new_symbols=max_new_symbols)
    if pending:
        logger.info("Updating %d pending stock files.", len(pending))

    for actual_start, grouped_codes in _group_codes_by_start(pending).items():
        for batch_codes in _batched(grouped_codes, batch_size):
            batch_df = fetch_daily_stocks(
                batch_codes,
                actual_start,
                end,
                client=client,
                retries=retries,
                retry_max_wait=retry_max_wait,
                batch_size=batch_size,
                window_days=window_days,
                skip_failed=True,
            )
            batch_failed = set(batch_df.attrs.get("failed_codes", []))
            for code in batch_codes:
                path, _actual_start, needs_adj_backfill = pending[code]
                if code in batch_failed:
                    failed[code] = "empty_or_failed_fetch"
                    continue
                if batch_df.empty:
                    new_df = pd.DataFrame(columns=[*DAILY_FIELDS, "adj_factor"])
                else:
                    new_df = batch_df[batch_df["ts_code"] == code].copy()
                if new_df.empty and path.exists():
                    written[code] = path
                    continue
                if new_df.empty:
                    failed[code] = "empty_or_failed_fetch"
                    continue
                if path.exists() and not needs_adj_backfill and not force_full:
                    old_df = pd.read_csv(path, parse_dates=["trade_date"])
                    new_df = pd.concat([old_df, new_df], ignore_index=True)
                new_df = normalize_daily_frame(new_df, default_ts_code=code)
                if "adj_factor" in new_df.columns:
                    new_df = new_df.dropna(subset=["adj_factor"])
                new_df = new_df.drop_duplicates(["ts_code", "trade_date"], keep=duplicate_keep)
                new_df.to_csv(path, index=False, encoding="utf-8-sig")
                written[code] = path
            logger.info("Updated %d/%d pending stock files.", len(written), len(pending))
    if failed:
        failed_path = target_dir / "failed_fetches.csv"
        pd.DataFrame({"ts_code": list(failed), "reason": list(failed.values())}).to_csv(
            failed_path, index=False, encoding="utf-8-sig"
        )
        logger.warning("Skipped %d symbols during data update. See %s", len(failed), failed_path)
    return written


def update_daily_data_resumable(
    stock_codes: Iterable[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    raw_dir: str | Path | None = None,
    chunk_size: int | None = None,
    sleep_seconds: float | None = None,
    progress_file: str | Path | None = None,
    max_chunks: int | None = None,
    include_existing: bool = False,
    force_full: bool = False,
) -> ResumableUpdateResult:
    """函数说明：更新 update_daily_data_resumable 主要逻辑。"""
    config = load_config()
    data_cfg = config.get("data", {})
    target_dir = resolve_path(raw_dir or data_cfg.get("raw_dir", "data/raw"))
    target_dir.mkdir(parents=True, exist_ok=True)

    start = start_date or data_cfg.get("history_start_date") or data_cfg["start_date"]
    end = resolve_target_date_value(end_date or data_cfg["end_date"], config=config)
    codes = list(stock_codes) if stock_codes is not None else fetch_stock_universe(date=end)
    codes = [str(code).upper() for code in dict.fromkeys(codes)]
    chunk_size = max(1, int(chunk_size or data_cfg.get("update_chunk_size", 20)))
    sleep_seconds = float(sleep_seconds if sleep_seconds is not None else data_cfg.get("update_sleep_seconds", 0))
    progress_path = resolve_path(progress_file or data_cfg.get("update_progress_file", "outputs/data_update_progress.json"))

    target_code_set = set(codes)
    progress_context = _update_progress_context(
        codes=codes,
        start_date=start,
        end_date=end,
        raw_dir=target_dir,
        include_existing=include_existing,
        force_full=force_full,
    )
    previous_progress = _read_update_progress(progress_path)
    initial_latest_for_reuse: set[str] | None = None
    if _can_reuse_complete_update_progress(previous_progress, progress_context):
        initial_latest_for_reuse = _fresh_stock_codes(target_dir, target_code_set, end)
        previous_confirmed = _confirmed_no_new_data_from_progress(previous_progress, target_code_set, initial_latest_for_reuse)
        if len(initial_latest_for_reuse | previous_confirmed) < len(codes):
            previous_progress = None
        else:
            refreshed_progress = dict(previous_progress or {})
            refreshed_progress.update(
                {
                    **progress_context,
                    "status": "complete",
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "chunk_size": chunk_size,
                    "sleep_seconds": sleep_seconds,
                    "last_chunk": [],
                    "current_symbol": "",
                    "last_error": "",
                    "latest_symbols": len(initial_latest_for_reuse),
                    "stale_or_missing_symbols": len(codes) - len(initial_latest_for_reuse),
                    "latest_coverage": _coverage_ratio(len(initial_latest_for_reuse), len(codes)),
                    "fresh_or_confirmed_symbols": len(initial_latest_for_reuse | previous_confirmed),
                    "remaining_unconfirmed_symbols": 0,
                }
            )
            _write_update_progress(progress_path, refreshed_progress)
            logger.info("Resumable update already complete for %s; reused %s.", end, progress_path)
            return _resumable_result_from_progress({}, progress_path, refreshed_progress)
    if previous_progress and not _progress_context_matches(previous_progress, progress_context):
        previous_progress = None

    initial_existing = _existing_stock_codes(target_dir) & target_code_set
    initial_latest = initial_latest_for_reuse if initial_latest_for_reuse is not None else _fresh_stock_codes(target_dir, target_code_set, end)
    initial_history_complete = _history_complete_stock_codes(target_dir, target_code_set, start, end) if force_full else set()
    latest_codes = set(initial_latest)
    confirmed_no_new_data: set[str] = set()
    if not include_existing and not force_full and previous_progress and str(previous_progress.get("target_end_date", "")) == end:
        previous_confirmed = previous_progress.get("confirmed_no_new_data", [])
        if not isinstance(previous_confirmed, list):
            previous_confirmed = []
        confirmed_no_new_data = {
            str(code).upper()
            for code in previous_confirmed
            if str(code).upper() in target_code_set and str(code).upper() not in initial_latest
        }
    if force_full:
        pending_codes = [
            code for code in codes if code not in initial_history_complete or _needs_adj_factor_backfill(target_dir / f"{code}.csv")
        ]
    elif include_existing:
        pending_codes = codes
    else:
        pending_codes = [
            code
            for code in codes
            if (
                (code not in initial_latest or _needs_adj_factor_backfill(target_dir / f"{code}.csv"))
                and code not in confirmed_no_new_data
            )
        ]
    written: dict[str, Path] = {}
    failed: dict[str, str] = {}
    processed_codes: set[str] = set()
    last_error = ""
    started_at = datetime.now().isoformat(timespec="seconds")
    logger.info(
        "Resumable update: %d target symbols, %d existing raw files, %d latest raw files, %d symbols pending.",
        len(codes),
        len(initial_existing),
        len(initial_latest),
        len(pending_codes),
    )
    _write_update_progress(
        progress_path,
        {
            **progress_context,
            "status": "running",
            "started_at": started_at,
            "updated_at": started_at,
            "target_end_date": end,
            "target_symbols": len(codes),
            "target_codes_count": len(codes),
            "initial_existing": len(initial_existing),
            "initial_latest_symbols": len(initial_latest),
            "initial_history_complete_symbols": len(initial_history_complete),
            "pending_symbols": len(pending_codes),
            "chunk_size": chunk_size,
            "sleep_seconds": sleep_seconds,
            "force_full": force_full,
            "confirmed_no_new_data_symbols": len(confirmed_no_new_data),
            "confirmed_no_new_data": sorted(confirmed_no_new_data),
            "completed_symbols": 0,
            "failed_symbols": 0,
            "remaining_symbols": len(pending_codes),
            "latest_symbols": len(initial_latest),
            "stale_or_missing_symbols": len(codes) - len(initial_latest),
            "latest_coverage": _coverage_ratio(len(initial_latest), len(codes)),
            "fresh_or_confirmed_symbols": len(initial_latest | confirmed_no_new_data),
            "remaining_unconfirmed_symbols": max(len(codes) - len(initial_latest | confirmed_no_new_data), 0),
            "last_chunk": [],
            "current_symbol": "",
            "last_error": "",
        },
    )

    chunks_run = 0
    for batch_codes in _batched(pending_codes, chunk_size):
        if max_chunks is not None and chunks_run >= max_chunks:
            break
        chunks_run += 1
        chunk_error = ""
        logger.info(
            "Updating missing-symbol chunk %d: %s",
            chunks_run,
            ",".join(batch_codes[:5]) + ("..." if len(batch_codes) > 5 else ""),
        )
        chunk_starts = {code: start for code in batch_codes}
        for code_start, grouped_codes in _group_chunk_codes_by_start(chunk_starts).items():
            batch_written: dict[str, Path] = {}
            try:
                batch_written = update_daily_data(
                    stock_codes=grouped_codes,
                    start_date=code_start,
                    end_date=end,
                    raw_dir=target_dir,
                    force_full=force_full,
                )
                written.update(batch_written)
            except Exception as exc:
                chunk_error = str(exc)
                last_error = chunk_error
                for code in grouped_codes:
                    failed[code] = chunk_error
                logger.error("Code group failed in chunk %d (%s): %s", chunks_run, ",".join(grouped_codes[:5]), exc)
            finally:
                processed_codes.update(grouped_codes)

            grouped_set = set(grouped_codes) & target_code_set
            latest_group = _fresh_stock_codes(target_dir, grouped_set, end)
            latest_codes.difference_update(grouped_set)
            latest_codes.update(latest_group)
            confirmed_no_new_data -= latest_codes
            for code in grouped_codes:
                if code not in latest_codes and code not in failed:
                    if code in batch_written and (target_dir / f"{code}.csv").exists():
                        confirmed_no_new_data.add(code)
                        logger.info("%s has no new rows through %s; marking as confirmed no-new-data.", code, end)
                    else:
                        reason = "not_latest" if (target_dir / f"{code}.csv").exists() else "not_written"
                        chunk_error = f"{code}: {reason}"
                        last_error = chunk_error
                        failed[code] = reason

            if include_existing:
                completed = len(processed_codes - set(failed))
                remaining = max(len(pending_codes) - len(processed_codes), 0)
            else:
                completed = len((latest_codes | confirmed_no_new_data) & set(pending_codes))
                remaining = max(len(pending_codes) - completed, 0)
            _write_update_progress(
                progress_path,
                {
                    **progress_context,
                    "status": "running",
                    "started_at": started_at,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "target_end_date": end,
                    "target_symbols": len(codes),
                    "target_codes_count": len(codes),
                    "initial_existing": len(initial_existing),
                    "initial_latest_symbols": len(initial_latest),
                    "pending_symbols": len(pending_codes),
                    "chunk_size": chunk_size,
                    "sleep_seconds": sleep_seconds,
                    "force_full": force_full,
                    "confirmed_no_new_data_symbols": len(confirmed_no_new_data),
                    "confirmed_no_new_data": sorted(confirmed_no_new_data),
                    "completed_symbols": completed,
                    "failed_symbols": len(failed),
                    "remaining_symbols": remaining,
                    "latest_symbols": len(latest_codes),
                    "stale_or_missing_symbols": len(codes) - len(latest_codes),
                    "latest_coverage": _coverage_ratio(len(latest_codes), len(codes)),
                    "fresh_or_confirmed_symbols": len(latest_codes | confirmed_no_new_data),
                    "remaining_unconfirmed_symbols": max(len(codes) - len(latest_codes | confirmed_no_new_data), 0),
                    "last_chunk": batch_codes,
                    "current_symbol": grouped_codes[-1] if grouped_codes else "",
                    "current_start_date": code_start,
                    "last_error": chunk_error,
                },
            )
        if remaining == 0:
            break
        if sleep_seconds > 0 and (max_chunks is None or chunks_run < max_chunks):
            logger.info("Sleeping %.1f seconds before next chunk.", sleep_seconds)
            time.sleep(sleep_seconds)

    latest_final = latest_codes
    confirmed_no_new_data -= latest_final
    if include_existing:
        completed_final = len(processed_codes - set(failed))
        remaining_final = max(len(pending_codes) - len(processed_codes), 0)
    else:
        completed_final = len((latest_final | confirmed_no_new_data) & set(pending_codes))
        remaining_final = max(len(pending_codes) - completed_final, 0)
    status = "error" if failed else ("complete" if remaining_final == 0 else "partial")
    if failed and not last_error:
        first_code, first_reason = next(iter(failed.items()))
        last_error = f"{first_code}: {first_reason}"
    _write_update_progress(
        progress_path,
        {
            **progress_context,
            "status": status,
            "started_at": started_at,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "target_end_date": end,
            "target_symbols": len(codes),
            "target_codes_count": len(codes),
            "initial_existing": len(initial_existing),
            "initial_latest_symbols": len(initial_latest),
            "pending_symbols": len(pending_codes),
            "chunk_size": chunk_size,
            "sleep_seconds": sleep_seconds,
            "force_full": force_full,
            "confirmed_no_new_data_symbols": len(confirmed_no_new_data),
            "confirmed_no_new_data": sorted(confirmed_no_new_data),
            "completed_symbols": completed_final,
            "failed_symbols": len(failed),
            "remaining_symbols": remaining_final,
            "latest_symbols": len(latest_final),
            "stale_or_missing_symbols": len(codes) - len(latest_final),
            "latest_coverage": _coverage_ratio(len(latest_final), len(codes)),
            "fresh_or_confirmed_symbols": len(latest_final | confirmed_no_new_data),
            "remaining_unconfirmed_symbols": max(len(codes) - len(latest_final | confirmed_no_new_data), 0),
            "last_chunk": [],
            "current_symbol": "",
            "last_error": last_error,
        },
    )
    return _resumable_result_from_progress(
        written,
        progress_path,
        {
            "status": status,
            "failed_symbols": len(failed),
            "remaining_symbols": remaining_final,
            "latest_symbols": len(latest_final),
            "fresh_or_confirmed_symbols": len(latest_final | confirmed_no_new_data),
            "last_error": last_error,
        },
    )


def _group_codes_by_start(pending: dict[str, tuple[Path, str, bool]]) -> dict[str, list[str]]:
    """函数说明：处理 group_codes_by_start 的内部辅助逻辑。"""
    grouped: dict[str, list[str]] = {}
    for code, (_path, actual_start, _needs_adj_backfill) in pending.items():
        grouped.setdefault(actual_start, []).append(code)
    return grouped


def _group_chunk_codes_by_start(chunk_starts: dict[str, str]) -> dict[str, list[str]]:
    """函数说明：处理 group_chunk_codes_by_start 的内部辅助逻辑。"""
    grouped: dict[str, list[str]] = {}
    for code, actual_start in chunk_starts.items():
        grouped.setdefault(actual_start, []).append(code)
    return grouped


def _limit_new_symbols_per_run(
    pending: dict[str, tuple[Path, str, bool]],
    max_new_symbols: int | None,
) -> dict[str, tuple[Path, str, bool]]:
    """函数说明：处理 limit_new_symbols_per_run 的内部辅助逻辑。"""
    if max_new_symbols is None or max_new_symbols < 0:
        return pending
    existing = {code: item for code, item in pending.items() if item[0].exists()}
    new_items = {code: item for code, item in pending.items() if not item[0].exists()}
    if len(new_items) <= max_new_symbols:
        return pending
    selected_new = dict(list(new_items.items())[:max_new_symbols])
    skipped = len(new_items) - len(selected_new)
    logger.info("Deferring %d new stock files to later runs.", skipped)
    return {**existing, **selected_new}


def _batched(values: list[str], batch_size: int) -> Iterable[list[str]]:
    """函数说明：处理 batched 的内部辅助逻辑。"""
    for index in range(0, len(values), batch_size):
        yield values[index : index + batch_size]


def _existing_stock_codes(raw_dir: Path) -> set[str]:
    """函数说明：处理 existing_stock_codes 的内部辅助逻辑。"""
    return {
        path.stem.upper()
        for path in raw_dir.glob("*.csv")
        if path.name.upper().endswith((".SZ.CSV", ".SH.CSV"))
    }


def _fresh_stock_codes(raw_dir: Path, codes: set[str], end_date: str) -> set[str]:
    """函数说明：处理 fresh_stock_codes 的内部辅助逻辑。"""
    target = pd.Timestamp(end_date).normalize()
    code_list = list(codes)

    def latest_for_code(code: str) -> tuple[str, pd.Timestamp | None]:
        """函数说明：处理 latest_for_code 主要逻辑。"""
        return code, _raw_latest_date(raw_dir / f"{code}.csv")

    fresh: set[str] = set()
    if len(code_list) >= 100:
        max_workers = min(32, len(code_list))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for code, latest in executor.map(latest_for_code, code_list):
                if latest is not None and latest >= target:
                    fresh.add(code)
    else:
        for code, latest in map(latest_for_code, code_list):
            if latest is not None and latest >= target:
                fresh.add(code)
    return fresh


def _history_complete_stock_codes(raw_dir: Path, codes: set[str], start_date: str, end_date: str) -> set[str]:
    """函数说明：处理 history_complete_stock_codes 的内部辅助逻辑。"""
    start_target = pd.Timestamp(start_date).normalize()
    start_tolerance = start_target + pd.Timedelta(days=31)
    end_target = pd.Timestamp(end_date).normalize()

    def coverage_for_code(code: str) -> tuple[str, pd.Timestamp | None, pd.Timestamp | None]:
        """函数说明：处理 coverage_for_code 主要逻辑。"""
        path = raw_dir / f"{code}.csv"
        return code, _raw_earliest_date(path), _raw_latest_date(path)

    complete: set[str] = set()
    code_list = list(codes)
    iterator = None
    if len(code_list) >= 100:
        max_workers = min(32, len(code_list))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            iterator = executor.map(coverage_for_code, code_list)
            for code, earliest, latest in iterator:
                if earliest is not None and latest is not None and earliest <= start_tolerance and latest >= end_target:
                    complete.add(code)
    else:
        for code, earliest, latest in map(coverage_for_code, code_list):
            if earliest is not None and latest is not None and earliest <= start_tolerance and latest >= end_target:
                complete.add(code)
    return complete


def _raw_earliest_date(path: Path) -> pd.Timestamp | None:
    """函数说明：处理 raw_earliest_date 的内部辅助逻辑。"""
    if not path.exists():
        return None
    try:
        rows = pd.read_csv(path, usecols=["trade_date"], parse_dates=["trade_date"], nrows=1)
    except (OSError, ValueError):
        return None
    if rows.empty:
        return None
    earliest = pd.to_datetime(rows["trade_date"], errors="coerce").min()
    if pd.isna(earliest):
        return None
    return pd.Timestamp(earliest).normalize()


def _raw_latest_date(path: Path) -> pd.Timestamp | None:
    """函数说明：处理 raw_latest_date 的内部辅助逻辑。"""
    if not path.exists():
        return None
    latest = _raw_latest_date_from_tail(path)
    if latest is not None:
        return latest
    try:
        dates = pd.read_csv(path, usecols=["trade_date"], parse_dates=["trade_date"])
    except (OSError, ValueError):
        return None
    if dates.empty:
        return None
    latest = pd.to_datetime(dates["trade_date"], errors="coerce").max()
    if pd.isna(latest):
        return None
    return pd.Timestamp(latest).normalize()


def _raw_latest_date_from_tail(path: Path, tail_bytes: int = 16384) -> pd.Timestamp | None:
    """函数说明：处理 raw_latest_date_from_tail 的内部辅助逻辑。"""
    try:
        with path.open("rb") as handle:
            header_bytes = handle.readline()
            header = header_bytes.decode("utf-8-sig", errors="ignore").strip()
            if "trade_date" not in header:
                return None
            handle.seek(0, 2)
            file_size = handle.tell()
            start = max(len(header_bytes), file_size - tail_bytes)
            handle.seek(start)
            chunk = handle.read().decode("utf-8-sig", errors="ignore")
    except OSError:
        return None

    lines = chunk.splitlines()
    if start > len(header_bytes) and lines:
        lines = lines[1:]
    if not lines:
        return None

    latest: pd.Timestamp | None = None
    reader = csv.DictReader(io.StringIO("\n".join([header, *lines[-50:]])))
    for row in reader:
        value = row.get("trade_date")
        if not value:
            continue
        current = _parse_trade_date_value(value)
        if current is None:
            continue
        if latest is None or current > latest:
            latest = current
    return latest


def _parse_trade_date_value(value: object) -> pd.Timestamp | None:
    """函数说明：解析 parse_trade_date_value 的内部辅助逻辑。"""
    text = str(value).strip()
    compact = text.replace("-", "")
    if len(compact) >= 8 and compact[:8].isdigit():
        digits = compact[:8]
        try:
            return pd.Timestamp(datetime(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))).normalize()
        except ValueError:
            return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize()


def _write_update_progress(path: Path, payload: dict[str, object]) -> None:
    """函数说明：写入 write_update_progress 的内部辅助逻辑。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_update_progress(path: Path) -> dict[str, object] | None:
    """函数说明：读取 read_update_progress 的内部辅助逻辑。"""
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _update_progress_context(
    *,
    codes: list[str],
    start_date: str,
    end_date: str,
    raw_dir: Path,
    include_existing: bool,
    force_full: bool,
) -> dict[str, object]:
    """函数说明：更新 update_progress_context 的内部辅助逻辑。"""
    return {
        "target_codes_hash": _target_codes_hash(codes),
        "target_codes_count": len(codes),
        "start_date": str(pd.Timestamp(start_date).date()),
        "end_date": str(pd.Timestamp(end_date).date()),
        "raw_dir": str(raw_dir.resolve()),
        "include_existing": bool(include_existing),
        "force_full": bool(force_full),
    }


def _target_codes_hash(codes: Iterable[str]) -> str:
    """函数说明：处理 target_codes_hash 的内部辅助逻辑。"""
    normalized = sorted(str(code).strip().upper() for code in codes if str(code).strip())
    payload = "\n".join(normalized).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _confirmed_no_new_data_from_progress(
    progress: dict[str, object] | None,
    target_code_set: set[str],
    latest_codes: set[str],
) -> set[str]:
    """函数说明：处理 confirmed_no_new_data_from_progress 的内部辅助逻辑。"""
    if not progress:
        return set()
    previous_confirmed = progress.get("confirmed_no_new_data", [])
    if not isinstance(previous_confirmed, list):
        return set()
    return {
        str(code).upper()
        for code in previous_confirmed
        if str(code).upper() in target_code_set and str(code).upper() not in latest_codes
    }


def _can_reuse_complete_update_progress(
    progress: dict[str, object] | None,
    expected_context: dict[str, object],
) -> bool:
    """函数说明：处理 can_reuse_complete_update_progress 的内部辅助逻辑。"""
    if not progress:
        return False
    if bool(expected_context.get("include_existing")) or bool(expected_context.get("force_full")):
        return False
    if str(progress.get("status", "")) != "complete":
        return False
    if not _progress_context_matches(progress, expected_context):
        return False
    target_symbols = _safe_int(expected_context.get("target_codes_count"), -1)
    if _safe_int(progress.get("target_symbols"), -1) != target_symbols:
        return False
    return _safe_int(progress.get("remaining_unconfirmed_symbols", progress.get("remaining_symbols", 1)), 1) == 0


def _progress_context_matches(progress: dict[str, object], expected_context: dict[str, object]) -> bool:
    """函数说明：处理 progress_context_matches 的内部辅助逻辑。"""
    for key in ["target_codes_hash", "target_codes_count", "start_date", "end_date", "raw_dir", "include_existing", "force_full"]:
        if progress.get(key) != expected_context.get(key):
            return False
    return True


def _safe_int(value: object, default: int) -> int:
    """函数说明：处理 safe_int 的内部辅助逻辑。"""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _daily_basic_trade_dates(
    start: pd.Timestamp,
    end: pd.Timestamp,
    trade_dates: Iterable[str | datetime] | None,
) -> list[pd.Timestamp]:
    """函数说明：处理 daily_basic_trade_dates 的内部辅助逻辑。"""
    if trade_dates is not None:
        dates = pd.DatetimeIndex(pd.to_datetime(list(trade_dates), errors="coerce")).dropna().normalize().unique().sort_values()
        return [pd.Timestamp(date) for date in dates if start <= pd.Timestamp(date) <= end]

    config = load_config()
    price_file = resolve_path(config.get("ic", {}).get("price_file", "data/prices/ohlcv_adjusted.parquet"))
    if price_file.exists():
        try:
            prices = pd.read_parquet(price_file, columns=[])
        except (ValueError, TypeError):
            prices = pd.read_parquet(price_file)
        dates = pd.DatetimeIndex(pd.to_datetime(prices.index, errors="coerce")).dropna().normalize().unique().sort_values()
        selected = [pd.Timestamp(date) for date in dates if start <= pd.Timestamp(date) <= end]
        if selected:
            return selected

    business_dates = pd.bdate_range(start, end)
    return [pd.Timestamp(date).normalize() for date in business_dates]


def _incremental_start(path: Path, configured_start: str) -> str:
    """函数说明：处理 incremental_start 的内部辅助逻辑。"""
    latest = _raw_latest_date(path)
    if latest is None:
        return configured_start
    return (latest + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def _needs_adj_factor_backfill(path: Path) -> bool:
    """函数说明：处理 needs_adj_factor_backfill 的内部辅助逻辑。"""
    if not path.exists():
        return False
    columns = pd.read_csv(path, nrows=0).columns
    return "adj_factor" not in columns
