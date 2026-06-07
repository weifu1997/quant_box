from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta
import logging
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from src.config_loader import load_config, resolve_path

logger = logging.getLogger(__name__)

AUTO_DATE_VALUES = {"auto", "latest", "latest_trade_date", "latest_trading_day"}
DEFAULT_CUTOFF_TIME = "20:00"
DEFAULT_TIMEZONE = "Asia/Shanghai"


@dataclass(frozen=True)
class TargetDateResolution:
    requested: str
    target_date: str
    latest_trade_date: str
    previous_trade_date: str
    cutoff_time: str
    cutoff_at: str
    now: str
    calendar_source: str
    reason: str
    calendar_warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def resolve_target_date_value(
    value: str | datetime | pd.Timestamp | None = None,
    config: dict | None = None,
    now: datetime | None = None,
    calendar: Iterable[str | datetime | pd.Timestamp] | pd.DatetimeIndex | None = None,
    price_df: pd.DataFrame | None = None,
    price_file: str | Path | None = None,
) -> str:
    return resolve_target_date(
        value=value,
        config=config,
        now=now,
        calendar=calendar,
        price_df=price_df,
        price_file=price_file,
    ).target_date


def resolve_target_date(
    value: str | datetime | pd.Timestamp | None = None,
    config: dict | None = None,
    now: datetime | None = None,
    calendar: Iterable[str | datetime | pd.Timestamp] | pd.DatetimeIndex | None = None,
    price_df: pd.DataFrame | None = None,
    price_file: str | Path | None = None,
) -> TargetDateResolution:
    cfg = config or load_config()
    data_cfg = cfg.get("data", {})
    requested = str(value if value is not None else data_cfg.get("end_date", "auto"))
    now_dt = _normalize_now(now, data_cfg.get("timezone", DEFAULT_TIMEZONE))
    cutoff = _parse_cutoff_time(str(data_cfg.get("target_date_cutoff_time", DEFAULT_CUTOFF_TIME)))

    if not _is_auto_date_value(requested):
        target = pd.Timestamp(requested).normalize()
        return TargetDateResolution(
            requested=requested,
            target_date=str(target.date()),
            latest_trade_date=str(target.date()),
            previous_trade_date="",
            cutoff_time=_format_cutoff(cutoff),
            cutoff_at="",
            now=now_dt.isoformat(timespec="seconds"),
            calendar_source="fixed",
            reason="fixed_end_date",
            calendar_warnings=[],
        )

    trade_calendar, source, calendar_warnings = _target_trade_calendar(
        cfg,
        now_dt=now_dt,
        calendar=calendar,
        price_df=price_df,
        price_file=price_file,
    )
    if trade_calendar.empty:
        raise ValueError("Unable to resolve auto target date because no trade calendar dates are available.")

    today = pd.Timestamp(now_dt.date())
    available = pd.DatetimeIndex(trade_calendar[trade_calendar <= today]).unique().sort_values()
    if available.empty:
        raise ValueError("Unable to resolve auto target date because the trade calendar has no dates before today.")

    latest = pd.Timestamp(available.max()).normalize()
    previous = pd.Timestamp(available[-2]).normalize() if len(available) >= 2 else latest
    cutoff_at = datetime.combine(latest.date(), cutoff, tzinfo=now_dt.tzinfo)
    if now_dt < cutoff_at:
        target = previous
        reason = "before_latest_trade_date_cutoff"
    else:
        target = latest
        reason = "after_latest_trade_date_cutoff"

    return TargetDateResolution(
        requested=requested,
        target_date=str(target.date()),
        latest_trade_date=str(latest.date()),
        previous_trade_date=str(previous.date()) if previous != latest else "",
        cutoff_time=_format_cutoff(cutoff),
        cutoff_at=cutoff_at.isoformat(timespec="seconds"),
        now=now_dt.isoformat(timespec="seconds"),
        calendar_source=source,
        reason=reason,
        calendar_warnings=calendar_warnings,
    )


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


def next_business_day(
    date: str | pd.Timestamp,
    config: dict | None = None,
    calendar: Iterable[str | datetime | pd.Timestamp] | pd.DatetimeIndex | None = None,
    price_df: pd.DataFrame | None = None,
    price_file: str | Path | None = None,
    strict: bool = False,
) -> pd.Timestamp:
    signal_ts = _required_date(date)
    explicit = _normalize_calendar(calendar)
    next_date = _next_from_calendar(explicit, signal_ts)
    if next_date is not None:
        return next_date

    cfg = config or load_config()
    file_calendar, _source = _configured_file_calendar(cfg)
    next_date = _next_from_calendar(file_calendar, signal_ts)
    if next_date is not None:
        return next_date

    configured_price_file = price_file or cfg.get("ic", {}).get("price_file", "data/prices/ohlcv_adjusted.parquet")
    prices = price_calendar(price_df=price_df, price_file=configured_price_file)
    next_date = _next_from_calendar(prices, signal_ts)
    if next_date is not None:
        return next_date

    next_trade = _a_trade_calendar_next_trade_date(date)
    if next_trade is not None:
        return next_trade

    if strict:
        raise ValueError(f"Unable to resolve next A-share trading day after {signal_ts.date()}.")
    logger.warning(
        "Falling back to weekday calendar for next business day after %s because no A-share trade calendar is available.",
        signal_ts.date(),
    )
    current = signal_ts + pd.Timedelta(days=1)
    while current.weekday() >= 5:
        current += pd.Timedelta(days=1)
    return current


def _is_auto_date_value(value: str) -> bool:
    return value.strip().lower() in AUTO_DATE_VALUES


def _normalize_now(now: datetime | None, timezone_name: str) -> datetime:
    tz = ZoneInfo(timezone_name or DEFAULT_TIMEZONE)
    if now is None:
        return datetime.now(tz)
    if now.tzinfo is None:
        return now.replace(tzinfo=tz)
    return now.astimezone(tz)


def _parse_cutoff_time(value: str) -> time:
    text = value.strip()
    for fmt in ["%H:%M:%S", "%H:%M"]:
        try:
            return datetime.strptime(text, fmt).time()
        except ValueError:
            pass
    raise ValueError(f"Invalid data.target_date_cutoff_time: {value!r}. Expected HH:MM.")


def _format_cutoff(value: time) -> str:
    return value.strftime("%H:%M")


def _target_trade_calendar(
    config: dict,
    now_dt: datetime,
    calendar: Iterable[str | datetime | pd.Timestamp] | pd.DatetimeIndex | None,
    price_df: pd.DataFrame | None,
    price_file: str | Path | None,
) -> tuple[pd.DatetimeIndex, str, list[str]]:
    warnings: list[str] = []
    explicit = _normalize_calendar(calendar)
    if not explicit.empty:
        return explicit, "explicit", warnings
    if calendar is not None:
        warnings.append("explicit_calendar_unavailable")

    remote = _tushare_trade_calendar(config, now_dt)
    if not remote.empty:
        return remote, "tushare_trade_cal", warnings
    warnings.append("tushare_trade_cal_unavailable")

    library_calendar = _a_trade_calendar(config, now_dt)
    if not library_calendar.empty:
        return library_calendar, "a_trade_calendar", warnings
    warnings.append("a_trade_calendar_unavailable")

    file_calendar, source = _configured_file_calendar(config)
    if not file_calendar.empty:
        return file_calendar, source, warnings
    warnings.append("calendar_file_unavailable")

    configured_price_file = price_file or config.get("ic", {}).get("price_file", "data/prices/ohlcv_adjusted.parquet")
    prices = price_calendar(price_df=price_df, price_file=configured_price_file)
    if not prices.empty:
        return prices, "price_calendar", warnings

    warnings.append("price_calendar_unavailable")
    return pd.DatetimeIndex([]), "", warnings


def _normalize_calendar(calendar: Iterable[str | datetime | pd.Timestamp] | pd.DatetimeIndex | None) -> pd.DatetimeIndex:
    if calendar is None:
        return pd.DatetimeIndex([])
    raw = pd.Series(list(calendar), dtype="object")
    text = raw.astype(str).str.strip()
    compact = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    fallback = pd.to_datetime(text, errors="coerce")
    dates = pd.Series(compact).where(pd.Series(compact).notna(), pd.Series(fallback))
    dates = pd.DatetimeIndex(dates.dropna()).normalize().unique().sort_values()
    return dates


def _next_from_calendar(calendar: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp | None:
    if calendar.empty:
        return None
    signal_ts = pd.Timestamp(date).normalize()
    pos = calendar.searchsorted(signal_ts, side="right")
    if pos >= len(calendar):
        return None
    return pd.Timestamp(calendar[pos]).normalize()


def _required_date(date: str | pd.Timestamp | None) -> pd.Timestamp:
    if date is None:
        raise ValueError("date is required")
    try:
        ts = pd.Timestamp(date)
    except Exception as exc:
        raise ValueError(f"Invalid date: {date!r}") from exc
    if pd.isna(ts):
        raise ValueError("date is required")
    return ts.normalize()


def _configured_file_calendar(config: dict) -> tuple[pd.DatetimeIndex, str]:
    data_cfg = config.get("data", {})
    qlib_cfg = config.get("qlib", {})
    candidates = [
        data_cfg.get("trading_calendar_file"),
        data_cfg.get("calendar_file"),
        Path(qlib_cfg.get("provider_uri", "data/qlib_data")) / "calendars" / "day.txt",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = resolve_path(candidate)
        calendar = _read_calendar_file(path)
        if not calendar.empty:
            return calendar, f"calendar_file:{path}"
    return pd.DatetimeIndex([]), ""


def _read_calendar_file(path: Path) -> pd.DatetimeIndex:
    if not path.exists():
        return pd.DatetimeIndex([])
    try:
        frame = pd.read_csv(path, header=None)
    except Exception as exc:
        logger.warning("Failed to read trading calendar file %s: %s", path, exc)
        return pd.DatetimeIndex([])
    if frame.empty:
        return pd.DatetimeIndex([])
    first_row = [str(value).strip().lower() for value in frame.iloc[0].tolist()]
    known_headers = {"cal_date", "trade_date", "date", "datetime", "is_open"}
    if set(first_row) & known_headers:
        frame.columns = first_row
        frame = frame.iloc[1:].reset_index(drop=True)
    lower_columns = {str(col).strip().lower(): col for col in frame.columns}
    if "is_open" in lower_columns:
        open_col = lower_columns["is_open"]
        frame = frame[_open_day_mask(frame[open_col])]
    date_col = next(
        (lower_columns[name] for name in ["cal_date", "trade_date", "date", "datetime"] if name in lower_columns),
        frame.columns[0],
    )
    return _normalize_calendar(frame[date_col])


def _tushare_trade_calendar(config: dict, now_dt: datetime) -> pd.DatetimeIndex:
    ts_cfg = config.get("tushare", {})
    http_url = str(ts_cfg.get("http_url", "") or "")
    if not http_url or "your-proxy-server" in http_url:
        return pd.DatetimeIndex([])
    token = ts_cfg.get("token") or None
    timeout = int(ts_cfg.get("timeout", 30))
    lookback_days = int(config.get("data", {}).get("trade_calendar_lookback_days", 90))
    start = (pd.Timestamp(now_dt.date()) - pd.Timedelta(days=lookback_days)).strftime("%Y%m%d")
    end = pd.Timestamp(now_dt.date()).strftime("%Y%m%d")
    payload = {
        "api_name": "trade_cal",
        "token": None if token == "your_token" else token,
        "params": {"exchange": "SSE", "start_date": start, "end_date": end},
        "fields": "cal_date,is_open",
    }
    try:
        import requests

        response = requests.post(http_url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if data.get("code", 0) != 0:
            raise RuntimeError(data.get("msg", data))
        raw = data.get("data", data)
        frame = pd.DataFrame(raw.get("items", []), columns=raw.get("fields", []))
    except Exception as exc:
        logger.warning("Falling back from Tushare trade calendar: %s", exc)
        return pd.DatetimeIndex([])
    if frame.empty or "cal_date" not in frame.columns:
        return pd.DatetimeIndex([])
    if "is_open" in frame.columns:
        frame = frame[_open_day_mask(frame["is_open"])]
    return _normalize_calendar(frame["cal_date"])


def _open_day_mask(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    return text.isin({"1", "1.0", "true", "t", "yes", "y", "open"})


def _a_trade_calendar(config: dict, now_dt: datetime) -> pd.DatetimeIndex:
    try:
        import a_trade_calendar
    except ImportError:
        return pd.DatetimeIndex([])
    data_cfg = config.get("data", {})
    lookback_days = int(data_cfg.get("trade_calendar_lookback_days", 90))
    today = pd.Timestamp(now_dt.date())
    start = (today - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")
    try:
        return _normalize_calendar(a_trade_calendar.get_trade_days(start, end))
    except Exception as exc:
        logger.warning("Falling back from a-trade-calendar: %s", exc)
        return pd.DatetimeIndex([])


def _a_trade_calendar_next_trade_date(date: str | pd.Timestamp) -> pd.Timestamp | None:
    try:
        import a_trade_calendar
    except ImportError:
        return None
    try:
        next_date = a_trade_calendar.get_next_trade_date(str(pd.Timestamp(date).date()), 1)
    except Exception as exc:
        logger.warning("Falling back from a-trade-calendar next trade date: %s", exc)
        return None
    return pd.Timestamp(next_date).normalize()
