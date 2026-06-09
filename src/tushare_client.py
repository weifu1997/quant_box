"""模块说明：封装 Tushare HTTP API 请求和返回数据解析。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import random
from typing import Iterable
from urllib.parse import urlparse

import pandas as pd

from src.config_loader import load_config


DAILY_FIELDS = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
]
ADJ_FACTOR_FIELDS = ["ts_code", "trade_date", "adj_factor"]
DAILY_BASIC_FIELDS = [
    "ts_code",
    "trade_date",
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "pe",
    "pe_ttm",
    "pb",
    "ps",
    "ps_ttm",
    "dv_ratio",
    "dv_ttm",
    "total_share",
    "float_share",
    "free_share",
    "total_mv",
    "circ_mv",
]
STOCK_BASIC_FIELDS = [
    "ts_code",
    "symbol",
    "name",
    "area",
    "industry",
    "market",
    "exchange",
    "list_status",
    "list_date",
    "delist_date",
]
INDEX_WEIGHT_FIELDS = ["index_code", "con_code", "trade_date", "weight"]
NAMECHANGE_FIELDS = ["ts_code", "name", "start_date", "end_date", "ann_date", "change_reason"]
ST_CALENDAR_FIELDS = ["ts_code", "st_start_date", "st_end_date", "name", "ann_date", "change_reason"]
MAINBOARD_PREFIXES = ("000", "001", "002", "003", "600", "601", "603", "605")


def _format_tushare_date(value: str | datetime | pd.Timestamp | None) -> str | None:
    """函数说明：处理 format_tushare_date 的内部辅助逻辑。"""
    if value is None:
        return None
    ts = pd.Timestamp(value)
    return ts.strftime("%Y%m%d")


def _parse_tushare_dates(series: pd.Series) -> pd.Series:
    """函数说明：解析 parse_tushare_dates 的内部辅助逻辑。"""
    text = series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    compact = text.str.replace("-", "", regex=False).str.replace("/", "", regex=False)
    yyyymmdd = compact.str.fullmatch(r"\d{8}")
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if yyyymmdd.any():
        parsed.loc[yyyymmdd] = pd.to_datetime(compact.loc[yyyymmdd], format="%Y%m%d", errors="coerce")
    if (~yyyymmdd).any():
        parsed.loc[~yyyymmdd] = pd.to_datetime(series.loc[~yyyymmdd], errors="coerce")
    return parsed


def _normalize_symbol(value: object) -> str:
    """函数说明：规范化 normalize_symbol 的内部辅助逻辑。"""
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def _normalize_symbol_series(series: pd.Series) -> pd.Series:
    """函数说明：规范化 normalize_symbol_series 的内部辅助逻辑。"""
    return series.astype("string").str.strip().str.upper()


def _unique_normalized_symbols(values: Iterable[object]) -> list[str]:
    """函数说明：处理 unique_normalized_symbols 的内部辅助逻辑。"""
    symbols: list[str] = []
    seen: set[str] = set()
    for value in values:
        symbol = _normalize_symbol(value)
        if symbol and symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)
    return symbols


def _parse_tushare_frame(data: dict) -> pd.DataFrame:
    """函数说明：解析 parse_tushare_frame 的内部辅助逻辑。"""
    payload = data.get("data", data)
    fields = payload.get("fields")
    items = payload.get("items")
    if fields is None or items is None:
        raise ValueError(f"Unexpected tushare response shape: {data}")
    return pd.DataFrame(items, columns=fields)


@dataclass
class TushareHttpClient:
    """类说明：封装 TushareHttpClient 相关数据和行为。"""
    http_url: str
    token: str | None = None
    timeout: int = 30

    @classmethod
    def from_config(cls, config: dict | None = None) -> "TushareHttpClient":
        """函数说明：处理 from_config 主要逻辑。"""
        cfg = config or load_config()
        ts_cfg = cfg.get("tushare", {})
        return cls(
            http_url=ts_cfg.get("http_url", ""),
            token=ts_cfg.get("token") or None,
            timeout=int(ts_cfg.get("timeout", 30)),
        )

    def call(self, api_name: str, params: dict | None = None, fields: Iterable[str] | str | None = None) -> pd.DataFrame:
        """函数说明：处理 call 主要逻辑。"""
        if not self.http_url or "your-proxy-server" in self.http_url:
            raise RuntimeError("Please configure tushare.http_url in config/settings.yaml first.")

        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("The 'requests' package is required for tushare HTTP access.") from exc

        if isinstance(fields, str):
            field_value = fields
        elif fields is None:
            field_value = None
        else:
            field_value = ",".join(fields)

        payload = {
            "api_name": api_name,
            "token": None if self.token == "your_token" else self.token,
            "params": params or {},
        }
        if field_value:
            payload["fields"] = field_value

        try:
            response = requests.post(self.http_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as exc:
            endpoint = describe_endpoint(self.http_url)
            raise RuntimeError(
                "Failed to connect to tushare HTTP proxy "
                f"({endpoint}). Check the full proxy URL/path, firewall/network permission, "
                "and whether the proxy service is running."
            ) from exc
        except ValueError as exc:
            raise RuntimeError("The tushare HTTP proxy returned a non-JSON response.") from exc
        if data.get("code", 0) != 0:
            raise RuntimeError(f"tushare error: {data.get('msg', data)}")
        return _parse_tushare_frame(data)

    def redacted_request_preview(
        self,
        api_name: str = "daily",
        params: dict | None = None,
        fields: Iterable[str] | str | None = None,
    ) -> dict:
        """函数说明：处理 redacted_request_preview 主要逻辑。"""
        if isinstance(fields, str):
            field_value = fields
        elif fields is None:
            field_value = None
        else:
            field_value = ",".join(fields)
        payload = {
            "api_name": api_name,
            "token": "***" if self.token else None,
            "params": params or {},
        }
        if field_value:
            payload["fields"] = field_value
        return {"url": describe_endpoint(self.http_url), "timeout": self.timeout, "payload": payload}


def _is_tushare_connection_error(exc: Exception) -> bool:
    """函数说明：判断 is_tushare_connection_error 是否成立。"""
    return "Failed to connect to tushare HTTP proxy" in str(exc)


def _retry_wait_seconds(attempt: int, retry_max_wait: float | None = None) -> float:
    """函数说明：处理 retry_wait_seconds 的内部辅助逻辑。"""
    wait_seconds = 2 ** (attempt - 1) + random.uniform(0, 1)
    if retry_max_wait is None:
        return wait_seconds
    return min(wait_seconds, max(float(retry_max_wait), 0.0))


def _date_windows(
    start_date: str | datetime,
    end_date: str | datetime,
    window_days: int | None,
) -> Iterable[tuple[str, str]]:
    """函数说明：处理 date_windows 的内部辅助逻辑。"""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if window_days is None or window_days <= 0 or start > end:
        yield start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        return

    current = start
    step = pd.Timedelta(days=window_days - 1)
    while current <= end:
        window_end = min(current + step, end)
        yield current.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d")
        current = window_end + pd.Timedelta(days=1)


def describe_endpoint(url: str) -> str:
    """函数说明：处理 describe_endpoint 主要逻辑。"""
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return "<invalid-url>"
    path = parsed.path or "/"
    return f"{parsed.scheme}://{parsed.hostname}:{parsed.port or _default_port(parsed.scheme)}{path}"


def _default_port(scheme: str) -> int:
    """函数说明：处理 default_port 的内部辅助逻辑。"""
    return 443 if scheme == "https" else 80
