from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from urllib.parse import urlparse
from pathlib import Path
import random
import time
from typing import Iterable

import pandas as pd

from src.config_loader import load_config, resolve_path


logger = logging.getLogger(__name__)

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


def _format_tushare_date(value: str | datetime | pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    return ts.strftime("%Y%m%d")


def _parse_tushare_frame(data: dict) -> pd.DataFrame:
    payload = data.get("data", data)
    fields = payload.get("fields")
    items = payload.get("items")
    if fields is None or items is None:
        raise ValueError(f"Unexpected tushare response shape: {data}")
    return pd.DataFrame(items, columns=fields)


@dataclass
class TushareHttpClient:
    http_url: str
    token: str | None = None
    timeout: int = 30

    @classmethod
    def from_config(cls, config: dict | None = None) -> "TushareHttpClient":
        cfg = config or load_config()
        ts_cfg = cfg.get("tushare", {})
        return cls(
            http_url=ts_cfg.get("http_url", ""),
            token=ts_cfg.get("token") or None,
            timeout=int(ts_cfg.get("timeout", 30)),
        )

    def call(self, api_name: str, params: dict | None = None, fields: Iterable[str] | str | None = None) -> pd.DataFrame:
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


def fetch_daily_stock(
    ts_code: str,
    start_date: str | datetime,
    end_date: str | datetime,
    client: TushareHttpClient | None = None,
    retries: int = 3,
) -> pd.DataFrame:
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
                wait_seconds = 2 ** (attempt - 1) + random.uniform(0, 1)
                logger.warning("Retrying %s daily data after error: %s", ts_code, exc)
                time.sleep(wait_seconds)
    raise ValueError(f"{ts_code} daily data response is invalid after {retries} attempts: {last_error}") from last_error


def merge_adj_factor(daily: pd.DataFrame, adj_factor: pd.DataFrame, default_ts_code: str | None = None) -> pd.DataFrame:
    daily = normalize_daily_frame(daily, default_ts_code=default_ts_code)
    adj = adj_factor.rename(columns={"date": "trade_date"}).copy()
    if "ts_code" not in adj.columns and default_ts_code:
        adj["ts_code"] = default_ts_code
    required = set(ADJ_FACTOR_FIELDS)
    if not required.issubset(adj.columns):
        missing = sorted(required - set(adj.columns))
        raise ValueError(f"Adj factor data is missing columns: {missing}")
    adj = adj[ADJ_FACTOR_FIELDS]
    adj["trade_date"] = pd.to_datetime(adj["trade_date"].astype(str), errors="coerce")
    adj["adj_factor"] = pd.to_numeric(adj["adj_factor"], errors="coerce")
    adj = adj.dropna(subset=["trade_date", "adj_factor"])
    merged = daily.merge(adj, on=["ts_code", "trade_date"], how="left")
    if merged["adj_factor"].isna().any():
        missing_count = int(merged["adj_factor"].isna().sum())
        raise ValueError(f"Missing adj_factor for {missing_count} daily rows.")
    return merged


def fetch_hs300_stocks(
    date: str | datetime | None = None,
    client: TushareHttpClient | None = None,
    local_file: str | Path | None = None,
) -> list[str]:
    config = load_config()
    local_path = resolve_path(local_file or config["data"]["constituents_file"])
    if local_path.exists():
        df = pd.read_csv(local_path)
        code_col = "ts_code" if "ts_code" in df.columns else "con_code"
        if code_col not in df.columns:
            raise ValueError(f"{local_path} must contain ts_code or con_code column.")
        return sorted(df[code_col].dropna().astype(str).unique().tolist())

    client = client or TushareHttpClient.from_config(config)
    params = {"index_code": "000300.SH"}
    if date is not None:
        params["trade_date"] = _format_tushare_date(date)
    df = client.call(
        "index_weight",
        params=params,
        fields=["index_code", "con_code", "trade_date", "weight"],
    )
    if df.empty:
        raise RuntimeError("No HS300 constituents returned. Add data/raw/hs300_constituents.csv or check proxy params.")
    return sorted(df["con_code"].dropna().astype(str).unique().tolist())


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
    renamed["trade_date"] = pd.to_datetime(renamed["trade_date"].astype(str), errors="coerce")
    for col in ["open", "high", "low", "close", "vol", "amount"]:
        renamed[col] = pd.to_numeric(renamed[col], errors="coerce")
    if "adj_factor" in df.columns:
        renamed["adj_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce")
    renamed = renamed.dropna(subset=["trade_date", "close"]).sort_values(["ts_code", "trade_date"])
    return renamed.reset_index(drop=True)


def update_daily_data(
    stock_codes: Iterable[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    raw_dir: str | Path | None = None,
) -> dict[str, Path]:
    config = load_config()
    data_cfg = config.get("data", {})
    duplicate_keep = str(data_cfg.get("duplicate_keep", "first"))
    target_dir = resolve_path(raw_dir or data_cfg.get("raw_dir", "data/raw"))
    target_dir.mkdir(parents=True, exist_ok=True)

    start = start_date or data_cfg["start_date"]
    end = end_date or data_cfg["end_date"]
    codes = list(stock_codes) if stock_codes is not None else fetch_hs300_stocks()
    client = TushareHttpClient.from_config(config)

    written: dict[str, Path] = {}
    for code in codes:
        path = target_dir / f"{code}.csv"
        needs_adj_backfill = _needs_adj_factor_backfill(path)
        actual_start = start if needs_adj_backfill else _incremental_start(path, start)
        if pd.Timestamp(actual_start) > pd.Timestamp(end):
            written[code] = path
            continue

        new_df = fetch_daily_stock(code, actual_start, end, client=client)
        if new_df.empty and path.exists():
            written[code] = path
            continue
        if new_df.empty:
            continue
        if path.exists() and not needs_adj_backfill:
            old_df = pd.read_csv(path, parse_dates=["trade_date"])
            new_df = pd.concat([old_df, new_df], ignore_index=True)
        new_df = normalize_daily_frame(new_df, default_ts_code=code)
        if "adj_factor" in new_df.columns:
            new_df = new_df.dropna(subset=["adj_factor"])
        new_df = new_df.drop_duplicates(["ts_code", "trade_date"], keep=duplicate_keep)
        new_df.to_csv(path, index=False, encoding="utf-8-sig")
        written[code] = path
    return written


def _incremental_start(path: Path, configured_start: str) -> str:
    if not path.exists():
        return configured_start
    df = pd.read_csv(path, usecols=["trade_date"], parse_dates=["trade_date"])
    if df.empty:
        return configured_start
    return (df["trade_date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def _needs_adj_factor_backfill(path: Path) -> bool:
    if not path.exists():
        return False
    columns = pd.read_csv(path, nrows=0).columns
    return "adj_factor" not in columns


def describe_endpoint(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return "<invalid-url>"
    path = parsed.path or "/"
    return f"{parsed.scheme}://{parsed.hostname}:{parsed.port or _default_port(parsed.scheme)}{path}"


def _default_port(scheme: str) -> int:
    return 443 if scheme == "https" else 80
