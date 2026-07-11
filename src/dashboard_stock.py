"""Read-only stock quote view model for the local Web dashboard."""

from __future__ import annotations

import csv
from datetime import datetime
import logging
import math
from pathlib import Path
import re
from typing import Any, Mapping

import pandas as pd

from src.common import normalize_instrument
from src.config_loader import load_config, resolve_path
from src.tushare_client import TushareHttpClient


logger = logging.getLogger(__name__)

STOCK_DETAIL_VERSION = 1
INSTRUMENT_PATTERN = re.compile(r"^\d{6}\.(?:SH|SZ|BJ)$")


def build_stock_detail(
    instrument: str,
    *,
    config: Mapping[str, Any] | None = None,
    client: TushareHttpClient | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return a live stock quote, falling back to the latest local daily row."""
    normalized = normalize_instrument(instrument)
    if not INSTRUMENT_PATTERN.fullmatch(normalized):
        raise ValueError(f"Invalid stock instrument: {instrument}")

    cfg = dict(config) if config is not None else load_config()
    retrieved_at = (now or datetime.now().astimezone()).astimezone().isoformat(timespec="seconds")
    names = load_instrument_name_map(cfg)
    quote_client = client or TushareHttpClient.from_config(cfg)
    try:
        frame = quote_client.call(
            "rt_k",
            params={"ts_code": normalized},
            fields=["ts_code", "name", "pre_close", "open", "high", "low", "close", "vol", "amount"],
        )
        return _live_detail(normalized, frame, names.get(normalized, ""), retrieved_at)
    except (RuntimeError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Live quote unavailable for %s; using local daily fallback: %s", normalized, exc)

    try:
        return _local_detail(normalized, cfg, names.get(normalized, ""), retrieved_at)
    except (FileNotFoundError, ValueError, OSError, pd.errors.ParserError) as exc:
        raise RuntimeError(
            f"Live quote is unavailable and no usable local daily price exists for {normalized}."
        ) from exc


def load_instrument_name_map(config: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Load the configured stock-universe code/name mapping."""
    cfg = dict(config) if config is not None else load_config()
    data_cfg = _mapping_value(cfg.get("data"))
    path = resolve_path(data_cfg.get("constituents_file", "data/raw/mainboard_a_stocks.csv"))
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            if "name" not in fieldnames:
                return {}
            code_column = next(
                (column for column in ["ts_code", "instrument", "ticker", "symbol"] if column in fieldnames),
                "",
            )
            if not code_column:
                return {}
            result: dict[str, str] = {}
            for row in reader:
                code = normalize_instrument(row.get(code_column, ""))
                name = str(row.get("name", "")).strip()
                if code and name:
                    result[code] = name
            return result
    except (OSError, csv.Error, UnicodeDecodeError):
        return {}


def _live_detail(instrument: str, frame: pd.DataFrame, fallback_name: str, retrieved_at: str) -> dict[str, Any]:
    if frame.empty:
        raise RuntimeError(f"Tushare rt_k returned no quote for {instrument}.")
    matches = pd.DataFrame()
    if "ts_code" in frame.columns:
        normalized_codes = frame["ts_code"].map(normalize_instrument)
        matches = frame.loc[normalized_codes == instrument]
    row = (matches if not matches.empty else frame).iloc[0]
    price = _required_positive_float(row.get("close"), "close")
    pre_close = _optional_positive_float(row.get("pre_close"))
    change = price - pre_close if pre_close is not None else None
    change_pct = (change / pre_close * 100.0) if change is not None and pre_close else None
    return {
        "version": STOCK_DETAIL_VERSION,
        "instrument": instrument,
        "name": _clean_stock_name(row.get("name")) or fallback_name,
        "status": "live",
        "is_live": True,
        "source": "tushare_rt_k",
        "price": price,
        "change": change,
        "change_pct": change_pct,
        "pre_close": pre_close,
        "open": _optional_positive_float(row.get("open")),
        "high": _optional_positive_float(row.get("high")),
        "low": _optional_positive_float(row.get("low")),
        "volume": _optional_non_negative_float(row.get("vol")),
        "amount": _optional_non_negative_float(row.get("amount")),
        "market_date": None,
        "retrieved_at": retrieved_at,
        "message": "实时行情接口返回最新价格，但接口未提供行情日期；非交易时段可能为最近一次收盘行情。",
    }


def _local_detail(
    instrument: str,
    config: Mapping[str, Any],
    fallback_name: str,
    retrieved_at: str,
) -> dict[str, Any]:
    data_cfg = _mapping_value(config.get("data"))
    raw_dir = resolve_path(data_cfg.get("raw_dir", "data/raw"))
    path = (raw_dir / f"{instrument}.csv").resolve()
    if path.parent != raw_dir.resolve():
        raise ValueError(f"Invalid local stock path for {instrument}.")
    if not path.exists():
        raise FileNotFoundError(f"Local daily price file not found: {path}")

    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"trade_date", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Local daily price is missing columns: {missing}")
    dates = pd.to_datetime(frame["trade_date"].astype(str), errors="coerce")
    closes = pd.to_numeric(frame["close"], errors="coerce")
    valid = dates.notna() & closes.notna() & (closes > 0)
    if not valid.any():
        raise ValueError(f"Local daily price has no usable rows for {instrument}.")
    usable = frame.loc[valid].copy()
    usable["_date"] = dates.loc[valid]
    usable["_close"] = closes.loc[valid]
    usable = usable.sort_values("_date")
    row = usable.iloc[-1]
    price = float(row["_close"])
    pre_close = _optional_positive_float(row.get("pre_close"))
    if pre_close is None and len(usable) > 1:
        pre_close = _optional_positive_float(usable.iloc[-2]["_close"])
    change = price - pre_close if pre_close is not None else None
    change_pct = (change / pre_close * 100.0) if change is not None and pre_close else None
    return {
        "version": STOCK_DETAIL_VERSION,
        "instrument": instrument,
        "name": fallback_name,
        "status": "fallback",
        "is_live": False,
        "source": "local_daily",
        "price": price,
        "change": change,
        "change_pct": change_pct,
        "pre_close": pre_close,
        "open": _optional_positive_float(row.get("open")),
        "high": _optional_positive_float(row.get("high")),
        "low": _optional_positive_float(row.get("low")),
        "volume": _optional_non_negative_float(row.get("vol", row.get("volume"))),
        "amount": _optional_non_negative_float(row.get("amount")),
        "market_date": pd.Timestamp(row["_date"]).date().isoformat(),
        "retrieved_at": retrieved_at,
        "message": "实时行情暂不可用，当前显示本地最新日线收盘价（非实时）。",
    }


def _required_positive_float(value: object, field: str) -> float:
    parsed = _optional_positive_float(value)
    if parsed is None:
        raise ValueError(f"Stock quote has no usable {field} value.")
    return parsed


def _optional_positive_float(value: object) -> float | None:
    parsed = _finite_float(value)
    return parsed if parsed is not None and parsed > 0 else None


def _optional_non_negative_float(value: object) -> float | None:
    parsed = _finite_float(value)
    return parsed if parsed is not None and parsed >= 0 else None


def _finite_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _clean_stock_name(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return "".join(str(value).split())


def _mapping_value(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}
