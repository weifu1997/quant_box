"""模块说明：生成数据健康报告并检查行情、因子和基础数据。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config_loader import load_config, resolve_path
from src.common import PRICE_FIELD_COLUMNS, is_stock_csv as _is_stock_csv, looks_like_field_table as _looks_like_field_table
from src.data_fetcher import _load_st_calendar, _raw_latest_date, filter_universe_frame
from src.trading_calendar import resolve_target_date_value


@dataclass
class DataHealthReport:
    """类说明：封装 DataHealthReport 相关数据和行为。"""
    generated_at: str
    requested_end_date: str
    target_symbols: int
    raw_stock_files: int
    raw_target_symbols: int
    price_panel_symbols: int
    price_target_symbols: int
    factor_symbols: int
    factor_target_symbols: int
    raw_latest_target_symbols: int
    price_latest_target_symbols: int
    factor_latest_target_symbols: int
    raw_target_coverage: float
    raw_latest_target_coverage: float
    price_target_coverage: float
    price_latest_target_coverage: float
    factor_target_coverage: float
    factor_latest_target_coverage: float
    raw_latest_date: str
    price_latest_date: str
    factor_latest_date: str
    min_raw_coverage: float
    min_price_coverage: float
    min_factor_coverage: float
    require_latest_end_date: bool
    is_healthy: bool
    issues: list[str]

    def to_dict(self) -> dict[str, Any]:
        """函数说明：处理 to_dict 主要逻辑。"""
        return asdict(self)


def build_data_health_report(
    config: dict | None = None,
    price_df: pd.DataFrame | None = None,
    factor_df: pd.DataFrame | None = None,
) -> DataHealthReport:
    """函数说明：构建 build_data_health_report 主要逻辑。"""
    cfg = config or load_config()
    data_cfg = cfg.get("data", {})
    quality_cfg = cfg.get("quality", {})
    raw_dir = resolve_path(data_cfg.get("raw_dir", "data/raw"))
    price_file = resolve_path(cfg.get("ic", {}).get("price_file", "data/prices/ohlcv_adjusted.parquet"))
    factor_file = resolve_path(cfg.get("factors", {}).get("cache_file", "data/factors/alpha158.parquet"))
    requested_end = resolve_target_date_value(data_cfg.get("end_date"), config=cfg)

    target_symbols = _target_symbols(cfg)
    raw_symbols = _raw_symbols(raw_dir)
    prices = price_df if price_df is not None else _read_parquet_if_exists(price_file)
    factors = factor_df if factor_df is not None else _read_parquet_if_exists(factor_file)
    price_symbols = _frame_symbols(prices)
    factor_symbols = _factor_symbols(factors)

    raw_target = raw_symbols & target_symbols if target_symbols else raw_symbols
    price_target = price_symbols & target_symbols if target_symbols else price_symbols
    factor_target = factor_symbols & target_symbols if target_symbols else factor_symbols

    raw_latest_by_symbol = _raw_latest_dates(raw_dir, raw_target)
    price_latest_by_symbol = _price_latest_dates(prices, price_target)
    factor_latest_by_symbol = _factor_latest_dates(factors, factor_target)
    raw_latest = min(raw_latest_by_symbol.values()) if raw_latest_by_symbol else None
    price_latest = min(price_latest_by_symbol.values()) if price_latest_by_symbol else None
    factor_latest = min(factor_latest_by_symbol.values()) if factor_latest_by_symbol else None

    min_raw = float(quality_cfg.get("min_raw_coverage", 0.95))
    min_price = float(quality_cfg.get("min_price_coverage", 0.95))
    min_factor = float(quality_cfg.get("min_factor_coverage", 0.99))
    require_latest = bool(quality_cfg.get("require_latest_end_date", True))

    raw_coverage = _ratio(len(raw_target), len(target_symbols))
    requested_ts = pd.Timestamp(requested_end)
    raw_latest_target_symbols = sum(1 for value in raw_latest_by_symbol.values() if value >= requested_ts)
    price_latest_target_symbols = sum(1 for value in price_latest_by_symbol.values() if value >= requested_ts)
    factor_latest_target_symbols = sum(1 for value in factor_latest_by_symbol.values() if value >= requested_ts)
    raw_latest_target_coverage = _ratio(raw_latest_target_symbols, len(target_symbols))
    price_latest_target_coverage = _ratio(price_latest_target_symbols, len(target_symbols))
    factor_latest_target_coverage = _ratio(factor_latest_target_symbols, len(target_symbols))
    price_coverage = _ratio(len(price_target), len(target_symbols))
    factor_coverage = _ratio(len(factor_target), len(target_symbols))
    issues: list[str] = []
    if not target_symbols:
        issues.append("target_universe_empty")
    if raw_coverage < min_raw:
        issues.append(f"raw_coverage_below_threshold:{raw_coverage:.4f}<{min_raw:.4f}")
    if price_coverage < min_price:
        issues.append(f"price_coverage_below_threshold:{price_coverage:.4f}<{min_price:.4f}")
    if factor_coverage < min_factor:
        issues.append(f"factor_coverage_below_threshold:{factor_coverage:.4f}<{min_factor:.4f}")
    if require_latest:
        if raw_latest_target_coverage < min_raw:
            issues.append(f"raw_latest_coverage_below_threshold:{raw_latest_target_coverage:.4f}<{min_raw:.4f}")
        if price_latest_target_coverage < min_price:
            issues.append(f"price_latest_coverage_below_threshold:{price_latest_target_coverage:.4f}<{min_price:.4f}")
        if factor_latest_target_coverage < min_factor:
            issues.append(f"factor_latest_coverage_below_threshold:{factor_latest_target_coverage:.4f}<{min_factor:.4f}")
        if price_latest_target_symbols == 0 and (price_latest is None or price_latest < requested_ts):
            issues.append(f"price_latest_before_end:{_date_text(price_latest)}<{requested_end}")
        if factor_latest_target_symbols == 0 and (factor_latest is None or factor_latest < requested_ts):
            issues.append(f"factor_latest_before_end:{_date_text(factor_latest)}<{requested_end}")

    return DataHealthReport(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        requested_end_date=requested_end,
        target_symbols=len(target_symbols),
        raw_stock_files=len(raw_symbols),
        raw_target_symbols=len(raw_target),
        price_panel_symbols=len(price_symbols),
        price_target_symbols=len(price_target),
        factor_symbols=len(factor_symbols),
        factor_target_symbols=len(factor_target),
        raw_latest_target_symbols=raw_latest_target_symbols,
        price_latest_target_symbols=price_latest_target_symbols,
        factor_latest_target_symbols=factor_latest_target_symbols,
        raw_target_coverage=raw_coverage,
        raw_latest_target_coverage=raw_latest_target_coverage,
        price_target_coverage=price_coverage,
        price_latest_target_coverage=price_latest_target_coverage,
        factor_target_coverage=factor_coverage,
        factor_latest_target_coverage=factor_latest_target_coverage,
        raw_latest_date=_date_text(raw_latest),
        price_latest_date=_date_text(price_latest),
        factor_latest_date=_date_text(factor_latest),
        min_raw_coverage=min_raw,
        min_price_coverage=min_price,
        min_factor_coverage=min_factor,
        require_latest_end_date=require_latest,
        is_healthy=not issues,
        issues=issues,
    )


def write_data_health_report(report: DataHealthReport, out_dir: str | Path) -> tuple[Path, Path]:
    """函数说明：写入 write_data_health_report 主要逻辑。"""
    output_dir = resolve_path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "data_health_report.json"
    csv_path = output_dir / "data_health_report.csv"
    json_path.write_text(_json_dumps(report.to_dict()), encoding="utf-8")
    row = report.to_dict()
    row["issues"] = ";".join(report.issues)
    pd.DataFrame([row]).to_csv(csv_path, index=False, encoding="utf-8-sig")
    return json_path, csv_path


def _target_symbols(config: dict) -> set[str]:
    """函数说明：处理 target_symbols 的内部辅助逻辑。"""
    data_cfg = config.get("data", {})
    universe_file = resolve_path(data_cfg.get("constituents_file", "data/raw/mainboard_a_stocks.csv"))
    if not universe_file.exists():
        return set()
    df = pd.read_csv(universe_file)
    exclude_st = bool(data_cfg.get("exclude_st", True))
    st_calendar = _load_st_calendar(data_cfg.get("st_calendar_file")) if exclude_st else None
    filtered = filter_universe_frame(
        df,
        universe=str(data_cfg.get("universe", "mainboard_a")),
        as_of_date=resolve_target_date_value(data_cfg.get("end_date"), config=config),
        exclude_st=exclude_st,
        st_calendar=st_calendar,
    )
    for column in ["ts_code", "con_code", "instrument", "code"]:
        if column in filtered.columns:
            return _normalize_symbols(filtered[column].dropna())
    return set()


def _raw_symbols(raw_dir: Path) -> set[str]:
    """函数说明：处理 raw_symbols 的内部辅助逻辑。"""
    if not raw_dir.exists():
        return set()
    return {path.stem.upper() for path in raw_dir.glob("*.csv") if _is_stock_csv(path)}


def _raw_latest_dates(raw_dir: Path, symbols: set[str]) -> dict[str, pd.Timestamp]:
    """函数说明：处理 raw_latest_dates 的内部辅助逻辑。"""
    latest_dates: dict[str, pd.Timestamp] = {}
    symbol_list = list(symbols)

    def latest_for_symbol(symbol: str) -> tuple[str, pd.Timestamp | None]:
        """函数说明：处理 latest_for_symbol 主要逻辑。"""
        return symbol, _raw_latest_date(raw_dir / f"{symbol}.csv")

    if len(symbol_list) >= 100:
        max_workers = min(32, len(symbol_list))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            latest_items = executor.map(latest_for_symbol, symbol_list)
            for symbol, current in latest_items:
                if current is not None:
                    latest_dates[symbol] = current
    else:
        for symbol, current in map(latest_for_symbol, symbol_list):
            if current is not None:
                latest_dates[symbol] = current
    return latest_dates


def _read_parquet_if_exists(path: Path) -> pd.DataFrame | None:
    """函数说明：读取 read_parquet_if_exists 的内部辅助逻辑。"""
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _frame_symbols(frame: pd.DataFrame | None) -> set[str]:
    """函数说明：处理 frame_symbols 的内部辅助逻辑。"""
    if frame is None or frame.empty:
        return set()
    if isinstance(frame.columns, pd.MultiIndex):
        return _normalize_symbols(frame.columns.get_level_values(-1))
    if _looks_like_field_table(frame.columns):
        raise ValueError("Non-MultiIndex price_df must be a close-price panel with instrument columns.")
    return _normalize_symbols(frame.columns)


def _factor_symbols(frame: pd.DataFrame | None) -> set[str]:
    """函数说明：处理 factor_symbols 的内部辅助逻辑。"""
    if frame is None or frame.empty or not isinstance(frame.index, pd.MultiIndex):
        return set()
    return _normalize_symbols(frame.index.get_level_values(1))


def _price_latest_dates(frame: pd.DataFrame | None, symbols: set[str]) -> dict[str, pd.Timestamp]:
    """函数说明：处理 price_latest_dates 的内部辅助逻辑。"""
    close = _close_price_frame(frame)
    if close.empty:
        return {}

    normalized = pd.Index([_normalize_symbol(column) for column in close.columns])
    keep = normalized != ""
    if symbols:
        keep &= normalized.isin(symbols)
    if not bool(keep.any()):
        return {}
    close = close.loc[:, keep].copy()
    close.columns = normalized[keep]

    date_values = close.index.to_numpy(dtype="datetime64[ns]")
    valid_dates = pd.DataFrame(
        np.where(close.notna().to_numpy(), date_values[:, None], np.datetime64("NaT")),
        columns=close.columns,
    )
    latest = pd.to_datetime(valid_dates.max(axis=0), errors="coerce").dropna()
    if latest.empty:
        return {}
    latest = latest.groupby(level=0).max()
    return {str(symbol): pd.Timestamp(date).normalize() for symbol, date in latest.items()}


def _close_price_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    """函数说明：处理 close_price_frame 的内部辅助逻辑。"""
    if frame is None or frame.empty:
        return pd.DataFrame()
    if isinstance(frame.columns, pd.MultiIndex):
        fields = frame.columns.get_level_values(0).astype(str).str.strip().str.lower()
        if "close" not in set(fields):
            return pd.DataFrame(index=frame.index)
        close = frame.loc[:, fields == "close"].copy()
        close.columns = close.columns.get_level_values(-1)
    else:
        if _looks_like_field_table(frame.columns):
            raise ValueError("Non-MultiIndex price_df must be a close-price panel with instrument columns.")
        close = frame.copy()

    dates = pd.to_datetime(close.index, errors="coerce")
    valid_dates = ~pd.isna(dates)
    close = close.loc[valid_dates].copy()
    raw_dates = pd.DatetimeIndex(dates[valid_dates])
    if not close.empty:
        order = np.argsort(raw_dates.to_numpy(), kind="mergesort")
        close = close.iloc[order].copy()
        raw_dates = raw_dates[order]
    close.index = raw_dates.normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _factor_latest_dates(frame: pd.DataFrame | None, symbols: set[str]) -> dict[str, pd.Timestamp]:
    """函数说明：处理 factor_latest_dates 的内部辅助逻辑。"""
    if frame is None or frame.empty or not isinstance(frame.index, pd.MultiIndex):
        return {}
    dates = pd.to_datetime(frame.index.get_level_values(0), errors="coerce")
    instruments = pd.Index(frame.index.get_level_values(1)).astype(str).str.strip().str.upper()
    valid = dates.notna() & (instruments != "")
    if symbols:
        valid &= instruments.isin(symbols)
    if not bool(valid.any()):
        return {}
    latest = pd.Series(pd.DatetimeIndex(dates[valid]).normalize(), index=instruments[valid]).groupby(level=0).max()
    return {str(symbol): pd.Timestamp(date).normalize() for symbol, date in latest.items()}


def _normalize_symbols(values: Any) -> set[str]:
    """函数说明：规范化 normalize_symbols 的内部辅助逻辑。"""
    symbols = pd.Index(values).dropna().astype(str).str.strip().str.upper()
    return set(symbol for symbol in symbols if symbol)


def _normalize_symbol(value: object) -> str:
    """函数说明：规范化 normalize_symbol 的内部辅助逻辑。"""
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def _ratio(part: int, whole: int) -> float:
    """函数说明：处理 ratio 的内部辅助逻辑。"""
    return float(part / whole) if whole else 0.0


def _date_text(value: pd.Timestamp | None) -> str:
    """函数说明：处理 date_text 的内部辅助逻辑。"""
    return "" if value is None else str(pd.Timestamp(value).date())


def _json_dumps(value: dict[str, Any]) -> str:
    """函数说明：处理 json_dumps 的内部辅助逻辑。"""
    import json

    return json.dumps(value, indent=2, ensure_ascii=False)
