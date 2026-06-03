from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config_loader import load_config, resolve_path
from src.data_fetcher import filter_universe_frame
from src.trading_calendar import latest_trade_date, resolve_target_date_value


@dataclass
class DataHealthReport:
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
    raw_target_coverage: float
    raw_latest_target_coverage: float
    price_target_coverage: float
    factor_target_coverage: float
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
        return asdict(self)


def build_data_health_report(
    config: dict | None = None,
    price_df: pd.DataFrame | None = None,
    factor_df: pd.DataFrame | None = None,
) -> DataHealthReport:
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
    raw_latest = min(raw_latest_by_symbol.values()) if raw_latest_by_symbol else None
    price_latest = latest_trade_date(price_df=prices) if prices is not None else None
    factor_latest = _factor_latest_date(factors)

    min_raw = float(quality_cfg.get("min_raw_coverage", 0.95))
    min_price = float(quality_cfg.get("min_price_coverage", 0.95))
    min_factor = float(quality_cfg.get("min_factor_coverage", 0.95))
    require_latest = bool(quality_cfg.get("require_latest_end_date", True))

    raw_coverage = _ratio(len(raw_target), len(target_symbols))
    requested_ts = pd.Timestamp(requested_end)
    raw_latest_target_symbols = sum(1 for value in raw_latest_by_symbol.values() if value >= requested_ts)
    raw_latest_target_coverage = _ratio(raw_latest_target_symbols, len(target_symbols))
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
        if price_latest is None or price_latest < requested_ts:
            issues.append(f"price_latest_before_end:{_date_text(price_latest)}<{requested_end}")
        if factor_latest is None or factor_latest < requested_ts:
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
        raw_target_coverage=raw_coverage,
        raw_latest_target_coverage=raw_latest_target_coverage,
        price_target_coverage=price_coverage,
        factor_target_coverage=factor_coverage,
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
    data_cfg = config.get("data", {})
    universe_file = resolve_path(data_cfg.get("constituents_file", "data/raw/mainboard_a_stocks.csv"))
    if not universe_file.exists():
        return set()
    df = pd.read_csv(universe_file)
    filtered = filter_universe_frame(
        df,
        universe=str(data_cfg.get("universe", "mainboard_a")),
        as_of_date=resolve_target_date_value(data_cfg.get("end_date"), config=config),
        exclude_st=bool(data_cfg.get("exclude_st", True)),
    )
    for column in ["ts_code", "con_code", "instrument", "code"]:
        if column in filtered.columns:
            return set(filtered[column].dropna().astype(str).str.upper())
    return set()


def _raw_symbols(raw_dir: Path) -> set[str]:
    if not raw_dir.exists():
        return set()
    return {path.stem.upper() for path in raw_dir.glob("*.csv") if _is_stock_csv(path)}


def _raw_latest_dates(raw_dir: Path, symbols: set[str]) -> dict[str, pd.Timestamp]:
    latest_dates: dict[str, pd.Timestamp] = {}
    for symbol in symbols:
        path = raw_dir / f"{symbol}.csv"
        if not path.exists():
            continue
        try:
            dates = pd.read_csv(path, usecols=["trade_date"], parse_dates=["trade_date"])
        except (OSError, ValueError):
            continue
        if dates.empty:
            continue
        current = pd.to_datetime(dates["trade_date"], errors="coerce").max()
        if pd.isna(current):
            continue
        latest_dates[symbol] = pd.Timestamp(current).normalize()
    return latest_dates


def _read_parquet_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _frame_symbols(frame: pd.DataFrame | None) -> set[str]:
    if frame is None or frame.empty:
        return set()
    if isinstance(frame.columns, pd.MultiIndex):
        return set(frame.columns.get_level_values(-1).astype(str).str.upper())
    return set(frame.columns.astype(str).str.upper())


def _factor_symbols(frame: pd.DataFrame | None) -> set[str]:
    if frame is None or frame.empty or not isinstance(frame.index, pd.MultiIndex):
        return set()
    return set(frame.index.get_level_values(1).astype(str).str.upper())


def _factor_latest_date(frame: pd.DataFrame | None) -> pd.Timestamp | None:
    if frame is None or frame.empty or not isinstance(frame.index, pd.MultiIndex):
        return None
    dates = pd.to_datetime(frame.index.get_level_values(0), errors="coerce")
    if dates.empty:
        return None
    return pd.Timestamp(dates.max()).normalize()


def _is_stock_csv(path: Path) -> bool:
    name = path.name.upper()
    return len(name) == len("000001.SZ.CSV") and name[:6].isdigit() and name[6:] in {".SZ.CSV", ".SH.CSV"}


def _ratio(part: int, whole: int) -> float:
    return float(part / whole) if whole else 0.0


def _date_text(value: pd.Timestamp | None) -> str:
    return "" if value is None else str(pd.Timestamp(value).date())


def _json_dumps(value: dict[str, Any]) -> str:
    import json

    return json.dumps(value, indent=2, ensure_ascii=False)
