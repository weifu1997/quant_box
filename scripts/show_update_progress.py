from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config, resolve_path
from src.data_fetcher import filter_universe_frame
from src.data_health import _raw_latest_dates, _raw_symbols
from src.trading_calendar import _configured_file_calendar, resolve_target_date_value


def main() -> None:
    parser = argparse.ArgumentParser(description="Show resumable raw data update progress.")
    parser.add_argument("--scan-raw", action="store_true", help="Scan raw CSV files to recompute latest-date coverage.")
    args = parser.parse_args()

    config = load_config()
    data_cfg = config.get("data", {})
    progress_path = resolve_path(data_cfg.get("update_progress_file", "outputs/data_update_progress.json"))
    raw_dir = resolve_path(data_cfg.get("raw_dir", "data/raw"))
    progress = _load_progress(progress_path)
    target_end = _resolve_target_end(config, progress, raw_dir)
    target_ts = pd.Timestamp(target_end).normalize()

    target_symbols = _target_symbols(config, target_end)
    raw_symbols = _raw_symbols(raw_dir)
    raw_target_symbols = raw_symbols & target_symbols if target_symbols else raw_symbols
    target_count = len(target_symbols)
    progress_freshness = _progress_freshness(progress, target_end)
    if args.scan_raw or progress_freshness is None:
        latest_dates = _raw_latest_dates(raw_dir, raw_target_symbols)
        latest_symbols = sum(1 for value in latest_dates.values() if value >= target_ts)
        stale_or_missing = max(target_count - latest_symbols, 0)
        latest_coverage = latest_symbols / target_count if target_count else 0.0
        raw_latest = _date_text(min(latest_dates.values()) if latest_dates else None)
        freshness_source = "raw_scan"
    else:
        latest_symbols = int(progress_freshness["latest_symbols"])
        stale_or_missing = int(progress_freshness["stale_or_missing_symbols"])
        latest_coverage = float(progress_freshness["latest_coverage"])
        raw_latest = "not_scanned"
        freshness_source = "progress_file"

    print("Raw data freshness:")
    print(f"  freshness_source: {freshness_source}")
    print(f"  target_end_date: {target_end}")
    print(f"  target_symbols: {target_count}")
    print(f"  raw_stock_files: {len(raw_symbols)}")
    print(f"  raw_target_symbols: {len(raw_target_symbols)}")
    print(f"  latest_symbols: {latest_symbols}")
    print(f"  stale_or_missing_symbols: {stale_or_missing}")
    print(f"  latest_coverage: {latest_coverage:.2%}")
    print(f"  earliest_latest_date: {raw_latest}")
    print()

    print(f"Progress file: {progress_path}")
    if progress is None:
        print("  progress file does not exist yet.")
        return

    for key in [
        "status",
        "target_end_date",
        "target_symbols",
        "initial_existing",
        "initial_latest_symbols",
        "pending_symbols",
        "completed_symbols",
        "remaining_symbols",
        "failed_symbols",
        "latest_symbols",
        "stale_or_missing_symbols",
        "latest_coverage",
        "current_symbol",
        "last_error",
        "updated_at",
    ]:
        if key in progress:
            print(f"  {key}: {_format_value(progress[key])}")


def _load_progress(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Progress file parse error: {exc}")
        return None
    return value if isinstance(value, dict) else None


def _progress_freshness(progress: dict[str, object] | None, target_end: str) -> dict[str, object] | None:
    if not progress or str(progress.get("target_end_date", "")) != target_end:
        return None
    keys = ["latest_symbols", "stale_or_missing_symbols", "latest_coverage"]
    if not all(key in progress for key in keys):
        return None
    return {key: progress[key] for key in keys}


def _resolve_target_end(config: dict, progress: dict[str, object] | None, raw_dir: Path) -> str:
    data_cfg = config.get("data", {})
    requested = str(data_cfg.get("end_date", "auto"))
    if requested.lower() not in {"auto", "latest", "latest_trade_date", "latest_trading_day"}:
        return resolve_target_date_value(requested, config=config)

    progress_target = progress.get("target_end_date") if progress else None
    if progress_target:
        return str(progress_target)

    file_calendar, _source = _configured_file_calendar(config)
    if not file_calendar.empty:
        return resolve_target_date_value(requested, config=config, calendar=file_calendar)

    raw_symbols = _raw_symbols(raw_dir)
    latest_dates = _raw_latest_dates(raw_dir, raw_symbols)
    if latest_dates:
        return str(max(latest_dates.values()).date())

    raise ValueError("Cannot infer target end date from progress, local calendar, or raw data.")


def _target_symbols(config: dict, target_end: str) -> set[str]:
    data_cfg = config.get("data", {})
    universe_file = resolve_path(data_cfg.get("constituents_file", "data/raw/mainboard_a_stocks.csv"))
    if not universe_file.exists():
        return set()
    df = pd.read_csv(universe_file)
    filtered = filter_universe_frame(
        df,
        universe=str(data_cfg.get("universe", "mainboard_a")),
        as_of_date=target_end,
        exclude_st=bool(data_cfg.get("exclude_st", True)),
    )
    for column in ["ts_code", "con_code", "instrument", "code"]:
        if column in filtered.columns:
            return set(filtered[column].dropna().astype(str).str.upper())
    return set()


def _date_text(value: pd.Timestamp | None) -> str:
    return "" if value is None else str(pd.Timestamp(value).date())


def _format_value(value: object) -> object:
    if isinstance(value, float) and 0 <= value <= 1:
        return f"{value:.2%}"
    return value


if __name__ == "__main__":
    main()
