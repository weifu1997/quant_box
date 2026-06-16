"""Command-line entry point for historical universe snapshots."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config
from src.data_fetcher import update_index_constituents_data
from src.trading_calendar import resolve_target_date_value
from src.universe_builder import build_historical_universe_from_file


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def _csv_values(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip().upper() for item in str(value).split(",") if item.strip()]


def main() -> None:
    config = load_config()
    builder_cfg = config.get("universe_builder", {})
    data_cfg = config.get("data", {})
    default_core = ",".join(builder_cfg.get("core_index_codes", ["000300.SH", "000905.SH"]))
    default_satellite = str(builder_cfg.get("satellite_index_code", "000852.SH"))
    default_index_file = builder_cfg.get("index_constituents_file", "data/raw/index_constituents.csv")
    default_output_file = builder_cfg.get("output_file", "data/raw/historical_universe.csv")

    parser = argparse.ArgumentParser(
        description="Fetch index_weight rows and build point-in-time universe snapshots."
    )
    parser.add_argument("--start-date", default=data_cfg.get("history_start_date", data_cfg.get("start_date", "2015-01-01")))
    parser.add_argument("--end-date", default=data_cfg.get("end_date", "auto"))
    parser.add_argument("--core-index-codes", default=default_core, help="Comma-separated core index codes kept in full.")
    parser.add_argument("--satellite-index-code", default=default_satellite, help="Index code ranked by weight.")
    parser.add_argument("--satellite-top-n", type=int, default=int(builder_cfg.get("satellite_top_n", 300)))
    parser.add_argument("--index-constituents-file", default=default_index_file)
    parser.add_argument("--out-file", default=default_output_file)
    parser.add_argument("--skip-fetch", action="store_true", help="Only rebuild snapshots from the cached index_weight CSV.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-index-windows", type=int)
    parser.add_argument("--index-window-days", type=int, default=31)
    parser.add_argument("--fail-on-index-error", action="store_true")
    args = parser.parse_args()

    end_date = resolve_target_date_value(args.end_date, config=config)
    core_codes = _csv_values(args.core_index_codes)
    index_codes = [*core_codes, args.satellite_index_code.strip().upper()]

    if not args.skip_fetch:
        for index_code in index_codes:
            path = update_index_constituents_data(
                index_code=index_code,
                start_date=args.start_date,
                end_date=end_date,
                out_file=args.index_constituents_file,
                sleep_seconds=args.sleep_seconds,
                max_windows=args.max_index_windows,
                window_days=args.index_window_days,
                skip_failed=not args.fail_on_index_error,
                fallback_index_codes=config.get("data_governance", {}).get("index_fallback_codes", [])
                if index_code == "000300.SH"
                else [],
            )
            logger.info("index_weight cache updated for %s: %s", index_code, path)

    out_path = build_historical_universe_from_file(
        index_constituents_file=args.index_constituents_file,
        output_file=args.out_file,
        config={
            **config,
            "universe_builder": {
                **builder_cfg,
                "core_index_codes": core_codes,
                "satellite_index_code": args.satellite_index_code,
                "satellite_top_n": args.satellite_top_n,
            },
        },
    )
    logger.info("historical universe snapshots written to %s", out_path)


if __name__ == "__main__":
    main()
