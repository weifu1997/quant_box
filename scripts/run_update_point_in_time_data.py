"""模块说明：提供 run_update_point_in_time_data 命令行入口。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config
from src.data_fetcher import update_daily_basic_data, update_index_constituents_data, update_st_calendar_data
from src.data_governance import build_data_governance_report, write_data_governance_report
from src.trading_calendar import resolve_target_date_value


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def _csv_values(value: str | None) -> list[str]:
    """函数说明：处理 csv_values 的内部辅助逻辑。"""
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def main() -> None:
    """函数说明：解析命令行参数并执行主流程。"""
    config = load_config()
    parser = argparse.ArgumentParser(
        description="Update point-in-time governance data: daily_basic, index constituents and ST calendar."
    )
    parser.add_argument("--start-date", default=config["data"].get("history_start_date", config["data"]["start_date"]))
    parser.add_argument("--end-date", default=config["data"]["end_date"])
    parser.add_argument("--index-code", default="000300.SH")
    parser.add_argument("--daily-basic-out-file", default=config["data"].get("daily_basic_file", "data/factors/daily_basic.parquet"))
    parser.add_argument("--index-out-file", default=config["data"].get("hs300_constituents_file", "data/raw/hs300_constituents.csv"))
    parser.add_argument(
        "--st-out-file",
        default=config.get("data_governance", {}).get("st_calendar_file")
        or config["data"].get("st_calendar_file", "data/raw/st_calendar.csv"),
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-dates", type=int, help="Fetch at most this many missing daily_basic trade dates.")
    parser.add_argument("--fail-on-daily-basic-error", action="store_true", help="Stop if any daily_basic date fails.")
    parser.add_argument("--max-index-windows", type=int, help="Fetch at most this many index_weight date windows.")
    parser.add_argument("--index-window-days", type=int, default=31, help="Days per index_weight request window.")
    parser.add_argument(
        "--fallback-index-codes",
        default=",".join(config.get("data_governance", {}).get("index_fallback_codes", [])),
        help="Comma-separated fallback index codes used when --index-code returns no rows for a window.",
    )
    parser.add_argument("--fail-on-index-error", action="store_true", help="Stop if any index_weight window fails.")
    parser.add_argument("--skip-daily-basic", action="store_true")
    parser.add_argument("--skip-index-constituents", action="store_true")
    parser.add_argument("--skip-st-calendar", action="store_true")
    args = parser.parse_args()

    end_date = resolve_target_date_value(args.end_date, config=config)
    if not args.skip_daily_basic:
        path = update_daily_basic_data(
            start_date=args.start_date,
            end_date=end_date,
            out_file=args.daily_basic_out_file,
            sleep_seconds=args.sleep_seconds,
            max_dates=args.max_dates,
            skip_failed=not args.fail_on_daily_basic_error,
        )
        logger.info("daily_basic cache written to %s", path)
    if not args.skip_index_constituents:
        path = update_index_constituents_data(
            index_code=args.index_code,
            start_date=args.start_date,
            end_date=end_date,
            out_file=args.index_out_file,
            sleep_seconds=args.sleep_seconds,
            max_windows=args.max_index_windows,
            window_days=args.index_window_days,
            skip_failed=not args.fail_on_index_error,
            fallback_index_codes=_csv_values(args.fallback_index_codes),
        )
        logger.info("index constituents cache written to %s", path)
    if not args.skip_st_calendar:
        path = update_st_calendar_data(out_file=args.st_out_file, coverage_end_date=end_date)
        logger.info("ST calendar written to %s", path)

    report = build_data_governance_report(config)
    report_path = write_data_governance_report(report, config.get("outputs", {}).get("dir", "outputs"))
    logger.info("data governance report written to %s", report_path)


if __name__ == "__main__":
    main()
