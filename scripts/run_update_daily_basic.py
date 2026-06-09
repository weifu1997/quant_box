"""模块说明：提供 run_update_daily_basic 命令行入口。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config
from src.data_fetcher import update_daily_basic_data
from src.trading_calendar import resolve_target_date_value


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """函数说明：解析命令行参数并执行主流程。"""
    config = load_config()
    parser = argparse.ArgumentParser(description="Fetch Tushare daily_basic market-cap data into a parquet cache.")
    parser.add_argument("--start-date", default=config["data"].get("history_start_date", config["data"]["start_date"]))
    parser.add_argument("--end-date", default=config["data"]["end_date"])
    parser.add_argument("--out-file", default=config["data"].get("daily_basic_file", "data/factors/daily_basic.parquet"))
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-dates", type=int, help="Fetch at most this many missing trade dates for resumable chunked updates.")
    args = parser.parse_args()

    end_date = resolve_target_date_value(args.end_date, config=config)
    path = update_daily_basic_data(
        start_date=args.start_date,
        end_date=end_date,
        out_file=args.out_file,
        sleep_seconds=args.sleep_seconds,
        max_dates=args.max_dates,
    )
    logger.info("daily_basic cache written to %s", path)


if __name__ == "__main__":
    main()
