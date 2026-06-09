"""模块说明：提供 run_update_data 命令行入口。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_fetcher import update_daily_data, update_daily_data_resumable

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """函数说明：解析命令行参数并执行主流程。"""
    parser = argparse.ArgumentParser(description="Fetch configured A-share universe daily data through a Tushare HTTP proxy.")
    parser.add_argument("--codes", nargs="*", help="Optional ts_codes. If omitted, configured universe constituents are used.")
    parser.add_argument("--start-date", help="Override config data.history_start_date.")
    parser.add_argument("--end-date", help="Override config data.end_date.")
    parser.add_argument("--chunk-size", type=int, help="Resumable missing-symbol chunk size when --codes is omitted.")
    parser.add_argument("--sleep-seconds", type=float, help="Seconds to sleep between resumable chunks when --codes is omitted.")
    parser.add_argument("--max-chunks", type=int, help="Stop after this many resumable chunks.")
    parser.add_argument("--include-existing", action="store_true", help="Also update existing raw files in resumable mode.")
    parser.add_argument("--force-full", action="store_true", help="Refetch each selected symbol from start/list date instead of incremental start.")
    parser.add_argument("--progress-file", help="Progress JSON path for resumable mode.")
    args = parser.parse_args()

    try:
        if args.codes:
            written = update_daily_data(
                stock_codes=args.codes,
                start_date=args.start_date,
                end_date=args.end_date,
                force_full=args.force_full,
            )
        else:
            written = update_daily_data_resumable(
                start_date=args.start_date,
                end_date=args.end_date,
                chunk_size=args.chunk_size,
                sleep_seconds=args.sleep_seconds,
                progress_file=args.progress_file,
                max_chunks=args.max_chunks,
                include_existing=args.include_existing,
                force_full=args.force_full,
            )
    except RuntimeError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    logger.info("Updated %d stock files.", len(written))
    for code, path in list(written.items())[:10]:
        logger.info("%s: %s", code, path)


if __name__ == "__main__":
    main()
