from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_fetcher import update_daily_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch HS300 daily data through a tushare HTTP proxy.")
    parser.add_argument("--codes", nargs="*", help="Optional ts_codes. If omitted, HS300 constituents are used.")
    parser.add_argument("--start-date", help="Override config data.start_date.")
    parser.add_argument("--end-date", help="Override config data.end_date.")
    args = parser.parse_args()

    try:
        written = update_daily_data(stock_codes=args.codes, start_date=args.start_date, end_date=args.end_date)
    except RuntimeError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    logger.info("Updated %d stock files.", len(written))
    for code, path in list(written.items())[:10]:
        logger.info("%s: %s", code, path)


if __name__ == "__main__":
    main()
