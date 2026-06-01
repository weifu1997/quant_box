from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_fetcher import update_daily_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch HS300 daily data through a tushare HTTP proxy.")
    parser.add_argument("--codes", nargs="*", help="Optional ts_codes. If omitted, HS300 constituents are used.")
    parser.add_argument("--start-date", help="Override config data.start_date.")
    parser.add_argument("--end-date", help="Override config data.end_date.")
    args = parser.parse_args()

    try:
        written = update_daily_data(stock_codes=args.codes, start_date=args.start_date, end_date=args.end_date)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Updated {len(written)} stock files.")
    for code, path in list(written.items())[:10]:
        print(f"{code}: {path}")


if __name__ == "__main__":
    main()
