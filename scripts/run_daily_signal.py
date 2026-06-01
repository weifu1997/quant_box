from __future__ import annotations

import argparse
from datetime import date
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.signal_generator import generate_signal, read_previous_holdings, save_signal

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate daily rebalance signal.")
    parser.add_argument("--date", default=date.today().isoformat(), help="Signal date, YYYY-MM-DD.")
    parser.add_argument("--previous-holdings", nargs="*", help="Optional previous holdings override.")
    args = parser.parse_args()

    previous = args.previous_holdings if args.previous_holdings is not None else read_previous_holdings()
    signal_df, holdings = generate_signal(args.date, previous_holdings=previous)
    signal_path, holdings_path = save_signal(signal_df, holdings, args.date)
    logger.info("Signal saved to %s", signal_path)
    logger.info("Latest holdings saved to %s", holdings_path)


if __name__ == "__main__":
    main()
