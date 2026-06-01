from __future__ import annotations

import argparse
from datetime import date
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.signal_generator import generate_signal, read_previous_holdings, save_signal


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate daily rebalance signal.")
    parser.add_argument("--date", default=date.today().isoformat(), help="Signal date, YYYY-MM-DD.")
    parser.add_argument("--previous-holdings", nargs="*", help="Optional previous holdings override.")
    args = parser.parse_args()

    previous = args.previous_holdings if args.previous_holdings is not None else read_previous_holdings()
    signal_df, holdings = generate_signal(args.date, previous_holdings=previous)
    signal_path, holdings_path = save_signal(signal_df, holdings, args.date)
    print(f"Signal saved to {signal_path}")
    print(f"Latest holdings saved to {holdings_path}")


if __name__ == "__main__":
    main()
