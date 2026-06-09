"""模块说明：提供 run_daily_signal 命令行入口。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.signal_generator import generate_signal, read_previous_holdings, save_candidate_signal, save_signal

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """函数说明：解析命令行参数并执行主流程。"""
    parser = argparse.ArgumentParser(description="Generate daily rebalance signal.")
    parser.add_argument("--date", default="latest", help="Signal date, YYYY-MM-DD, or latest factor date.")
    parser.add_argument("--previous-holdings", nargs="*", help="Optional previous holdings override.")
    parser.add_argument(
        "--official",
        action="store_true",
        help="Write official signal files and overwrite latest holdings. Default writes candidate files only.",
    )
    args = parser.parse_args()

    previous = args.previous_holdings if args.previous_holdings is not None else read_previous_holdings()
    signal_df, holdings = generate_signal(args.date, previous_holdings=previous)
    output_date = _signal_output_date(signal_df, args.date)
    if args.official:
        signal_path, holdings_path = save_signal(signal_df, holdings, output_date)
        logger.info("Official signal saved to %s", signal_path)
        logger.info("Latest holdings saved to %s", holdings_path)
    else:
        signal_path, holdings_path = save_candidate_signal(signal_df, holdings, output_date)
        logger.info("Candidate signal saved to %s", signal_path)
        logger.info("Candidate holdings saved to %s", holdings_path)


def _signal_output_date(signal_df, requested_date: str) -> str:
    """函数说明：处理 signal_output_date 的内部辅助逻辑。"""
    if not signal_df.empty and "date" in signal_df.columns:
        return str(signal_df["date"].iloc[0])
    signal_date = getattr(signal_df, "attrs", {}).get("signal_date")
    if signal_date:
        return str(signal_date)
    return requested_date


if __name__ == "__main__":
    main()
