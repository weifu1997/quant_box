"""Command line entrypoint for fundamental quality/dividend/debt screening."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config
from src.fundamental_data import build_fundamental_screen, write_fundamental_screen_outputs


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Parse arguments and write the fundamental screen report."""

    config = load_config()
    screen_cfg = config.get("fundamental_screen", {})
    parser = argparse.ArgumentParser(description="Generate a conservative fundamental screen and explanation report.")
    parser.add_argument("--date", default="latest", help="Screen date, YYYY-MM-DD, or latest.")
    parser.add_argument("--top", type=int, default=int(screen_cfg.get("top_n", 30)), help="Number of passed candidates in the report.")
    parser.add_argument("--csv-file", help="Optional output CSV path.")
    parser.add_argument("--report-file", help="Optional output Markdown path.")
    args = parser.parse_args()

    result = build_fundamental_screen(config=config, as_of=args.date)
    csv_path, report_path = write_fundamental_screen_outputs(
        result,
        config.get("outputs", {}).get("dir", "outputs"),
        top_n=args.top,
        csv_file=args.csv_file,
        report_file=args.report_file,
    )
    logger.info("Fundamental screen CSV written to %s", csv_path)
    logger.info("Fundamental screen report written to %s", report_path)


if __name__ == "__main__":
    main()
