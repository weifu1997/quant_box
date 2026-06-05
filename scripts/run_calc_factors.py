from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config, resolve_path
from src.factor_calculator import load_or_compute_factors
from src.trading_calendar import resolve_target_date_value

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description="Compute and cache Qlib Alpha158 factors.")
    parser.add_argument("--start-date", default=config["data"].get("history_start_date", config["data"]["start_date"]))
    parser.add_argument("--end-date", default=config["data"]["end_date"])
    parser.add_argument("--force", action="store_true", help="Recompute even if the factor cache exists.")
    args = parser.parse_args()

    end_date = resolve_target_date_value(args.end_date, config=config)
    config["data"]["end_date"] = end_date
    factors = load_or_compute_factors(args.start_date, end_date, force=args.force)
    cache_path = resolve_path(config["factors"]["cache_file"])
    logger.info("Saved factors to %s", cache_path)
    logger.info("Shape: %s", factors.shape)


if __name__ == "__main__":
    main()
