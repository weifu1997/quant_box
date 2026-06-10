"""Command line entrypoint for updating fundamental data caches."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config
from src.fundamental_data import update_fundamental_data
from src.trading_calendar import resolve_target_date_value


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def _csv_symbols(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def main() -> None:
    """Parse arguments and update fundamental caches."""

    config = load_config()
    data_cfg = config.get("data", {})
    fundamental_cfg = config.get("fundamentals", {})
    parser = argparse.ArgumentParser(description="Update Tushare fundamental caches for quality/dividend/debt screening.")
    parser.add_argument("--start-date", default=data_cfg.get("history_start_date") or data_cfg.get("start_date"))
    parser.add_argument("--end-date", default=data_cfg.get("end_date", "auto"))
    parser.add_argument("--symbols", help="Comma-separated instruments. Defaults to the configured stock universe.")
    parser.add_argument("--max-symbols", type=int, help="Fetch at most this many symbols; useful for smoke tests.")
    parser.add_argument("--missing-only", action="store_true", help="Fetch only symbols missing from each fundamental cache.")
    parser.add_argument("--sleep-seconds", type=float, default=fundamental_cfg.get("update_sleep_seconds", 0.0))
    parser.add_argument("--skip-fina-indicator", action="store_true", help="Skip fina_indicator cache update.")
    parser.add_argument("--skip-dividend", action="store_true", help="Skip dividend cache update.")
    parser.add_argument("--fail-on-error", action="store_true", help="Stop on the first failed symbol.")
    args = parser.parse_args()

    end_date = resolve_target_date_value(args.end_date, config=config)
    paths = update_fundamental_data(
        start_date=args.start_date,
        end_date=end_date,
        symbols=_csv_symbols(args.symbols),
        sleep_seconds=args.sleep_seconds,
        max_symbols=args.max_symbols,
        missing_only=args.missing_only,
        skip_failed=not args.fail_on_error,
        update_fina_indicator=not args.skip_fina_indicator,
        update_dividend=not args.skip_dividend,
        config=config,
    )
    for name, path in paths.items():
        logger.info("%s cache written to %s", name, path)


if __name__ == "__main__":
    main()
