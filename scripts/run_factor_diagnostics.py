"""Build factor IC stability and quantile-return diagnostics."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._shared import requested_factor_columns
from src.config_loader import load_config, resolve_path
from src.factor_calculator import load_or_compute_factors
from src.factor_diagnostics import build_factor_diagnostics, write_factor_diagnostics
from src.trading_calendar import resolve_target_date_value


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    config = load_config()
    ic_cfg = config.get("ic", {})
    parser = argparse.ArgumentParser(description="Build IC, yearly IC, and factor group-return diagnostics.")
    parser.add_argument("--start-date", default=config["data"]["start_date"])
    parser.add_argument("--end-date", default=config["data"]["end_date"])
    parser.add_argument("--factor-file", default=config["factors"]["cache_file"])
    parser.add_argument("--price-file", default=ic_cfg.get("price_file", "data/prices/ohlcv_adjusted.parquet"))
    parser.add_argument("--out-dir", default=config.get("outputs", {}).get("dir", "outputs"))
    parser.add_argument(
        "--factor-groups",
        default=config.get("strategy", {}).get("factor_group", "momentum"),
        help="Comma-separated factor groups to diagnose. Use all for every cached factor.",
    )
    parser.add_argument("--horizon", type=int, default=ic_cfg.get("horizon", 1))
    parser.add_argument("--method", default=ic_cfg.get("method", "spearman"))
    parser.add_argument("--min-obs", type=int, default=ic_cfg.get("min_obs", 20))
    parser.add_argument("--quantiles", type=int, default=5)
    args = parser.parse_args()

    end_date = resolve_target_date_value(args.end_date, config=config)
    price_path = resolve_path(args.price_file)
    if not price_path.exists():
        raise FileNotFoundError(f"Price file not found: {price_path}. Run scripts/run_convert_data.py first.")
    prices = pd.read_parquet(price_path)
    factor_columns = _requested_factor_columns(args.factor_file, args.factor_groups, config)
    if factor_columns is None:
        logger.info("Factor diagnostics requested all available factor columns.")
    else:
        logger.info("Factor diagnostics requested %s factor columns.", len(factor_columns))
    factors = load_or_compute_factors(args.start_date, end_date, cache_file=args.factor_file, columns=factor_columns)
    tables = build_factor_diagnostics(
        factors,
        prices,
        horizon=args.horizon,
        method=args.method,
        min_obs=args.min_obs,
        quantiles=args.quantiles,
    )
    paths = write_factor_diagnostics(tables, args.out_dir)
    logger.info(
        "Factor diagnostics written: ic_summary=%s yearly=%s group_returns=%s",
        paths["factor_ic_summary"],
        paths["factor_ic_yearly"],
        paths["factor_group_returns"],
    )
    print(
        "Factor diagnostics: "
        f"daily_ic={len(tables['daily_ic'])}; "
        f"yearly_rows={len(tables['yearly_ic'])}; "
        f"group_return_rows={len(tables['group_returns'])}; "
        f"out_dir={resolve_path(args.out_dir)}"
    )


def _requested_factor_columns(factor_file: str, factor_groups: str, config: dict) -> list[str] | None:
    groups = [group.strip() for group in str(factor_groups).split(",") if group.strip()]
    normalized = {group.lower() for group in groups}
    if not groups or normalized.intersection({"all", "ic_weighted"}):
        return None
    requested: set[str] = set()
    for group in groups:
        columns = requested_factor_columns(
            factor_file,
            {**config.get("strategy", {}), "factor_group": group},
            config.get("dynamic_ic_selector", {}),
            config.get("ml_strategy", {}),
            config.get("regime_score_blend", {}),
            config.get("regime_score_filter", {}),
        )
        if columns is None:
            return None
        requested.update(columns)
    return sorted(requested) if requested else None


if __name__ == "__main__":
    main()
