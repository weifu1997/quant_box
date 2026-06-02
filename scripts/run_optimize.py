from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config, resolve_path
from src.factor_calculator import load_or_compute_factors
from src.factor_ic import calculate_factor_ic, make_ic_weights, summarize_ic
from src.optimizer import DEFAULT_GRID, run_parameter_grid, run_walk_forward_optimization
from src.trading_calendar import resolve_target_date_value

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def _csv_values(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description="Run parameter grid search for the ranking strategy.")
    parser.add_argument("--start-date", default=config["data"]["start_date"])
    parser.add_argument("--end-date", default=config["data"]["end_date"])
    parser.add_argument("--factor-file", default=config["factors"]["cache_file"])
    parser.add_argument("--price-file", default="data/prices/ohlcv.parquet")
    parser.add_argument("--factor-groups", default="ic_weighted,momentum")
    parser.add_argument("--top-n", default="5,7,10")
    parser.add_argument("--max-turnover", default="1")
    parser.add_argument("--rank-buffer", default="10,20")
    parser.add_argument("--rebalance-freq", default="weekly,monthly")
    parser.add_argument("--ic-top-k", type=int, default=30)
    parser.add_argument("--ic-window", type=int, default=config.get("ic", {}).get("window", 252))
    parser.add_argument("--ic-min-periods", type=int, default=config.get("ic", {}).get("min_periods", 60))
    parser.add_argument("--ic-min-abs", type=float, default=config.get("ic", {}).get("min_abs_ic", 0.02))
    parser.add_argument("--ic-corr-threshold", type=float, default=config.get("ic", {}).get("corr_threshold", 0.7))
    parser.add_argument("--ic-weight-smoothing", type=float, default=config.get("ic", {}).get("weight_smoothing", 0.0))
    parser.add_argument("--ic-max-weight-turnover", type=float, default=config.get("ic", {}).get("max_weight_turnover"))
    parser.add_argument("--turnover-penalty", type=float, default=0.02)
    parser.add_argument("--cost-penalty", type=float, default=1.0)
    parser.add_argument("--rolling-ic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--walk-forward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--test-months", type=int, default=12)
    parser.add_argument("--step-months", type=int, default=6)
    parser.add_argument("--output", default="outputs/optimization_results.csv")
    args = parser.parse_args()
    end_date = resolve_target_date_value(args.end_date, config=config)
    config["data"]["end_date"] = end_date

    factors = load_or_compute_factors(args.start_date, end_date, cache_file=args.factor_file)
    prices = pd.read_parquet(resolve_path(args.price_file))

    grid = {
        **DEFAULT_GRID,
        "factor_group": _csv_values(args.factor_groups, str),
        "top_n": _csv_values(args.top_n, int),
        "max_turnover": _csv_values(args.max_turnover, int),
        "rank_buffer": _csv_values(args.rank_buffer, int),
        "rebalance_freq": _csv_values(args.rebalance_freq, str),
    }

    ic_weights = None
    if "ic_weighted" in grid["factor_group"] and not args.walk_forward and not args.rolling_ic:
        ic_df = calculate_factor_ic(factors, prices)
        ic_summary = summarize_ic(ic_df)
        ic_weights = make_ic_weights(ic_summary, top_k=args.ic_top_k, min_abs_ic=args.ic_min_abs)
        ic_summary_path = resolve_path("outputs/factor_ic_summary.csv")
        ic_summary_path.parent.mkdir(parents=True, exist_ok=True)
        ic_summary.to_csv(ic_summary_path, encoding="utf-8-sig")

    base_config = {**config["backtest"], **config["strategy"]}
    if args.walk_forward:
        results = run_walk_forward_optimization(
            factors,
            prices,
            base_config=base_config,
            start_date=args.start_date,
            end_date=end_date,
            grid=grid,
            train_years=args.train_years,
            test_months=args.test_months,
            step_months=args.step_months,
            use_rolling_ic=args.rolling_ic,
            ic_window=args.ic_window,
            ic_min_periods=args.ic_min_periods,
            ic_min_abs=args.ic_min_abs,
            ic_corr_threshold=args.ic_corr_threshold,
            ic_top_k=args.ic_top_k,
            ic_weight_smoothing=args.ic_weight_smoothing,
            ic_max_weight_turnover=args.ic_max_weight_turnover,
            turnover_penalty=args.turnover_penalty,
            cost_penalty=args.cost_penalty,
        )
    else:
        results = run_parameter_grid(
            factors,
            prices,
            base_config=base_config,
            start_date=args.start_date,
            end_date=end_date,
            grid=grid,
            ic_weights=ic_weights,
            use_rolling_ic=args.rolling_ic,
            ic_window=args.ic_window,
            ic_min_periods=args.ic_min_periods,
            ic_min_abs=args.ic_min_abs,
            ic_corr_threshold=args.ic_corr_threshold,
            ic_top_k=args.ic_top_k,
            ic_weight_smoothing=args.ic_weight_smoothing,
            ic_max_weight_turnover=args.ic_max_weight_turnover,
            turnover_penalty=args.turnover_penalty,
            cost_penalty=args.cost_penalty,
        )
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("Optimization results saved to %s", output_path)
    logger.info("Top results:\n%s", results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
