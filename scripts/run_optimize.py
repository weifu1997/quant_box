from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config, resolve_path
from src.factor_calculator import load_or_compute_factors
from src.factor_ic import calculate_factor_ic, make_ic_weights, summarize_ic
from src.optimizer import DEFAULT_GRID, run_parameter_grid


def _csv_values(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description="Run parameter grid search for the ranking strategy.")
    parser.add_argument("--start-date", default=config["data"]["start_date"])
    parser.add_argument("--end-date", default=config["data"]["end_date"])
    parser.add_argument("--factor-file", default=config["factors"]["cache_file"])
    parser.add_argument("--price-file", default="data/prices/close.parquet")
    parser.add_argument("--factor-groups", default="momentum,volatility,all,ic_weighted")
    parser.add_argument("--top-n", default="7,10,15")
    parser.add_argument("--max-turnover", default="1,2")
    parser.add_argument("--rank-buffer", default="0,5,10")
    parser.add_argument("--rebalance-freq", default="daily,weekly")
    parser.add_argument("--ic-top-k", type=int, default=30)
    parser.add_argument("--output", default="outputs/optimization_results.csv")
    args = parser.parse_args()

    factors = load_or_compute_factors(args.start_date, args.end_date, cache_file=args.factor_file)
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
    if "ic_weighted" in grid["factor_group"]:
        ic_df = calculate_factor_ic(factors, prices)
        ic_summary = summarize_ic(ic_df)
        ic_weights = make_ic_weights(ic_summary, top_k=args.ic_top_k)
        ic_summary_path = resolve_path("outputs/factor_ic_summary.csv")
        ic_summary_path.parent.mkdir(parents=True, exist_ok=True)
        ic_summary.to_csv(ic_summary_path, encoding="utf-8-sig")

    base_config = {**config["backtest"], **config["strategy"]}
    results = run_parameter_grid(
        factors,
        prices,
        base_config=base_config,
        start_date=args.start_date,
        end_date=args.end_date,
        grid=grid,
        ic_weights=ic_weights,
    )
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Optimization results saved to {output_path}")
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
