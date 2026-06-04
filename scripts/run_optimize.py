from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config, resolve_path
from src.factor_calculator import factor_cache_columns, load_or_compute_factors
from src.factor_ic import calculate_factor_ic, make_ic_weights, summarize_ic
from src.optimizer import BASELINE_GRID, DEFAULT_GRID, run_parameter_grid, run_walk_forward_grid_validation, run_walk_forward_optimization
from src.scoring import DEFAULT_DYNAMIC_IC_CANDIDATES, DYNAMIC_IC_SELECTOR_GROUPS
from src.strategy import factor_columns_for_method
from src.trading_calendar import resolve_target_date_value

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def _csv_values(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _grid_values(value: str | None, defaults: list, cast):
    if value is None:
        return list(defaults)
    return _csv_values(value, cast)


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description="Run parameter grid search for the ranking strategy.")
    parser.add_argument("--start-date", default=config["data"]["start_date"])
    parser.add_argument("--end-date", default=config["data"]["end_date"])
    parser.add_argument("--factor-file", default=config["factors"]["cache_file"])
    parser.add_argument("--price-file", default=config.get("ic", {}).get("price_file", "data/prices/ohlcv_adjusted.parquet"))
    parser.add_argument("--factor-groups", help="Comma-separated factor groups. Defaults to the fast baseline grid.")
    parser.add_argument("--top-n", help="Comma-separated portfolio sizes. Defaults to the fast baseline grid.")
    parser.add_argument("--max-turnover", help="Comma-separated max turnover values. Defaults to the fast baseline grid.")
    parser.add_argument("--rank-buffer", help="Comma-separated rank buffer values. Defaults to the fast baseline grid.")
    parser.add_argument("--rebalance-freq", help="Comma-separated rebalance frequencies. Defaults to the fast baseline grid.")
    parser.add_argument("--full-grid", action="store_true", help="Use the full default grid instead of the fast baseline grid.")
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
    parser.add_argument(
        "--selection-walk-forward",
        action="store_true",
        help="Use slower nested walk-forward parameter selection. Default walk-forward mode validates every grid row out of sample.",
    )
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--test-months", type=int, default=12)
    parser.add_argument("--step-months", type=int, default=12)
    parser.add_argument("--output", default="outputs/optimization_results.csv")
    args = parser.parse_args()
    end_date = resolve_target_date_value(args.end_date, config=config)
    config["data"]["end_date"] = end_date

    grid_defaults = DEFAULT_GRID if args.full_grid else BASELINE_GRID
    grid = {
        **grid_defaults,
        "factor_group": _grid_values(args.factor_groups, grid_defaults["factor_group"], str),
        "top_n": _grid_values(args.top_n, grid_defaults["top_n"], int),
        "max_turnover": _grid_values(args.max_turnover, grid_defaults["max_turnover"], int),
        "rank_buffer": _grid_values(args.rank_buffer, grid_defaults["rank_buffer"], int),
        "rebalance_freq": _grid_values(args.rebalance_freq, grid_defaults["rebalance_freq"], str),
    }
    total_combinations = 1
    for values in grid.values():
        total_combinations *= len(values)
    logger.info("Optimization grid has %s combinations: %s", total_combinations, grid)
    factor_columns = _requested_factor_columns(args.factor_file, grid["factor_group"], config.get("dynamic_ic_selector", {}))
    if factor_columns is None:
        logger.info("Loading all factor columns.")
    else:
        logger.info("Loading %s factor columns for groups: %s", len(factor_columns), ",".join(grid["factor_group"]))
    factors = load_or_compute_factors(args.start_date, end_date, cache_file=args.factor_file, columns=factor_columns)
    prices = pd.read_parquet(resolve_path(args.price_file))

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    on_result = _progress_writer(output_path)

    ic_weights = None
    if "ic_weighted" in grid["factor_group"] and not args.walk_forward and not args.rolling_ic:
        ic_df = calculate_factor_ic(factors, prices)
        ic_summary = summarize_ic(ic_df)
        ic_weights = make_ic_weights(ic_summary, top_k=args.ic_top_k, min_abs_ic=args.ic_min_abs)
        ic_summary_path = resolve_path("outputs/factor_ic_summary.csv")
        ic_summary_path.parent.mkdir(parents=True, exist_ok=True)
        ic_summary.to_csv(ic_summary_path, encoding="utf-8-sig")

    base_config = {**config["backtest"], **config["strategy"]}
    if args.walk_forward and args.selection_walk_forward:
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
            on_result=on_result,
        )
    elif args.walk_forward:
        results = run_walk_forward_grid_validation(
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
            on_result=on_result,
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
            on_result=on_result,
        )
    results.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("Optimization results saved to %s", output_path)
    logger.info("Top results:\n%s", results.head(10).to_string(index=False))


def _requested_factor_columns(factor_file: str, factor_groups: list[str], dynamic_cfg: dict | None = None) -> list[str] | None:
    groups = {group.strip().lower() for group in factor_groups}
    if not groups or groups.intersection({"all", "ic_weighted"}):
        return None
    available_columns = factor_cache_columns(factor_file)
    if not available_columns:
        return None
    requested: set[str] = set()
    for group in groups:
        if group in DYNAMIC_IC_SELECTOR_GROUPS:
            candidates = (dynamic_cfg or {}).get("candidates", DEFAULT_DYNAMIC_IC_CANDIDATES)
            for candidate in candidates:
                requested.update(str(column) for column in factor_columns_for_method(available_columns, _strip_direction_prefix(str(candidate))))
        else:
            requested.update(str(column) for column in factor_columns_for_method(available_columns, group))
    return sorted(requested) if requested else None


def _strip_direction_prefix(value: str) -> str:
    lowered = value.strip().lower()
    for prefix in ("low_", "inverse_", "short_"):
        if lowered.startswith(prefix):
            return value.strip()[len(prefix) :]
    return value


def _progress_writer(output_path: Path):
    count = 0

    def write_progress(row: dict[str, object], results: pd.DataFrame) -> None:
        nonlocal count
        count += 1
        results.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(
            "Finished optimization row %s: factor_group=%s top_n=%s max_turnover=%s rank_buffer=%s rebalance_freq=%s annual_return=%.4f sharpe=%.4f",
            count,
            row.get("factor_group"),
            row.get("top_n"),
            row.get("max_turnover"),
            row.get("rank_buffer"),
            row.get("rebalance_freq"),
            _number(row.get("annual_return")),
            _number(row.get("sharpe")),
        )

    return write_progress


def _number(value: object) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return 0.0
    return float(parsed)


if __name__ == "__main__":
    main()
