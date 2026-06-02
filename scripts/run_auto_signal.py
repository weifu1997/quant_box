from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.auto_tuning import apply_strategy_params, select_stable_params, summarize_parameter_validation
from src.backtest import run_backtest
from src.config_loader import load_config, resolve_path
from src.data_converter import convert_to_qlib_format
from src.data_fetcher import update_daily_data_resumable
from src.factor_calculator import load_or_compute_factors
from src.optimizer import DEFAULT_GRID, run_walk_forward_grid_validation
from src.scoring import build_strategy_scores
from src.signal_generator import generate_signal, read_previous_holdings, save_signal
from src.strategy import resample_signals
from src.universe_coverage import summarize_universe_coverage

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def _csv_values(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description="Run data refresh, automatic walk-forward tuning, backtest and latest signal.")
    parser.add_argument("--skip-update", action="store_true", help="Skip Tushare data update.")
    parser.add_argument("--skip-convert", action="store_true", help="Skip raw-to-price conversion.")
    parser.add_argument("--skip-factor", action="store_true", help="Skip factor recomputation when cache is valid.")
    parser.add_argument("--skip-optimize", action="store_true", help="Use current config strategy instead of automatic tuning.")
    parser.add_argument("--start-date", default=config["data"]["start_date"])
    parser.add_argument("--end-date", default=config["data"]["end_date"])
    parser.add_argument("--date", default="latest", help="Signal date, YYYY-MM-DD, or latest.")
    parser.add_argument("--chunk-size", type=int, default=config["data"].get("update_chunk_size", 20))
    parser.add_argument("--sleep-seconds", type=float, default=config["data"].get("update_sleep_seconds", 90))
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--factor-groups", default="ic_weighted,momentum")
    parser.add_argument("--top-n", default="5,7,10")
    parser.add_argument("--max-turnover", default="1")
    parser.add_argument("--rank-buffer", default="10,20")
    parser.add_argument("--rebalance-freq", default="weekly,monthly")
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--test-months", type=int, default=12)
    parser.add_argument("--step-months", type=int, default=6)
    parser.add_argument("--turnover-penalty", type=float, default=0.02)
    parser.add_argument("--cost-penalty", type=float, default=1.0)
    args = parser.parse_args()

    if not args.skip_update:
        logger.info("Updating raw stock data, including existing files.")
        update_daily_data_resumable(
            start_date=args.start_date,
            end_date=args.end_date,
            chunk_size=args.chunk_size,
            sleep_seconds=args.sleep_seconds,
            max_chunks=args.max_chunks,
            include_existing=True,
        )

    if not args.skip_convert:
        logger.info("Converting raw data to Qlib provider and price panels.")
        convert_to_qlib_format()

    factor_file = config["factors"]["cache_file"]
    logger.info("Loading or computing factors.")
    factors = load_or_compute_factors(args.start_date, args.end_date, cache_file=factor_file, force=not args.skip_factor)

    price_path = resolve_path(config.get("ic", {}).get("price_file", "data/prices/ohlcv.parquet"))
    if not price_path.exists():
        raise FileNotFoundError(f"Price file not found: {price_path}. Run conversion first.")
    prices = pd.read_parquet(price_path)

    out_dir = resolve_path(config["outputs"].get("dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_config = config
    selected_params: dict[str, Any] = dict(config.get("strategy", {}))
    validation = pd.DataFrame()
    summary = pd.DataFrame()
    if not args.skip_optimize:
        grid = {
            **DEFAULT_GRID,
            "factor_group": _csv_values(args.factor_groups, str),
            "top_n": _csv_values(args.top_n, int),
            "max_turnover": _csv_values(args.max_turnover, int),
            "rank_buffer": _csv_values(args.rank_buffer, int),
            "rebalance_freq": _csv_values(args.rebalance_freq, str),
        }
        logger.info("Running automatic walk-forward grid validation.")
        validation = run_walk_forward_grid_validation(
            factors,
            prices,
            base_config={**config["backtest"], **config["strategy"]},
            start_date=args.start_date,
            end_date=args.end_date,
            grid=grid,
            train_years=args.train_years,
            test_months=args.test_months,
            step_months=args.step_months,
            turnover_penalty=args.turnover_penalty,
            cost_penalty=args.cost_penalty,
            use_rolling_ic=True,
            ic_window=int(config.get("ic", {}).get("window", 252)),
            ic_min_periods=int(config.get("ic", {}).get("min_periods", 60)),
            ic_min_abs=float(config.get("ic", {}).get("min_abs_ic", 0.02)),
            ic_corr_threshold=float(config.get("ic", {}).get("corr_threshold", 0.7)),
            ic_top_k=int(config.get("ic", {}).get("top_k", 30)),
            ic_weight_smoothing=float(config.get("ic", {}).get("weight_smoothing", 0.0)),
            ic_max_weight_turnover=config.get("ic", {}).get("max_weight_turnover"),
        )
        validation.to_csv(out_dir / "auto_validation_windows.csv", index=False, encoding="utf-8-sig")
        summary = summarize_parameter_validation(validation)
        summary.to_csv(out_dir / "auto_parameter_summary.csv", index=False, encoding="utf-8-sig")
        selected_params = select_stable_params(summary)
        selected_config = apply_strategy_params(config, selected_params)

    logger.info("Selected strategy params: %s", selected_params)
    scores = build_strategy_scores(factors, selected_config, price_df=prices)
    scores = resample_signals(scores, selected_config["strategy"].get("rebalance_freq", "daily"))
    result = run_backtest(
        scores,
        prices,
        args.start_date,
        args.end_date,
        {**selected_config["backtest"], **selected_config["strategy"]},
    )
    result.equity_curve.to_csv(out_dir / "auto_backtest_equity.csv", encoding="utf-8-sig")
    result.holdings.to_csv(out_dir / "auto_backtest_holdings.csv", index=False, encoding="utf-8-sig")
    result.trades.to_csv(out_dir / "auto_backtest_trades.csv", index=False, encoding="utf-8-sig")

    previous = read_previous_holdings(selected_config["outputs"]["holdings_file"])
    signal_df, holdings = generate_signal(args.date, previous_holdings=previous, factor_file=factor_file, config=selected_config)
    output_date = signal_df["date"].iloc[0] if args.date.lower() == "latest" and not signal_df.empty else args.date
    signal_path, holdings_path = save_signal(signal_df, holdings, output_date, config=selected_config)

    coverage = summarize_universe_coverage(selected_config, price_df=prices)
    report = {
        "selected_params": selected_params,
        "backtest_metrics": result.metrics,
        "universe_coverage": coverage,
        "signal_path": str(signal_path),
        "holdings_path": str(holdings_path),
        "validation_windows": int(len(validation)),
        "validation_param_sets": int(len(summary)),
        "signal_date": str(output_date),
    }
    (out_dir / "auto_selected_params.json").write_text(json.dumps(selected_params, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "auto_backtest_metrics.json").write_text(json.dumps(result.metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "auto_signal_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Auto signal saved to %s", signal_path)
    logger.info("Auto report saved to %s", out_dir / "auto_signal_report.json")


if __name__ == "__main__":
    main()
