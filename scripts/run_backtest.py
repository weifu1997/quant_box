"""模块说明：提供 run_backtest 命令行入口。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import run_backtest
from src.config_loader import load_config, resolve_path
from src.factor_calculator import load_or_compute_factors
from src.market_regime import apply_defensive_timing_to_backtest_config
from src.scoring import build_strategy_scores
from src.selection_constraints import apply_selection_constraints_to_backtest_config
from src.strategy import resample_signals
from src.trading_calendar import resolve_target_date_value
from src.universe_coverage import summarize_universe_coverage
from scripts._shared import requested_factor_columns, strip_direction_prefix

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """函数说明：解析命令行参数并执行主流程。"""
    config = load_config()
    parser = argparse.ArgumentParser(description="Run a lightweight ranking-strategy backtest.")
    parser.add_argument("--start-date", default=config["data"]["start_date"])
    parser.add_argument("--end-date", default=config["data"]["end_date"])
    parser.add_argument("--factor-file", default=config["factors"]["cache_file"])
    parser.add_argument("--price-file", default=config.get("ic", {}).get("price_file", "data/prices/ohlcv_adjusted.parquet"))
    parser.add_argument("--benchmark-file", help="Optional benchmark close parquet/csv for alpha, beta and IR.")
    args = parser.parse_args()
    end_date = resolve_target_date_value(args.end_date, config=config)
    config["data"]["end_date"] = end_date

    price_file = resolve_path(args.price_file)
    if not price_file.exists():
        raise FileNotFoundError(f"Price file not found: {price_file}. Run scripts/run_convert_data.py first.")
    prices = pd.read_parquet(price_file)
    factor_columns = _requested_factor_columns(
        args.factor_file,
        config.get("strategy", {}),
        config.get("dynamic_ic_selector", {}),
        config.get("ml_strategy", {}),
        config.get("regime_score_blend", {}),
        config.get("regime_score_filter", {}),
    )
    if factor_columns is None:
        logger.info("Loading all factor columns.")
    else:
        logger.info("Loading %s factor columns for factor_group=%s.", len(factor_columns), config.get("strategy", {}).get("factor_group"))
    factors = load_or_compute_factors(args.start_date, end_date, cache_file=args.factor_file, columns=factor_columns)
    scores = build_strategy_scores(factors, config, price_df=prices)
    scores = resample_signals(scores, config["strategy"].get("rebalance_freq", "daily"))

    bt_config = apply_defensive_timing_to_backtest_config({**config["backtest"], **config["strategy"]}, prices, config)
    bt_config = apply_selection_constraints_to_backtest_config(bt_config, config)
    if args.benchmark_file:
        benchmark_path = resolve_path(args.benchmark_file)
        if benchmark_path.suffix.lower() == ".csv":
            benchmark = pd.read_csv(benchmark_path, index_col=0).iloc[:, 0]
            benchmark.index = pd.to_datetime(benchmark.index)
        else:
            benchmark_df = pd.read_parquet(benchmark_path)
            benchmark = benchmark_df.iloc[:, 0] if isinstance(benchmark_df, pd.DataFrame) else benchmark_df
            benchmark.index = pd.to_datetime(benchmark.index)
        bt_config["benchmark_curve"] = benchmark
    result = run_backtest(scores, prices, args.start_date, end_date, bt_config)

    out_dir = resolve_path(config["outputs"].get("dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    result.equity_curve.to_csv(out_dir / "backtest_equity.csv", encoding="utf-8-sig")
    result.holdings.to_csv(out_dir / "backtest_holdings.csv", index=False, encoding="utf-8-sig")
    result.trades.to_csv(out_dir / "backtest_trades.csv", index=False, encoding="utf-8-sig")
    (out_dir / "backtest_metrics.json").write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
    coverage = summarize_universe_coverage(config, price_df=prices)
    (out_dir / "universe_coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")

    logger.info("Backtest finished.")
    for key, value in result.metrics.items():
        logger.info("%s: %.6f", key, value)
    logger.info(
        "Universe coverage: %d/%d target symbols in price panel (%.2f%%).",
        coverage["price_target_symbols"],
        coverage["target_symbols"],
        coverage["price_target_coverage"] * 100,
    )


def _requested_factor_columns(
    factor_file: str,
    strategy_cfg: dict,
    dynamic_cfg: dict | None = None,
    ml_cfg: dict | None = None,
    score_blend_cfg: dict | None = None,
    score_filter_cfg: dict | None = None,
) -> list[str] | None:
    """函数说明：处理 requested_factor_columns 的内部辅助逻辑。"""
    return requested_factor_columns(factor_file, strategy_cfg, dynamic_cfg, ml_cfg, score_blend_cfg, score_filter_cfg)


def _strip_direction_prefix(value: str) -> str:
    """函数说明：去除 strip_direction_prefix 的内部辅助逻辑。"""
    return strip_direction_prefix(value)


if __name__ == "__main__":
    main()
