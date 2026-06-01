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
from src.strategy import composite_factor, resample_signals

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description="Run a lightweight ranking-strategy backtest.")
    parser.add_argument("--start-date", default=config["data"]["start_date"])
    parser.add_argument("--end-date", default=config["data"]["end_date"])
    parser.add_argument("--factor-file", default=config["factors"]["cache_file"])
    parser.add_argument("--price-file", default="data/prices/ohlcv.parquet")
    parser.add_argument("--benchmark-file", help="Optional benchmark close parquet/csv for alpha, beta and IR.")
    args = parser.parse_args()

    factors = load_or_compute_factors(args.start_date, args.end_date, cache_file=args.factor_file)
    scores = composite_factor(factors, method=config["strategy"].get("factor_group", "momentum"))
    scores = resample_signals(scores, config["strategy"].get("rebalance_freq", "daily"))
    price_file = resolve_path(args.price_file)
    if not price_file.exists():
        raise FileNotFoundError(f"Price file not found: {price_file}. Run scripts/run_convert_data.py first.")
    prices = pd.read_parquet(price_file)

    bt_config = {**config["backtest"], **config["strategy"]}
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
    result = run_backtest(scores, prices, args.start_date, args.end_date, bt_config)

    out_dir = resolve_path(config["outputs"].get("dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    result.equity_curve.to_csv(out_dir / "backtest_equity.csv", encoding="utf-8-sig")
    result.holdings.to_csv(out_dir / "backtest_holdings.csv", index=False, encoding="utf-8-sig")
    result.trades.to_csv(out_dir / "backtest_trades.csv", index=False, encoding="utf-8-sig")
    (out_dir / "backtest_metrics.json").write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")

    logger.info("Backtest finished.")
    for key, value in result.metrics.items():
        logger.info("%s: %.6f", key, value)


if __name__ == "__main__":
    main()
