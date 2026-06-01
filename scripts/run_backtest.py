from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import run_backtest
from src.config_loader import load_config, resolve_path
from src.factor_calculator import load_or_compute_factors
from src.strategy import composite_factor


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description="Run a lightweight ranking-strategy backtest.")
    parser.add_argument("--start-date", default=config["data"]["start_date"])
    parser.add_argument("--end-date", default=config["data"]["end_date"])
    parser.add_argument("--factor-file", default=config["factors"]["cache_file"])
    parser.add_argument("--price-file", default="data/prices/close.parquet")
    args = parser.parse_args()

    factors = load_or_compute_factors(args.start_date, args.end_date, cache_file=args.factor_file)
    scores = composite_factor(factors, method=config["strategy"].get("factor_group", "momentum"))
    price_file = resolve_path(args.price_file)
    if not price_file.exists():
        raise FileNotFoundError(f"Price file not found: {price_file}. Run scripts/run_convert_data.py first.")
    prices = pd.read_parquet(price_file)

    bt_config = {**config["backtest"], **config["strategy"]}
    result = run_backtest(scores, prices, args.start_date, args.end_date, bt_config)

    out_dir = resolve_path(config["outputs"].get("dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    result.equity_curve.to_csv(out_dir / "backtest_equity.csv", encoding="utf-8-sig")
    result.holdings.to_csv(out_dir / "backtest_holdings.csv", index=False, encoding="utf-8-sig")
    result.trades.to_csv(out_dir / "backtest_trades.csv", index=False, encoding="utf-8-sig")
    (out_dir / "backtest_metrics.json").write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")

    print("Backtest finished.")
    for key, value in result.metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
