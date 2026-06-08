from __future__ import annotations

import argparse
from copy import deepcopy
import logging
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._shared import (
    probe_factor_columns,
    probe_symbols,
    read_factor_subset,
    read_price_subset,
    read_selected_params,
)
from src.auto_tuning import apply_strategy_params
from src.config_loader import load_config, resolve_path
from src.fast_monthly_backtest import prepare_fast_period_data, run_fast_prepared_backtest
from src.market_regime import apply_defensive_timing_to_backtest_config
from src.scoring import build_strategy_scores
from src.selection_constraints import apply_selection_constraints_to_backtest_config
from src.strategy import resample_signals
from src.trading_calendar import resolve_target_date_value

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description="Run a fast approximate probe for rebalance drift thresholds.")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default=config["data"].get("end_date", "auto"))
    parser.add_argument("--thresholds", default="0.0,0.02,0.05")
    parser.add_argument("--factor-file", default=config["factors"]["cache_file"])
    parser.add_argument("--price-file", default=config.get("ic", {}).get("price_file", "data/prices/ohlcv_adjusted.parquet"))
    parser.add_argument("--selected-params", default="outputs/auto_selected_params.json")
    parser.add_argument(
        "--symbol-source",
        choices=["auto_trades", "all"],
        default="auto_trades",
        help="auto_trades limits the probe to instruments already seen in auto backtest trades/holdings.",
    )
    parser.add_argument("--max-symbols", type=int, default=500)
    parser.add_argument("--output", default="outputs/rebalance_drift_fast_probe.csv")
    args = parser.parse_args()

    end_date = resolve_target_date_value(args.end_date, config=config)
    selected = read_selected_params(args.selected_params)
    if selected:
        config = apply_strategy_params(config, selected)
    symbols = probe_symbols(args.symbol_source, args.max_symbols)
    factor_columns = probe_factor_columns(args.factor_file, config)
    logger.info("Loading factor columns: %s", factor_columns or "all")
    factors = read_factor_subset(args.factor_file, factor_columns, args.start_date, end_date, symbols)
    if symbols:
        symbols = sorted(set(factors.index.get_level_values(1).astype(str)))
    logger.info("Probe factor shape: %s over %s symbols.", factors.shape, len(symbols) if symbols else "all")

    fields = ["close"]
    if bool(config.get("liquidity_filter", {}).get("enabled", False)):
        fields.append(str(config.get("liquidity_filter", {}).get("field", "amount")).lower())
    prices = read_price_subset(args.price_file, fields, symbols, args.start_date, end_date)
    logger.info("Probe price shape: %s.", prices.shape)

    scores = build_strategy_scores(factors, config, price_df=prices)
    scores = resample_signals(scores, config["strategy"].get("rebalance_freq", "daily"))
    base_bt_config = apply_defensive_timing_to_backtest_config({**config["backtest"], **config["strategy"]}, prices, config)
    base_bt_config = apply_selection_constraints_to_backtest_config(base_bt_config, config)
    prepared = prepare_fast_period_data(scores, prices, args.start_date, end_date)

    rows: list[dict[str, object]] = []
    for threshold in _csv_floats(args.thresholds):
        bt_config = deepcopy(base_bt_config)
        bt_config["rebalance_drift_threshold"] = threshold
        result = run_fast_prepared_backtest(prepared, bt_config)
        row = {
            "rebalance_drift_threshold": threshold,
            "approximate": True,
            "start_date": args.start_date,
            "end_date": end_date,
            "symbol_source": args.symbol_source,
            "symbol_count": len(symbols) if symbols else 0,
            "periods": len(prepared.periods),
            **result.metrics,
        }
        rows.append(row)
        logger.info(
            "threshold=%.4f annual_return=%.4f max_drawdown=%.4f annual_weight_turnover=%.4f",
            threshold,
            float(row.get("annual_return", 0.0)),
            float(row.get("max_drawdown", 0.0)),
            float(row.get("annual_weight_turnover", 0.0)),
        )

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("Fast drift probe saved to %s", output_path)


def _read_selected_params(path_value: str | Path) -> dict[str, object]:
    return read_selected_params(path_value)


def _probe_symbols(source: str, max_symbols: int) -> list[str]:
    return probe_symbols(source, max_symbols)


def _probe_factor_columns(factor_file: str | Path, config: dict) -> list[str] | None:
    return probe_factor_columns(factor_file, config)


def _read_factor_subset(
    factor_file: str | Path,
    factor_columns: list[str] | None,
    start_date: str,
    end_date: str,
    symbols: list[str],
) -> pd.DataFrame:
    return read_factor_subset(factor_file, factor_columns, start_date, end_date, symbols)


def _read_price_subset(
    price_file: str | Path,
    fields: Iterable[str],
    symbols: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    return read_price_subset(price_file, fields, symbols, start_date, end_date)


def _csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


if __name__ == "__main__":
    main()
