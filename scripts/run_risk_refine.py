from __future__ import annotations

import argparse
from copy import deepcopy
from itertools import product
import logging
from pathlib import Path
import sys
import time
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_backtest import _requested_factor_columns
from src.backtest import run_backtest
from src.config_loader import load_config, resolve_path
from src.factor_calculator import load_or_compute_factors
from src.scoring import build_strategy_scores
from src.strategy import resample_signals
from src.trading_calendar import resolve_target_date_value

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    config = load_config()
    quality = config.get("quality", {})
    parser = argparse.ArgumentParser(
        description="Run a bounded exact risk-parameter refinement around promising liquidity candidates."
    )
    parser.add_argument("--start-date", default=config["data"]["start_date"])
    parser.add_argument("--end-date", default=config["data"]["end_date"])
    parser.add_argument("--factor-file", default=config["factors"]["cache_file"])
    parser.add_argument("--price-file", default=config.get("ic", {}).get("price_file", "data/prices/ohlcv_adjusted.parquet"))
    parser.add_argument("--liquidity-quantiles", default="0.50")
    parser.add_argument("--top-n", default=str(config.get("strategy", {}).get("top_n", 15)))
    parser.add_argument("--rank-buffer", default=str(config.get("strategy", {}).get("rank_buffer", 20)))
    parser.add_argument("--circuit-breaker-drawdown", default="0.10,0.12,0.14")
    parser.add_argument("--cooldown-days", default="10,20,40")
    parser.add_argument("--target-annual-return", type=float, default=quality.get("target_annual_return", 0.20))
    parser.add_argument("--drawdown-limit", type=float, default=quality.get("max_backtest_drawdown_limit", -0.40))
    parser.add_argument("--max-seconds", type=float, default=900.0)
    parser.add_argument("--resume", action="store_true", help="Skip parameter rows already present in the output CSV.")
    parser.add_argument("--output", default="outputs/risk_refine_results.csv")
    args = parser.parse_args()

    start_time = time.monotonic()
    end_date = resolve_target_date_value(args.end_date, config=config)
    prices = pd.read_parquet(resolve_path(args.price_file))
    factor_columns = _requested_factor_columns(args.factor_file, config.get("strategy", {}), config.get("dynamic_ic_selector", {}))
    factors = load_or_compute_factors(args.start_date, end_date, cache_file=args.factor_file, columns=factor_columns)

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_keys(output_path) if args.resume else set()
    rows = _read_existing(output_path) if args.resume else pd.DataFrame()

    combos = list(
        product(
            _csv_values(args.liquidity_quantiles, float),
            _csv_values(args.top_n, int),
            _csv_values(args.rank_buffer, int),
            _csv_values(args.circuit_breaker_drawdown, float),
            _csv_values(args.cooldown_days, int),
        )
    )
    logger.info("Risk refinement grid has %s exact backtest candidates.", len(combos))

    score_cache: dict[float, pd.Series] = {}
    for idx, (liquidity_quantile, top_n, rank_buffer, circuit_drawdown, cooldown_days) in enumerate(combos, start=1):
        key = _combo_key(liquidity_quantile, top_n, rank_buffer, circuit_drawdown, cooldown_days)
        if key in completed:
            logger.info("Skipping completed row %s/%s: %s.", idx, len(combos), key)
            continue
        if args.max_seconds > 0 and time.monotonic() - start_time >= args.max_seconds:
            logger.warning("Stopping before row %s because max-seconds=%.0f was reached.", idx, args.max_seconds)
            break

        logger.info("Running row %s/%s: %s.", idx, len(combos), key)
        scores = score_cache.get(liquidity_quantile)
        if scores is None:
            scoring_config = deepcopy(config)
            scoring_config.setdefault("liquidity_filter", {})
            scoring_config["liquidity_filter"]["quantile"] = liquidity_quantile
            scores = build_strategy_scores(factors, scoring_config, price_df=prices)
            scores = resample_signals(scores, scoring_config["strategy"].get("rebalance_freq", "daily"))
            score_cache[liquidity_quantile] = scores

        bt_config = {**config["backtest"], **config["strategy"]}
        bt_config.update(
            {
                "top_n": top_n,
                "rank_buffer": rank_buffer,
                "circuit_breaker_drawdown": circuit_drawdown,
                "circuit_breaker_cooldown_days": cooldown_days,
            }
        )

        row_start = time.monotonic()
        result = run_backtest(scores, prices, args.start_date, end_date, bt_config)
        row = {
            "liquidity_quantile": liquidity_quantile,
            "top_n": top_n,
            "rank_buffer": rank_buffer,
            "circuit_breaker_drawdown": circuit_drawdown,
            "circuit_breaker_cooldown_days": cooldown_days,
            "seconds": time.monotonic() - row_start,
            **result.metrics,
        }
        row["annual_return_gap"] = float(row.get("annual_return", 0.0)) - args.target_annual_return
        row["drawdown_buffer"] = float(row.get("max_drawdown", 0.0)) - args.drawdown_limit
        row["meets_target"] = bool(row["annual_return_gap"] >= 0 and row["drawdown_buffer"] >= 0)
        rows = pd.concat([rows, pd.DataFrame([row])], ignore_index=True)
        rows.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(
            "Saved row %s: annual_return=%.4f max_drawdown=%.4f sharpe=%.4f meets_target=%s.",
            idx,
            _number(row.get("annual_return")),
            _number(row.get("max_drawdown")),
            _number(row.get("sharpe")),
            row["meets_target"],
        )

    if not rows.empty:
        logger.info("Best rows by target fit:\n%s", _best_rows(rows, args.target_annual_return, args.drawdown_limit).to_string(index=False))
    logger.info("Risk refinement results saved to %s", output_path)


def _csv_values(value: str, cast: Callable[[str], Any]) -> list[Any]:
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def _combo_key(
    liquidity_quantile: float,
    top_n: int,
    rank_buffer: int,
    circuit_drawdown: float,
    cooldown_days: int,
) -> tuple[float, int, int, float, int]:
    return (round(float(liquidity_quantile), 6), int(top_n), int(rank_buffer), round(float(circuit_drawdown), 6), int(cooldown_days))


def _completed_keys(output_path: Path) -> set[tuple[float, int, int, float, int]]:
    frame = _read_existing(output_path)
    if frame.empty:
        return set()
    required = [
        "liquidity_quantile",
        "top_n",
        "rank_buffer",
        "circuit_breaker_drawdown",
        "circuit_breaker_cooldown_days",
    ]
    if any(column not in frame.columns for column in required):
        return set()
    return {
        _combo_key(
            row["liquidity_quantile"],
            row["top_n"],
            row["rank_buffer"],
            row["circuit_breaker_drawdown"],
            row["circuit_breaker_cooldown_days"],
        )
        for _, row in frame.iterrows()
    }


def _read_existing(output_path: Path) -> pd.DataFrame:
    if not output_path.exists():
        return pd.DataFrame()
    return pd.read_csv(output_path)


def _best_rows(rows: pd.DataFrame, target_annual_return: float, drawdown_limit: float) -> pd.DataFrame:
    frame = rows.copy()
    for column in ["annual_return", "max_drawdown", "sharpe", "calmar"]:
        frame[column] = pd.to_numeric(frame.get(column, 0.0), errors="coerce").fillna(0.0)
    frame["target_distance"] = (frame["annual_return"] - target_annual_return).clip(upper=0).abs()
    frame["drawdown_shortfall"] = (drawdown_limit - frame["max_drawdown"]).clip(lower=0)
    return frame.sort_values(
        ["meets_target", "drawdown_shortfall", "target_distance", "max_drawdown", "sharpe"],
        ascending=[False, True, True, False, False],
    ).head(10)


def _number(value: object) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return 0.0
    return float(parsed)


if __name__ == "__main__":
    main()
