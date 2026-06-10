"""Run a formal backtest from precomputed monthly selector weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._shared import yearly_stats
from scripts.run_goal_audit import audit_yearly_goal, goal_thresholds, write_audit_outputs
from src.backtest import run_backtest
from src.config_loader import load_config, resolve_path
from src.factor_calculator import load_or_compute_factors
from src.market_regime import apply_defensive_timing_to_backtest_config
from src.scoring import _apply_liquidity_filter
from src.selection_constraints import apply_selection_constraints_to_backtest_config
from src.strategy import composite_factor, resample_signals
from src.trading_calendar import resolve_target_date_value


SELECTOR_WEIGHT_MAP: dict[str, tuple[str, float]] = {
    "w_roc60": ("ROC60", 1.0),
    "w_roc30": ("ROC30", 1.0),
    "w_beta20": ("BETA20", 1.0),
    "w_beta60": ("BETA60", 1.0),
    "w_rsqr20": ("RSQR20", 1.0),
    "w_low0": ("LOW0", 1.0),
    "w_klen_inv": ("KLEN", -1.0),
    "w_max60_inv": ("MAX60", -1.0),
    "w_db_turnover_f_inv": ("DB_turnover_rate_f", -1.0),
    "w_db_total_mv_inv": ("DB_total_mv", -1.0),
    "w_db_circ_mv_inv": ("DB_circ_mv", -1.0),
    "w_db_pb_inv": ("DB_pb", -1.0),
    "w_db_dv_ttm": ("DB_dv_ttm", 1.0),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal backtest from precomputed selector weight CSV.")
    parser.add_argument("--selector-file", required=True)
    parser.add_argument("--factor-file", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--max-turnover", type=int, default=1)
    parser.add_argument("--rank-buffer", type=int, default=20)
    parser.add_argument("--rebalance-freq", default="monthly")
    parser.add_argument("--min-obs", type=int, default=5)
    parser.add_argument("--no-defensive-timing", action="store_true")
    args = parser.parse_args()

    config = load_config()
    start_date = args.start_date or config["data"]["start_date"]
    end_date = resolve_target_date_value(args.end_date or config["data"]["end_date"], config=config)
    output_prefix = resolve_path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    selector = pd.read_csv(resolve_path(args.selector_file))
    weights = selector_weights_from_frame(selector)
    columns = sorted({column for series in weights.values() for column in series.index})
    if not columns:
        raise ValueError("selector weights did not reference any known factor columns")

    factors = load_or_compute_factors(start_date, end_date, cache_file=args.factor_file, columns=columns)
    signed_factors = apply_selector_directions(factors, weights)
    prices = pd.read_parquet(resolve_path(config.get("ic", {}).get("price_file", "data/prices/ohlcv_adjusted.parquet")))
    scores = composite_factor(
        signed_factors,
        method="ic_weighted",
        factor_weights_dynamic=weights,
        min_obs=args.min_obs,
    )
    scores = _apply_liquidity_filter(scores, prices, config.get("liquidity_filter", {}))
    scores = resample_signals(scores, args.rebalance_freq)

    bt_config = {
        **config.get("backtest", {}),
        **config.get("strategy", {}),
        "top_n": args.top_n,
        "max_turnover": args.max_turnover,
        "rank_buffer": args.rank_buffer,
        "rebalance_freq": args.rebalance_freq,
    }
    timing_config = dict(config)
    if args.no_defensive_timing:
        timing_config.setdefault("defensive_timing", {})["enabled"] = False
    bt_config = apply_defensive_timing_to_backtest_config(bt_config, prices, timing_config)
    bt_config = apply_selection_constraints_to_backtest_config(bt_config, config)

    result = run_backtest(scores, prices, start_date, end_date, bt_config)
    yearly = yearly_stats(result.equity_curve, bt_config)
    return_target, drawdown_limit = goal_thresholds(config)
    audited_yearly, audit_summary = audit_yearly_goal(
        yearly,
        return_target=return_target,
        drawdown_limit=drawdown_limit,
    )

    result.equity_curve.to_csv(Path(str(output_prefix) + "_equity.csv"), encoding="utf-8-sig")
    result.holdings.to_csv(Path(str(output_prefix) + "_holdings.csv"), index=False, encoding="utf-8-sig")
    result.trades.to_csv(Path(str(output_prefix) + "_trades.csv"), index=False, encoding="utf-8-sig")
    audited_yearly.to_csv(Path(str(output_prefix) + "_years.csv"), index=False, encoding="utf-8-sig")
    selector.to_csv(Path(str(output_prefix) + "_selector.csv"), index=False, encoding="utf-8-sig")

    payload = {
        "metrics": result.metrics,
        "audit": audit_summary,
        "selector_file": str(resolve_path(args.selector_file)),
        "factor_file": str(resolve_path(args.factor_file)),
        "weight_dates": len(weights),
        "factor_columns": columns,
    }
    Path(str(output_prefix) + "_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    write_audit_outputs(
        output_prefix=Path(str(output_prefix) + "_audit"),
        yearly=audited_yearly,
        summary=audit_summary,
        metrics=result.metrics,
    )
    print(
        f"selector_weight annual={result.metrics.get('annual_return', 0.0):.4f} "
        f"dd={result.metrics.get('max_drawdown', 0.0):.4f} "
        f"yearly={audit_summary['year_return_pass_count']}/{audit_summary['year_drawdown_pass_count']} "
        f"goal={audit_summary['is_goal_met']}"
    )
    print(f"wrote prefix: {output_prefix}")


def selector_weights_from_frame(frame: pd.DataFrame) -> dict[pd.Timestamp, pd.Series]:
    if "date" not in frame.columns:
        raise ValueError("selector file must include a date column")
    result: dict[pd.Timestamp, pd.Series] = {}
    for _, row in frame.iterrows():
        date = pd.Timestamp(row.get("date")).normalize()
        weights: dict[str, float] = {}
        for weight_column, (factor_column, _direction) in SELECTOR_WEIGHT_MAP.items():
            if weight_column not in frame.columns:
                continue
            value = pd.to_numeric(pd.Series([row.get(weight_column)]), errors="coerce").iloc[0]
            if pd.notna(value) and float(value) != 0.0:
                weights[factor_column] = float(value)
        if weights:
            series = pd.Series(weights, dtype=float)
            total = float(series.abs().sum())
            if total > 0:
                result[date] = series / total
    return result


def apply_selector_directions(factors: pd.DataFrame, weights: dict[pd.Timestamp, pd.Series]) -> pd.DataFrame:
    columns = sorted({column for series in weights.values() for column in series.index})
    signed = factors[columns].copy()
    directions = {
        factor_column: direction
        for _weight_column, (factor_column, direction) in SELECTOR_WEIGHT_MAP.items()
        if factor_column in signed.columns
    }
    for column, direction in directions.items():
        signed[column] = pd.to_numeric(signed[column], errors="coerce").astype("float32") * float(direction)
    return signed


if __name__ == "__main__":
    main()
