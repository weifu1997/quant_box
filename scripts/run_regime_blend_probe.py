from __future__ import annotations

import argparse
from copy import deepcopy
import logging
from pathlib import Path
import sys
from typing import Any

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
    parser = argparse.ArgumentParser(description="Run a fast approximate probe for regime score-blend weights.")
    parser.add_argument("--start-date", default=config["data"].get("start_date", "2015-01-01"))
    parser.add_argument("--end-date", default=config["data"].get("end_date", "auto"))
    parser.add_argument("--factor-file", default=config["factors"]["cache_file"])
    parser.add_argument("--price-file", default=config.get("ic", {}).get("price_file", "data/prices/ohlcv_adjusted.parquet"))
    parser.add_argument("--selected-params", default="outputs/auto_selected_params.json")
    parser.add_argument("--bull-defensive-weights", default="0.0")
    parser.add_argument("--sideways-defensive-weights", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--bear-defensive-weights", default="0.5,0.75,1.0")
    parser.add_argument("--liquidity-sides", default=str(config.get("liquidity_filter", {}).get("side", "high")))
    parser.add_argument("--liquidity-quantiles", default=str(config.get("liquidity_filter", {}).get("quantile", 0.20)))
    parser.add_argument("--bear-drawdown-thresholds", default=str(config.get("market_regime", {}).get("bear_drawdown_threshold", "none")))
    parser.add_argument("--sideways-exposures", default=str(config.get("defensive_timing", {}).get("sideways_exposure", 0.60)))
    parser.add_argument("--bear-exposures", default=str(config.get("defensive_timing", {}).get("bear_exposure", 0.30)))
    parser.add_argument("--rebalance-drift-thresholds", default=str(config.get("strategy", {}).get("rebalance_drift_threshold", 0.0)))
    parser.add_argument(
        "--symbol-source",
        choices=["auto_trades", "all"],
        default="auto_trades",
        help="auto_trades limits the probe to instruments already seen in auto backtest trades/holdings.",
    )
    parser.add_argument("--max-symbols", type=int, default=700)
    parser.add_argument("--output", default="outputs/regime_blend_fast_probe.csv")
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

    rows: list[dict[str, Any]] = []
    score_cache: dict[tuple[Any, ...], pd.Series] = {}
    for liquidity_side in _csv_values(args.liquidity_sides, str):
        for liquidity_quantile in _csv_values(args.liquidity_quantiles, float):
            for bear_drawdown_threshold in _csv_optional_values(args.bear_drawdown_thresholds, float):
                for bull_weight in _csv_values(args.bull_defensive_weights, float):
                    for sideways_weight in _csv_values(args.sideways_defensive_weights, float):
                        for bear_weight in _csv_values(args.bear_defensive_weights, float):
                            run_config = _with_probe_overrides(
                                config,
                                liquidity_side=liquidity_side,
                                liquidity_quantile=liquidity_quantile,
                                bear_drawdown_threshold=bear_drawdown_threshold,
                                bull_defensive_weight=bull_weight,
                                sideways_defensive_weight=sideways_weight,
                                bear_defensive_weight=bear_weight,
                            )
                            score_key = _score_key(
                                liquidity_side,
                                liquidity_quantile,
                                bear_drawdown_threshold,
                                bull_weight,
                                sideways_weight,
                                bear_weight,
                            )
                            scores = score_cache.get(score_key)
                            if scores is None:
                                scores = build_strategy_scores(factors, run_config, price_df=prices)
                                scores = resample_signals(scores, run_config["strategy"].get("rebalance_freq", "daily"))
                                score_cache[score_key] = scores
                            prepared = prepare_fast_period_data(scores, prices, args.start_date, end_date)
                            for sideways_exposure in _csv_values(args.sideways_exposures, float):
                                for bear_exposure in _csv_values(args.bear_exposures, float):
                                    for drift_threshold in _csv_values(args.rebalance_drift_thresholds, float):
                                        bt_config = apply_defensive_timing_to_backtest_config(
                                            {**run_config["backtest"], **run_config["strategy"]},
                                            prices,
                                            _with_timing_probe(run_config, sideways_exposure, bear_exposure),
                                        )
                                        bt_config["rebalance_drift_threshold"] = drift_threshold
                                        bt_config = apply_selection_constraints_to_backtest_config(bt_config, run_config)
                                        result = run_fast_prepared_backtest(prepared, bt_config)
                                        row = {
                                            "approximate": True,
                                            "start_date": args.start_date,
                                            "end_date": end_date,
                                            "symbol_source": args.symbol_source,
                                            "symbol_count": len(symbols) if symbols else 0,
                                            "periods": len(prepared.periods),
                                            "liquidity_side": str(liquidity_side).strip().lower(),
                                            "liquidity_quantile": liquidity_quantile,
                                            "bear_drawdown_threshold": bear_drawdown_threshold,
                                            "bull_defensive_weight": bull_weight,
                                            "sideways_defensive_weight": sideways_weight,
                                            "bear_defensive_weight": bear_weight,
                                            "sideways_exposure": sideways_exposure,
                                            "bear_exposure": bear_exposure,
                                            "rebalance_drift_threshold": drift_threshold,
                                            **result.metrics,
                                        }
                                        rows.append(row)
                                        logger.info(
                                            "liq=%s/%s blend=(%.2f,%.2f,%.2f) exp=(%.2f,%.2f) drift=%.3f "
                                            "annual=%.4f dd=%.4f turnover=%.4f",
                                            row["liquidity_side"],
                                            liquidity_quantile,
                                            bull_weight,
                                            sideways_weight,
                                            bear_weight,
                                            sideways_exposure,
                                            bear_exposure,
                                            drift_threshold,
                                            float(row.get("annual_return", 0.0)),
                                            float(row.get("max_drawdown", 0.0)),
                                            float(row.get("annual_weight_turnover", 0.0)),
                                        )

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("Fast regime-blend probe saved to %s", output_path)


def _with_probe_overrides(
    config: dict[str, Any],
    *,
    liquidity_side: str,
    liquidity_quantile: float,
    bear_drawdown_threshold: float | None,
    bull_defensive_weight: float,
    sideways_defensive_weight: float,
    bear_defensive_weight: float,
) -> dict[str, Any]:
    result = deepcopy(config)
    result.setdefault("liquidity_filter", {})
    result["liquidity_filter"]["enabled"] = True
    result["liquidity_filter"]["side"] = str(liquidity_side).strip().lower()
    result["liquidity_filter"]["quantile"] = float(liquidity_quantile)
    result.setdefault("market_regime", {})["bear_drawdown_threshold"] = bear_drawdown_threshold
    result.setdefault("regime_score_blend", {})["enabled"] = True
    result["regime_score_blend"]["bull_defensive_weight"] = float(bull_defensive_weight)
    result["regime_score_blend"]["sideways_defensive_weight"] = float(sideways_defensive_weight)
    result["regime_score_blend"]["bear_defensive_weight"] = float(bear_defensive_weight)
    return result


def _with_timing_probe(config: dict[str, Any], sideways_exposure: float, bear_exposure: float) -> dict[str, Any]:
    result = deepcopy(config)
    result.setdefault("defensive_timing", {})
    result["defensive_timing"]["sideways_exposure"] = float(sideways_exposure)
    result["defensive_timing"]["bear_exposure"] = float(bear_exposure)
    return result


def _score_key(
    liquidity_side: str,
    liquidity_quantile: float,
    bear_drawdown_threshold: float | None,
    bull_weight: float,
    sideways_weight: float,
    bear_weight: float,
) -> tuple[Any, ...]:
    return (
        str(liquidity_side).strip().lower(),
        round(float(liquidity_quantile), 6),
        _optional_key(bear_drawdown_threshold),
        round(float(bull_weight), 6),
        round(float(sideways_weight), 6),
        round(float(bear_weight), 6),
    )


def _csv_values(value: str, cast):
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def _csv_optional_values(value: str, cast):
    values: list[Any] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower() in {"none", "null", "off"}:
            values.append(None)
        else:
            values.append(cast(item))
    return values


def _optional_key(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 6)


if __name__ == "__main__":
    main()
