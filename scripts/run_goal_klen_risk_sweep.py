"""Focused fast sweep for the active yearly-return/drawdown goal."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._shared import dated_output_path, yearly_stats
from scripts.run_goal_fast_factor_screen import (
    _filter_scores,
    _filter_scores_by_selection_risk,
    _liquidity_row,
    _parse_liquidity_mode,
    _read_factor_subset,
    _read_price_fields,
    _screen_config,
    _screen_quality_fields,
    _screen_yearly_quality_fields,
    _selection_risk_price_fields,
    _sorted,
)
from src.config_loader import load_config, resolve_path
from src.fast_monthly_backtest import prepare_fast_period_data, run_fast_prepared_backtest
from src.market_regime import defensive_exposure_schedule, detect_market_regime
from src.scoring import build_strategy_scores
from src.strategy import resample_signals
from src.trading_calendar import resolve_target_date_value


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep KLEN-style risk overlays for the active goal.")
    parser.add_argument("--output", default=dated_output_path("goal_klen_risk_sweep"))
    parser.add_argument("--factor", default="KLEN")
    parser.add_argument("--direction", choices=["long_high", "long_low"], default="long_low")
    parser.add_argument("--factors", default="", help="Comma-separated factors to sweep; overrides --factor.")
    parser.add_argument("--directions", default="", help="Comma-separated directions: long_high,long_low; defaults to --direction.")
    parser.add_argument("--top-n", default="7,15,30")
    parser.add_argument("--liquidity-modes", default="high:0.65,high:0.80")
    parser.add_argument("--sideways-exposures", default="0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--bear-exposures", default="0.0,0.1,0.2,0.3,0.5,0.8")
    parser.add_argument("--bull-exposures", default="1.0")
    parser.add_argument("--ma-windows", default="90")
    parser.add_argument("--momentum-windows", default="5")
    parser.add_argument("--bear-drawdowns", default="none")
    parser.add_argument("--lag-days", default="1")
    args = parser.parse_args()

    config = load_config()
    start_date = config["data"]["start_date"]
    end_date = resolve_target_date_value(config["data"]["end_date"], config=config)
    trade_price_field = str(config.get("backtest", {}).get("trade_price_field", "close")).lower()
    liquidity_modes = [_parse_liquidity_mode(value) for value in args.liquidity_modes.split(",") if value.strip()]
    price_fields = {"close", trade_price_field}
    if any(bool(mode.get("enabled", False)) for mode in liquidity_modes):
        price_fields.add(str(config.get("liquidity_filter", {}).get("field", "amount")).lower())
    price_fields.update(_selection_risk_price_fields(config))

    started = time.monotonic()
    prices = _read_price_fields(
        config.get("ic", {}).get("price_file", "data/prices/ohlcv_adjusted.parquet"),
        sorted(price_fields),
        start_date,
        end_date,
    )
    top_ns = _float_list(args.top_n, value_type=int)
    factors_to_run = _csv_strings(args.factors) or [args.factor]
    directions_to_run = _csv_strings(args.directions) or [args.direction]
    bull_exposures = _float_list(args.bull_exposures)
    sideways_exposures = _float_list(args.sideways_exposures)
    bear_exposures = _float_list(args.bear_exposures)
    ma_windows = _float_list(args.ma_windows, value_type=int)
    momentum_windows = _float_list(args.momentum_windows, value_type=int)
    bear_drawdowns = _optional_float_list(args.bear_drawdowns)
    lag_days_values = _float_list(args.lag_days, value_type=int)

    rows: list[dict[str, object]] = []
    out_path = resolve_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    regime_items = []
    for ma_window in ma_windows:
        for momentum_window in momentum_windows:
            for bear_drawdown in bear_drawdowns:
                for lag_days in lag_days_values:
                    regime_config = _regime_config(config, ma_window, momentum_window, bear_drawdown, lag_days)
                    regime_items.append((regime_config, detect_market_regime(prices, regime_config), ma_window, momentum_window, bear_drawdown, lag_days))

    for factor in factors_to_run:
        factors = _read_factor_subset(config["factors"]["cache_file"], [factor], start_date, end_date)
        for direction in directions_to_run:
            factor_group = f"factor:{factor}" if direction == "long_high" else f"inverse_factor:{factor}"
            raw_scores = build_strategy_scores(factors, _screen_config(config, factor_group, {"enabled": False}), price_df=prices)
            for liquidity_mode in liquidity_modes:
                scores = _filter_scores(raw_scores, prices, config, liquidity_mode)
                scores = resample_signals(scores, "monthly")
                scores = _filter_scores_by_selection_risk(scores, prices, config)
                prepared = prepare_fast_period_data(scores, prices, start_date, end_date, trade_price_field=trade_price_field)
                for regime_config, regimes, ma_window, momentum_window, bear_drawdown, lag_days in regime_items:
                    for top_n in top_ns:
                        for bull in bull_exposures:
                            for sideways in sideways_exposures:
                                for bear in bear_exposures:
                                    bt_config = {
                                        **config["backtest"],
                                        "top_n": int(top_n),
                                        "max_turnover": 1,
                                        "rank_buffer": 20,
                                        "rebalance_freq": "monthly",
                                        "rebalance_drift_threshold": 0.02,
                                        "exposure_schedule": _exposure_schedule(regimes, regime_config, prices.index, bull, sideways, bear),
                                    }
                                    result = run_fast_prepared_backtest(prepared, bt_config)
                                    row = {
                                        "factor_group": factor_group,
                                        "factor": factor,
                                        "direction": direction,
                                        **_liquidity_row(liquidity_mode),
                                        "top_n": int(top_n),
                                        "regime_ma_window": int(ma_window),
                                        "regime_momentum_window": int(momentum_window),
                                        "regime_bear_drawdown": "" if bear_drawdown is None else float(bear_drawdown),
                                        "regime_lag_days": int(lag_days),
                                        "bull_exposure": bull,
                                        "sideways_exposure": sideways,
                                        "bear_exposure": bear,
                                        **result.metrics,
                                    }
                                    row.update(_screen_quality_fields(row, config))
                                    row.update(_screen_yearly_quality_fields(yearly_stats(result.equity_curve, bt_config), config))
                                    rows.append(row)
                frame = _sorted(rows)
                frame.to_csv(out_path, index=False, encoding="utf-8-sig")
                best = frame.iloc[0]
                print(
                    f"screened {factor}/{direction}; "
                    f"{len(rows)} variants in {time.monotonic() - started:.1f}s; "
                    f"best annual={best.get('annual_return', 0.0):.4f} "
                    f"dd={best.get('max_drawdown', 0.0):.4f} "
                    f"yearly={best.get('year_ann_pass', 0)}/{best.get('year_dd_pass', 0)}",
                    flush=True,
                )
    print(f"Saved sweep to {out_path}")


def _exposure_schedule(
    regimes: pd.Series,
    config: dict,
    dates: pd.Index,
    bull: float,
    sideways: float,
    bear: float,
) -> pd.Series:
    timing_config = {
        **config,
        "defensive_timing": {
            **config.get("defensive_timing", {}),
            "enabled": True,
            "bull_exposure": float(bull),
            "sideways_exposure": float(sideways),
            "bear_exposure": float(bear),
        },
    }
    return defensive_exposure_schedule(regimes, timing_config, dates)


def _regime_config(
    config: dict,
    ma_window: int,
    momentum_window: int,
    bear_drawdown: float | None,
    lag_days: int,
) -> dict:
    regime_cfg = {
        **config.get("market_regime", {}),
        "ma_window": int(ma_window),
        "momentum_window": int(momentum_window),
        "bear_drawdown_threshold": bear_drawdown,
        "lag_days": int(lag_days),
    }
    return {**config, "market_regime": regime_cfg}


def _float_list(raw_value: str, value_type: type = float) -> list:
    values = [value.strip() for value in raw_value.split(",") if value.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value.")
    return [value_type(value) for value in values]


def _optional_float_list(raw_value: str) -> list[float | None]:
    values = [value.strip().lower() for value in raw_value.split(",") if value.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value.")
    result: list[float | None] = []
    for value in values:
        result.append(None if value in {"none", "null", "na"} else float(value))
    return result


def _csv_strings(raw_value: str) -> list[str]:
    return [value.strip() for value in raw_value.split(",") if value.strip()]


if __name__ == "__main__":
    main()
