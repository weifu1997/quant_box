from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_backtest import _requested_factor_columns
from src.backtest import run_backtest
from src.config_loader import load_config, resolve_path
from src.factor_calculator import load_or_compute_factors
from src.market_regime import apply_defensive_timing_to_backtest_config
from src.research_diagnostics import build_research_diagnostics, write_research_diagnostics
from src.scoring import build_strategy_scores
from src.selection_constraints import apply_selection_constraints_to_backtest_config
from src.strategy import resample_signals
from src.trading_calendar import resolve_target_date_value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run selected full-history formal candidates for the active goal.")
    parser.add_argument("--output", default="outputs/goal_formal_candidate_summary_20260606.csv")
    parser.add_argument("--start-index", type=int, default=1, help="1-based candidate index to start from.")
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--skip-diagnostics", action="store_true", help="Skip candidate-level research diagnostics.")
    parser.add_argument("--resume", action="store_true", help="Skip candidates already present in the output CSV.")
    args = parser.parse_args()

    config = load_config()
    start_date = config["data"]["start_date"]
    end_date = resolve_target_date_value(config["data"]["end_date"], config=config)
    prices = pd.read_parquet(resolve_path(config.get("ic", {}).get("price_file", "data/prices/ohlcv_adjusted.parquet")))

    out_path = resolve_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = _candidate_specs()
    candidates = candidates[max(0, args.start_index - 1) :]
    if args.max_candidates:
        candidates = candidates[: args.max_candidates]

    factor_cache: dict[tuple[str, ...] | None, pd.DataFrame] = {}
    score_cache: dict[str, pd.Series] = {}
    rows, completed_candidates = _load_existing_candidate_rows(out_path) if args.resume else ([], set())
    for idx, candidate in enumerate(candidates, start=1):
        if candidate["name"] in completed_candidates:
            print(f"{idx}/{len(candidates)} {candidate['name']}: skipped existing result", flush=True)
            continue
        started = time.monotonic()
        strategy = _strategy_config(config, candidate)
        factor_config = _scoring_config(config, strategy, candidate)
        factor_columns = _requested_factor_columns(
            config["factors"]["cache_file"],
            strategy,
            factor_config.get("dynamic_ic_selector", {}),
            factor_config.get("ml_strategy", {}),
            factor_config.get("regime_score_blend", {}),
            factor_config.get("regime_score_filter", {}),
        )
        factor_key = None if factor_columns is None else tuple(sorted(factor_columns))
        if factor_key not in factor_cache:
            factor_cache[factor_key] = load_or_compute_factors(
                start_date,
                end_date,
                cache_file=config["factors"]["cache_file"],
                columns=factor_columns,
            )
        scoring_key = _score_key(candidate)
        if scoring_key not in score_cache:
            scoring_config = factor_config
            scores = build_strategy_scores(factor_cache[factor_key], scoring_config, price_df=prices)
            score_cache[scoring_key] = resample_signals(scores, strategy.get("rebalance_freq", "daily"))

        timing_config = _timing_config(config, candidate)
        bt_config = apply_defensive_timing_to_backtest_config({**config["backtest"], **strategy}, prices, timing_config)
        for key, value in candidate.get("backtest", {}).items():
            if value == "__remove__":
                bt_config.pop(key, None)
            else:
                bt_config[key] = value
        bt_config = apply_selection_constraints_to_backtest_config(bt_config, config)

        result = run_backtest(score_cache[scoring_key], prices, start_date, end_date, bt_config)
        yearly = _yearly_stats(result.equity_curve)
        row = {
            "candidate": candidate["name"],
            "seconds": time.monotonic() - started,
            **candidate.get("recorded_hint", {}),
            **result.metrics,
            "year_ann_pass": int((yearly["annual_return"] >= 0.20).sum()) if not yearly.empty else 0,
            "year_dd_pass": int((yearly["max_drawdown"] >= -0.20).sum()) if not yearly.empty else 0,
        }
        row.update(_quality_flags(row, config.get("quality", {})))
        prefix = out_path.with_name(f"{out_path.stem}_{candidate['name']}")
        row.update(_write_candidate_artifacts(prefix, result, yearly, prices, config, write_diagnostics=not args.skip_diagnostics))
        Path(str(prefix) + "_metrics.json").write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(
            f"{idx}/{len(candidates)} {candidate['name']}: annual={row['annual_return']:.4f} "
            f"dd={row['max_drawdown']:.4f} acceptable={row['is_acceptable']} yearly={row['year_ann_pass']}/{row['year_dd_pass']}",
            flush=True,
        )

    print(f"Saved summary to {out_path}")


def _load_existing_candidate_rows(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    if not path.exists() or path.stat().st_size == 0:
        return [], set()
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return [], set()
    if frame.empty or "candidate" not in frame.columns:
        return [], set()
    rows = frame.to_dict(orient="records")
    candidates = {str(value) for value in frame["candidate"].dropna().astype(str)}
    return rows, candidates


def _candidate_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "momentum_smoke_targetvol_none",
            "strategy": {"factor_group": "momentum", "top_n": 7, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "recorded_hint": {"hint_annual_return": 0.274169, "hint_max_drawdown": -0.136058},
        },
        {
            "name": "momentum_smoke_targetvol_020",
            "strategy": {"factor_group": "momentum", "top_n": 7, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": 0.20, "max_leverage": 1.0},
            "recorded_hint": {"hint_annual_return": 0.274205, "hint_max_drawdown": -0.135933},
        },
        {
            "name": "ic_weighted_oos_hit",
            "strategy": {"factor_group": "ic_weighted", "top_n": 5, "rank_buffer": 10, "rebalance_freq": "monthly"},
            "recorded_hint": {"hint_annual_return": 0.339701, "hint_max_drawdown": -0.199027},
        },
        {
            "name": "all_weekly_smoke",
            "strategy": {"factor_group": "all", "top_n": 10, "rank_buffer": 0, "rebalance_freq": "weekly"},
            "recorded_hint": {"hint_annual_return": 5.571895, "hint_max_drawdown": -0.125099},
        },
        {
            "name": "dynamic_lowliq_no_timing_take065",
            "strategy": {"factor_group": "dynamic_ic_selector", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 0.65, "circuit_breaker_drawdown": None},
            "recorded_hint": {"hint_annual_return": 0.226552, "hint_max_drawdown": -0.507504},
        },
        {
            "name": "momentum_highliq_no_timing_no_stop",
            "strategy": {"factor_group": "momentum", "top_n": 7, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "high", "quantile": 0.20},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": None, "circuit_breaker_drawdown": None},
        },
        {
            "name": "momentum_noliq_no_timing_no_stop",
            "strategy": {"factor_group": "momentum", "top_n": 7, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": False},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": None, "circuit_breaker_drawdown": None},
        },
        {
            "name": "momentum_lowliq_no_timing_no_stop",
            "strategy": {"factor_group": "momentum", "top_n": 7, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": None, "circuit_breaker_drawdown": None},
        },
        {
            "name": "momentum_lowliq_no_timing_take035",
            "strategy": {"factor_group": "momentum", "top_n": 7, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 0.35, "circuit_breaker_drawdown": None},
        },
        {
            "name": "momentum_lowliq_no_timing_take065",
            "strategy": {"factor_group": "momentum", "top_n": 7, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 0.65, "circuit_breaker_drawdown": None},
        },
        {
            "name": "momentum_lowliq_no_timing_take100",
            "strategy": {"factor_group": "momentum", "top_n": 7, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 1.0, "circuit_breaker_drawdown": None},
        },
        {
            "name": "momentum_lowliq_timing_take035",
            "strategy": {"factor_group": "momentum", "top_n": 7, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "defensive_timing": True,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 0.35, "circuit_breaker_drawdown": None},
        },
        {
            "name": "momentum_lowliq_no_timing_take035_cb12_cash",
            "strategy": {"factor_group": "momentum", "top_n": 7, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 0.35, "circuit_breaker_drawdown": 0.12, "circuit_breaker_cooldown_days": 60, "circuit_breaker_target_exposure": 0.0, "rebalance_drift_threshold": 0.02},
        },
        {
            "name": "momentum_lowliq_no_timing_take035_cb12_30",
            "strategy": {"factor_group": "momentum", "top_n": 7, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 0.35, "circuit_breaker_drawdown": 0.12, "circuit_breaker_cooldown_days": 60, "circuit_breaker_target_exposure": 0.30, "rebalance_drift_threshold": 0.02},
        },
        {
            "name": "momentum_lowliq_no_timing_take035_vol10",
            "strategy": {"factor_group": "momentum", "top_n": 7, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": 0.10, "max_leverage": 1.0},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 0.35, "circuit_breaker_drawdown": None, "rebalance_drift_threshold": 0.02},
        },
        {
            "name": "momentum_lowliq_no_timing_take035_vol15",
            "strategy": {"factor_group": "momentum", "top_n": 7, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": 0.15, "max_leverage": 1.0},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 0.35, "circuit_breaker_drawdown": None, "rebalance_drift_threshold": 0.02},
        },
        {
            "name": "momentum_lowliq_no_timing_take035_vol20",
            "strategy": {"factor_group": "momentum", "top_n": 7, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": 0.20, "max_leverage": 1.0},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 0.35, "circuit_breaker_drawdown": None, "rebalance_drift_threshold": 0.02},
        },
        {
            "name": "momentum_lowliq_no_timing_take035_top15",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 0.35, "circuit_breaker_drawdown": None, "rebalance_drift_threshold": 0.02},
        },
        {
            "name": "momentum_lowliq_no_timing_take035_top20",
            "strategy": {"factor_group": "momentum", "top_n": 20, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 0.35, "circuit_breaker_drawdown": None, "rebalance_drift_threshold": 0.02},
        },
        {
            "name": "momentum_lowliq_no_timing_take035_top30",
            "strategy": {"factor_group": "momentum", "top_n": 30, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 0.35, "circuit_breaker_drawdown": None, "rebalance_drift_threshold": 0.02},
        },
        {
            "name": "momentum_lowliq_q20_no_timing_take035_top15",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.20},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 0.35, "circuit_breaker_drawdown": None, "rebalance_drift_threshold": 0.02},
        },
        {
            "name": "momentum_lowliq_q50_no_timing_take035_top15",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.50},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 0.35, "circuit_breaker_drawdown": None, "rebalance_drift_threshold": 0.02},
        },
        {
            "name": "momentum_lowliq_q50_no_timing_take035_top7",
            "strategy": {"factor_group": "momentum", "top_n": 7, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.50},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 0.35, "circuit_breaker_drawdown": None, "rebalance_drift_threshold": 0.02},
        },
        {
            "name": "momentum_lowliq_no_timing_take035_top15_indcap25",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__", "max_industry_weight": 0.25},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 0.35, "circuit_breaker_drawdown": None, "rebalance_drift_threshold": 0.02},
        },
        {
            "name": "momentum_lowliq_no_timing_take035_top15_indcap20",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__", "max_industry_weight": 0.20},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 0.35, "circuit_breaker_drawdown": None, "rebalance_drift_threshold": 0.02},
        },
        {
            "name": "momentum_lowliq_no_timing_take035_top15_bear_roc_filter",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_filter": {
                "enabled": True,
                "rules": [{"regime": "bear", "components": [{"column": "ROC20", "direction": 1.0}], "min_score": 0.0}],
            },
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 0.35, "circuit_breaker_drawdown": None, "rebalance_drift_threshold": 0.02},
        },
        {
            "name": "momentum_lowliq_no_timing_take035_top15_bear_def_filter",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_filter": {
                "enabled": True,
                "rules": [
                    {
                        "regime": "bear",
                        "components": [
                            {"column": "ROC20", "direction": 1.0},
                            {"column": "STD20", "direction": -1.0},
                            {"column": "BETA20", "direction": -1.0},
                        ],
                        "min_score": 0.0,
                    }
                ],
            },
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 0.35, "circuit_breaker_drawdown": None, "rebalance_drift_threshold": 0.02},
        },
        {
            "name": "momentum_lowliq_take035_top7_dd10_cash",
            "strategy": {"factor_group": "momentum", "top_n": 7, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "defensive_timing": True,
            "market_regime": _drawdown_only_regime(0.10),
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.0},
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 0.35, "circuit_breaker_drawdown": None, "rebalance_drift_threshold": 0.02},
        },
        {
            "name": "momentum_lowliq_take035_top7_dd12_cash",
            "strategy": {"factor_group": "momentum", "top_n": 7, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "defensive_timing": True,
            "market_regime": _drawdown_only_regime(0.12),
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.0},
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 0.35, "circuit_breaker_drawdown": None, "rebalance_drift_threshold": 0.02},
        },
        {
            "name": "momentum_lowliq_take035_top7_dd15_cash",
            "strategy": {"factor_group": "momentum", "top_n": 7, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "defensive_timing": True,
            "market_regime": _drawdown_only_regime(0.15),
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.0},
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 0.35, "circuit_breaker_drawdown": None, "rebalance_drift_threshold": 0.02},
        },
        {
            "name": "momentum_lowliq_take035_top15_dd12_cash",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "defensive_timing": True,
            "market_regime": _drawdown_only_regime(0.12),
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.0},
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 0.35, "circuit_breaker_drawdown": None, "rebalance_drift_threshold": 0.02},
        },
        {
            "name": "klen_lowliq_q20_no_timing_top15_no_stop",
            "strategy": {"factor_group": "inverse_factor:KLEN", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.20},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": None, "circuit_breaker_drawdown": None, "rebalance_drift_threshold": 0.02},
            "recorded_hint": {"hint_annual_return": 0.877824, "hint_max_drawdown": -0.289837, "hint_source": "goal_fast_factor_screen_smoke2_20260607"},
        },
        {
            "name": "klen_lowliq_q20_no_timing_top15_take035",
            "strategy": {"factor_group": "inverse_factor:KLEN", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.20},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 0.35, "circuit_breaker_drawdown": None, "rebalance_drift_threshold": 0.02},
        },
        {
            "name": "beta10_lowliq_q20_no_timing_top15_no_stop",
            "strategy": {"factor_group": "factor:BETA10", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.20},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": None, "circuit_breaker_drawdown": None, "rebalance_drift_threshold": 0.02},
            "recorded_hint": {"hint_annual_return": 0.883235, "hint_max_drawdown": -0.220997, "hint_source": "goal_fast_factor_screen_first30_lowq20_top15_20260607"},
        },
        {
            "name": "beta5_lowliq_q20_no_timing_top15_no_stop",
            "strategy": {"factor_group": "factor:BETA5", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.20},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": None, "circuit_breaker_drawdown": None, "rebalance_drift_threshold": 0.02},
            "recorded_hint": {"hint_annual_return": 0.900879, "hint_max_drawdown": -0.227155, "hint_source": "goal_fast_factor_screen_first30_lowq20_top15_20260607"},
        },
        {
            "name": "kmid_lowliq_q20_no_timing_top15_no_stop",
            "strategy": {"factor_group": "inverse_factor:KMID", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.20},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": None, "circuit_breaker_drawdown": None, "rebalance_drift_threshold": 0.02},
            "recorded_hint": {"hint_annual_return": 0.216744, "hint_max_drawdown": -0.337811, "hint_source": "goal_fast_factor_screen_smoke2_blend_directionfix_20260607"},
        },
        {
            "name": "klen_lowliq_q20_no_blend_top15_no_stop",
            "strategy": {"factor_group": "inverse_factor:KLEN", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.20},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": None, "circuit_breaker_drawdown": None, "rebalance_drift_threshold": 0.02},
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 0.35, "circuit_breaker_drawdown": None, "rebalance_drift_threshold": 0.02},
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top30_take035",
            "strategy": {"factor_group": "momentum", "top_n": 30, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 0.35, "circuit_breaker_drawdown": None, "rebalance_drift_threshold": 0.02},
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top50_take035",
            "strategy": {"factor_group": "momentum", "top_n": 50, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 0.35, "circuit_breaker_drawdown": None, "rebalance_drift_threshold": 0.02},
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_cb12_30",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 0.35, "circuit_breaker_drawdown": 0.12, "circuit_breaker_cooldown_days": 60, "circuit_breaker_target_exposure": 0.30, "rebalance_drift_threshold": 0.02},
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_cb12_cash",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 0.35, "circuit_breaker_drawdown": 0.12, "circuit_breaker_cooldown_days": 60, "circuit_breaker_target_exposure": 0.0, "rebalance_drift_threshold": 0.02},
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_vol15",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": 0.15, "max_leverage": 1.0},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 0.35, "circuit_breaker_drawdown": None, "rebalance_drift_threshold": 0.02},
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_ma_side1_bear02",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": True,
            "market_regime": {"high_volatility_threshold": 999.0, "bear_drawdown_threshold": None},
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.20},
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 0.35, "circuit_breaker_drawdown": None, "rebalance_drift_threshold": 0.02},
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_ma_side1_bear03",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": True,
            "market_regime": {"high_volatility_threshold": 999.0, "bear_drawdown_threshold": None},
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.30},
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 0.35, "circuit_breaker_drawdown": None, "rebalance_drift_threshold": 0.02},
            "recorded_hint": {"hint_annual_return": 0.2103, "hint_max_drawdown": -0.1887, "hint_source": "post_processed_no_blend_equity_ma_only"},
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_ma_side1_bear04",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": True,
            "market_regime": {"high_volatility_threshold": 999.0, "bear_drawdown_threshold": None},
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.40},
            "backtest": {"stop_loss_pct": None, "take_profit_pct": 0.35, "circuit_breaker_drawdown": None, "rebalance_drift_threshold": 0.02},
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_stop05_take035",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": 0.05, "take_profit_pct": 0.35, "circuit_breaker_drawdown": None, "rebalance_drift_threshold": 0.02},
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_stop08_take035",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": 0.08, "take_profit_pct": 0.35, "circuit_breaker_drawdown": None, "rebalance_drift_threshold": 0.02},
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_stop12_take035",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": False,
            "backtest": {"stop_loss_pct": 0.12, "take_profit_pct": 0.35, "circuit_breaker_drawdown": None, "rebalance_drift_threshold": 0.02},
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_ma90_side02_bear02",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": False,
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "equity_overlay": {
                    "enabled": True,
                    "min_periods": 60,
                    "ma_window": 90,
                    "momentum_window": 5,
                    "drawdown_window": 60,
                    "drawdown_cut": 0.15,
                    "bull_exposure": 1.0,
                    "sideways_exposure": 0.20,
                    "bear_exposure": 0.20,
                    "rebalance_threshold": 0.05,
                },
            },
            "recorded_hint": {"hint_annual_return": 0.2201, "hint_max_drawdown": -0.1833, "hint_source": "post_processed_no_blend_equity_overlay"},
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_ma90_side02_bear04",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": False,
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "equity_overlay": {
                    "enabled": True,
                    "min_periods": 60,
                    "ma_window": 90,
                    "momentum_window": 5,
                    "drawdown_window": 60,
                    "drawdown_cut": 0.15,
                    "bull_exposure": 1.0,
                    "sideways_exposure": 0.20,
                    "bear_exposure": 0.40,
                    "rebalance_threshold": 0.05,
                },
            },
            "recorded_hint": {"hint_annual_return": 0.2266, "hint_max_drawdown": -0.1854, "hint_source": "post_processed_no_blend_equity_overlay"},
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side02_bear02",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": False,
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "equity_overlay": {
                    "enabled": True,
                    "min_periods": 60,
                    "ma_window": 90,
                    "momentum_window": 5,
                    "drawdown_window": 60,
                    "drawdown_cut": 0.15,
                    "bull_exposure": 1.0,
                    "sideways_exposure": 0.20,
                    "bear_exposure": 0.20,
                    "rebalance_threshold": 0.05,
                    "rebalance_on_signal_only": True,
                },
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side04_bear04",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": False,
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "equity_overlay": {
                    "enabled": True,
                    "min_periods": 60,
                    "ma_window": 90,
                    "momentum_window": 5,
                    "drawdown_window": 60,
                    "drawdown_cut": 0.15,
                    "bull_exposure": 1.0,
                    "sideways_exposure": 0.40,
                    "bear_exposure": 0.40,
                    "rebalance_threshold": 0.05,
                    "rebalance_on_signal_only": True,
                },
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side045_bear045",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": False,
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.45, bear_exposure=0.45),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": False,
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_indcap25",
            "strategy": {
                "factor_group": "momentum",
                "top_n": 15,
                "rank_buffer": 20,
                "rebalance_freq": "monthly",
                "target_vol": "__remove__",
                "max_industry_weight": 0.25,
            },
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": False,
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_indcap20",
            "strategy": {
                "factor_group": "momentum",
                "top_n": 15,
                "rank_buffer": 20,
                "rebalance_freq": "monthly",
                "target_vol": "__remove__",
                "max_industry_weight": 0.20,
            },
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": False,
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_marketdd08_schedsig_side06_bear02",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "market_regime": _market_drawdown_regime(0.08),
            "defensive_timing": True,
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 0.60, "bear_exposure": 0.20},
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "exposure_schedule_rebalance_on_signal_only": True,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_marketdd10_schedsig_side06_bear03",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "market_regime": _market_drawdown_regime(0.10),
            "defensive_timing": True,
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 0.60, "bear_exposure": 0.30},
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "exposure_schedule_rebalance_on_signal_only": True,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_marketdd12_schedsig_side1_bear05",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "market_regime": _market_drawdown_regime(0.12),
            "defensive_timing": True,
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.50},
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "exposure_schedule_rebalance_on_signal_only": True,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_marketdd15_schedsig_side1_bear06",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "market_regime": _market_drawdown_regime(0.15),
            "defensive_timing": True,
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.60},
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "exposure_schedule_rebalance_on_signal_only": True,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_fastdd20_10_schedsig_side1_bear03",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "market_regime": _market_fast_drawdown_regime(threshold=0.10, drawdown_window=20),
            "defensive_timing": True,
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.30},
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "exposure_schedule_rebalance_on_signal_only": True,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_fastdd20_12_schedsig_side1_bear03",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "market_regime": _market_fast_drawdown_regime(threshold=0.12, drawdown_window=20),
            "defensive_timing": True,
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.30},
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "exposure_schedule_rebalance_on_signal_only": True,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_fastdd40_12_schedsig_side1_bear05",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "market_regime": _market_fast_drawdown_regime(threshold=0.12, drawdown_window=40),
            "defensive_timing": True,
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.50},
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "exposure_schedule_rebalance_on_signal_only": True,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma60_side04_bear04",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": False,
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "equity_overlay": _overlay_signal_config(ma_window=60, sideways_exposure=0.40, bear_exposure=0.40),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma120_side04_bear04",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": False,
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "equity_overlay": _overlay_signal_config(ma_window=120, sideways_exposure=0.40, bear_exposure=0.40),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_cb18_30",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": False,
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": 0.18,
                "circuit_breaker_cooldown_days": 60,
                "circuit_breaker_target_exposure": 0.30,
                "rebalance_drift_threshold": 0.02,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_cb20_30",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": False,
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": 0.20,
                "circuit_breaker_cooldown_days": 60,
                "circuit_breaker_target_exposure": 0.30,
                "rebalance_drift_threshold": 0.02,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_globaldd12_side035_bear02",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": False,
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "equity_overlay": _overlay_signal_config(
                    ma_window=90,
                    sideways_exposure=0.35,
                    bear_exposure=0.20,
                    drawdown_window=10000,
                    drawdown_cut=0.12,
                ),
            },
            "recorded_hint": {"hint_annual_return": 0.2162, "hint_max_drawdown": -0.1873, "hint_source": "post_processed_global_drawdown_overlay"},
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_globaldd12_side04_bear02",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": False,
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "equity_overlay": _overlay_signal_config(
                    ma_window=90,
                    sideways_exposure=0.40,
                    bear_exposure=0.20,
                    drawdown_window=10000,
                    drawdown_cut=0.12,
                ),
            },
            "recorded_hint": {"hint_annual_return": 0.2184, "hint_max_drawdown": -0.2008, "hint_source": "post_processed_global_drawdown_overlay"},
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_turn3_buf0_take035_overlay_signal_ma90_side05_bear05",
            "strategy": {"factor_group": "momentum", "top_n": 15, "max_turnover": 3, "rank_buffer": 0, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": False,
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_turn5_buf0_take035_overlay_signal_ma90_side05_bear05",
            "strategy": {"factor_group": "momentum", "top_n": 15, "max_turnover": 5, "rank_buffer": 0, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": False,
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_vol16",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": 0.16, "max_leverage": 1.0},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": False,
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_vol18",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": 0.18, "max_leverage": 1.0},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": False,
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q50_no_blend_top15_take035_overlay_signal_ma90_side05_bear05",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.50},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": False,
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q50_no_blend_top30_take035_overlay_signal_ma90_side05_bear05",
            "strategy": {"factor_group": "momentum", "top_n": 30, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.50},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": False,
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top20_take035_overlay_signal_ma90_side05_bear05",
            "strategy": {"factor_group": "momentum", "top_n": 20, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": False,
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top25_take035_overlay_signal_ma90_side05_bear05",
            "strategy": {"factor_group": "momentum", "top_n": 25, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": False,
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top30_take035_overlay_signal_ma90_side05_bear05",
            "strategy": {"factor_group": "momentum", "top_n": 30, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": False,
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_bear_def_keep70",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "regime_score_filter": _defensive_filter(["bear"], keep_top_fraction=0.70),
            "defensive_timing": False,
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_bear_def_keep50",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "regime_score_filter": _defensive_filter(["bear"], keep_top_fraction=0.50),
            "defensive_timing": False,
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_sidebear_def_keep70",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "regime_score_filter": _defensive_filter(["sideways", "bear"], keep_top_fraction=0.70),
            "defensive_timing": False,
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_bear_lowvol_keep70",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "regime_score_filter": _defensive_filter(
                ["bear"],
                keep_top_fraction=0.70,
                components=[{"column": "STD20", "direction": -1.0}, {"column": "BETA20", "direction": -1.0}],
            ),
            "defensive_timing": False,
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_ma90_schedsig_bear06",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "market_regime": _ma_only_regime(ma_window=90, momentum_window=5),
            "defensive_timing": True,
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.60},
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "exposure_schedule_rebalance_on_signal_only": True,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_ma90_schedsig_bear06_selrisk3",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "market_regime": _ma_only_regime(ma_window=90, momentum_window=5),
            "defensive_timing": True,
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.60},
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "exposure_schedule_rebalance_on_signal_only": True,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
                "selection_risk_filter": _selection_risk_filter(lookback_sessions=3),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_ma90_schedsig_bear06_selrisk5",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "market_regime": _ma_only_regime(ma_window=90, momentum_window=5),
            "defensive_timing": True,
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.60},
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "exposure_schedule_rebalance_on_signal_only": True,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
                "selection_risk_filter": _selection_risk_filter(lookback_sessions=5),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_ma90_schedsig_bear08",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "market_regime": _ma_only_regime(ma_window=90, momentum_window=5),
            "defensive_timing": True,
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.80},
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "exposure_schedule_rebalance_on_signal_only": True,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_defaultregime_schedsig_bear06",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": True,
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.60},
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "exposure_schedule_rebalance_on_signal_only": True,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_defaultregime_schedsig_bear08",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "defensive_timing": True,
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.80},
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "exposure_schedule_rebalance_on_signal_only": True,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_marketdd18_schedsig_side1_bear08",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "market_regime": _market_drawdown_regime(0.18),
            "defensive_timing": True,
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.80},
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "exposure_schedule_rebalance_on_signal_only": True,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_marketdd20_schedsig_side1_bear08",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "market_regime": _market_drawdown_regime(0.20),
            "defensive_timing": True,
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.80},
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "exposure_schedule_rebalance_on_signal_only": True,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_bearblend025_top15_take035_overlay_signal_ma90_side05_bear05",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": _defensive_blend(sideways_weight=0.0, bear_weight=0.25),
            "defensive_timing": False,
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_bearblend040_top15_take035_overlay_signal_ma90_side05_bear05",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": _defensive_blend(sideways_weight=0.0, bear_weight=0.40),
            "defensive_timing": False,
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_sideblend015_bearblend035_top15_take035_overlay_signal_ma90_side05_bear05",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": _defensive_blend(sideways_weight=0.15, bear_weight=0.35),
            "defensive_timing": False,
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_ma90_schedsig_bear055",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "market_regime": _ma_only_regime(ma_window=90, momentum_window=5),
            "defensive_timing": True,
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.55},
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "exposure_schedule_rebalance_on_signal_only": True,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_ma90_schedsig_bear058",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "market_regime": _ma_only_regime(ma_window=90, momentum_window=5),
            "defensive_timing": True,
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.58},
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "exposure_schedule_rebalance_on_signal_only": True,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_ma80_schedsig_bear06",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "market_regime": _ma_only_regime(ma_window=80, momentum_window=5),
            "defensive_timing": True,
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.60},
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "exposure_schedule_rebalance_on_signal_only": True,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_ma100_schedsig_bear06",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "market_regime": _ma_only_regime(ma_window=100, momentum_window=5),
            "defensive_timing": True,
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.60},
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "exposure_schedule_rebalance_on_signal_only": True,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_ma90_schedsig_bear06_overlay_cut12",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "market_regime": _ma_only_regime(ma_window=90, momentum_window=5),
            "defensive_timing": True,
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.60},
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "rebalance_drift_threshold": 0.02,
                "exposure_schedule_rebalance_on_signal_only": True,
                "equity_overlay": _overlay_signal_config(
                    ma_window=90,
                    sideways_exposure=0.50,
                    bear_exposure=0.50,
                    drawdown_cut=0.12,
                ),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_ma90_schedsig_bear06_cb18_30",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "market_regime": _ma_only_regime(ma_window=90, momentum_window=5),
            "defensive_timing": True,
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.60},
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": 0.18,
                "circuit_breaker_cooldown_days": 30,
                "circuit_breaker_target_exposure": 0.30,
                "rebalance_drift_threshold": 0.02,
                "exposure_schedule_rebalance_on_signal_only": True,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_ma90_schedsig_bear06_cb18_50",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "market_regime": _ma_only_regime(ma_window=90, momentum_window=5),
            "defensive_timing": True,
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.60},
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": 0.18,
                "circuit_breaker_cooldown_days": 30,
                "circuit_breaker_target_exposure": 0.50,
                "rebalance_drift_threshold": 0.02,
                "exposure_schedule_rebalance_on_signal_only": True,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_ma90_schedsig_bear06_cb19_30",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "market_regime": _ma_only_regime(ma_window=90, momentum_window=5),
            "defensive_timing": True,
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.60},
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": 0.19,
                "circuit_breaker_cooldown_days": 30,
                "circuit_breaker_target_exposure": 0.30,
                "rebalance_drift_threshold": 0.02,
                "exposure_schedule_rebalance_on_signal_only": True,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_ma90_schedsig_bear06_cb20_30",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "market_regime": _ma_only_regime(ma_window=90, momentum_window=5),
            "defensive_timing": True,
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.60},
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": 0.20,
                "circuit_breaker_cooldown_days": 30,
                "circuit_breaker_target_exposure": 0.30,
                "rebalance_drift_threshold": 0.02,
                "exposure_schedule_rebalance_on_signal_only": True,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_ma90_schedsig_bear06_annualdd18_30",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "market_regime": _ma_only_regime(ma_window=90, momentum_window=5),
            "defensive_timing": True,
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.60},
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "annual_drawdown_guard": {"enabled": True, "drawdown": 0.18, "target_exposure": 0.30},
                "rebalance_drift_threshold": 0.02,
                "exposure_schedule_rebalance_on_signal_only": True,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_ma90_schedsig_bear06_annualdd18_50",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "market_regime": _ma_only_regime(ma_window=90, momentum_window=5),
            "defensive_timing": True,
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.60},
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "annual_drawdown_guard": {"enabled": True, "drawdown": 0.18, "target_exposure": 0.50},
                "rebalance_drift_threshold": 0.02,
                "exposure_schedule_rebalance_on_signal_only": True,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_ma90_schedsig_bear06_annualdd19_30",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "market_regime": _ma_only_regime(ma_window=90, momentum_window=5),
            "defensive_timing": True,
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.60},
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "annual_drawdown_guard": {"enabled": True, "drawdown": 0.19, "target_exposure": 0.30},
                "rebalance_drift_threshold": 0.02,
                "exposure_schedule_rebalance_on_signal_only": True,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_ma90_schedsig_bear06_annualdd19_50",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "market_regime": _ma_only_regime(ma_window=90, momentum_window=5),
            "defensive_timing": True,
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.60},
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "annual_drawdown_guard": {"enabled": True, "drawdown": 0.19, "target_exposure": 0.50},
                "rebalance_drift_threshold": 0.02,
                "exposure_schedule_rebalance_on_signal_only": True,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_ma90_schedsig_bear06_annualdd18_rel15_50",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "market_regime": _ma_only_regime(ma_window=90, momentum_window=5),
            "defensive_timing": True,
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.60},
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "annual_drawdown_guard": {
                    "enabled": True,
                    "drawdown": 0.18,
                    "release_drawdown": 0.15,
                    "target_exposure": 0.50,
                },
                "rebalance_drift_threshold": 0.02,
                "exposure_schedule_rebalance_on_signal_only": True,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_ma90_schedsig_bear06_annualdd18_rel12_50",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "market_regime": _ma_only_regime(ma_window=90, momentum_window=5),
            "defensive_timing": True,
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.60},
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "annual_drawdown_guard": {
                    "enabled": True,
                    "drawdown": 0.18,
                    "release_drawdown": 0.12,
                    "target_exposure": 0.50,
                },
                "rebalance_drift_threshold": 0.02,
                "exposure_schedule_rebalance_on_signal_only": True,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_ma90_schedsig_bear06_annualdd19_rel15_50",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "market_regime": _ma_only_regime(ma_window=90, momentum_window=5),
            "defensive_timing": True,
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.60},
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "annual_drawdown_guard": {
                    "enabled": True,
                    "drawdown": 0.19,
                    "release_drawdown": 0.15,
                    "target_exposure": 0.50,
                },
                "rebalance_drift_threshold": 0.02,
                "exposure_schedule_rebalance_on_signal_only": True,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
        {
            "name": "momentum_lowliq_q35_no_blend_top15_take035_overlay_signal_ma90_side05_bear05_ma90_schedsig_bear06_annualdd18_rel15_30",
            "strategy": {"factor_group": "momentum", "top_n": 15, "rank_buffer": 20, "rebalance_freq": "monthly", "target_vol": "__remove__"},
            "liquidity_filter": {"enabled": True, "side": "low", "quantile": 0.35},
            "regime_score_blend": {"enabled": False},
            "market_regime": _ma_only_regime(ma_window=90, momentum_window=5),
            "defensive_timing": True,
            "defensive_timing_config": {"bull_exposure": 1.0, "sideways_exposure": 1.0, "bear_exposure": 0.60},
            "backtest": {
                "stop_loss_pct": None,
                "take_profit_pct": 0.35,
                "circuit_breaker_drawdown": None,
                "annual_drawdown_guard": {
                    "enabled": True,
                    "drawdown": 0.18,
                    "release_drawdown": 0.15,
                    "target_exposure": 0.30,
                },
                "rebalance_drift_threshold": 0.02,
                "exposure_schedule_rebalance_on_signal_only": True,
                "equity_overlay": _overlay_signal_config(ma_window=90, sideways_exposure=0.50, bear_exposure=0.50),
            },
        },
    ]


def _selection_risk_filter(lookback_sessions: int) -> dict[str, Any]:
    return {
        "enabled": True,
        "lookback_sessions": int(lookback_sessions),
        "required_price_fields": ["open", "close"],
        "max_missing_price_sessions": 0,
        "max_limit_down_days": 0,
        "limit_down_buffer": 0.0,
        "require_positive_volume": True,
    }


def _strategy_config(config: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    strategy = deepcopy(config["strategy"])
    for key, value in candidate.get("strategy", {}).items():
        if value == "__remove__":
            strategy.pop(key, None)
        else:
            strategy[key] = value
    return strategy


def _scoring_config(config: dict[str, Any], strategy: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(config)
    result["strategy"] = strategy
    for key in ["liquidity_filter", "market_regime", "regime_score_blend", "regime_score_filter"]:
        if key in candidate:
            result[key] = {**result.get(key, {}), **candidate[key]}
    return result


def _timing_config(config: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(config)
    if "market_regime" in candidate:
        result["market_regime"] = {**result.get("market_regime", {}), **candidate["market_regime"]}
    result.setdefault("defensive_timing", {})["enabled"] = bool(candidate.get("defensive_timing", True))
    if "defensive_timing_config" in candidate:
        result["defensive_timing"] = {**result.get("defensive_timing", {}), **candidate["defensive_timing_config"]}
    return result


def _drawdown_only_regime(threshold: float) -> dict[str, Any]:
    return {
        "bear_drawdown_threshold": threshold,
        "drawdown_window": 252,
        "bear_momentum_max": -999.0,
        "high_volatility_threshold": 999.0,
    }


def _market_drawdown_regime(threshold: float) -> dict[str, Any]:
    return {
        "ma_window": 90,
        "momentum_window": 5,
        "volatility_window": 20,
        "min_periods": 60,
        "bear_drawdown_threshold": threshold,
        "drawdown_window": 252,
        "high_volatility_threshold": 999.0,
        "lag_days": 1,
    }


def _ma_only_regime(ma_window: int, momentum_window: int) -> dict[str, Any]:
    return {
        "ma_window": ma_window,
        "momentum_window": momentum_window,
        "high_volatility_threshold": 999.0,
        "bear_drawdown_threshold": None,
        "lag_days": 1,
    }


def _market_fast_drawdown_regime(*, threshold: float, drawdown_window: int) -> dict[str, Any]:
    return {
        "ma_window": 90,
        "momentum_window": 5,
        "volatility_window": 20,
        "min_periods": min(60, max(1, int(drawdown_window))),
        "bear_drawdown_threshold": threshold,
        "drawdown_window": drawdown_window,
        "bear_momentum_max": -999.0,
        "high_volatility_threshold": 999.0,
        "lag_days": 1,
    }


def _overlay_signal_config(
    *,
    ma_window: int,
    sideways_exposure: float,
    bear_exposure: float,
    drawdown_window: int = 60,
    drawdown_cut: float = 0.15,
) -> dict[str, Any]:
    return {
        "enabled": True,
        "min_periods": 60,
        "ma_window": ma_window,
        "momentum_window": 5,
        "drawdown_window": drawdown_window,
        "drawdown_cut": drawdown_cut,
        "bull_exposure": 1.0,
        "sideways_exposure": sideways_exposure,
        "bear_exposure": bear_exposure,
        "rebalance_threshold": 0.05,
        "rebalance_on_signal_only": True,
    }


def _defensive_filter(
    regimes: list[str],
    *,
    keep_top_fraction: float,
    components: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    selected_components = components or [
        {"column": "ROC20", "direction": 1.0},
        {"column": "STD20", "direction": -1.0},
        {"column": "BETA20", "direction": -1.0},
    ]
    return {
        "enabled": True,
        "rules": [
            {
                "regime": regime,
                "components": selected_components,
                "min_score": -1.0,
                "keep_top_fraction": keep_top_fraction,
            }
            for regime in regimes
        ],
    }


def _defensive_blend(*, sideways_weight: float, bear_weight: float) -> dict[str, Any]:
    return {
        "enabled": True,
        "bull_defensive_weight": 0.0,
        "sideways_defensive_weight": sideways_weight,
        "bear_defensive_weight": bear_weight,
        "defensive_components": [
            {"column": "STD20", "direction": -1.0},
            {"column": "BETA20", "direction": -1.0},
            {"column": "ROC20", "direction": 1.0},
        ],
    }


def _score_key(candidate: dict[str, Any]) -> str:
    strategy_items = tuple(sorted(candidate.get("strategy", {}).items()))
    liquidity_items = tuple(sorted(candidate.get("liquidity_filter", {}).items()))
    market_items = json.dumps(candidate.get("market_regime", {}), sort_keys=True, default=str)
    blend_items = json.dumps(candidate.get("regime_score_blend", {}), sort_keys=True, default=str)
    filter_items = json.dumps(candidate.get("regime_score_filter", {}), sort_keys=True, default=str)
    return repr((strategy_items, liquidity_items, market_items, blend_items, filter_items))


def _quality_flags(metrics: dict[str, Any], quality_cfg: dict[str, Any]) -> dict[str, Any]:
    annual_return = float(metrics.get("annual_return", 0.0) or 0.0)
    max_drawdown = float(metrics.get("max_drawdown", 0.0) or 0.0)
    annual_turnover = float(metrics.get("annual_turnover", 0.0) or 0.0)
    annual_cost = float(metrics.get("annual_trade_cost_ratio", metrics.get("trade_cost_ratio", 0.0)) or 0.0)
    return_threshold = float(quality_cfg.get("min_backtest_annual_return", quality_cfg.get("target_annual_return", 0.20)))
    drawdown_limit = float(quality_cfg.get("max_backtest_drawdown_limit", quality_cfg.get("max_drawdown_limit", -0.20)))
    turnover_limit = float(quality_cfg.get("max_annual_turnover", 20.0))
    cost_limit = float(quality_cfg.get("max_annual_trade_cost_ratio", 0.2))
    flags = {
        "annual_return_pass": annual_return >= return_threshold,
        "drawdown_pass": max_drawdown >= drawdown_limit,
        "annual_turnover_pass": annual_turnover <= turnover_limit,
        "annual_trade_cost_ratio_pass": annual_cost <= cost_limit,
        "annual_return_threshold": return_threshold,
        "drawdown_limit": drawdown_limit,
        "annual_turnover_limit": turnover_limit,
        "annual_trade_cost_ratio_limit": cost_limit,
    }
    flags["is_acceptable"] = bool(
        flags["annual_return_pass"]
        and flags["drawdown_pass"]
        and flags["annual_turnover_pass"]
        and flags["annual_trade_cost_ratio_pass"]
    )
    return flags


def _write_candidate_artifacts(
    prefix: Path,
    result,
    yearly: pd.DataFrame,
    prices: pd.DataFrame,
    config: dict[str, Any],
    write_diagnostics: bool,
) -> dict[str, str]:
    paths = {
        "equity_path": str(prefix) + "_equity.csv",
        "years_path": str(prefix) + "_years.csv",
        "trades_path": str(prefix) + "_trades.csv",
        "holdings_path": str(prefix) + "_holdings.csv",
    }
    result.equity_curve.to_csv(paths["equity_path"], encoding="utf-8-sig")
    yearly.to_csv(paths["years_path"], index=False, encoding="utf-8-sig")
    result.trades.to_csv(paths["trades_path"], index=False, encoding="utf-8-sig")
    result.holdings.to_csv(paths["holdings_path"], index=False, encoding="utf-8-sig")
    if write_diagnostics:
        diagnostics, tables = build_research_diagnostics(result.equity_curve, result.holdings, result.trades, prices, config)
        research_paths = write_research_diagnostics(
            diagnostics,
            tables,
            prefix.parent,
            prefix=f"{prefix.name}_research",
        )
        paths["research_diagnostics_path"] = research_paths.get("research_diagnostics", "")
    return paths


def _yearly_stats(equity_curve: pd.Series) -> pd.DataFrame:
    if equity_curve.empty:
        return pd.DataFrame(columns=["year", "days", "annual_return", "max_drawdown"])
    rows: list[dict[str, Any]] = []
    curve = equity_curve.sort_index()
    for year, group in curve.groupby(curve.index.year):
        if len(group) <= 1 or float(group.iloc[0]) <= 0:
            continue
        total_return = float(group.iloc[-1] / group.iloc[0] - 1.0)
        annual_return = float((1.0 + total_return) ** (252 / max(len(group) - 1, 1)) - 1.0) if total_return > -1 else -1.0
        max_drawdown = float((group / group.cummax() - 1.0).min())
        rows.append({"year": int(year), "days": int(len(group)), "annual_return": annual_return, "max_drawdown": max_drawdown})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
