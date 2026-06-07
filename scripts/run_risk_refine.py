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
from src.data_coverage import build_yearly_equity_coverage
from src.factor_calculator import load_or_compute_factors
from src.market_regime import apply_defensive_timing_to_backtest_config
from src.scoring import build_strategy_scores
from src.selection_constraints import apply_selection_constraints_to_backtest_config
from src.strategy import resample_signals
from src.trading_calendar import resolve_target_date_value

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)
LIQUIDITY_DISABLED_SIDES = {"none", "off", "disabled", "false", "0"}


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
    parser.add_argument("--factor-groups", default=str(config.get("strategy", {}).get("factor_group", "momentum")))
    parser.add_argument("--dynamic-candidates", default="", help="Comma-separated dynamic_ic_selector candidates to use for this run.")
    parser.add_argument("--liquidity-sides", default=str(config.get("liquidity_filter", {}).get("side", "high")))
    parser.add_argument("--liquidity-quantiles", default="0.20,0.30")
    parser.add_argument("--top-n", default=str(config.get("strategy", {}).get("top_n", 15)))
    parser.add_argument("--max-turnover", default=str(config.get("strategy", {}).get("max_turnover", 1)))
    parser.add_argument("--rank-buffer", default=str(config.get("strategy", {}).get("rank_buffer", 30)))
    parser.add_argument("--max-industry-weight", default=str(config.get("strategy", {}).get("max_industry_weight", "none")))
    parser.add_argument("--score-weighted", default=str(config.get("strategy", {}).get("score_weighted", config.get("backtest", {}).get("score_weighted", False))))
    parser.add_argument("--stop-loss-pct", default=str(config.get("strategy", {}).get("stop_loss_pct", "none")))
    parser.add_argument("--take-profit-pct", default=str(config.get("strategy", {}).get("take_profit_pct", "none")))
    parser.add_argument("--circuit-breaker-drawdown", default="0.06,0.08,0.10")
    parser.add_argument("--cooldown-days", default="5,10")
    parser.add_argument("--circuit-breaker-target-exposure", default="0.0,0.30")
    parser.add_argument("--rebalance-drift-threshold", default="0.0,0.02,0.05")
    parser.add_argument("--annual-guard-drawdown", default="none", help="Comma-separated per-year drawdown triggers, or none.")
    parser.add_argument("--annual-guard-target-exposure", default="0.0", help="Comma-separated target exposures after annual guard triggers.")
    parser.add_argument("--annual-guard-release-drawdown", default="none", help="Comma-separated same-year release drawdowns, or none.")
    parser.add_argument("--equity-overlay", choices=["config", "enabled", "disabled"], default="config")
    parser.add_argument("--overlay-sideways-exposure", default=str(config.get("backtest", {}).get("equity_overlay", {}).get("sideways_exposure", 0.5)))
    parser.add_argument("--overlay-bear-exposure", default=str(config.get("backtest", {}).get("equity_overlay", {}).get("bear_exposure", 0.5)))
    parser.add_argument("--overlay-drawdown-cut", default=str(config.get("backtest", {}).get("equity_overlay", {}).get("drawdown_cut", 0.15)))
    parser.add_argument("--bull-exposure", default=str(config.get("defensive_timing", {}).get("bull_exposure", 1.0)))
    parser.add_argument("--sideways-exposure", default=str(config.get("defensive_timing", {}).get("sideways_exposure", 0.60)))
    parser.add_argument("--bear-exposure", default=str(config.get("defensive_timing", {}).get("bear_exposure", 0.30)))
    parser.add_argument("--bear-drawdown-threshold", default=str(config.get("market_regime", {}).get("bear_drawdown_threshold", "none")))
    parser.add_argument("--bull-defensive-weight", default=str(config.get("regime_score_blend", {}).get("bull_defensive_weight", 0.0)))
    parser.add_argument("--sideways-defensive-weight", default=str(config.get("regime_score_blend", {}).get("sideways_defensive_weight", 0.5)))
    parser.add_argument("--bear-defensive-weight", default=str(config.get("regime_score_blend", {}).get("bear_defensive_weight", 1.0)))
    parser.add_argument(
        "--defensive-timing",
        choices=["config", "enabled", "disabled"],
        default="config",
        help="Whether exact backtests should use the configured defensive exposure schedule.",
    )
    parser.add_argument("--target-annual-return", type=float, default=quality.get("target_annual_return", 0.20))
    parser.add_argument("--drawdown-limit", type=float, default=quality.get("max_backtest_drawdown_limit", -0.20))
    parser.add_argument("--max-seconds", type=float, default=900.0)
    parser.add_argument("--resume", action="store_true", help="Skip parameter rows already present in the output CSV.")
    parser.add_argument("--output", default="outputs/risk_refine_results.csv")
    args = parser.parse_args()

    start_time = time.monotonic()
    config = _with_dynamic_candidates_override(config, args.dynamic_candidates)
    end_date = resolve_target_date_value(args.end_date, config=config)
    prices = pd.read_parquet(resolve_path(args.price_file))
    factor_groups = _factor_group_values(args.factor_groups, config)
    factor_columns = _requested_factor_columns_for_groups(args.factor_file, config, factor_groups)
    factors = load_or_compute_factors(args.start_date, end_date, cache_file=args.factor_file, columns=factor_columns)

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_keys(output_path) if args.resume else set()
    rows = _read_existing(output_path) if args.resume else pd.DataFrame()

    combos = list(
        product(
            factor_groups,
            _csv_values(args.liquidity_sides, str),
            _csv_values(args.liquidity_quantiles, float),
            _csv_values(args.top_n, int),
            _csv_values(args.max_turnover, int),
            _csv_values(args.rank_buffer, int),
            _csv_optional_values(args.max_industry_weight, float),
            _csv_bool_values(args.score_weighted),
            _csv_optional_values(args.stop_loss_pct, float),
            _csv_optional_values(args.take_profit_pct, float),
            _csv_optional_values(args.circuit_breaker_drawdown, float),
            _csv_values(args.cooldown_days, int),
            _csv_values(args.circuit_breaker_target_exposure, float),
            _csv_values(args.rebalance_drift_threshold, float),
            _csv_optional_values(args.annual_guard_drawdown, float),
            _csv_values(args.annual_guard_target_exposure, float),
            _csv_optional_values(args.annual_guard_release_drawdown, float),
            _csv_values(args.equity_overlay, str),
            _csv_values(args.overlay_sideways_exposure, float),
            _csv_values(args.overlay_bear_exposure, float),
            _csv_values(args.overlay_drawdown_cut, float),
            _csv_values(args.bull_exposure, float),
            _csv_values(args.sideways_exposure, float),
            _csv_values(args.bear_exposure, float),
            _csv_optional_values(args.bear_drawdown_threshold, float),
            _csv_values(args.bull_defensive_weight, float),
            _csv_values(args.sideways_defensive_weight, float),
            _csv_values(args.bear_defensive_weight, float),
        )
    )
    logger.info("Risk refinement grid has %s exact backtest candidates.", len(combos))

    score_cache: dict[tuple[str, str, float, float | None, float, float, float], pd.Series] = {}
    for idx, (
        factor_group,
        liquidity_side,
        liquidity_quantile,
        top_n,
        max_turnover,
        rank_buffer,
        max_industry_weight,
        score_weighted,
        stop_loss,
        take_profit,
        circuit_drawdown,
        cooldown_days,
        target_exposure,
        rebalance_drift_threshold,
        annual_guard_drawdown,
        annual_guard_target_exposure,
        annual_guard_release_drawdown,
        equity_overlay_mode,
        overlay_sideways_exposure,
        overlay_bear_exposure,
        overlay_drawdown_cut,
        bull_exposure,
        sideways_exposure,
        bear_exposure,
        bear_drawdown_threshold,
        bull_defensive_weight,
        sideways_defensive_weight,
        bear_defensive_weight,
    ) in enumerate(combos, start=1):
        liquidity_enabled, liquidity_side_key, liquidity_quantile_key = _liquidity_filter_state(liquidity_side, liquidity_quantile)
        annual_guard_enabled, annual_guard_drawdown_key, annual_guard_target_key, annual_guard_release_key = _annual_guard_state(
            annual_guard_drawdown,
            annual_guard_target_exposure,
            annual_guard_release_drawdown,
        )
        overlay_mode_key, overlay_sideways_key, overlay_bear_key, overlay_drawdown_key = _equity_overlay_state(
            equity_overlay_mode,
            overlay_sideways_exposure,
            overlay_bear_exposure,
            overlay_drawdown_cut,
        )
        key = _combo_key(
            factor_group,
            liquidity_side_key,
            liquidity_quantile_key,
            top_n,
            max_turnover,
            rank_buffer,
            max_industry_weight,
            score_weighted,
            stop_loss,
            take_profit,
            circuit_drawdown,
            cooldown_days,
            target_exposure,
            rebalance_drift_threshold,
            annual_guard_enabled,
            annual_guard_drawdown_key,
            annual_guard_target_key,
            annual_guard_release_key,
            overlay_mode_key,
            overlay_sideways_key,
            overlay_bear_key,
            overlay_drawdown_key,
            args.defensive_timing,
            bull_exposure,
            sideways_exposure,
            bear_exposure,
            bear_drawdown_threshold,
            bull_defensive_weight,
            sideways_defensive_weight,
            bear_defensive_weight,
        )
        if key in completed:
            logger.info("Skipping completed row %s/%s: %s.", idx, len(combos), key)
            continue
        if args.max_seconds > 0 and time.monotonic() - start_time >= args.max_seconds:
            logger.warning("Stopping before row %s because max-seconds=%.0f was reached.", idx, args.max_seconds)
            break

        logger.info("Running row %s/%s: %s.", idx, len(combos), key)
        score_key = (
            str(factor_group).strip().lower(),
            liquidity_side_key,
            liquidity_quantile_key,
            _optional_key(bear_drawdown_threshold),
            round(float(bull_defensive_weight), 6),
            round(float(sideways_defensive_weight), 6),
            round(float(bear_defensive_weight), 6),
        )
        scores = score_cache.get(score_key)
        if scores is None:
            scoring_config = deepcopy(config)
            scoring_config.setdefault("strategy", {})["factor_group"] = factor_group
            scoring_config.setdefault("liquidity_filter", {})
            scoring_config["liquidity_filter"]["enabled"] = liquidity_enabled
            if liquidity_enabled:
                scoring_config["liquidity_filter"]["side"] = score_key[1]
                scoring_config["liquidity_filter"]["quantile"] = liquidity_quantile_key
            scoring_config.setdefault("market_regime", {})["bear_drawdown_threshold"] = bear_drawdown_threshold
            scoring_config.setdefault("regime_score_blend", {})["bull_defensive_weight"] = bull_defensive_weight
            scoring_config.setdefault("regime_score_blend", {})["sideways_defensive_weight"] = sideways_defensive_weight
            scoring_config.setdefault("regime_score_blend", {})["bear_defensive_weight"] = bear_defensive_weight
            scores = build_strategy_scores(factors, scoring_config, price_df=prices)
            scores = resample_signals(scores, scoring_config["strategy"].get("rebalance_freq", "daily"))
            score_cache[score_key] = scores

        timing_config = _with_timing_overrides(
            config,
            args.defensive_timing,
            bull_exposure,
            sideways_exposure,
            bear_exposure,
            bear_drawdown_threshold,
        )
        bt_config = apply_defensive_timing_to_backtest_config({**config["backtest"], **config["strategy"]}, prices, timing_config)
        bt_config = _with_equity_overlay_overrides(
            bt_config,
            overlay_mode_key,
            overlay_sideways_key,
            overlay_bear_key,
            overlay_drawdown_key,
        )
        bt_config.update(
            {
                "top_n": top_n,
                "max_turnover": max_turnover,
                "rank_buffer": rank_buffer,
                "max_industry_weight": max_industry_weight,
                "score_weighted": score_weighted,
                "stop_loss_pct": stop_loss,
                "take_profit_pct": take_profit,
                "circuit_breaker_drawdown": circuit_drawdown,
                "circuit_breaker_cooldown_days": cooldown_days,
                "circuit_breaker_target_exposure": target_exposure,
                "rebalance_drift_threshold": rebalance_drift_threshold,
                "annual_drawdown_guard": _annual_guard_config(
                    annual_guard_enabled,
                    annual_guard_drawdown_key,
                    annual_guard_target_key,
                    annual_guard_release_key,
                ),
            }
        )
        bt_config = apply_selection_constraints_to_backtest_config(bt_config, config)

        row_start = time.monotonic()
        result = run_backtest(scores, prices, args.start_date, end_date, bt_config)
        yearly = _yearly_stats(result.equity_curve)
        yearly_coverage = build_yearly_equity_coverage(result.equity_curve, args.start_date, end_date)
        row = {
            "factor_group": factor_group,
            "liquidity_enabled": liquidity_enabled,
            "liquidity_side": score_key[1],
            "liquidity_quantile": liquidity_quantile_key,
            "top_n": top_n,
            "max_turnover": max_turnover,
            "rank_buffer": rank_buffer,
            "max_industry_weight": max_industry_weight,
            "score_weighted": score_weighted,
            "stop_loss_pct": stop_loss,
            "take_profit_pct": take_profit,
            "circuit_breaker_drawdown": circuit_drawdown,
            "circuit_breaker_cooldown_days": cooldown_days,
            "circuit_breaker_target_exposure": target_exposure,
            "rebalance_drift_threshold": rebalance_drift_threshold,
            "annual_guard_enabled": annual_guard_enabled,
            "annual_guard_drawdown": annual_guard_drawdown_key,
            "annual_guard_target_exposure": annual_guard_target_key,
            "annual_guard_release_drawdown": annual_guard_release_key,
            "equity_overlay": overlay_mode_key,
            "overlay_sideways_exposure": overlay_sideways_key,
            "overlay_bear_exposure": overlay_bear_key,
            "overlay_drawdown_cut": overlay_drawdown_key,
            "bull_exposure": bull_exposure,
            "sideways_exposure": sideways_exposure,
            "bear_exposure": bear_exposure,
            "bear_drawdown_threshold": bear_drawdown_threshold,
            "bull_defensive_weight": bull_defensive_weight,
            "sideways_defensive_weight": sideways_defensive_weight,
            "bear_defensive_weight": bear_defensive_weight,
            "defensive_timing": args.defensive_timing,
            "seconds": time.monotonic() - row_start,
            **result.metrics,
        }
        row.update(_target_quality_fields(result.metrics, yearly, yearly_coverage, quality, args.target_annual_return, args.drawdown_limit))
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


def _factor_group_values(value: str, config: dict[str, Any]) -> list[str]:
    values = _csv_values(value, str)
    if values:
        return values
    return [str(config.get("strategy", {}).get("factor_group", "momentum"))]


def _with_dynamic_candidates_override(config: dict[str, Any], candidates_value: str) -> dict[str, Any]:
    candidates = _csv_values(candidates_value, str)
    if not candidates:
        return config
    result = deepcopy(config)
    result.setdefault("dynamic_ic_selector", {})["candidates"] = candidates
    return result


def _liquidity_filter_state(side: str, quantile: float) -> tuple[bool, str, float]:
    normalized = str(side).strip().lower()
    if normalized in LIQUIDITY_DISABLED_SIDES:
        return False, "none", 0.0
    return True, normalized, float(quantile)


def _annual_guard_state(
    drawdown: float | None,
    target_exposure: float,
    release_drawdown: float | None,
) -> tuple[bool, float | None, float, float | None]:
    if drawdown is None:
        return False, None, 0.0, None
    return True, abs(float(drawdown)), max(0.0, min(float(target_exposure), 1.0)), release_drawdown


def _annual_guard_config(
    enabled: bool,
    drawdown: float | None,
    target_exposure: float,
    release_drawdown: float | None,
) -> dict[str, object]:
    if not enabled:
        return {"enabled": False}
    result: dict[str, object] = {
        "enabled": True,
        "drawdown": drawdown,
        "target_exposure": target_exposure,
    }
    if release_drawdown is not None:
        result["release_drawdown"] = release_drawdown
    return result


def _equity_overlay_state(
    mode: str,
    sideways_exposure: float,
    bear_exposure: float,
    drawdown_cut: float,
) -> tuple[str, float, float, float]:
    normalized = str(mode).strip().lower()
    if normalized == "disabled":
        return "disabled", 1.0, 1.0, 0.0
    if normalized not in {"config", "enabled"}:
        raise ValueError(f"Unsupported equity overlay mode: {mode}")
    return (
        normalized,
        max(0.0, min(float(sideways_exposure), 1.0)),
        max(0.0, min(float(bear_exposure), 1.0)),
        abs(float(drawdown_cut)),
    )


def _with_equity_overlay_overrides(
    bt_config: dict[str, Any],
    mode: str,
    sideways_exposure: float,
    bear_exposure: float,
    drawdown_cut: float,
) -> dict[str, Any]:
    result = dict(bt_config)
    if mode == "config":
        return result
    if mode == "disabled":
        result["equity_overlay"] = {"enabled": False}
        return result
    overlay = dict(result.get("equity_overlay", {}))
    overlay.update(
        {
            "enabled": True,
            "sideways_exposure": sideways_exposure,
            "bear_exposure": bear_exposure,
            "drawdown_cut": drawdown_cut,
        }
    )
    result["equity_overlay"] = overlay
    return result


def _requested_factor_columns_for_groups(factor_file: str, config: dict[str, Any], factor_groups: list[str]) -> list[str] | None:
    requested: set[str] = set()
    for factor_group in factor_groups:
        strategy_cfg = dict(config.get("strategy", {}))
        strategy_cfg["factor_group"] = factor_group
        columns = _requested_factor_columns(
            factor_file,
            strategy_cfg,
            config.get("dynamic_ic_selector", {}),
            config.get("ml_strategy", {}),
            config.get("regime_score_blend", {}),
            config.get("regime_score_filter", {}),
        )
        if columns is None:
            return None
        requested.update(str(column) for column in columns)
    return sorted(requested) if requested else None


def _csv_values(value: str, cast: Callable[[str], Any]) -> list[Any]:
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def _csv_optional_values(value: str, cast: Callable[[str], Any]) -> list[Any]:
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


def _csv_bool_values(value: str) -> list[bool]:
    return [_bool_value(item.strip()) for item in str(value).split(",") if item.strip()]


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "n", "off", "disabled"}:
        return False
    raise ValueError(f"Expected boolean value, got {value!r}")


def _with_timing_overrides(
    config: dict[str, Any],
    mode: str,
    bull_exposure: float,
    sideways_exposure: float,
    bear_exposure: float,
    bear_drawdown_threshold: float | None,
) -> dict[str, Any]:
    result = deepcopy(config)
    result.setdefault("defensive_timing", {})
    if mode != "config":
        result["defensive_timing"]["enabled"] = mode == "enabled"
    result["defensive_timing"]["bull_exposure"] = bull_exposure
    result["defensive_timing"]["sideways_exposure"] = sideways_exposure
    result["defensive_timing"]["bear_exposure"] = bear_exposure
    result.setdefault("market_regime", {})["bear_drawdown_threshold"] = bear_drawdown_threshold
    return result


def _combo_key(
    factor_group: str,
    liquidity_side: str,
    liquidity_quantile: float,
    top_n: int,
    max_turnover: int,
    rank_buffer: int,
    max_industry_weight: float | None,
    score_weighted: bool,
    stop_loss: float | None,
    take_profit: float | None,
    circuit_drawdown: float | None,
    cooldown_days: int,
    target_exposure: float,
    rebalance_drift_threshold: float,
    annual_guard_enabled: bool,
    annual_guard_drawdown: float | None,
    annual_guard_target_exposure: float,
    annual_guard_release_drawdown: float | None,
    equity_overlay_mode: str,
    overlay_sideways_exposure: float,
    overlay_bear_exposure: float,
    overlay_drawdown_cut: float,
    defensive_timing: str,
    bull_exposure: float,
    sideways_exposure: float,
    bear_exposure: float,
    bear_drawdown_threshold: float | None,
    bull_defensive_weight: float,
    sideways_defensive_weight: float,
    bear_defensive_weight: float,
) -> tuple[
    str,
    str,
    float,
    int,
    int,
    int,
    float | None,
    bool,
    float | None,
    float | None,
    float | None,
    int,
    float,
    float,
    bool,
    float | None,
    float,
    float | None,
    str,
    float,
    float,
    float,
    str,
    float,
    float,
    float,
    float | None,
    float,
    float,
    float,
]:
    return (
        str(factor_group).strip().lower(),
        str(liquidity_side).strip().lower(),
        round(float(liquidity_quantile), 6),
        int(top_n),
        int(max_turnover),
        int(rank_buffer),
        _optional_key(max_industry_weight),
        _bool_value(score_weighted),
        _optional_key(stop_loss),
        _optional_key(take_profit),
        _optional_key(circuit_drawdown),
        int(cooldown_days),
        round(float(target_exposure), 6),
        round(float(rebalance_drift_threshold), 6),
        _bool_value(annual_guard_enabled),
        _optional_key(annual_guard_drawdown),
        round(float(annual_guard_target_exposure), 6),
        _optional_key(annual_guard_release_drawdown),
        str(equity_overlay_mode).strip().lower(),
        round(float(overlay_sideways_exposure), 6),
        round(float(overlay_bear_exposure), 6),
        round(float(overlay_drawdown_cut), 6),
        str(defensive_timing).strip().lower(),
        round(float(bull_exposure), 6),
        round(float(sideways_exposure), 6),
        round(float(bear_exposure), 6),
        _optional_key(bear_drawdown_threshold),
        round(float(bull_defensive_weight), 6),
        round(float(sideways_defensive_weight), 6),
        round(float(bear_defensive_weight), 6),
    )


def _optional_key(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 6)


def _completed_keys(
    output_path: Path,
) -> set[
    tuple[
        str,
        str,
        float,
        int,
        int,
        int,
        float | None,
        bool,
        float | None,
        float | None,
        float | None,
        int,
        float,
        float,
        bool,
        float | None,
        float,
        float | None,
        str,
        float,
        float,
        float,
        str,
        float,
        float,
        float,
        float | None,
        float,
        float,
        float,
    ]
]:
    frame = _read_existing(output_path)
    if frame.empty:
        return set()
    required = [
        "factor_group",
        "liquidity_side",
        "liquidity_quantile",
        "top_n",
        "max_turnover",
        "rank_buffer",
        "max_industry_weight",
        "score_weighted",
        "stop_loss_pct",
        "take_profit_pct",
        "circuit_breaker_drawdown",
        "circuit_breaker_cooldown_days",
        "circuit_breaker_target_exposure",
        "rebalance_drift_threshold",
        "annual_guard_enabled",
        "annual_guard_drawdown",
        "annual_guard_target_exposure",
        "annual_guard_release_drawdown",
        "equity_overlay",
        "overlay_sideways_exposure",
        "overlay_bear_exposure",
        "overlay_drawdown_cut",
        "defensive_timing",
        "bull_exposure",
        "sideways_exposure",
        "bear_exposure",
        "bear_drawdown_threshold",
        "bull_defensive_weight",
        "sideways_defensive_weight",
        "bear_defensive_weight",
    ]
    if any(column not in frame.columns for column in required):
        return set()
    return {
        _combo_key(
            row["factor_group"],
            row["liquidity_side"],
            row["liquidity_quantile"],
            row["top_n"],
            row["max_turnover"],
            row["rank_buffer"],
            row["max_industry_weight"],
            row["score_weighted"],
            row["stop_loss_pct"],
            row["take_profit_pct"],
            row["circuit_breaker_drawdown"],
            row["circuit_breaker_cooldown_days"],
            row["circuit_breaker_target_exposure"],
            row["rebalance_drift_threshold"],
            row["annual_guard_enabled"],
            row["annual_guard_drawdown"],
            row["annual_guard_target_exposure"],
            row["annual_guard_release_drawdown"],
            row["equity_overlay"],
            row["overlay_sideways_exposure"],
            row["overlay_bear_exposure"],
            row["overlay_drawdown_cut"],
            row["defensive_timing"],
            row["bull_exposure"],
            row["sideways_exposure"],
            row["bear_exposure"],
            row["bear_drawdown_threshold"],
            row["bull_defensive_weight"],
            row["sideways_defensive_weight"],
            row["bear_defensive_weight"],
        )
        for _, row in frame.iterrows()
    }


def _read_existing(output_path: Path) -> pd.DataFrame:
    if not output_path.exists():
        return pd.DataFrame()
    return pd.read_csv(output_path)


def _best_rows(rows: pd.DataFrame, target_annual_return: float, drawdown_limit: float) -> pd.DataFrame:
    frame = rows.copy()
    for column in ["annual_return", "max_drawdown", "sharpe", "calmar", "min_year_annual_return", "worst_year_drawdown"]:
        frame[column] = _numeric_column(frame, column, 0.0)
    min_yearly_return = _numeric_column(frame, "min_yearly_annual_return", target_annual_return)
    max_yearly_drawdown = _numeric_column(frame, "max_yearly_drawdown_limit", drawdown_limit)
    frame["year_coverage_pass"] = _bool_column(frame, "year_coverage_pass", False)
    frame["target_distance"] = (frame["annual_return"] - target_annual_return).clip(upper=0).abs()
    frame["drawdown_shortfall"] = (drawdown_limit - frame["max_drawdown"]).clip(lower=0)
    frame["yearly_return_shortfall"] = (min_yearly_return - frame["min_year_annual_return"]).clip(lower=0)
    frame["yearly_drawdown_shortfall"] = (max_yearly_drawdown - frame["worst_year_drawdown"]).clip(lower=0)
    return frame.sort_values(
        [
            "meets_target",
            "year_coverage_pass",
            "yearly_return_shortfall",
            "yearly_drawdown_shortfall",
            "drawdown_shortfall",
            "target_distance",
            "max_drawdown",
            "sharpe",
        ],
        ascending=[False, False, True, True, True, True, False, False],
    ).head(10)


def _numeric_column(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(float(default), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(float(default))


def _bool_column(frame: pd.DataFrame, column: str, default: bool) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(bool(default), index=frame.index, dtype=bool)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(bool(default)).astype(bool)
    normalized = values.fillna(str(default)).astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y"})


def _target_quality_fields(
    metrics: dict[str, Any],
    yearly: pd.DataFrame,
    yearly_coverage: pd.DataFrame,
    quality_cfg: dict[str, Any],
    target_annual_return: float,
    drawdown_limit: float,
) -> dict[str, Any]:
    yearly_fields = _yearly_quality_fields(yearly, yearly_coverage, quality_cfg, target_annual_return, drawdown_limit)
    annual_return_gap = _number(metrics.get("annual_return")) - float(target_annual_return)
    drawdown_buffer = _number(metrics.get("max_drawdown")) - float(drawdown_limit)
    yearly_return_gap = float(yearly_fields["min_year_annual_return"]) - float(yearly_fields["min_yearly_annual_return"])
    yearly_drawdown_buffer = float(yearly_fields["worst_year_drawdown"]) - float(yearly_fields["max_yearly_drawdown_limit"])
    return {
        **yearly_fields,
        "annual_return_gap": annual_return_gap,
        "drawdown_buffer": drawdown_buffer,
        "yearly_return_gap": yearly_return_gap,
        "yearly_drawdown_buffer": yearly_drawdown_buffer,
        "meets_target": bool(
            annual_return_gap >= 0
            and drawdown_buffer >= 0
            and bool(yearly_fields["year_coverage_pass"])
            and bool(yearly_fields["yearly_annual_return_pass"])
            and bool(yearly_fields["yearly_drawdown_pass"])
        ),
    }


def _yearly_quality_fields(
    yearly: pd.DataFrame,
    yearly_coverage: pd.DataFrame | None,
    quality_cfg: dict[str, Any],
    target_annual_return: float,
    drawdown_limit: float,
) -> dict[str, Any]:
    min_yearly_return = float(quality_cfg.get("min_yearly_annual_return", target_annual_return))
    max_yearly_drawdown = float(quality_cfg.get("max_yearly_drawdown_limit", drawdown_limit))
    fields: dict[str, Any] = {
        "year_count": 0,
        "expected_year_count": 0,
        "year_ann_pass": 0,
        "year_dd_pass": 0,
        "min_year_annual_return": 0.0,
        "worst_year_drawdown": 0.0,
        "min_yearly_annual_return": min_yearly_return,
        "max_yearly_drawdown_limit": max_yearly_drawdown,
        "year_coverage_pass": False,
        "missing_years": "",
        "years_below_return_target": "",
        "years_breaching_drawdown_limit": "",
        "yearly_annual_return_pass": False,
        "yearly_drawdown_pass": False,
    }
    if yearly is not None and not yearly.empty:
        annual = pd.to_numeric(yearly.get("annual_return", pd.Series(dtype=float)), errors="coerce")
        drawdown = pd.to_numeric(yearly.get("max_drawdown", pd.Series(dtype=float)), errors="coerce")
        valid = annual.notna() & drawdown.notna()
        fields["year_count"] = int(valid.sum())
        if int(valid.sum()):
            fields["year_ann_pass"] = int((annual[valid] >= min_yearly_return).sum())
            fields["year_dd_pass"] = int((drawdown[valid] >= max_yearly_drawdown).sum())
            fields["min_year_annual_return"] = float(annual[valid].min())
            fields["worst_year_drawdown"] = float(drawdown[valid].min())
            fields["yearly_annual_return_pass"] = bool(fields["year_ann_pass"] == fields["year_count"])
            fields["yearly_drawdown_pass"] = bool(fields["year_dd_pass"] == fields["year_count"])
            if "year" in yearly.columns:
                years = pd.to_numeric(yearly["year"], errors="coerce")
                return_fail_years = years[valid & (annual < min_yearly_return)].dropna().astype(int).astype(str)
                drawdown_fail_years = years[valid & (drawdown < max_yearly_drawdown)].dropna().astype(int).astype(str)
                fields["years_below_return_target"] = ",".join(return_fail_years.tolist())
                fields["years_breaching_drawdown_limit"] = ",".join(drawdown_fail_years.tolist())
    if yearly_coverage is None:
        fields["expected_year_count"] = int(fields["year_count"])
        fields["year_coverage_pass"] = bool(fields["year_count"] > 0)
        return fields
    fields["expected_year_count"] = int(len(yearly_coverage))
    if yearly_coverage.empty:
        return fields
    if "passes_min_days" in yearly_coverage.columns:
        coverage = yearly_coverage["passes_min_days"].fillna(False).astype(bool)
    elif "has_equity" in yearly_coverage.columns:
        coverage = yearly_coverage["has_equity"].fillna(False).astype(bool)
    else:
        coverage = pd.Series(False, index=yearly_coverage.index)
    missing: list[str] = []
    if not coverage.all() and "year" in yearly_coverage.columns:
        missing.extend(yearly_coverage.loc[~coverage, "year"].dropna().astype(int).astype(str).tolist())
    if "year" in yearly_coverage.columns:
        expected_years = set(yearly_coverage["year"].dropna().astype(int).astype(str))
        observed_years = set(yearly["year"].dropna().astype(int).astype(str)) if yearly is not None and not yearly.empty and "year" in yearly.columns else set()
        missing.extend(sorted(expected_years - observed_years))
    missing = sorted(set(missing))
    fields["missing_years"] = ",".join(missing)
    fields["year_coverage_pass"] = bool(coverage.all() and not missing)
    return fields


def _yearly_stats(equity_curve: pd.Series) -> pd.DataFrame:
    if equity_curve.empty:
        return pd.DataFrame(columns=["year", "days", "annual_return", "max_drawdown"])
    rows: list[dict[str, object]] = []
    equity = equity_curve.sort_index().astype(float)
    equity.index = pd.to_datetime(equity.index).normalize()
    for year, group in equity.groupby(equity.index.year):
        if len(group) <= 1 or float(group.iloc[0]) <= 0:
            continue
        total_return = float(group.iloc[-1] / group.iloc[0] - 1.0)
        calendar_days = max(int((group.index[-1] - group.index[0]).days), 1)
        annual_return = float((1.0 + total_return) ** (365.25 / calendar_days) - 1.0) if total_return > -1 else -1.0
        drawdown = group / group.cummax() - 1.0
        rows.append(
            {
                "year": int(year),
                "days": int(len(group)),
                "annual_return": annual_return,
                "max_drawdown": float(drawdown.min()),
            }
        )
    return pd.DataFrame(rows)


def _number(value: object) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return 0.0
    return float(parsed)


if __name__ == "__main__":
    main()
