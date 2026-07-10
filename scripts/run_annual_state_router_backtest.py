"""Backtest an annual market-state router from real score sources."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._shared import dated_output_path, yearly_stats
from scripts.run_annual_state_router_probe import route_for_date
from scripts.run_fundamental_quality_backtest import month_end_signal_dates
from scripts.run_goal_fast_factor_screen import _read_factor_subset
from scripts.run_goal_audit import audit_yearly_goal, goal_thresholds, write_audit_outputs
from scripts.run_quality_selector_gate_backtest import build_quality_scores, daily_score_for_date
from scripts.run_selector_weight_backtest import apply_selector_directions, selector_weights_from_frame
from src.backtest import run_backtest
from src.config_loader import load_config, resolve_path
from src.market_regime import _benchmark_close
from src.risk_policy import RiskPolicy
from src.scoring import _apply_liquidity_filter, build_strategy_scores
from src.strategy import composite_factor, resample_signals
from src.trading_calendar import resolve_target_date_value


DEFAULT_EXTENDED_FACTOR_FILE = "data/factors/codex_goal_extended_factors_20260610.parquet"
DEFAULT_INDUSTRY_FACTOR_FILE = "data/factors/codex_goal_industry_momentum_factors_20260611.parquet"
DEFAULT_SELECTOR_FILE = "outputs/selector_weight_lb63_top5_posprop_top5_formal_20260611_selector.csv"


@dataclass(frozen=True)
class ScoreSourceDefinition:
    name: str
    kind: str
    factor_group: str = ""
    factor_file: str = ""
    selector_file: str = ""
    top_n: int = 5
    max_turnover: int = 1
    rank_buffer: int = 20
    liquidity_quantile: float | None = None


@dataclass(frozen=True)
class RoutedScoreRun:
    scores: pd.Series
    score_routes: pd.DataFrame
    year_routes: pd.DataFrame


def apply_research_config_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    result = dict(config)
    max_industry_weight = getattr(args, "max_industry_weight", None)
    if max_industry_weight is not None:
        strategy = dict(result.get("strategy", {}))
        strategy["max_industry_weight"] = float(max_industry_weight)
        result["strategy"] = strategy
    if getattr(args, "rebalance_after_risk_exit", False):
        strategy = dict(result.get("strategy", {}))
        strategy["rebalance_after_risk_exit"] = True
        result["strategy"] = strategy
    risk_exit_min_positions = getattr(args, "risk_exit_min_positions", None)
    if risk_exit_min_positions is not None:
        strategy = dict(result.get("strategy", {}))
        strategy["risk_exit_min_positions"] = int(risk_exit_min_positions)
        reasons = parse_reason_list(getattr(args, "risk_exit_min_positions_reasons", ""))
        if reasons:
            strategy["risk_exit_min_positions_reasons"] = sorted(reasons)
        result["strategy"] = strategy

    backtest = dict(result.get("backtest", {}))
    overlay = dict(backtest.get("equity_overlay", {}))
    if args.equity_overlay_sideways_exposure is not None:
        overlay["sideways_exposure"] = float(args.equity_overlay_sideways_exposure)
    if args.equity_overlay_bear_exposure is not None:
        overlay["bear_exposure"] = float(args.equity_overlay_bear_exposure)
    if args.equity_overlay_drawdown_cut is not None:
        overlay["drawdown_cut"] = float(args.equity_overlay_drawdown_cut)
    if overlay:
        backtest["equity_overlay"] = overlay
    if backtest:
        result["backtest"] = backtest

    defensive = dict(result.get("defensive_timing", {}))
    if args.defensive_sideways_exposure is not None:
        defensive["sideways_exposure"] = float(args.defensive_sideways_exposure)
    if args.defensive_bear_exposure is not None:
        defensive["bear_exposure"] = float(args.defensive_bear_exposure)
    if defensive:
        result["defensive_timing"] = defensive
    return result


def research_config_overrides_payload(args: argparse.Namespace) -> dict[str, float | bool | str]:
    values = {
        "max_industry_weight": getattr(args, "max_industry_weight", None),
        "rebalance_after_risk_exit": getattr(args, "rebalance_after_risk_exit", False),
        "risk_exit_min_positions": getattr(args, "risk_exit_min_positions", None),
        "equity_overlay_sideways_exposure": args.equity_overlay_sideways_exposure,
        "equity_overlay_bear_exposure": args.equity_overlay_bear_exposure,
        "equity_overlay_drawdown_cut": args.equity_overlay_drawdown_cut,
        "defensive_sideways_exposure": args.defensive_sideways_exposure,
        "defensive_bear_exposure": args.defensive_bear_exposure,
    }
    payload = {key: value if isinstance(value, bool) else float(value) for key, value in values.items() if value is not None and value is not False}
    reasons = "+".join(sorted(parse_reason_list(getattr(args, "risk_exit_min_positions_reasons", ""))))
    if reasons:
        payload["risk_exit_min_positions_reasons"] = reasons
    return payload


def source_top_n_overrides_payload(args: argparse.Namespace) -> dict[str, int]:
    values = {
        "beta": getattr(args, "beta_top_n", None),
        "beta20": getattr(args, "beta20_top_n", None),
    }
    return {source: int(value) for source, value in values.items() if value is not None}


def apply_source_top_n_overrides(
    definitions: dict[str, ScoreSourceDefinition],
    overrides: dict[str, int],
) -> dict[str, ScoreSourceDefinition]:
    result = dict(definitions)
    for source, raw_top_n in overrides.items():
        if source not in result:
            raise ValueError(f"source_top_n override references unknown source: {source}")
        top_n = int(raw_top_n)
        if top_n < 1:
            raise ValueError("source_top_n overrides must be >= 1.")
        definition = result[source]
        result[source] = replace(
            definition,
            top_n=top_n,
            max_turnover=min(top_n, int(definition.max_turnover)),
        )
    return result


def parse_reason_list(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    normalized = str(value or "").replace("+", ",")
    return {item.strip() for item in normalized.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest annual market-state routing over reproducible score sources."
    )
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--factor-file", default=DEFAULT_EXTENDED_FACTOR_FILE)
    parser.add_argument("--industry-factor-file", default=DEFAULT_INDUSTRY_FACTOR_FILE)
    parser.add_argument("--selector-file", default=DEFAULT_SELECTOR_FILE)
    parser.add_argument("--initial-source", default="beta")
    parser.add_argument("--missing-ret252-exposure", type=float, default=0.65)
    parser.add_argument("--flat-negative-exposure", type=float, default=0.90)
    parser.add_argument("--fallback-source", default="")
    parser.add_argument("--full-turnover-on-route-change", action="store_true")
    parser.add_argument("--use-defensive-timing", action="store_true")
    parser.add_argument("--include-expanded-sources", action="store_true")
    parser.add_argument("--moderate-positive-source", default="")
    parser.add_argument("--moderate-positive-ret252-min", type=float, default=0.20)
    parser.add_argument("--moderate-positive-exposure", type=float, default=1.0)
    parser.add_argument("--moderate-low-source", default="")
    parser.add_argument("--moderate-low-ret252-min", type=float, default=0.18)
    parser.add_argument("--moderate-low-ret252-max", type=float, default=0.20)
    parser.add_argument("--moderate-low-exposure", type=float, default=1.0)
    parser.add_argument("--moderate-lower-source", default="")
    parser.add_argument("--moderate-lower-ret252-min", type=float, default=0.16)
    parser.add_argument("--moderate-lower-ret252-max", type=float, default=0.18)
    parser.add_argument("--moderate-lower-exposure", type=float, default=1.0)
    parser.add_argument("--strong-trailing-exposure", type=float, default=1.0)
    parser.add_argument("--disable-equity-overlay", action="store_true")
    parser.add_argument("--equity-overlay-sideways-exposure", type=float, default=None)
    parser.add_argument("--equity-overlay-bear-exposure", type=float, default=None)
    parser.add_argument("--equity-overlay-drawdown-cut", type=float, default=None)
    parser.add_argument("--defensive-sideways-exposure", type=float, default=None)
    parser.add_argument("--defensive-bear-exposure", type=float, default=None)
    parser.add_argument("--max-industry-weight", type=float, default=None)
    parser.add_argument("--rebalance-after-risk-exit", action="store_true")
    parser.add_argument("--risk-exit-min-positions", type=int, default=None)
    parser.add_argument("--risk-exit-min-positions-reasons", default="")
    parser.add_argument("--beta-top-n", type=int, default=None)
    parser.add_argument("--beta20-top-n", type=int, default=None)
    parser.add_argument("--turnover-boost-reasons", default="")
    parser.add_argument("--turnover-boost-max-turnover", type=int, default=2)
    parser.add_argument("--turnover-boost-rank-buffer", type=int, default=10)
    parser.add_argument("--output-prefix", default=dated_output_path("annual_state_router_backtest", suffix=""))
    args = parser.parse_args()

    config = apply_research_config_overrides(load_config(), args)
    start_date = args.start_date or config["data"]["start_date"]
    end_date = resolve_target_date_value(args.end_date or config["data"]["end_date"], config=config)
    output_prefix = resolve_path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    prices = pd.read_parquet(resolve_path(config.get("ic", {}).get("price_file", "data/prices/ohlcv_adjusted.parquet")))
    signal_dates = month_end_signal_dates(prices.index, start_date=start_date, end_date=end_date)
    source_definitions = default_source_definitions(
        factor_file=args.factor_file,
        industry_factor_file=args.industry_factor_file,
        selector_file=args.selector_file,
        include_expanded_sources=bool(args.include_expanded_sources),
    )
    source_definitions = apply_source_top_n_overrides(source_definitions, source_top_n_overrides_payload(args))
    score_sources = build_score_sources(
        config=config,
        prices=prices,
        signal_dates=signal_dates,
        start_date=start_date,
        end_date=end_date,
        source_definitions=source_definitions,
    )
    benchmark = _benchmark_close(prices, config, config.get("market_regime", {})).dropna().sort_index()
    if benchmark.empty:
        raise ValueError("Benchmark close series is empty; cannot route by annual market state.")

    routed = run_annual_state_score_router(
        score_sources=score_sources,
        source_definitions=source_definitions,
        price_dates=pd.DatetimeIndex(pd.to_datetime(prices.index).normalize()),
        benchmark=benchmark,
        initial_source=args.initial_source,
        missing_ret252_exposure=args.missing_ret252_exposure,
        flat_negative_exposure=args.flat_negative_exposure,
        fallback_source=args.fallback_source or None,
        moderate_positive_source=args.moderate_positive_source or None,
        moderate_positive_ret252_min=args.moderate_positive_ret252_min,
        moderate_positive_exposure=args.moderate_positive_exposure,
        moderate_low_source=args.moderate_low_source or None,
        moderate_low_ret252_min=args.moderate_low_ret252_min,
        moderate_low_ret252_max=args.moderate_low_ret252_max,
        moderate_low_exposure=args.moderate_low_exposure,
        moderate_lower_source=args.moderate_lower_source or None,
        moderate_lower_ret252_min=args.moderate_lower_ret252_min,
        moderate_lower_ret252_max=args.moderate_lower_ret252_max,
        moderate_lower_exposure=args.moderate_lower_exposure,
        strong_trailing_exposure=args.strong_trailing_exposure,
        turnover_boost_reasons=parse_reason_list(args.turnover_boost_reasons),
        turnover_boost_max_turnover=args.turnover_boost_max_turnover,
        turnover_boost_rank_buffer=args.turnover_boost_rank_buffer,
    )
    if routed.scores.empty:
        raise ValueError("Routed score panel is empty.")

    bt_config = routed_backtest_config(
        config=config,
        prices=prices,
        routed=routed,
        source_definitions=source_definitions,
        full_turnover_on_route_change=bool(args.full_turnover_on_route_change),
        use_defensive_timing=bool(args.use_defensive_timing),
        disable_equity_overlay=bool(args.disable_equity_overlay),
    )
    result = run_backtest(routed.scores, prices, start_date, end_date, bt_config)
    yearly = yearly_stats(result.equity_curve, bt_config)
    return_target, drawdown_limit = goal_thresholds(config)
    audited_yearly, audit_summary = audit_yearly_goal(
        yearly,
        return_target=return_target,
        drawdown_limit=drawdown_limit,
    )
    full_gate = full_gate_summary(
        metrics=result.metrics,
        audit_summary=audit_summary,
        config=config,
        return_target=return_target,
        drawdown_limit=drawdown_limit,
    )

    result.equity_curve.to_csv(Path(str(output_prefix) + "_equity.csv"), encoding="utf-8-sig")
    result.holdings.to_csv(Path(str(output_prefix) + "_holdings.csv"), index=False, encoding="utf-8-sig")
    result.trades.to_csv(Path(str(output_prefix) + "_trades.csv"), index=False, encoding="utf-8-sig")
    routed.score_routes.to_csv(Path(str(output_prefix) + "_score_routes.csv"), index=False, encoding="utf-8-sig")
    routed.year_routes.to_csv(Path(str(output_prefix) + "_year_routes.csv"), index=False, encoding="utf-8-sig")
    audited_yearly.to_csv(Path(str(output_prefix) + "_years.csv"), index=False, encoding="utf-8-sig")
    payload = {
        "metrics": result.metrics,
        "audit": audit_summary,
        "full_gate": full_gate,
        "source_definitions": {name: asdict(definition) for name, definition in source_definitions.items()},
        "score_rows": int(len(routed.scores)),
        "score_route_counts": routed.score_routes["source"].value_counts().to_dict() if not routed.score_routes.empty else {},
        "full_turnover_on_route_change": bool(args.full_turnover_on_route_change),
        "use_defensive_timing": bool(args.use_defensive_timing),
        "include_expanded_sources": bool(args.include_expanded_sources),
        "moderate_positive_source": args.moderate_positive_source or None,
        "moderate_positive_ret252_min": args.moderate_positive_ret252_min,
        "moderate_positive_exposure": args.moderate_positive_exposure,
        "moderate_low_source": args.moderate_low_source or None,
        "moderate_low_ret252_min": args.moderate_low_ret252_min,
        "moderate_low_ret252_max": args.moderate_low_ret252_max,
        "moderate_low_exposure": args.moderate_low_exposure,
        "moderate_lower_source": args.moderate_lower_source or None,
        "moderate_lower_ret252_min": args.moderate_lower_ret252_min,
        "moderate_lower_ret252_max": args.moderate_lower_ret252_max,
        "moderate_lower_exposure": args.moderate_lower_exposure,
        "strong_trailing_exposure": args.strong_trailing_exposure,
        "disable_equity_overlay": bool(args.disable_equity_overlay),
        "research_config_overrides": research_config_overrides_payload(args),
        "source_top_n_overrides": source_top_n_overrides_payload(args),
        "turnover_boost_reasons": sorted(parse_reason_list(args.turnover_boost_reasons)),
        "turnover_boost_max_turnover": args.turnover_boost_max_turnover,
        "turnover_boost_rank_buffer": args.turnover_boost_rank_buffer,
        "note": (
            "Research backtest: annual routing uses benchmark data before the first trading day of each "
            "calendar year and builds holdings/trades from source score panels. Source set and route "
            "thresholds remain exploratory until separately validated."
        ),
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
        f"annual_state_router_backtest annual={result.metrics.get('annual_return', 0.0):.4f} "
        f"dd={result.metrics.get('max_drawdown', 0.0):.4f} "
        f"yearly={audit_summary['year_return_pass_count']}/{audit_summary['year_drawdown_pass_count']} "
        f"full_goal={full_gate['is_full_goal_met']}"
    )
    print(f"wrote prefix: {output_prefix}")


def default_source_definitions(
    *,
    factor_file: str,
    industry_factor_file: str,
    selector_file: str,
    include_expanded_sources: bool = False,
) -> dict[str, ScoreSourceDefinition]:
    definitions = {
        "beta": ScoreSourceDefinition(
            name="beta",
            kind="factor",
            factor_group="factor:BETA60",
            factor_file=factor_file,
            top_n=5,
            max_turnover=1,
            rank_buffer=10,
            liquidity_quantile=0.65,
        ),
        "db_size": ScoreSourceDefinition(
            name="db_size",
            kind="factor",
            factor_group="inverse_factor:DB_circ_mv",
            factor_file=factor_file,
            top_n=7,
            max_turnover=1,
            rank_buffer=15,
            liquidity_quantile=0.80,
        ),
        "quality": ScoreSourceDefinition(
            name="quality",
            kind="quality",
            top_n=5,
            max_turnover=5,
            rank_buffer=20,
        ),
        "selector": ScoreSourceDefinition(
            name="selector",
            kind="selector",
            factor_file=factor_file,
            selector_file=selector_file,
            top_n=5,
            max_turnover=1,
            rank_buffer=20,
            liquidity_quantile=0.65,
        ),
        "industry": ScoreSourceDefinition(
            name="industry",
            kind="factor",
            factor_group="factor:IND_ROC120",
            factor_file=industry_factor_file,
            top_n=10,
            max_turnover=1,
            rank_buffer=20,
            liquidity_quantile=0.65,
        ),
    }
    if include_expanded_sources:
        definitions.update(
            {
                "roc60": ScoreSourceDefinition(
                    name="roc60",
                    kind="factor",
                    factor_group="factor:ROC60",
                    factor_file=factor_file,
                    top_n=7,
                    max_turnover=1,
                    rank_buffer=20,
                    liquidity_quantile=0.65,
                ),
                "db_total": ScoreSourceDefinition(
                    name="db_total",
                    kind="factor",
                    factor_group="inverse_factor:DB_total_mv",
                    factor_file=factor_file,
                    top_n=10,
                    max_turnover=1,
                    rank_buffer=20,
                    liquidity_quantile=None,
                ),
                "beta20": ScoreSourceDefinition(
                    name="beta20",
                    kind="factor",
                    factor_group="factor:BETA20",
                    factor_file=factor_file,
                    top_n=5,
                    max_turnover=1,
                    rank_buffer=10,
                    liquidity_quantile=0.80,
                ),
                "rsqr20": ScoreSourceDefinition(
                    name="rsqr20",
                    kind="factor",
                    factor_group="factor:RSQR20",
                    factor_file=factor_file,
                    top_n=7,
                    max_turnover=1,
                    rank_buffer=10,
                    liquidity_quantile=0.80,
                ),
            }
        )
    return definitions


def build_score_sources(
    *,
    config: dict[str, Any],
    prices: pd.DataFrame,
    signal_dates: list[pd.Timestamp],
    start_date: str,
    end_date: str,
    source_definitions: dict[str, ScoreSourceDefinition],
    progress_callback: Callable[[str, int, int, str], None] | None = None,
    signal_dates_by_source: dict[str, list[pd.Timestamp]] | None = None,
) -> dict[str, pd.Series]:
    sources: dict[str, pd.Series] = {}
    total = len(source_definitions)
    for index, (name, definition) in enumerate(source_definitions.items(), start=1):
        if progress_callback is not None:
            progress_callback(name, index, total, "running")
        if definition.kind == "factor":
            sources[name] = build_factor_source_scores(
                config=config,
                prices=prices,
                source=definition,
                start_date=start_date,
                end_date=end_date,
            )
        elif definition.kind == "quality":
            source_dates = (signal_dates_by_source or {}).get(name, signal_dates)
            sources[name] = build_quality_scores(config, source_dates)
        elif definition.kind == "selector":
            sources[name] = build_selector_source_scores(
                config=config,
                prices=prices,
                source=definition,
                start_date=start_date,
                end_date=end_date,
            )
        else:
            raise ValueError(f"Unsupported score source kind: {definition.kind}")
        if progress_callback is not None:
            progress_callback(name, index, total, "complete")
    return sources


def build_factor_source_scores(
    *,
    config: dict[str, Any],
    prices: pd.DataFrame,
    source: ScoreSourceDefinition,
    start_date: str,
    end_date: str,
) -> pd.Series:
    if not source.factor_group:
        raise ValueError(f"Factor source {source.name} is missing factor_group.")
    column = source.factor_group.split(":", 1)[1] if ":" in source.factor_group else source.factor_group
    strategy = {**config.get("strategy", {}), "factor_group": source.factor_group, "rebalance_freq": "monthly"}
    scoring_config = dict(config)
    scoring_config["strategy"] = strategy
    if source.liquidity_quantile is not None:
        scoring_config["liquidity_filter"] = {
            **config.get("liquidity_filter", {}),
            "enabled": True,
            "side": "high",
            "quantile": float(source.liquidity_quantile),
        }
    factors = _read_factor_subset(source.factor_file, [column], start_date, end_date)
    scores = build_strategy_scores(factors, scoring_config, price_df=prices)
    return resample_signals(scores, "monthly")


def build_selector_source_scores(
    *,
    config: dict[str, Any],
    prices: pd.DataFrame,
    source: ScoreSourceDefinition,
    start_date: str,
    end_date: str,
) -> pd.Series:
    selector = pd.read_csv(resolve_path(source.selector_file))
    weights = selector_weights_from_frame(selector)
    columns = sorted({column for series in weights.values() for column in series.index})
    if not columns:
        raise ValueError(f"Selector source {source.name} did not reference any known factor columns.")
    factors = _read_factor_subset(source.factor_file, columns, start_date, end_date)
    signed_factors = apply_selector_directions(factors, weights)
    scores = composite_factor(
        signed_factors,
        method="ic_weighted",
        factor_weights_dynamic=weights,
        min_obs=int(config.get("strategy", {}).get("min_cross_section_obs", 5)),
    )
    if source.liquidity_quantile is not None:
        liquidity_config = {
            **config.get("liquidity_filter", {}),
            "enabled": True,
            "side": "high",
            "quantile": float(source.liquidity_quantile),
        }
        scores = _apply_liquidity_filter(scores, prices, liquidity_config)
    return resample_signals(scores, "monthly")


def run_annual_state_score_router(
    *,
    score_sources: dict[str, pd.Series],
    source_definitions: dict[str, ScoreSourceDefinition],
    price_dates: pd.DatetimeIndex,
    benchmark: pd.Series,
    initial_source: str,
    missing_ret252_exposure: float,
    flat_negative_exposure: float,
    signal_dates: list[pd.Timestamp] | None = None,
    fallback_source: str | None = None,
    moderate_positive_source: str | None = None,
    moderate_positive_ret252_min: float = 0.20,
    moderate_positive_exposure: float = 1.0,
    moderate_low_source: str | None = None,
    moderate_low_ret252_min: float = 0.18,
    moderate_low_ret252_max: float = 0.20,
    moderate_low_exposure: float = 1.0,
    moderate_lower_source: str | None = None,
    moderate_lower_ret252_min: float = 0.16,
    moderate_lower_ret252_max: float = 0.18,
    moderate_lower_exposure: float = 1.0,
    strong_trailing_exposure: float = 1.0,
    turnover_boost_reasons: set[str] | None = None,
    turnover_boost_max_turnover: int = 2,
    turnover_boost_rank_buffer: int = 10,
) -> RoutedScoreRun:
    if initial_source not in score_sources:
        raise ValueError(f"initial_source is not in score sources: {initial_source}")
    dates = (
        sorted({pd.Timestamp(date).normalize() for date in signal_dates})
        if signal_dates is not None
        else routed_signal_dates(score_sources)
    )
    normalized_price_dates = pd.DatetimeIndex(pd.to_datetime(price_dates).normalize()).unique().sort_values()
    signal_trade_dates = signal_trade_date_map(dates, normalized_price_dates)
    year_routes = annual_route_decisions(
        years=sorted({int(trade_date.year) for trade_date in signal_trade_dates.values()}),
        price_dates=normalized_price_dates,
        benchmark=benchmark,
        initial_source=initial_source,
        missing_ret252_exposure=missing_ret252_exposure,
        flat_negative_exposure=flat_negative_exposure,
        moderate_positive_source=moderate_positive_source,
        moderate_positive_ret252_min=moderate_positive_ret252_min,
        moderate_positive_exposure=moderate_positive_exposure,
        moderate_low_source=moderate_low_source,
        moderate_low_ret252_min=moderate_low_ret252_min,
        moderate_low_ret252_max=moderate_low_ret252_max,
        moderate_low_exposure=moderate_low_exposure,
        moderate_lower_source=moderate_lower_source,
        moderate_lower_ret252_min=moderate_lower_ret252_min,
        moderate_lower_ret252_max=moderate_lower_ret252_max,
        moderate_lower_exposure=moderate_lower_exposure,
        strong_trailing_exposure=strong_trailing_exposure,
    )
    route_by_year = {int(row["year"]): row for row in year_routes}
    boost_reasons = set(turnover_boost_reasons or set())
    parts: list[pd.Series] = []
    rows: list[dict[str, Any]] = []
    for date in dates:
        trade_date = signal_trade_dates.get(date)
        if trade_date is None:
            continue
        route_year = int(trade_date.year)
        decision = route_by_year[route_year]
        source = str(decision["source"])
        if source not in score_sources:
            raise ValueError(f"Routed source is not in score sources: {source}")
        actual_source = source
        daily = latest_score_on_or_before(score_sources[source], date)
        fallback_used = False
        if daily.empty and fallback_source:
            if fallback_source not in score_sources:
                raise ValueError(f"fallback_source is not in score sources: {fallback_source}")
            actual_source = fallback_source
            daily = latest_score_on_or_before(score_sources[fallback_source], date)
            fallback_used = True
        if daily.empty:
            raise ValueError(f"No scores for source={source} date={date.date()}.")
        daily.index = pd.MultiIndex.from_product([[date], daily.index.astype(str)], names=["date", "instrument"])
        parts.append(daily.rename("score"))
        definition = source_definitions[actual_source]
        top_n = int(definition.top_n)
        max_turnover = int(definition.max_turnover)
        rank_buffer = int(definition.rank_buffer)
        if str(decision["reason"]) in boost_reasons:
            max_turnover = min(top_n, max(1, int(turnover_boost_max_turnover)))
            rank_buffer = max(0, int(turnover_boost_rank_buffer))
        rows.append(
            {
                "date": date.date().isoformat(),
                "trade_date": trade_date.date().isoformat(),
                "signal_year": int(date.year),
                "year": route_year,
                "source": actual_source,
                "routed_source": source,
                "reason": decision["reason"],
                "fallback_used": fallback_used,
                "scores": int(daily.notna().sum()),
                "top_n": top_n,
                "max_turnover": max_turnover,
                "rank_buffer": rank_buffer,
                "exposure": float(decision["exposure"]),
            }
        )
    scores = pd.concat(parts).sort_index().rename("score") if parts else pd.Series(dtype=float, name="score")
    return RoutedScoreRun(scores=scores, score_routes=pd.DataFrame(rows), year_routes=pd.DataFrame(route_by_year.values()))


def latest_score_on_or_before(scores: pd.Series, date: pd.Timestamp) -> pd.Series:
    """Read the newest point-in-time score cross-section available by a signal date."""
    if scores.empty or not isinstance(scores.index, pd.MultiIndex):
        return pd.Series(dtype=float, name=scores.name)
    target = pd.Timestamp(date).normalize()
    dates = pd.DatetimeIndex(pd.to_datetime(scores.index.get_level_values(0)).normalize())
    eligible = dates[dates <= target]
    if eligible.empty:
        return pd.Series(dtype=float, name=scores.name)
    return daily_score_for_date(scores, pd.Timestamp(eligible.max()).normalize())


def routed_signal_dates(score_sources: dict[str, pd.Series]) -> list[pd.Timestamp]:
    dates: set[pd.Timestamp] = set()
    for source, scores in score_sources.items():
        if scores.empty:
            raise ValueError(f"Score source is empty: {source}")
        if not isinstance(scores.index, pd.MultiIndex):
            raise ValueError(f"Score source must use MultiIndex date/instrument: {source}")
        dates.update(pd.Timestamp(value).normalize() for value in pd.to_datetime(scores.index.get_level_values(0)).unique())
    return sorted(dates)


def signal_trade_date_map(
    signal_dates: list[pd.Timestamp],
    price_dates: pd.DatetimeIndex,
) -> dict[pd.Timestamp, pd.Timestamp]:
    normalized_price_dates = pd.DatetimeIndex(pd.to_datetime(price_dates).normalize()).unique().sort_values()
    result: dict[pd.Timestamp, pd.Timestamp] = {}
    for raw_date in sorted(pd.Timestamp(date).normalize() for date in signal_dates):
        pos = normalized_price_dates.searchsorted(raw_date, side="right")
        if pos >= len(normalized_price_dates):
            continue
        result[raw_date] = pd.Timestamp(normalized_price_dates[pos]).normalize()
    return result


def annual_route_decisions(
    *,
    years: list[int],
    price_dates: pd.DatetimeIndex,
    benchmark: pd.Series,
    initial_source: str,
    missing_ret252_exposure: float,
    flat_negative_exposure: float,
    moderate_positive_source: str | None = None,
    moderate_positive_ret252_min: float = 0.20,
    moderate_positive_exposure: float = 1.0,
    moderate_low_source: str | None = None,
    moderate_low_ret252_min: float = 0.18,
    moderate_low_ret252_max: float = 0.20,
    moderate_low_exposure: float = 1.0,
    moderate_lower_source: str | None = None,
    moderate_lower_ret252_min: float = 0.16,
    moderate_lower_ret252_max: float = 0.18,
    moderate_lower_exposure: float = 1.0,
    strong_trailing_exposure: float = 1.0,
) -> list[dict[str, Any]]:
    benchmark = pd.to_numeric(benchmark, errors="coerce").dropna().sort_index()
    benchmark.index = pd.to_datetime(benchmark.index).normalize()
    benchmark_returns = benchmark.pct_change(fill_method=None)
    normalized_price_dates = pd.DatetimeIndex(pd.to_datetime(price_dates).normalize()).unique().sort_values()
    rows: list[dict[str, Any]] = []
    for year in sorted(set(int(year) for year in years)):
        year_dates = normalized_price_dates[normalized_price_dates.year == year]
        if year_dates.empty:
            raise ValueError(f"No price dates found for route year {year}.")
        route = route_for_date(
            benchmark=benchmark,
            benchmark_returns=benchmark_returns,
            date=pd.Timestamp(year_dates[0]).normalize(),
            initial_source=initial_source,
            missing_ret252_exposure=missing_ret252_exposure,
            flat_negative_exposure=flat_negative_exposure,
        )
        rows.append(
            adjust_route_decision(
                route,
                moderate_positive_source=moderate_positive_source,
                moderate_positive_ret252_min=moderate_positive_ret252_min,
                moderate_positive_exposure=moderate_positive_exposure,
                moderate_low_source=moderate_low_source,
                moderate_low_ret252_min=moderate_low_ret252_min,
                moderate_low_ret252_max=moderate_low_ret252_max,
                moderate_low_exposure=moderate_low_exposure,
                moderate_lower_source=moderate_lower_source,
                moderate_lower_ret252_min=moderate_lower_ret252_min,
                moderate_lower_ret252_max=moderate_lower_ret252_max,
                moderate_lower_exposure=moderate_lower_exposure,
                strong_trailing_exposure=strong_trailing_exposure,
            )
        )
    return rows


def adjust_route_decision(
    route: dict[str, Any],
    *,
    moderate_positive_source: str | None,
    moderate_positive_ret252_min: float,
    moderate_positive_exposure: float = 1.0,
    moderate_low_source: str | None = None,
    moderate_low_ret252_min: float = 0.18,
    moderate_low_ret252_max: float = 0.20,
    moderate_low_exposure: float = 1.0,
    moderate_lower_source: str | None = None,
    moderate_lower_ret252_min: float = 0.16,
    moderate_lower_ret252_max: float = 0.18,
    moderate_lower_exposure: float = 1.0,
    strong_trailing_exposure: float = 1.0,
) -> dict[str, Any]:
    result = dict(route)
    if result.get("reason") == "strong_trailing_market":
        result["exposure"] = float(result.get("exposure", 1.0)) * float(strong_trailing_exposure)
    ret252 = pd.to_numeric(pd.Series([result.get("ret252")]), errors="coerce").iloc[0]
    if (
        moderate_lower_source
        and result.get("reason") == "default_beta"
        and pd.notna(ret252)
        and float(ret252) >= float(moderate_lower_ret252_min)
        and float(ret252) < float(moderate_lower_ret252_max)
    ):
        result["source"] = moderate_lower_source
        result["reason"] = f"moderate_lower_{moderate_lower_source}"
        result["exposure"] = float(result.get("exposure", 1.0)) * float(moderate_lower_exposure)
        return result
    if (
        moderate_low_source
        and result.get("reason") == "default_beta"
        and pd.notna(ret252)
        and float(ret252) >= float(moderate_low_ret252_min)
        and float(ret252) < float(moderate_low_ret252_max)
    ):
        result["source"] = moderate_low_source
        result["reason"] = f"moderate_low_{moderate_low_source}"
        result["exposure"] = float(result.get("exposure", 1.0)) * float(moderate_low_exposure)
        return result
    if (
        moderate_positive_source
        and result.get("reason") == "default_beta"
        and pd.notna(ret252)
        and float(ret252) >= float(moderate_positive_ret252_min)
    ):
        result["source"] = moderate_positive_source
        result["reason"] = f"moderate_positive_{moderate_positive_source}"
        result["exposure"] = float(result.get("exposure", 1.0)) * float(moderate_positive_exposure)
    return result


def routed_backtest_config(
    *,
    config: dict[str, Any],
    prices: pd.DataFrame,
    routed: RoutedScoreRun,
    source_definitions: dict[str, ScoreSourceDefinition],
    full_turnover_on_route_change: bool,
    use_defensive_timing: bool,
    disable_equity_overlay: bool = False,
) -> dict[str, Any]:
    max_top_n = max(int(definition.top_n) for definition in source_definitions.values())
    max_turnover = max(int(definition.max_turnover) for definition in source_definitions.values())
    max_rank_buffer = max(int(definition.rank_buffer) for definition in source_definitions.values())
    bt_config = {
        **config.get("backtest", {}),
        **config.get("strategy", {}),
        "top_n": max_top_n,
        "max_turnover": max_turnover,
        "rank_buffer": max_rank_buffer,
        "rebalance_freq": "monthly",
        "selection_schedule": selection_schedule_from_routes(
            routed.score_routes,
            full_turnover_on_route_change=full_turnover_on_route_change,
        ),
        "exposure_schedule": exposure_schedule_from_year_routes(routed.year_routes),
    }
    router_cfg = config.get("annual_state_router", {}) if isinstance(config.get("annual_state_router", {}), dict) else {}
    strategy_cfg = config.get("strategy", {}) if isinstance(config.get("strategy", {}), dict) else {}
    min_position_reasons = parse_reason_list(
        strategy_cfg.get("risk_exit_min_positions_reasons")
        or router_cfg.get("risk_exit_min_positions_reasons")
        or []
    )
    min_positions = strategy_cfg.get("risk_exit_min_positions", router_cfg.get("risk_exit_min_positions"))
    if min_positions is not None and min_position_reasons:
        bt_config["risk_exit_min_positions"] = 0
        bt_config["risk_exit_min_positions_schedule"] = risk_exit_min_positions_schedule_from_routes(
            routed.score_routes,
            min_positions=int(min_positions),
            reasons=min_position_reasons,
        )
    if use_defensive_timing:
        from src.market_regime import apply_defensive_timing_to_backtest_config

        bt_config = apply_defensive_timing_to_backtest_config(bt_config, prices, config)
    if disable_equity_overlay and isinstance(bt_config.get("equity_overlay"), dict):
        bt_config["equity_overlay"] = {**bt_config["equity_overlay"], "enabled": False}
    return RiskPolicy(config).apply_to_backtest_config(bt_config)


def selection_schedule_from_routes(
    routes: pd.DataFrame,
    *,
    full_turnover_on_route_change: bool,
) -> dict[str, dict[str, int]]:
    if routes.empty:
        return {}
    result: dict[str, dict[str, int]] = {}
    previous_source: str | None = None
    for _, row in routes.sort_values("date").iterrows():
        source = str(row["source"])
        top_n = int(row["top_n"])
        max_turnover = int(row["max_turnover"])
        rank_buffer = int(row["rank_buffer"])
        if full_turnover_on_route_change and previous_source is not None and source != previous_source:
            max_turnover = top_n
            rank_buffer = 0
        result[str(row["date"])] = {
            "top_n": top_n,
            "max_turnover": max_turnover,
            "rank_buffer": rank_buffer,
        }
        previous_source = source
    return result


def exposure_schedule_from_year_routes(year_routes: pd.DataFrame) -> dict[str, float]:
    if year_routes.empty:
        return {}
    return {
        str(row["decision_date"]): float(row["exposure"])
        for _, row in year_routes.sort_values("decision_date").iterrows()
    }


def risk_exit_min_positions_schedule_from_routes(
    routes: pd.DataFrame,
    *,
    min_positions: int,
    reasons: set[str],
) -> dict[str, int]:
    if routes.empty or min_positions <= 0 or not reasons:
        return {}
    result: dict[str, int] = {}
    for _, row in routes.sort_values("date").iterrows():
        if str(row.get("reason", "")) in reasons:
            result[str(row["date"])] = int(min_positions)
    return result


def full_gate_summary(
    *,
    metrics: dict[str, Any],
    audit_summary: dict[str, Any],
    config: dict[str, Any],
    return_target: float,
    drawdown_limit: float,
) -> dict[str, bool]:
    quality_cfg = config.get("quality", {})
    turnover_limit = float(quality_cfg.get("max_annual_turnover", float("inf")))
    trade_cost_limit = float(quality_cfg.get("max_annual_trade_cost_ratio", float("inf")))
    annual_return_pass = bool(metrics.get("annual_return", 0.0) >= return_target)
    max_drawdown_pass = bool(metrics.get("max_drawdown", 0.0) >= drawdown_limit)
    annual_turnover_pass = bool(metrics.get("annual_turnover", 0.0) <= turnover_limit)
    annual_trade_cost_ratio_pass = bool(metrics.get("annual_trade_cost_ratio", 0.0) <= trade_cost_limit)
    return {
        "annual_return_pass": annual_return_pass,
        "max_drawdown_pass": max_drawdown_pass,
        "annual_turnover_pass": annual_turnover_pass,
        "annual_trade_cost_ratio_pass": annual_trade_cost_ratio_pass,
        "is_full_goal_met": bool(
            annual_return_pass
            and max_drawdown_pass
            and annual_turnover_pass
            and annual_trade_cost_ratio_pass
            and audit_summary["is_goal_met"]
        ),
    }


if __name__ == "__main__":
    main()
