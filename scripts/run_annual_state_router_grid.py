"""Run a resumable grid over annual state-router score backtests."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._shared import dated_output_path, yearly_stats
from scripts.run_annual_state_router_backtest import (
    DEFAULT_EXTENDED_FACTOR_FILE,
    DEFAULT_INDUSTRY_FACTOR_FILE,
    DEFAULT_SELECTOR_FILE,
    ScoreSourceDefinition,
    RoutedScoreRun,
    apply_research_config_overrides,
    build_score_sources,
    default_source_definitions,
    full_gate_summary,
    routed_backtest_config,
    run_annual_state_score_router,
)
from scripts.run_fundamental_quality_backtest import month_end_signal_dates
from scripts.run_goal_audit import audit_yearly_goal, goal_thresholds
from src.backtest import run_backtest
from src.config_loader import load_config, resolve_path
from src.market_regime import _benchmark_close
from src.trading_calendar import resolve_target_date_value


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable annual state-router grid search.")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="2026-06-09")
    parser.add_argument("--factor-file", default=DEFAULT_EXTENDED_FACTOR_FILE)
    parser.add_argument("--industry-factor-file", default=DEFAULT_INDUSTRY_FACTOR_FILE)
    parser.add_argument("--selector-file", default=DEFAULT_SELECTOR_FILE)
    parser.add_argument("--cache-dir", default="outputs/router_score_cache")
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--output", default=dated_output_path("annual_state_router_grid", suffix=".csv"))
    parser.add_argument("--max-combinations", type=int, default=0)
    parser.add_argument("--missing-ret252-exposures", default="0.70,0.72,0.75")
    parser.add_argument("--strong-trailing-exposures", default="0.80,0.85")
    parser.add_argument("--moderate-positive-sources", default="roc60,db_total,beta20,rsqr20")
    parser.add_argument("--moderate-positive-ret252-mins", default="0.20,0.22")
    parser.add_argument("--moderate-low-sources", default="none,db_total,beta20,rsqr20")
    parser.add_argument("--moderate-low-ret252-mins", default="0.18")
    parser.add_argument("--moderate-low-ret252-maxs", default="0.20")
    parser.add_argument("--moderate-low-exposures", default="1.0")
    parser.add_argument("--turnover-modes", default="default,turnover2,rank10")
    parser.add_argument("--turnover-boost-reason-sets", default="none")
    parser.add_argument("--turnover-boost-max-turnovers", default="2")
    parser.add_argument("--turnover-boost-rank-buffers", default="10")
    parser.add_argument("--equity-overlay-sideways-exposures", default="none,0.60,0.70")
    parser.add_argument("--equity-overlay-bear-exposures", default="none,0.60")
    parser.add_argument("--defensive-bear-exposures", default="none")
    parser.add_argument("--max-industry-weights", default="none")
    parser.add_argument("--disable-equity-overlay", action="store_true")
    parser.add_argument("--write-hit-prefix", default="")
    args = parser.parse_args()

    config = load_config()
    start_date = args.start_date or config["data"]["start_date"]
    end_date = resolve_target_date_value(args.end_date or config["data"]["end_date"], config=config)
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    prices = pd.read_parquet(resolve_path(config.get("ic", {}).get("price_file", "data/prices/ohlcv_adjusted.parquet")))
    signal_dates = month_end_signal_dates(prices.index, start_date=start_date, end_date=end_date)
    base_definitions = default_source_definitions(
        factor_file=args.factor_file,
        industry_factor_file=args.industry_factor_file,
        selector_file=args.selector_file,
        include_expanded_sources=True,
    )
    score_sources = load_or_build_score_sources(
        config=config,
        prices=prices,
        signal_dates=signal_dates,
        start_date=start_date,
        end_date=end_date,
        source_definitions=base_definitions,
        cache_dir=resolve_path(args.cache_dir),
        force=bool(args.force_rebuild_cache),
    )
    benchmark = _benchmark_close(prices, config, config.get("market_regime", {})).dropna().sort_index()
    if benchmark.empty:
        raise ValueError("Benchmark close series is empty; cannot route by annual market state.")

    return_target, drawdown_limit = goal_thresholds(config)
    seen = completed_keys(output)
    count = 0
    for combo in iter_grid(args):
        key = combo_key(combo)
        if key in seen:
            continue
        count += 1
        if args.max_combinations > 0 and count > args.max_combinations:
            break
        definitions = definitions_for_turnover_mode(base_definitions, combo["turnover_mode"])
        routed = run_annual_state_score_router(
            score_sources=score_sources,
            source_definitions=definitions,
            price_dates=pd.DatetimeIndex(pd.to_datetime(prices.index).normalize()),
            benchmark=benchmark,
            initial_source="beta",
            missing_ret252_exposure=combo["missing_ret252_exposure"],
            flat_negative_exposure=0.90,
            moderate_positive_source=combo["moderate_positive_source"],
            moderate_positive_ret252_min=combo["moderate_positive_ret252_min"],
            moderate_low_source=combo["moderate_low_source"],
            moderate_low_ret252_min=combo["moderate_low_ret252_min"],
            moderate_low_ret252_max=combo["moderate_low_ret252_max"],
            moderate_low_exposure=combo["moderate_low_exposure"],
            strong_trailing_exposure=combo["strong_trailing_exposure"],
            turnover_boost_reasons=parse_reason_set(combo["turnover_boost_reasons"]),
            turnover_boost_max_turnover=combo["turnover_boost_max_turnover"],
            turnover_boost_rank_buffer=combo["turnover_boost_rank_buffer"],
        )
        combo_config = apply_research_config_overrides(config, namespace_for_combo(combo))
        bt_config = routed_backtest_config(
            config=combo_config,
            prices=prices,
            routed=routed,
            source_definitions=definitions,
            full_turnover_on_route_change=True,
            use_defensive_timing=True,
            disable_equity_overlay=bool(args.disable_equity_overlay),
        )
        result = run_backtest(routed.scores, prices, start_date, end_date, bt_config)
        yearly = yearly_stats(result.equity_curve, bt_config)
        audited_yearly, audit_summary = audit_yearly_goal(
            yearly,
            return_target=return_target,
            drawdown_limit=drawdown_limit,
        )
        full_gate = full_gate_summary(
            metrics=result.metrics,
            audit_summary=audit_summary,
            config=combo_config,
            return_target=return_target,
            drawdown_limit=drawdown_limit,
        )
        row = {
            "key": key,
            **combo,
            "annual_return": result.metrics.get("annual_return", 0.0),
            "max_drawdown": result.metrics.get("max_drawdown", 0.0),
            "annual_turnover": result.metrics.get("annual_turnover", 0.0),
            "annual_trade_cost_ratio": result.metrics.get("annual_trade_cost_ratio", 0.0),
            "year_return_pass_count": audit_summary["year_return_pass_count"],
            "year_drawdown_pass_count": audit_summary["year_drawdown_pass_count"],
            "min_yearly_annual_return": audit_summary["min_yearly_annual_return"],
            "worst_yearly_drawdown": audit_summary["worst_yearly_drawdown"],
            "failed_years": ",".join(str(year) for year in audit_summary["failed_years"]),
            "full_goal": full_gate["is_full_goal_met"],
            "score_route_counts": json.dumps(
                routed.score_routes["source"].value_counts().to_dict(),
                ensure_ascii=False,
                sort_keys=True,
            ),
        }
        append_row(output, row)
        print(
            f"{count}: goal={row['full_goal']} years={row['year_return_pass_count']}/"
            f"{row['year_drawdown_pass_count']} annual={row['annual_return']:.4f} "
            f"dd={row['max_drawdown']:.4f} cost={row['annual_trade_cost_ratio']:.4f} "
            f"failed={row['failed_years']} key={key}",
            flush=True,
        )
        if full_gate["is_full_goal_met"] and args.write_hit_prefix:
            write_hit_outputs(
                prefix=resolve_path(args.write_hit_prefix),
                result=result,
                routed=routed,
                yearly=audited_yearly,
                audit_summary=audit_summary,
                full_gate=full_gate,
                combo=combo,
            )
            break


def load_or_build_score_sources(
    *,
    config: dict[str, Any],
    prices: pd.DataFrame,
    signal_dates: list[pd.Timestamp],
    start_date: str,
    end_date: str,
    source_definitions: dict[str, ScoreSourceDefinition],
    cache_dir: Path,
    force: bool,
) -> dict[str, pd.Series]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    sources: dict[str, pd.Series] = {}
    missing: dict[str, ScoreSourceDefinition] = {}
    for name, definition in source_definitions.items():
        path = cache_dir / f"{name}_{pd.Timestamp(end_date).date().isoformat()}.parquet"
        if path.exists() and not force:
            frame = pd.read_parquet(path)
            sources[name] = frame["score"].rename("score")
            print(f"loaded score cache: {path}", flush=True)
        else:
            missing[name] = definition
    for name, definition in missing.items():
        print(f"building score source: {name}", flush=True)
        built = build_score_sources(
            config=config,
            prices=prices,
            signal_dates=signal_dates,
            start_date=start_date,
            end_date=end_date,
            source_definitions={name: definition},
        )[name]
        path = cache_dir / f"{name}_{pd.Timestamp(end_date).date().isoformat()}.parquet"
        built.to_frame("score").to_parquet(path)
        sources[name] = built
        print(f"wrote score cache: {path} rows={len(built)}", flush=True)
    return sources


def iter_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    combos: list[dict[str, Any]] = []
    for missing in parse_float_list(args.missing_ret252_exposures):
        for strong in parse_float_list(args.strong_trailing_exposures):
            for high_source in parse_source_list(args.moderate_positive_sources):
                for high_min in parse_float_list(args.moderate_positive_ret252_mins):
                    for low_source in parse_source_list(args.moderate_low_sources):
                        for low_min in parse_float_list(args.moderate_low_ret252_mins):
                            for low_max in parse_float_list(args.moderate_low_ret252_maxs):
                                low_exposures = parse_float_list(args.moderate_low_exposures)
                                if low_source is None:
                                    low_exposures = [1.0]
                                for low_exposure in low_exposures:
                                    for turnover_mode in parse_str_list(args.turnover_modes):
                                        for boost_reasons in parse_reason_set_list(args.turnover_boost_reason_sets):
                                            boost_turnovers = parse_int_list(args.turnover_boost_max_turnovers)
                                            boost_buffers = parse_int_list(args.turnover_boost_rank_buffers)
                                            if boost_reasons == "none":
                                                boost_turnovers = [0]
                                                boost_buffers = [0]
                                            for boost_turnover in boost_turnovers:
                                                for boost_buffer in boost_buffers:
                                                    for overlay_side in parse_optional_float_list(args.equity_overlay_sideways_exposures):
                                                        for overlay_bear in parse_optional_float_list(args.equity_overlay_bear_exposures):
                                                            for defensive_bear in parse_optional_float_list(args.defensive_bear_exposures):
                                                                for max_industry_weight in parse_optional_float_list(args.max_industry_weights):
                                                                    combos.append(
                                                                        {
                                                                            "missing_ret252_exposure": missing,
                                                                            "strong_trailing_exposure": strong,
                                                                            "moderate_positive_source": high_source,
                                                                            "moderate_positive_ret252_min": high_min,
                                                                            "moderate_low_source": low_source,
                                                                            "moderate_low_ret252_min": low_min,
                                                                            "moderate_low_ret252_max": low_max,
                                                                            "moderate_low_exposure": low_exposure,
                                                                            "turnover_mode": turnover_mode,
                                                                            "turnover_boost_reasons": boost_reasons,
                                                                            "turnover_boost_max_turnover": boost_turnover,
                                                                            "turnover_boost_rank_buffer": boost_buffer,
                                                                            "equity_overlay_sideways_exposure": overlay_side,
                                                                            "equity_overlay_bear_exposure": overlay_bear,
                                                                            "equity_overlay_drawdown_cut": None,
                                                                            "defensive_sideways_exposure": None,
                                                                            "defensive_bear_exposure": defensive_bear,
                                                                            "max_industry_weight": max_industry_weight,
                                                                        }
                                                                    )
    return combos


def definitions_for_turnover_mode(
    definitions: dict[str, ScoreSourceDefinition],
    mode: str,
) -> dict[str, ScoreSourceDefinition]:
    mode = str(mode or "default").strip().lower()
    if mode == "default":
        return dict(definitions)
    result = dict(definitions)
    if mode == "turnover2":
        for name, definition in list(result.items()):
            result[name] = replace(
                definition,
                max_turnover=min(int(definition.top_n), max(2, int(definition.max_turnover))),
                rank_buffer=min(int(definition.rank_buffer), 10),
            )
        return result
    if mode == "rank10":
        for name, definition in list(result.items()):
            result[name] = replace(definition, rank_buffer=min(int(definition.rank_buffer), 10))
        return result
    if mode == "full":
        for name, definition in list(result.items()):
            result[name] = replace(definition, max_turnover=int(definition.top_n), rank_buffer=0)
        return result
    raise ValueError(f"Unsupported turnover mode: {mode}")


def namespace_for_combo(combo: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        equity_overlay_sideways_exposure=combo["equity_overlay_sideways_exposure"],
        equity_overlay_bear_exposure=combo["equity_overlay_bear_exposure"],
        equity_overlay_drawdown_cut=combo["equity_overlay_drawdown_cut"],
        defensive_sideways_exposure=combo["defensive_sideways_exposure"],
        defensive_bear_exposure=combo["defensive_bear_exposure"],
        max_industry_weight=combo["max_industry_weight"],
    )


def completed_keys(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(path, usecols=["key"])
    return set(frame["key"].astype(str))


def append_row(path: Path, row: dict[str, Any]) -> None:
    frame = pd.DataFrame([row])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")


def combo_key(combo: dict[str, Any]) -> str:
    return "|".join(f"{key}={combo[key]}" for key in sorted(combo))


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_source_list(value: str) -> list[str | None]:
    return [None if item.lower() in {"", "none", "null"} else item for item in parse_str_list(value)]


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in parse_str_list(value)]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in parse_str_list(value)]


def parse_optional_float_list(value: str) -> list[float | None]:
    result: list[float | None] = []
    for item in parse_str_list(value):
        result.append(None if item.lower() in {"none", "null"} else float(item))
    return result


def parse_reason_set_list(value: str) -> list[str]:
    result: list[str] = []
    for raw in str(value).split(";"):
        item = raw.strip()
        if not item:
            continue
        if item.lower() in {"none", "null"}:
            result.append("none")
            continue
        reasons = sorted({part.strip() for part in item.replace(",", "+").split("+") if part.strip()})
        if reasons:
            result.append("+".join(reasons))
    return result or ["none"]


def parse_reason_set(value: str) -> set[str]:
    if str(value).strip().lower() in {"", "none", "null"}:
        return set()
    return {part.strip() for part in str(value).replace(",", "+").split("+") if part.strip()}


def write_hit_outputs(
    *,
    prefix: Path,
    result: Any,
    routed: RoutedScoreRun,
    yearly: pd.DataFrame,
    audit_summary: dict[str, Any],
    full_gate: dict[str, Any],
    combo: dict[str, Any],
) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    result.equity_curve.to_csv(Path(str(prefix) + "_equity.csv"), encoding="utf-8-sig")
    result.holdings.to_csv(Path(str(prefix) + "_holdings.csv"), index=False, encoding="utf-8-sig")
    result.trades.to_csv(Path(str(prefix) + "_trades.csv"), index=False, encoding="utf-8-sig")
    routed.score_routes.to_csv(Path(str(prefix) + "_score_routes.csv"), index=False, encoding="utf-8-sig")
    routed.year_routes.to_csv(Path(str(prefix) + "_year_routes.csv"), index=False, encoding="utf-8-sig")
    yearly.to_csv(Path(str(prefix) + "_years.csv"), index=False, encoding="utf-8-sig")
    Path(str(prefix) + "_metrics.json").write_text(
        json.dumps(
            {
                "metrics": result.metrics,
                "audit": audit_summary,
                "full_gate": full_gate,
                "combo": combo,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
