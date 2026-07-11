"""Run a resumable grid over annual state-router score backtests."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
from itertools import product
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
    ANNUAL_ROUTER_ENGINE_CONTRACT,
    ScoreSourceDefinition,
    RoutedScoreRun,
    apply_research_config_overrides,
    apply_source_top_n_overrides,
    build_score_sources,
    configured_source_definitions,
    full_gate_summary,
    routed_backtest_config,
    run_annual_state_score_router,
    source_top_n_overrides_payload,
)
from scripts.run_fundamental_quality_backtest import month_end_signal_dates
from scripts.run_goal_audit import audit_yearly_goal, goal_thresholds
from src.backtest import run_backtest
from src.config_loader import load_config, resolve_path
from src.market_regime import _benchmark_close
from src.research_diagnostics import build_exposure_diagnostics
from src.trading_calendar import resolve_target_date_value


COMBO_KEY_FIELDS = tuple(
    sorted(
        {
            "beta20_top_n",
            "beta_top_n",
            "defensive_bear_exposure",
            "defensive_sideways_exposure",
            "equity_overlay_bear_exposure",
            "equity_overlay_drawdown_cut",
            "equity_overlay_sideways_exposure",
            "max_industry_weight",
            "missing_ret252_exposure",
            "moderate_low_exposure",
            "moderate_low_ret252_max",
            "moderate_low_ret252_min",
            "moderate_low_source",
            "moderate_lower_exposure",
            "moderate_lower_ret252_max",
            "moderate_lower_ret252_min",
            "moderate_lower_source",
            "moderate_positive_exposure",
            "moderate_positive_ret252_min",
            "moderate_positive_source",
            "rebalance_after_risk_exit",
            "risk_exit_min_positions",
            "risk_exit_min_positions_reasons",
            "strong_trailing_exposure",
            "turnover_boost_max_turnover",
            "turnover_boost_rank_buffer",
            "turnover_boost_reasons",
            "turnover_mode",
        }
    )
)
INT_COMBO_KEY_FIELDS = {
    "beta20_top_n",
    "beta_top_n",
    "risk_exit_min_positions",
    "turnover_boost_max_turnover",
    "turnover_boost_rank_buffer",
}
BOOL_COMBO_KEY_FIELDS = {"rebalance_after_risk_exit"}
COMBO_KEY_LEGACY_DEFAULTS = {
    "moderate_lower_exposure": 1.0,
    "moderate_lower_ret252_max": 0.18,
    "moderate_lower_ret252_min": 0.16,
    "moderate_lower_source": None,
    "moderate_positive_exposure": 1.0,
    "risk_exit_min_positions_reasons": None,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable annual state-router grid search.")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="2026-06-09")
    parser.add_argument("--factor-file", default="")
    parser.add_argument("--industry-factor-file", default="")
    parser.add_argument("--selector-file", default="")
    parser.add_argument("--cache-dir", default="outputs/router_score_cache")
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--output", default=dated_output_path("annual_state_router_grid", suffix=".csv"))
    parser.add_argument("--max-combinations", type=int, default=0)
    parser.add_argument("--missing-ret252-exposures", default="0.70,0.72,0.75")
    parser.add_argument("--strong-trailing-exposures", default="0.80,0.85")
    parser.add_argument("--moderate-positive-sources", default="roc60,db_total,beta20,rsqr20")
    parser.add_argument("--moderate-positive-ret252-mins", default="0.20,0.22")
    parser.add_argument("--moderate-positive-exposures", default="1.0")
    parser.add_argument("--moderate-low-sources", default="none,db_total,beta20,rsqr20")
    parser.add_argument("--moderate-low-ret252-mins", default="0.18")
    parser.add_argument("--moderate-low-ret252-maxs", default="0.20")
    parser.add_argument("--moderate-low-exposures", default="1.0")
    parser.add_argument("--moderate-lower-sources", default="none")
    parser.add_argument("--moderate-lower-ret252-mins", default="0.16")
    parser.add_argument("--moderate-lower-ret252-maxs", default="0.18")
    parser.add_argument("--moderate-lower-exposures", default="1.0")
    parser.add_argument("--turnover-modes", default="default,turnover2,rank10")
    parser.add_argument("--turnover-boost-reason-sets", default="none")
    parser.add_argument("--turnover-boost-max-turnovers", default="2")
    parser.add_argument("--turnover-boost-rank-buffers", default="10")
    parser.add_argument("--equity-overlay-sideways-exposures", default="none,0.60,0.70")
    parser.add_argument("--equity-overlay-bear-exposures", default="none,0.60")
    parser.add_argument("--defensive-bear-exposures", default="none")
    parser.add_argument("--max-industry-weights", default="none")
    parser.add_argument("--rebalance-after-risk-exit-options", default="false")
    parser.add_argument("--risk-exit-min-positions-options", default="none")
    parser.add_argument("--risk-exit-min-positions-reason-sets", default="none")
    parser.add_argument("--beta-top-ns", default="none")
    parser.add_argument("--beta20-top-ns", default="none")
    parser.add_argument("--include-exposure-diagnostics", action="store_true")
    parser.add_argument("--disable-equity-overlay", action="store_true")
    parser.add_argument("--write-hit-prefix", default="")
    parser.add_argument("--detail-dir", default="")
    args = parser.parse_args()

    config = load_config()
    start_date = args.start_date or config["data"]["start_date"]
    end_date = resolve_target_date_value(args.end_date or config["data"]["end_date"], config=config)
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    prices = pd.read_parquet(resolve_path(config.get("ic", {}).get("price_file", "data/prices/ohlcv_adjusted.parquet")))
    signal_dates = month_end_signal_dates(prices.index, start_date=start_date, end_date=end_date)
    base_definitions = configured_source_definitions(
        config,
        factor_file=args.factor_file or None,
        industry_factor_file=args.industry_factor_file or None,
        selector_file=args.selector_file or None,
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
        combo_definitions = apply_source_top_n_overrides(base_definitions, source_top_n_overrides_payload(namespace_for_combo(combo)))
        definitions = definitions_for_turnover_mode(combo_definitions, combo["turnover_mode"])
        routed = run_annual_state_score_router(
            score_sources=score_sources,
            source_definitions=definitions,
            price_dates=pd.DatetimeIndex(pd.to_datetime(prices.index).normalize()),
            benchmark=benchmark,
            signal_dates=signal_dates,
            initial_source="beta",
            missing_ret252_exposure=combo["missing_ret252_exposure"],
            flat_negative_exposure=0.90,
            moderate_positive_source=combo["moderate_positive_source"],
            moderate_positive_ret252_min=combo["moderate_positive_ret252_min"],
            moderate_positive_exposure=combo["moderate_positive_exposure"],
            moderate_low_source=combo["moderate_low_source"],
            moderate_low_ret252_min=combo["moderate_low_ret252_min"],
            moderate_low_ret252_max=combo["moderate_low_ret252_max"],
            moderate_low_exposure=combo["moderate_low_exposure"],
            moderate_lower_source=combo["moderate_lower_source"],
            moderate_lower_ret252_min=combo["moderate_lower_ret252_min"],
            moderate_lower_ret252_max=combo["moderate_lower_ret252_max"],
            moderate_lower_exposure=combo["moderate_lower_exposure"],
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
        if args.include_exposure_diagnostics:
            row.update(grid_exposure_fields(result.holdings, combo_config))
        append_row(output, row)
        if args.detail_dir:
            write_candidate_detail(
                detail_dir=resolve_path(args.detail_dir),
                key=key,
                metrics=result.metrics,
                yearly=audited_yearly,
                audit_summary=audit_summary,
                full_gate=full_gate,
                combo=combo,
                source_definitions=definitions,
            )
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
                source_definitions=definitions,
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
        path = _score_cache_path(cache_dir, name, definition, start_date, end_date)
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
        path = _score_cache_path(cache_dir, name, definition, start_date, end_date)
        built.to_frame("score").to_parquet(path)
        sources[name] = built
        print(f"wrote score cache: {path} rows={len(built)}", flush=True)
    return sources


def _score_cache_path(
    cache_dir: Path,
    name: str,
    definition: ScoreSourceDefinition,
    start_date: str,
    end_date: str,
) -> Path:
    payload = {
        "engine_contract": ANNUAL_ROUTER_ENGINE_CONTRACT,
        "source": asdict(definition),
        "start_date": str(pd.Timestamp(start_date).date()),
        "end_date": str(pd.Timestamp(end_date).date()),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{name}_{pd.Timestamp(end_date).date().isoformat()}_{digest}.parquet"


def iter_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    combos: list[dict[str, Any]] = []
    common_iter = product(
        parse_float_list(args.missing_ret252_exposures),
        parse_float_list(args.strong_trailing_exposures),
        parse_source_list(args.moderate_positive_sources),
        parse_float_list(args.moderate_positive_ret252_mins),
        parse_float_list(args.moderate_positive_exposures),
        parse_source_list(args.moderate_low_sources),
        parse_float_list(args.moderate_low_ret252_mins),
        parse_float_list(args.moderate_low_ret252_maxs),
        parse_source_list(args.moderate_lower_sources),
        parse_float_list(args.moderate_lower_ret252_mins),
        parse_float_list(args.moderate_lower_ret252_maxs),
        parse_str_list(args.turnover_modes),
        parse_reason_set_list(args.turnover_boost_reason_sets),
        parse_optional_float_list(args.equity_overlay_sideways_exposures),
        parse_optional_float_list(args.equity_overlay_bear_exposures),
        parse_optional_float_list(args.defensive_bear_exposures),
        parse_optional_float_list(args.max_industry_weights),
        parse_bool_list(args.rebalance_after_risk_exit_options),
        parse_optional_int_list(args.risk_exit_min_positions_options),
        parse_reason_set_list(args.risk_exit_min_positions_reason_sets),
        parse_optional_int_list(args.beta_top_ns),
        parse_optional_int_list(args.beta20_top_ns),
    )
    for (
        missing,
        strong,
        high_source,
        high_min,
        high_exposure,
        low_source,
        low_min,
        low_max,
        lower_source,
        lower_min,
        lower_max,
        turnover_mode,
        boost_reasons,
        overlay_side,
        overlay_bear,
        defensive_bear,
        max_industry_weight,
        rebalance_after_risk_exit,
        risk_exit_min_positions,
        risk_exit_min_positions_reasons,
        beta_top_n,
        beta20_top_n,
    ) in common_iter:
        low_exposures = [1.0] if low_source is None else parse_float_list(args.moderate_low_exposures)
        lower_exposures = [1.0] if lower_source is None else parse_float_list(args.moderate_lower_exposures)
        boost_turnovers = parse_int_list(args.turnover_boost_max_turnovers)
        boost_buffers = parse_int_list(args.turnover_boost_rank_buffers)
        if boost_reasons == "none":
            boost_turnovers = [0]
            boost_buffers = [0]
        for low_exposure, lower_exposure, boost_turnover, boost_buffer in product(
            low_exposures,
            lower_exposures,
            boost_turnovers,
            boost_buffers,
        ):
            combos.append(
                {
                    "missing_ret252_exposure": missing,
                    "strong_trailing_exposure": strong,
                    "moderate_positive_source": high_source,
                    "moderate_positive_ret252_min": high_min,
                    "moderate_positive_exposure": high_exposure,
                    "moderate_low_source": low_source,
                    "moderate_low_ret252_min": low_min,
                    "moderate_low_ret252_max": low_max,
                    "moderate_low_exposure": low_exposure,
                    "moderate_lower_source": lower_source,
                    "moderate_lower_ret252_min": lower_min,
                    "moderate_lower_ret252_max": lower_max,
                    "moderate_lower_exposure": lower_exposure,
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
                    "rebalance_after_risk_exit": rebalance_after_risk_exit,
                    "risk_exit_min_positions": risk_exit_min_positions,
                    "risk_exit_min_positions_reasons": None if risk_exit_min_positions_reasons == "none" else risk_exit_min_positions_reasons,
                    "beta_top_n": beta_top_n,
                    "beta20_top_n": beta20_top_n,
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
        rebalance_after_risk_exit=combo["rebalance_after_risk_exit"],
        risk_exit_min_positions=combo["risk_exit_min_positions"],
        risk_exit_min_positions_reasons=combo["risk_exit_min_positions_reasons"] or "",
        beta_top_n=combo["beta_top_n"],
        beta20_top_n=combo["beta20_top_n"],
    )


def completed_keys(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        frame = pd.read_csv(path)
    except pd.errors.ParserError:
        frame = pd.read_csv(path, engine="python", on_bad_lines="skip")
    keys: set[str] = set()
    if "key" in frame.columns:
        keys.update(str(value) for value in frame["key"].dropna().tolist())
    for _, row in frame.iterrows():
        keys.add(combo_key({field: row.get(field, COMBO_KEY_LEGACY_DEFAULTS.get(field)) for field in COMBO_KEY_FIELDS}))
    return keys


def append_row(path: Path, row: dict[str, Any]) -> None:
    frame = pd.DataFrame([row])
    if not path.exists() or path.stat().st_size == 0:
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return
    try:
        existing = pd.read_csv(path)
    except pd.errors.ParserError:
        existing = pd.read_csv(path, engine="python", on_bad_lines="skip")
    columns = list(dict.fromkeys([*existing.columns.tolist(), *frame.columns.tolist()]))
    combined = existing.reindex(columns=columns).astype(object).copy()
    combined.loc[len(combined), columns] = frame.reindex(columns=columns).iloc[0].tolist()
    combined.to_csv(path, index=False, encoding="utf-8-sig")


def combo_key(combo: dict[str, Any]) -> str:
    return "|".join(f"{key}={_combo_key_value(key, combo.get(key))}" for key in COMBO_KEY_FIELDS)


def _combo_key_value(key: str, value: Any) -> str:
    if _is_missing(value):
        return "None"
    if key in BOOL_COMBO_KEY_FIELDS:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y"}:
                return "True"
            if normalized in {"0", "false", "no", "n"}:
                return "False"
        return str(bool(value))
    if key in INT_COMBO_KEY_FIELDS:
        return str(int(float(value)))
    return str(value)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


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


def parse_optional_int_list(value: str) -> list[int | None]:
    result: list[int | None] = []
    for item in parse_str_list(value):
        result.append(None if item.lower() in {"none", "null"} else int(item))
    return result


def parse_bool_list(value: str) -> list[bool]:
    result: list[bool] = []
    for item in parse_str_list(value):
        normalized = item.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            result.append(True)
        elif normalized in {"0", "false", "no", "n"}:
            result.append(False)
        else:
            raise ValueError(f"Unsupported boolean value: {item}")
    return result or [False]


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
    source_definitions: dict[str, ScoreSourceDefinition],
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
                "engine_contract": ANNUAL_ROUTER_ENGINE_CONTRACT,
                "source_definitions": {name: asdict(definition) for name, definition in source_definitions.items()},
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )


def write_candidate_detail(
    *,
    detail_dir: Path,
    key: str,
    metrics: dict[str, Any],
    yearly: pd.DataFrame,
    audit_summary: dict[str, Any],
    full_gate: dict[str, Any],
    combo: dict[str, Any],
    source_definitions: dict[str, ScoreSourceDefinition],
) -> None:
    detail_dir.mkdir(parents=True, exist_ok=True)
    slug = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    yearly.to_csv(detail_dir / f"{slug}_years.csv", index=False, encoding="utf-8-sig")
    (detail_dir / f"{slug}_metrics.json").write_text(
        json.dumps(
            {
                "key": key,
                "metrics": metrics,
                "audit": audit_summary,
                "full_gate": full_gate,
                "combo": combo,
                "engine_contract": ANNUAL_ROUTER_ENGINE_CONTRACT,
                "source_definitions": {name: asdict(definition) for name, definition in source_definitions.items()},
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )


def grid_exposure_fields(holdings: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    summary, _ = build_exposure_diagnostics(holdings, config)
    fields: dict[str, Any] = {
        "latest_position_count": summary.get("latest_position_count"),
        "latest_max_industry_weight": summary.get("latest_max_industry_weight"),
        "latest_top_position_weight": summary.get("latest_top_position_weight"),
        "market_cap_matched_weight": summary.get("market_cap_matched_weight"),
        "market_cap_staleness_days": summary.get("market_cap_staleness_days"),
    }
    for row in summary.get("market_cap_buckets", []):
        if not isinstance(row, dict):
            continue
        bucket = str(row.get("bucket") or "").strip().lower()
        if bucket:
            fields[f"market_cap_{bucket}_weight"] = row.get("weight")
            fields[f"market_cap_{bucket}_position_count"] = row.get("position_count")
    return {key: value for key, value in fields.items() if value is not None}


if __name__ == "__main__":
    main()
