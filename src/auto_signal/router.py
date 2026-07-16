"""Annual-router integration specific to the auto-signal workflow."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.annual_router import (
    ANNUAL_ROUTER_ENGINE_CONTRACT,
    ScoreSourceDefinition,
    annual_route_decisions,
    definitions_for_turnover_mode,
    routed_backtest_config,
    run_annual_state_score_router,
    signal_trade_date_map,
)
from src.auto_signal.models import AnnualStateRouterRuntime
from src.auto_signal.quality import formal_candidate_quality_report, quality_number
from src.auto_signal.status import stage
from src.auto_tuning import ParameterQualityReport
from src.config_loader import resolve_path
from src.market_regime import _benchmark_close


@dataclass(frozen=True)
class RouterServices:
    configured_source_definitions: Callable[..., dict[str, ScoreSourceDefinition]]
    build_score_sources: Callable[..., dict[str, pd.Series]]
    month_end_signal_dates: Callable[..., list[pd.Timestamp]]


def annual_state_router_enabled(config: dict[str, Any]) -> bool:
    router_cfg = config.get("annual_state_router", {})
    return isinstance(router_cfg, dict) and bool(router_cfg.get("enabled", False))


def annual_state_router_cfg(config: dict[str, Any]) -> dict[str, Any]:
    router_cfg = config.get("annual_state_router", {})
    return router_cfg if isinstance(router_cfg, dict) else {}


def optional_router_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def router_reason_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    normalized = str(value).replace("+", ",")
    return {item.strip() for item in normalized.split(",") if item.strip()}


def annual_state_router_selected_params(config: dict[str, Any]) -> dict[str, Any]:
    router_cfg = annual_state_router_cfg(config)
    keys = [
        "initial_source",
        "missing_ret252_exposure",
        "flat_negative_exposure",
        "moderate_positive_source",
        "moderate_positive_ret252_min",
        "moderate_positive_exposure",
        "moderate_low_source",
        "moderate_low_ret252_min",
        "moderate_low_ret252_max",
        "moderate_low_exposure",
        "moderate_lower_source",
        "moderate_lower_ret252_min",
        "moderate_lower_ret252_max",
        "moderate_lower_exposure",
        "strong_trailing_exposure",
        "turnover_boost_max_turnover",
        "turnover_boost_rank_buffer",
        "risk_exit_min_positions",
        "turnover_mode",
        "full_turnover_on_route_change",
        "use_defensive_timing",
        "disable_equity_overlay",
    ]
    selected = {key: router_cfg.get(key) for key in keys if key in router_cfg}
    selected["strategy_mode"] = "annual_state_router"
    selected["include_expanded_sources"] = bool(router_cfg.get("include_expanded_sources", True))
    selected["turnover_boost_reasons"] = sorted(router_reason_set(router_cfg.get("turnover_boost_reasons")))
    selected["risk_exit_min_positions_reasons"] = sorted(router_reason_set(router_cfg.get("risk_exit_min_positions_reasons")))
    return selected


def annual_state_router_source_definitions(
    config: dict[str, Any],
    *,
    services: RouterServices,
) -> dict[str, ScoreSourceDefinition]:
    router_cfg = annual_state_router_cfg(config)
    definitions = services.configured_source_definitions(config)
    definitions = definitions_for_turnover_mode(definitions, str(router_cfg.get("turnover_mode", "default")))
    required_names = {
        "beta",
        "db_size",
        "quality",
        "selector",
        "industry",
        *{
            str(router_cfg.get(key) or "").strip()
            for key in (
                "initial_source",
                "fallback_source",
                "moderate_positive_source",
                "moderate_low_source",
                "moderate_lower_source",
            )
        },
    }
    required_names.discard("")
    missing = sorted(required_names - definitions.keys())
    if missing:
        raise ValueError(f"annual_state_router references unknown score sources: {', '.join(missing)}")
    return definitions


def build_annual_state_router_runtime(
    *,
    config: dict[str, Any],
    prices: pd.DataFrame,
    start_date: str,
    end_date: str,
    out_dir: Path,
    artifacts: list[Path],
    status: dict[str, Any],
    services: RouterServices,
) -> AnnualStateRouterRuntime:
    router_cfg = annual_state_router_cfg(config)
    stage(status, out_dir, "annual_state_router", "running", "building score sources")
    signal_dates = services.month_end_signal_dates(prices.index, start_date=start_date, end_date=end_date)
    if not signal_dates:
        raise ValueError("annual_state_router has no signal dates in the requested backtest window.")
    source_definitions = annual_state_router_source_definitions(config, services=services)
    benchmark = _benchmark_close(prices, config, config.get("market_regime", {})).dropna().sort_index()
    if benchmark.empty:
        raise ValueError("annual_state_router benchmark close series is empty.")
    normalized_price_dates = pd.DatetimeIndex(pd.to_datetime(prices.index).normalize()).unique().sort_values()
    trade_dates = signal_trade_date_map(signal_dates, normalized_price_dates)
    route_preview = annual_route_decisions(
        years=sorted({int(trade_date.year) for trade_date in trade_dates.values()}),
        price_dates=normalized_price_dates,
        benchmark=benchmark,
        initial_source=str(router_cfg.get("initial_source", "beta")),
        missing_ret252_exposure=float(router_cfg.get("missing_ret252_exposure", 0.65)),
        flat_negative_exposure=float(router_cfg.get("flat_negative_exposure", 0.90)),
        moderate_positive_source=optional_router_text(router_cfg.get("moderate_positive_source")),
        moderate_positive_ret252_min=float(router_cfg.get("moderate_positive_ret252_min", 0.20)),
        moderate_positive_exposure=float(router_cfg.get("moderate_positive_exposure", 1.0)),
        moderate_low_source=optional_router_text(router_cfg.get("moderate_low_source")),
        moderate_low_ret252_min=float(router_cfg.get("moderate_low_ret252_min", 0.18)),
        moderate_low_ret252_max=float(router_cfg.get("moderate_low_ret252_max", 0.20)),
        moderate_low_exposure=float(router_cfg.get("moderate_low_exposure", 1.0)),
        moderate_lower_source=optional_router_text(router_cfg.get("moderate_lower_source")),
        moderate_lower_ret252_min=float(router_cfg.get("moderate_lower_ret252_min", 0.16)),
        moderate_lower_ret252_max=float(router_cfg.get("moderate_lower_ret252_max", 0.18)),
        moderate_lower_exposure=float(router_cfg.get("moderate_lower_exposure", 1.0)),
        strong_trailing_exposure=float(router_cfg.get("strong_trailing_exposure", 1.0)),
    )
    route_by_year = {int(row["year"]): row for row in route_preview}
    signal_dates_by_source: dict[str, list[pd.Timestamp]] = {}
    for signal_date, trade_date in trade_dates.items():
        source = str(route_by_year[int(trade_date.year)]["source"])
        signal_dates_by_source.setdefault(source, []).append(signal_date)
    fallback_source = optional_router_text(router_cfg.get("fallback_source"))
    if fallback_source:
        signal_dates_by_source[fallback_source] = list(signal_dates)
    source_definitions = {
        name: definition for name, definition in source_definitions.items() if name in signal_dates_by_source
    }
    score_sources = services.build_score_sources(
        config=config,
        prices=prices,
        signal_dates=signal_dates,
        start_date=start_date,
        end_date=end_date,
        source_definitions=source_definitions,
        signal_dates_by_source=signal_dates_by_source,
        progress_callback=lambda name, index, total, state: stage(
            status,
            out_dir,
            "annual_state_router",
            "running",
            f"score source {index}/{total}: {name} ({state})",
        ),
    )

    stage(status, out_dir, "annual_state_router", "running", "routing annual state scores")
    routed = run_annual_state_score_router(
        score_sources=score_sources,
        source_definitions=source_definitions,
        price_dates=normalized_price_dates,
        benchmark=benchmark,
        signal_dates=signal_dates,
        initial_source=str(router_cfg.get("initial_source", "beta")),
        missing_ret252_exposure=float(router_cfg.get("missing_ret252_exposure", 0.65)),
        flat_negative_exposure=float(router_cfg.get("flat_negative_exposure", 0.90)),
        fallback_source=optional_router_text(router_cfg.get("fallback_source")),
        moderate_positive_source=optional_router_text(router_cfg.get("moderate_positive_source")),
        moderate_positive_ret252_min=float(router_cfg.get("moderate_positive_ret252_min", 0.20)),
        moderate_positive_exposure=float(router_cfg.get("moderate_positive_exposure", 1.0)),
        moderate_low_source=optional_router_text(router_cfg.get("moderate_low_source")),
        moderate_low_ret252_min=float(router_cfg.get("moderate_low_ret252_min", 0.18)),
        moderate_low_ret252_max=float(router_cfg.get("moderate_low_ret252_max", 0.20)),
        moderate_low_exposure=float(router_cfg.get("moderate_low_exposure", 1.0)),
        moderate_lower_source=optional_router_text(router_cfg.get("moderate_lower_source")),
        moderate_lower_ret252_min=float(router_cfg.get("moderate_lower_ret252_min", 0.16)),
        moderate_lower_ret252_max=float(router_cfg.get("moderate_lower_ret252_max", 0.18)),
        moderate_lower_exposure=float(router_cfg.get("moderate_lower_exposure", 1.0)),
        strong_trailing_exposure=float(router_cfg.get("strong_trailing_exposure", 1.0)),
        turnover_boost_reasons=router_reason_set(router_cfg.get("turnover_boost_reasons")),
        turnover_boost_max_turnover=int(router_cfg.get("turnover_boost_max_turnover", 2)),
        turnover_boost_rank_buffer=int(router_cfg.get("turnover_boost_rank_buffer", 10)),
    )
    if routed.scores.empty:
        raise ValueError("annual_state_router produced an empty routed score panel.")
    bt_config = routed_backtest_config(
        config=config,
        prices=prices,
        routed=routed,
        source_definitions=source_definitions,
        full_turnover_on_route_change=bool(router_cfg.get("full_turnover_on_route_change", False)),
        use_defensive_timing=bool(router_cfg.get("use_defensive_timing", False)),
        disable_equity_overlay=bool(router_cfg.get("disable_equity_overlay", False)),
    )
    score_routes_path = out_dir / "auto_annual_state_router_score_routes.csv"
    year_routes_path = out_dir / "auto_annual_state_router_year_routes.csv"
    routed.score_routes.to_csv(score_routes_path, index=False, encoding="utf-8-sig")
    routed.year_routes.to_csv(year_routes_path, index=False, encoding="utf-8-sig")
    artifacts.extend([score_routes_path, year_routes_path])
    stage(status, out_dir, "annual_state_router", "complete", f"{len(routed.score_routes)} routed signal dates")
    return AnnualStateRouterRuntime(
        routed=routed,
        source_definitions=source_definitions,
        backtest_config=bt_config,
        files={
            "annual_state_router_score_routes": str(score_routes_path),
            "annual_state_router_year_routes": str(year_routes_path),
        },
    )


def annual_state_router_signal_config(
    config: dict[str, Any],
    runtime: AnnualStateRouterRuntime,
    signal_date_arg: str,
) -> dict[str, Any]:
    effective_date = effective_router_score_date(runtime.routed.scores, signal_date_arg)
    route = router_route_for_date(runtime.routed.score_routes, effective_date)
    result = deepcopy(config)
    strategy = dict(result.get("strategy", {}))
    strategy.update(
        {
            "top_n": int(route["top_n"]),
            "max_turnover": int(route["max_turnover"]),
            "rank_buffer": int(route["rank_buffer"]),
            "rebalance_freq": "monthly",
        }
    )
    result["strategy"] = strategy
    return result


def effective_router_score_date(scores: pd.Series, signal_date_arg: str) -> pd.Timestamp:
    if scores.empty or not isinstance(scores.index, pd.MultiIndex):
        raise ValueError("annual_state_router scores must use MultiIndex date/instrument.")
    dates = pd.DatetimeIndex(pd.to_datetime(scores.index.get_level_values(0)).normalize()).unique().sort_values()
    if dates.empty:
        raise ValueError("annual_state_router score panel has no dated rows.")
    arg = str(signal_date_arg).strip().lower()
    if arg in {"", "none", "latest"}:
        return pd.Timestamp(dates.max()).normalize()
    requested = pd.Timestamp(signal_date_arg).normalize()
    eligible = dates[dates <= requested]
    if eligible.empty:
        raise ValueError(f"No annual_state_router score date is available on or before {requested.date()}.")
    return pd.Timestamp(eligible.max()).normalize()


def router_route_for_date(routes: pd.DataFrame, score_date: pd.Timestamp) -> pd.Series:
    if routes.empty or "date" not in routes.columns:
        raise ValueError("annual_state_router score routes are empty.")
    frame = routes.copy()
    frame["_date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    matched = frame[frame["_date"] == pd.Timestamp(score_date).normalize()]
    if matched.empty:
        raise ValueError(f"No annual_state_router route found for score date {score_date.date()}.")
    return matched.iloc[-1]


def annual_state_router_quality(
    config: dict[str, Any],
    quality_config: dict,
    *,
    services: RouterServices,
) -> ParameterQualityReport | None:
    if not annual_state_router_enabled(config):
        return None
    router_cfg = annual_state_router_cfg(config)
    metrics_file = router_cfg.get("evidence_metrics_file")
    if not metrics_file:
        return formal_candidate_quality_report(quality_config, ["annual_state_router_evidence_metrics_file_missing"])
    metrics_path = resolve_path(metrics_file)
    if not metrics_path.exists():
        return formal_candidate_quality_report(
            quality_config,
            [f"annual_state_router_evidence_metrics_file_not_found:{metrics_path}"],
        )
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    audit = payload.get("audit", {}) if isinstance(payload, dict) else {}
    full_gate = payload.get("full_gate", {}) if isinstance(payload, dict) else {}
    yearly = annual_state_router_evidence_yearly(router_cfg)

    annual_return = quality_number(metrics.get("annual_return"), 0.0)
    max_drawdown = quality_number(metrics.get("max_drawdown"), 0.0)
    sharpe = quality_number(metrics.get("sharpe"), 0.0)
    annual_turnover = quality_number(metrics.get("annual_turnover"), 0.0)
    annual_trade_cost_ratio = quality_number(metrics.get("annual_trade_cost_ratio"), 0.0)
    windows = int(quality_number(audit.get("year_count"), 0.0))
    positive_return_rate = 1.0 if annual_return > 0 else 0.0
    annual_return_mean = annual_return
    annual_return_min = quality_number(audit.get("min_yearly_annual_return"), annual_return)
    max_drawdown_worst = quality_number(audit.get("worst_yearly_drawdown"), max_drawdown)
    if not yearly.empty and {"annual_return", "max_drawdown"}.issubset(yearly.columns):
        yearly_returns = pd.to_numeric(yearly["annual_return"], errors="coerce").dropna()
        yearly_drawdowns = pd.to_numeric(yearly["max_drawdown"], errors="coerce").dropna()
        if not yearly_returns.empty:
            windows = int(len(yearly_returns))
            positive_return_rate = float((yearly_returns > 0).mean())
            annual_return_mean = float(yearly_returns.mean())
            annual_return_min = float(yearly_returns.min())
        if not yearly_drawdowns.empty:
            max_drawdown_worst = float(yearly_drawdowns.min())

    min_windows = int(quality_config.get("min_validation_windows", 3))
    min_positive = float(quality_config.get("min_positive_return_rate", 0.5))
    min_return = float(quality_config.get("min_optimizer_annual_return", quality_config.get("target_annual_return", 0.20)))
    min_sharpe = float(quality_config.get("min_sharpe_mean", 0.0))
    max_drawdown_limit = float(quality_config.get("max_drawdown_limit", -0.20))
    max_turnover = float(quality_config.get("max_annual_turnover", 20.0))
    max_cost = float(quality_config.get("max_annual_trade_cost_ratio", 0.2))

    combo = payload.get("combo", {}) if isinstance(payload, dict) else {}
    issues = annual_state_router_combo_issues(router_cfg, combo)
    issues.extend(annual_state_router_evidence_provenance_issues(config, payload, services=services))
    if not bool(full_gate.get("is_full_goal_met", False)):
        issues.append("annual_state_router_evidence_full_gate_not_met")
    if windows < min_windows:
        issues.append(f"validation_windows_below_threshold:{windows}<{min_windows}")
    if positive_return_rate < min_positive:
        issues.append(f"positive_return_rate_below_threshold:{positive_return_rate:.4f}<{min_positive:.4f}")
    if annual_return_mean < min_return:
        issues.append(f"annual_return_mean_below_threshold:{annual_return_mean:.4f}<{min_return:.4f}")
    if annual_return_min < min_return:
        issues.append(f"annual_return_min_below_threshold:{annual_return_min:.4f}<{min_return:.4f}")
    if sharpe < min_sharpe:
        issues.append(f"sharpe_mean_below_threshold:{sharpe:.4f}<{min_sharpe:.4f}")
    if max_drawdown_worst < max_drawdown_limit:
        issues.append(f"max_drawdown_worse_than_limit:{max_drawdown_worst:.4f}<{max_drawdown_limit:.4f}")
    if annual_turnover > max_turnover:
        issues.append(f"annual_turnover_above_threshold:{annual_turnover:.4f}>{max_turnover:.4f}")
    if annual_trade_cost_ratio > max_cost:
        issues.append(f"annual_trade_cost_ratio_above_threshold:{annual_trade_cost_ratio:.4f}>{max_cost:.4f}")

    return ParameterQualityReport(
        is_acceptable=not issues,
        issues=issues,
        windows=windows,
        positive_return_rate=positive_return_rate,
        annual_return_mean=annual_return_mean,
        annual_return_min=annual_return_min,
        sharpe_mean=sharpe,
        max_drawdown_worst=max_drawdown_worst,
        annual_turnover_mean=annual_turnover,
        annual_trade_cost_ratio_mean=annual_trade_cost_ratio,
        min_validation_windows=min_windows,
        min_positive_return_rate=min_positive,
        min_optimizer_annual_return=min_return,
        min_sharpe_mean=min_sharpe,
        max_drawdown_limit=max_drawdown_limit,
        max_annual_turnover=max_turnover,
        max_annual_trade_cost_ratio=max_cost,
    )


def annual_state_router_evidence_provenance_issues(
    config: dict[str, Any],
    payload: Any,
    *,
    services: RouterServices,
) -> list[str]:
    if not isinstance(payload, dict):
        return ["annual_state_router_evidence_provenance_missing"]
    if payload.get("engine_contract") != ANNUAL_ROUTER_ENGINE_CONTRACT:
        return ["annual_state_router_evidence_engine_contract_mismatch"]
    observed = payload.get("source_definitions")
    if not isinstance(observed, dict) or not observed:
        return ["annual_state_router_evidence_source_definitions_missing"]
    expected = {
        name: definition.__dict__
        for name, definition in annual_state_router_source_definitions(config, services=services).items()
    }
    issues: list[str] = []
    for name, definition in expected.items():
        actual = observed.get(name)
        if not isinstance(actual, dict):
            issues.append(f"annual_state_router_evidence_source_missing:{name}")
            continue
        for key, value in definition.items():
            actual_value = actual.get(key)
            if key in {"factor_file", "selector_file"}:
                expected_value = str(resolve_path(value)) if value else ""
                observed_value = str(resolve_path(actual_value)) if actual_value else ""
                matches = expected_value == observed_value
            else:
                matches = actual_value == value
            if not matches:
                issues.append(f"annual_state_router_evidence_source_mismatch:{name}.{key}")
    return issues


def annual_state_router_evidence_yearly(router_cfg: dict[str, Any]) -> pd.DataFrame:
    years_file = router_cfg.get("evidence_years_file")
    if not years_file:
        return pd.DataFrame()
    years_path = resolve_path(years_file)
    return pd.DataFrame() if not years_path.exists() else pd.read_csv(years_path)


def annual_state_router_combo_issues(router_cfg: dict[str, Any], combo: Any) -> list[str]:
    if not isinstance(combo, dict) or not combo:
        return ["annual_state_router_evidence_combo_missing"]
    issues: list[str] = []
    numeric_keys = [
        "missing_ret252_exposure",
        "strong_trailing_exposure",
        "moderate_positive_ret252_min",
        "moderate_positive_exposure",
        "moderate_low_ret252_min",
        "moderate_low_ret252_max",
        "moderate_low_exposure",
        "moderate_lower_ret252_min",
        "moderate_lower_ret252_max",
        "moderate_lower_exposure",
        "turnover_boost_max_turnover",
        "turnover_boost_rank_buffer",
        "risk_exit_min_positions",
    ]
    for key in numeric_keys:
        if key in router_cfg and key in combo:
            expected = quality_number(router_cfg.get(key), float("nan"))
            observed = quality_number(combo.get(key), float("nan"))
            if pd.notna(expected) and pd.notna(observed) and abs(expected - observed) > 1e-9:
                issues.append(f"annual_state_router_evidence_combo_mismatch:{key}")
    for key in ["moderate_positive_source", "moderate_low_source", "moderate_lower_source", "turnover_mode"]:
        if key in router_cfg and key in combo and str(router_cfg.get(key) or "") != str(combo.get(key) or ""):
            issues.append(f"annual_state_router_evidence_combo_mismatch:{key}")
    if router_reason_set(router_cfg.get("turnover_boost_reasons")) != router_reason_set(combo.get("turnover_boost_reasons")):
        issues.append("annual_state_router_evidence_combo_mismatch:turnover_boost_reasons")
    if router_reason_set(router_cfg.get("risk_exit_min_positions_reasons")) != router_reason_set(
        combo.get("risk_exit_min_positions_reasons")
    ):
        issues.append("annual_state_router_evidence_combo_mismatch:risk_exit_min_positions_reasons")
    return issues
