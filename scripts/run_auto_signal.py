"""模块说明：提供 run_auto_signal 命令行入口。"""

from __future__ import annotations

import argparse
import gc
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.auto_tuning import (
    ParameterQualityReport,
    apply_strategy_params,
    assess_backtest_quality,
    assess_parameter_quality,
    select_stable_params,
    summarize_parameter_validation,
)
from src.adj_factor_metadata import build_adj_factor_metadata, write_adj_factor_metadata
from src.backtest import BacktestResult, run_backtest
from src.config_loader import load_config, resolve_path
from src.data_converter import convert_to_qlib_format
from src.data_fetcher import update_daily_data_resumable
from src.data_governance import build_data_governance_report, write_data_governance_report
from src.data_health import build_data_health_report, write_data_health_report
from src.factor_calculator import load_or_compute_factors
from src.failure_analysis import build_failure_analysis_artifacts, build_yearly_breakdown, write_failure_analysis_artifacts
from src.fundamental_data import (
    build_fundamental_screen,
    summarize_fundamental_screen_result,
    write_fundamental_screen_outputs,
)
from src.manual_orders import (
    generate_fill_feedback_template,
    generate_manual_orders,
    generate_order_confirmation_template,
    load_account_state,
    load_current_holdings,
    save_execution_templates,
    save_manual_orders,
    validate_account_inputs,
)
from src.market_regime import _benchmark_close, apply_defensive_timing_to_backtest_config
from src.optimizer import BASELINE_GRID, DEFAULT_GRID, OptimizationTimeoutError, run_walk_forward_grid_validation
from src.reporting import archive_run, signal_action_summary, write_daily_signal_report
from src.research_diagnostics import build_research_diagnostics, write_research_diagnostics
from src.scoring import build_strategy_scores
from src.risk_policy import RiskPolicy
from src.signal_generator import generate_signal, read_signal_previous_holdings, save_signal
from src.strategy import resample_signals
from src.trading_calendar import next_business_day, next_trade_date, resolve_target_date
from scripts.run_annual_state_router_backtest import (
    ANNUAL_ROUTER_ENGINE_CONTRACT,
    RoutedScoreRun,
    ScoreSourceDefinition,
    annual_route_decisions,
    build_score_sources,
    configured_source_definitions,
    routed_backtest_config,
    run_annual_state_score_router,
    signal_trade_date_map,
)
from scripts.run_annual_state_router_grid import definitions_for_turnover_mode
from scripts.run_fundamental_quality_backtest import month_end_signal_dates

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class OptimizationStageResult:
    """Result bundle produced by the automatic parameter optimization stage."""

    selected_config: dict[str, Any]
    selected_params: dict[str, Any]
    selected_params_status: str
    validation: pd.DataFrame
    summary: pd.DataFrame


@dataclass
class DataPreparationStageResult:
    """Result bundle produced by data refresh, factor loading, and governance checks."""

    factor_file: str
    factors: pd.DataFrame
    prices: pd.DataFrame
    data_health: Any
    data_governance: Any
    data_gate: bool
    governance_gate: bool
    health_json: Path
    governance_path: Path


@dataclass
class BacktestStageResult:
    """Result bundle produced by historical backtest and research diagnostics."""

    result: BacktestResult
    backtest_runtime_config: dict[str, Any]
    annual_state_router: "AnnualStateRouterRuntime | None"
    backtest_quality: ParameterQualityReport
    backtest_quality_gate: bool
    backtest_quality_path: Path
    metrics_path: Path
    research_diagnostics: dict[str, Any]
    research_tables: dict[str, pd.DataFrame]
    research_files: dict[str, str]


@dataclass
class AnnualStateRouterRuntime:
    """Runtime artifacts for the annual state router used by backtest and signal generation."""

    routed: RoutedScoreRun
    source_definitions: dict[str, ScoreSourceDefinition]
    backtest_config: dict[str, Any]
    files: dict[str, str]


@dataclass
class SignalStageResult:
    """Result bundle produced by signal generation and execution template stages."""

    signal_df: pd.DataFrame
    target_holdings: list[str]
    output_date: str
    intended_trade_date: str
    previous_holdings_source: str
    account: Any
    is_executable: bool
    block_reasons: list[str]
    quality_warnings: list[str]
    failure_analysis: dict[str, Any]
    failure_files: dict[str, str]
    fundamental_screen: dict[str, Any]
    fundamental_files: dict[str, str]
    signal_path: Path
    holdings_path: Path
    orders_path: Path
    execution_files: dict[str, str]


@dataclass
class ReportStageResult:
    """Result bundle produced by report writing and optional archiving."""

    report_path: Path
    markdown_path: Path
    report: dict[str, Any]


def _csv_values(value: str, cast):
    """函数说明：处理 csv_values 的内部辅助逻辑。"""
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _csv_optional_values(value: str, cast):
    """函数说明：处理 csv_optional_values 的内部辅助逻辑。"""
    values = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower() in {"none", "null", "off"}:
            values.append(None)
        else:
            values.append(cast(item))
    return values


def _grid_values(value: str | None, defaults: list, cast):
    """函数说明：处理 grid_values 的内部辅助逻辑。"""
    if value is None:
        return list(defaults)
    return _csv_values(value, cast)


def _maybe_add_grid_values(grid: dict[str, list], key: str, value: str | None, cast) -> None:
    """函数说明：处理 maybe_add_grid_values 的内部辅助逻辑。"""
    if value is not None:
        grid[key] = _csv_optional_values(value, cast)


def _grid_has_enabled_value(grid: dict[str, list], key: str) -> bool:
    """函数说明：处理 grid_has_enabled_value 的内部辅助逻辑。"""
    return any(value is not None for value in grid.get(key, []))


def _annual_state_router_enabled(config: dict[str, Any]) -> bool:
    router_cfg = config.get("annual_state_router", {})
    return isinstance(router_cfg, dict) and bool(router_cfg.get("enabled", False))


def _annual_state_router_cfg(config: dict[str, Any]) -> dict[str, Any]:
    router_cfg = config.get("annual_state_router", {})
    return router_cfg if isinstance(router_cfg, dict) else {}


def _optional_router_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _router_reason_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    normalized = str(value).replace("+", ",")
    return {item.strip() for item in normalized.split(",") if item.strip()}


def _annual_state_router_selected_params(config: dict[str, Any]) -> dict[str, Any]:
    router_cfg = _annual_state_router_cfg(config)
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
    selected["turnover_boost_reasons"] = sorted(_router_reason_set(router_cfg.get("turnover_boost_reasons")))
    selected["risk_exit_min_positions_reasons"] = sorted(_router_reason_set(router_cfg.get("risk_exit_min_positions_reasons")))
    return selected


def _annual_state_router_source_definitions(config: dict[str, Any]) -> dict[str, ScoreSourceDefinition]:
    router_cfg = _annual_state_router_cfg(config)
    definitions = configured_source_definitions(config)
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


def _build_annual_state_router_runtime(
    *,
    config: dict[str, Any],
    prices: pd.DataFrame,
    start_date: str,
    end_date: str,
    out_dir: Path,
    artifacts: list[Path],
    status: dict[str, Any],
) -> AnnualStateRouterRuntime:
    router_cfg = _annual_state_router_cfg(config)
    _stage(status, out_dir, "annual_state_router", "running", "building score sources")
    signal_dates = month_end_signal_dates(prices.index, start_date=start_date, end_date=end_date)
    if not signal_dates:
        raise ValueError("annual_state_router has no signal dates in the requested backtest window.")
    source_definitions = _annual_state_router_source_definitions(config)
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
        moderate_positive_source=_optional_router_text(router_cfg.get("moderate_positive_source")),
        moderate_positive_ret252_min=float(router_cfg.get("moderate_positive_ret252_min", 0.20)),
        moderate_positive_exposure=float(router_cfg.get("moderate_positive_exposure", 1.0)),
        moderate_low_source=_optional_router_text(router_cfg.get("moderate_low_source")),
        moderate_low_ret252_min=float(router_cfg.get("moderate_low_ret252_min", 0.18)),
        moderate_low_ret252_max=float(router_cfg.get("moderate_low_ret252_max", 0.20)),
        moderate_low_exposure=float(router_cfg.get("moderate_low_exposure", 1.0)),
        moderate_lower_source=_optional_router_text(router_cfg.get("moderate_lower_source")),
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
    fallback_source = _optional_router_text(router_cfg.get("fallback_source"))
    if fallback_source:
        signal_dates_by_source[fallback_source] = list(signal_dates)
    source_definitions = {
        name: definition for name, definition in source_definitions.items() if name in signal_dates_by_source
    }
    score_sources = build_score_sources(
        config=config,
        prices=prices,
        signal_dates=signal_dates,
        start_date=start_date,
        end_date=end_date,
        source_definitions=source_definitions,
        signal_dates_by_source=signal_dates_by_source,
        progress_callback=lambda name, index, total, state: _stage(
            status,
            out_dir,
            "annual_state_router",
            "running",
            f"score source {index}/{total}: {name} ({state})",
        ),
    )

    _stage(status, out_dir, "annual_state_router", "running", "routing annual state scores")
    routed = run_annual_state_score_router(
        score_sources=score_sources,
        source_definitions=source_definitions,
        price_dates=normalized_price_dates,
        benchmark=benchmark,
        signal_dates=signal_dates,
        initial_source=str(router_cfg.get("initial_source", "beta")),
        missing_ret252_exposure=float(router_cfg.get("missing_ret252_exposure", 0.65)),
        flat_negative_exposure=float(router_cfg.get("flat_negative_exposure", 0.90)),
        fallback_source=_optional_router_text(router_cfg.get("fallback_source")),
        moderate_positive_source=_optional_router_text(router_cfg.get("moderate_positive_source")),
        moderate_positive_ret252_min=float(router_cfg.get("moderate_positive_ret252_min", 0.20)),
        moderate_positive_exposure=float(router_cfg.get("moderate_positive_exposure", 1.0)),
        moderate_low_source=_optional_router_text(router_cfg.get("moderate_low_source")),
        moderate_low_ret252_min=float(router_cfg.get("moderate_low_ret252_min", 0.18)),
        moderate_low_ret252_max=float(router_cfg.get("moderate_low_ret252_max", 0.20)),
        moderate_low_exposure=float(router_cfg.get("moderate_low_exposure", 1.0)),
        moderate_lower_source=_optional_router_text(router_cfg.get("moderate_lower_source")),
        moderate_lower_ret252_min=float(router_cfg.get("moderate_lower_ret252_min", 0.16)),
        moderate_lower_ret252_max=float(router_cfg.get("moderate_lower_ret252_max", 0.18)),
        moderate_lower_exposure=float(router_cfg.get("moderate_lower_exposure", 1.0)),
        strong_trailing_exposure=float(router_cfg.get("strong_trailing_exposure", 1.0)),
        turnover_boost_reasons=_router_reason_set(router_cfg.get("turnover_boost_reasons")),
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
    _stage(status, out_dir, "annual_state_router", "complete", f"{len(routed.score_routes)} routed signal dates")
    return AnnualStateRouterRuntime(
        routed=routed,
        source_definitions=source_definitions,
        backtest_config=bt_config,
        files={
            "annual_state_router_score_routes": str(score_routes_path),
            "annual_state_router_year_routes": str(year_routes_path),
        },
    )


def _annual_state_router_signal_config(
    config: dict[str, Any],
    runtime: AnnualStateRouterRuntime,
    signal_date_arg: str,
) -> dict[str, Any]:
    effective_date = _effective_router_score_date(runtime.routed.scores, signal_date_arg)
    route = _router_route_for_date(runtime.routed.score_routes, effective_date)
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


def _effective_router_score_date(scores: pd.Series, signal_date_arg: str) -> pd.Timestamp:
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


def _router_route_for_date(routes: pd.DataFrame, score_date: pd.Timestamp) -> pd.Series:
    if routes.empty or "date" not in routes.columns:
        raise ValueError("annual_state_router score routes are empty.")
    frame = routes.copy()
    frame["_date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    matched = frame[frame["_date"] == pd.Timestamp(score_date).normalize()]
    if matched.empty:
        raise ValueError(f"No annual_state_router route found for score date {score_date.date()}.")
    return matched.iloc[-1]


def _annual_state_router_quality(config: dict[str, Any], quality_config: dict) -> ParameterQualityReport | None:
    if not _annual_state_router_enabled(config):
        return None
    router_cfg = _annual_state_router_cfg(config)
    metrics_file = router_cfg.get("evidence_metrics_file")
    if not metrics_file:
        return _formal_candidate_quality_report(quality_config, ["annual_state_router_evidence_metrics_file_missing"])
    metrics_path = resolve_path(metrics_file)
    if not metrics_path.exists():
        return _formal_candidate_quality_report(
            quality_config,
            [f"annual_state_router_evidence_metrics_file_not_found:{metrics_path}"],
        )
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    audit = payload.get("audit", {}) if isinstance(payload, dict) else {}
    full_gate = payload.get("full_gate", {}) if isinstance(payload, dict) else {}
    yearly = _annual_state_router_evidence_yearly(router_cfg)

    annual_return = _quality_number(metrics.get("annual_return"), 0.0)
    max_drawdown = _quality_number(metrics.get("max_drawdown"), 0.0)
    sharpe = _quality_number(metrics.get("sharpe"), 0.0)
    annual_turnover = _quality_number(metrics.get("annual_turnover"), 0.0)
    annual_trade_cost_ratio = _quality_number(metrics.get("annual_trade_cost_ratio"), 0.0)
    windows = int(_quality_number(audit.get("year_count"), 0.0))
    positive_return_rate = 1.0 if annual_return > 0 else 0.0
    annual_return_mean = annual_return
    annual_return_min = _quality_number(audit.get("min_yearly_annual_return"), annual_return)
    max_drawdown_worst = _quality_number(audit.get("worst_yearly_drawdown"), max_drawdown)
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
    issues = _annual_state_router_combo_issues(router_cfg, combo)
    issues.extend(_annual_state_router_evidence_provenance_issues(config, payload))
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


def _annual_state_router_evidence_provenance_issues(config: dict[str, Any], payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return ["annual_state_router_evidence_provenance_missing"]
    if payload.get("engine_contract") != ANNUAL_ROUTER_ENGINE_CONTRACT:
        return ["annual_state_router_evidence_engine_contract_mismatch"]
    observed = payload.get("source_definitions")
    if not isinstance(observed, dict) or not observed:
        return ["annual_state_router_evidence_source_definitions_missing"]
    expected = {
        name: definition.__dict__
        for name, definition in _annual_state_router_source_definitions(config).items()
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


def _annual_state_router_evidence_yearly(router_cfg: dict[str, Any]) -> pd.DataFrame:
    years_file = router_cfg.get("evidence_years_file")
    if not years_file:
        return pd.DataFrame()
    years_path = resolve_path(years_file)
    if not years_path.exists():
        return pd.DataFrame()
    return pd.read_csv(years_path)


def _annual_state_router_combo_issues(router_cfg: dict[str, Any], combo: Any) -> list[str]:
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
            expected = _quality_number(router_cfg.get(key), float("nan"))
            observed = _quality_number(combo.get(key), float("nan"))
            if pd.notna(expected) and pd.notna(observed) and abs(expected - observed) > 1e-9:
                issues.append(f"annual_state_router_evidence_combo_mismatch:{key}")
    for key in ["moderate_positive_source", "moderate_low_source", "moderate_lower_source", "turnover_mode"]:
        if key in router_cfg and key in combo and str(router_cfg.get(key) or "") != str(combo.get(key) or ""):
            issues.append(f"annual_state_router_evidence_combo_mismatch:{key}")
    expected_reasons = _router_reason_set(router_cfg.get("turnover_boost_reasons"))
    observed_reasons = _router_reason_set(combo.get("turnover_boost_reasons"))
    if expected_reasons != observed_reasons:
        issues.append("annual_state_router_evidence_combo_mismatch:turnover_boost_reasons")
    expected_min_position_reasons = _router_reason_set(router_cfg.get("risk_exit_min_positions_reasons"))
    observed_min_position_reasons = _router_reason_set(combo.get("risk_exit_min_positions_reasons"))
    if expected_min_position_reasons != observed_min_position_reasons:
        issues.append("annual_state_router_evidence_combo_mismatch:risk_exit_min_positions_reasons")
    return issues


def _run_data_preparation_stage(
    args: argparse.Namespace,
    config: dict[str, Any],
    end_date: str,
    out_dir: Path,
    status: dict[str, Any],
    artifacts: list[Path],
) -> DataPreparationStageResult:
    """Run update, conversion, factor loading, and data quality stages."""
    update_info: dict[str, Any] | None = None
    if not args.skip_update:
        _stage(status, out_dir, "update_data", "running")
        logger.info("Updating raw stock data that is missing or stale.")
        update_result = update_daily_data_resumable(
            start_date=args.start_date,
            end_date=end_date,
            chunk_size=args.chunk_size,
            sleep_seconds=args.sleep_seconds,
            max_chunks=args.max_chunks,
            include_existing=args.include_existing,
        )
        update_info = _update_result_status(update_result)
        update_state = str(update_info.get("status") or "complete")
        if update_state not in {"complete", "partial", "error"}:
            update_state = "complete"
        _stage(status, out_dir, "update_data", update_state, _update_status_message(update_info))
        if update_state == "error" and not args.allow_unhealthy:
            raise RuntimeError(f"Data update failed: {update_info.get('last_error') or update_info}")
    else:
        _stage(status, out_dir, "update_data", "skipped")

    if not args.skip_convert:
        if _can_reuse_conversion_outputs(update_info, config, end_date):
            logger.info("Skipping conversion because no raw files changed and conversion outputs cover %s.", end_date)
            _stage(status, out_dir, "convert_data", "skipped", "cache_current_no_raw_changes")
        else:
            _stage(status, out_dir, "convert_data", "running")
            logger.info("Converting raw data to Qlib provider and price panels.")
            convert_to_qlib_format()
            _stage(status, out_dir, "convert_data", "complete")
    else:
        _stage(status, out_dir, "convert_data", "skipped")

    factor_file = config["factors"]["cache_file"]
    _stage(status, out_dir, "compute_factors", "running")
    logger.info("Loading or computing factors.")
    factor_path = resolve_path(factor_file)
    if args.skip_factor:
        if not factor_path.exists():
            raise FileNotFoundError(f"Factor cache not found: {factor_path}")
        factors = pd.read_parquet(factor_path)
    else:
        factors = load_or_compute_factors(args.start_date, end_date, cache_file=factor_file, force=args.force_factor)
    _stage(status, out_dir, "compute_factors", "complete")

    price_path = resolve_path(config.get("ic", {}).get("price_file", "data/prices/ohlcv_adjusted.parquet"))
    if not price_path.exists():
        raise FileNotFoundError(f"Price file not found: {price_path}. Run conversion first.")
    prices = pd.read_parquet(price_path)

    _stage(status, out_dir, "data_health", "running")
    data_health = build_data_health_report(config, price_df=prices, factor_df=factors)
    health_json, health_csv = write_data_health_report(data_health, out_dir)
    artifacts.extend([health_json, health_csv])
    _stage(status, out_dir, "data_health", "complete", "healthy" if data_health.is_healthy else ",".join(data_health.issues))
    data_gate = data_health.is_healthy or args.allow_unhealthy

    if not args.skip_adj_factor_meta:
        _stage(status, out_dir, "adj_factor_meta", "running")
        adj_factor_meta = build_adj_factor_metadata(config)
        adj_factor_meta_path = write_adj_factor_metadata(adj_factor_meta, config)
        artifacts.append(adj_factor_meta_path)
        _stage(
            status,
            out_dir,
            "adj_factor_meta",
            "complete",
            f"{adj_factor_meta.files_with_adj_factor}/{adj_factor_meta.raw_file_count}",
        )
    else:
        _stage(status, out_dir, "adj_factor_meta", "skipped")

    _stage(status, out_dir, "data_governance", "running")
    data_governance = build_data_governance_report(config)
    governance_path = write_data_governance_report(data_governance, out_dir)
    artifacts.append(governance_path)
    _stage(
        status,
        out_dir,
        "data_governance",
        "complete",
        "point_in_time_ready" if data_governance.is_point_in_time_ready else ",".join(data_governance.issues),
    )
    governance_gate = data_governance.is_point_in_time_ready

    return DataPreparationStageResult(
        factor_file=factor_file,
        factors=factors,
        prices=prices,
        data_health=data_health,
        data_governance=data_governance,
        data_gate=data_gate,
        governance_gate=governance_gate,
        health_json=health_json,
        governance_path=governance_path,
    )


def _run_optimization_stage(
    args: argparse.Namespace,
    config: dict[str, Any],
    factors: pd.DataFrame,
    prices: pd.DataFrame,
    end_date: str,
    out_dir: Path,
    status: dict[str, Any],
    artifacts: list[Path],
) -> OptimizationStageResult:
    """Run or skip automatic parameter validation and selection."""
    selected_config = config
    selected_params: dict[str, Any] = dict(config.get("strategy", {}))
    selected_params_status = "current_config"
    validation = pd.DataFrame()
    summary = pd.DataFrame()
    if _annual_state_router_enabled(config):
        _stage(status, out_dir, "optimize_params", "skipped", "annual_state_router_enabled")
        selected_params = _annual_state_router_selected_params(config)
        return OptimizationStageResult(config, selected_params, "annual_state_router", validation, summary)
    if args.skip_optimize:
        _stage(status, out_dir, "optimize_params", "skipped")
        return OptimizationStageResult(config, selected_params, "skipped", validation, summary)

    _stage(status, out_dir, "optimize_params", "running")
    grid_defaults = DEFAULT_GRID if args.full_grid else BASELINE_GRID
    grid = {
        **grid_defaults,
        "factor_group": _grid_values(args.factor_groups, grid_defaults["factor_group"], str),
        "top_n": _grid_values(args.top_n, grid_defaults["top_n"], int),
        "max_turnover": _grid_values(args.max_turnover, grid_defaults["max_turnover"], int),
        "rank_buffer": _grid_values(args.rank_buffer, grid_defaults["rank_buffer"], int),
        "rebalance_freq": _grid_values(args.rebalance_freq, grid_defaults["rebalance_freq"], str),
    }
    _maybe_add_grid_values(grid, "max_weight_per_stock", args.max_weight_per_stock, float)
    _maybe_add_grid_values(grid, "stop_loss_pct", args.stop_loss_pct, float)
    _maybe_add_grid_values(grid, "take_profit_pct", args.take_profit_pct, float)
    _maybe_add_grid_values(grid, "circuit_breaker_drawdown", args.circuit_breaker_drawdown, float)
    _maybe_add_grid_values(grid, "circuit_breaker_cooldown_days", args.circuit_breaker_cooldown_days, int)
    _maybe_add_grid_values(grid, "circuit_breaker_target_exposure", args.circuit_breaker_target_exposure, float)
    _maybe_add_grid_values(grid, "target_vol", args.target_vol, float)
    _maybe_add_grid_values(grid, "max_industry_weight", args.max_industry_weight, float)
    _maybe_add_grid_values(grid, "rebalance_drift_threshold", args.rebalance_drift_threshold, float)
    param_columns = list(grid)
    total_combinations = 1
    for values in grid.values():
        total_combinations *= len(values)
    logger.info("Automatic validation grid has %s combinations: %s", total_combinations, grid)
    logger.info("Running automatic walk-forward grid validation.")
    _stage(status, out_dir, "optimize_params", "running", f"0 results; {total_combinations} combinations per validation window")

    def on_validation_result(row: dict[str, object], frame: pd.DataFrame) -> None:
        """函数说明：处理 on_validation_result 主要逻辑。"""
        _stage(status, out_dir, "optimize_params", "running", _validation_progress_message(row, len(frame)))

    risk_policy = RiskPolicy(config)
    base_bt_config = apply_defensive_timing_to_backtest_config({**config["backtest"], **config["strategy"]}, prices, config)
    base_bt_config = risk_policy.apply_to_backtest_config(
        base_bt_config,
        force_industry_map=_grid_has_enabled_value(grid, "max_industry_weight"),
    )
    validation_path = out_dir / "auto_validation_windows.csv"
    summary_path = out_dir / "auto_parameter_summary.csv"
    try:
        validation = run_walk_forward_grid_validation(
            factors,
            prices,
            base_config=base_bt_config,
            start_date=args.start_date,
            end_date=end_date,
            grid=grid,
            train_years=args.train_years,
            test_months=args.test_months,
            step_months=args.step_months,
            turnover_penalty=args.turnover_penalty,
            cost_penalty=args.cost_penalty,
            target_annual_return=args.target_annual_return,
            min_annual_return=args.min_annual_return,
            drawdown_limit=args.drawdown_limit,
            drawdown_penalty=args.drawdown_penalty,
            use_rolling_ic=True,
            ic_horizon=int(config.get("ic", {}).get("horizon", 1)),
            ic_method=str(config.get("ic", {}).get("method", "spearman")),
            ic_min_obs=int(config.get("ic", {}).get("min_obs", 20)),
            ic_window=int(config.get("ic", {}).get("window", 252)),
            ic_min_periods=int(config.get("ic", {}).get("min_periods", 60)),
            ic_min_abs=float(config.get("ic", {}).get("min_abs_ic", 0.02)),
            ic_corr_threshold=float(config.get("ic", {}).get("corr_threshold", 0.7)),
            ic_top_k=int(config.get("ic", {}).get("top_k", 30)),
            ic_weight_smoothing=float(config.get("ic", {}).get("weight_smoothing", 0.0)),
            ic_max_weight_turnover=config.get("ic", {}).get("max_weight_turnover"),
            scoring_config=config,
            on_result=on_validation_result,
            timeout_seconds=args.optimize_timeout_seconds,
            max_grid_combinations=args.max_optimize_combinations,
        )
    except OptimizationTimeoutError as exc:
        validation = exc.partial_results
        validation.to_csv(validation_path, index=False, encoding="utf-8-sig")
        summary = summarize_parameter_validation(validation, param_columns=param_columns)
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        artifacts.extend([validation_path, summary_path])
        status["optimizer_timeout"] = {
            "message": str(exc),
            "completed_windows": exc.completed_windows,
            "completed_combinations": exc.completed_combinations,
            "validation_path": str(validation_path),
            "summary_path": str(summary_path),
        }
        _stage(status, out_dir, "optimize_params", "timeout", str(exc))
        raise
    validation.to_csv(validation_path, index=False, encoding="utf-8-sig")
    summary = summarize_parameter_validation(validation, param_columns=param_columns)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    artifacts.extend([validation_path, summary_path])
    if not summary.empty:
        try:
            selected_params = select_stable_params(
                summary,
                config.get("quality", {}),
                param_columns=param_columns,
                strict=True,
            )
            selected_config = apply_strategy_params(config, selected_params)
            selected_params_status = "selected_acceptable_params"
        except ValueError as exc:
            if str(exc) != "no_acceptable_params":
                raise
            selected_params = {}
            selected_config = config
            selected_params_status = "no_acceptable_params"
    _stage(status, out_dir, "optimize_params", "complete")
    return OptimizationStageResult(selected_config, selected_params, selected_params_status, validation, summary)


def _run_backtest_stage(
    args: argparse.Namespace,
    selected_config: dict[str, Any],
    factors: pd.DataFrame,
    prices: pd.DataFrame,
    end_date: str,
    out_dir: Path,
    status: dict[str, Any],
    artifacts: list[Path],
) -> BacktestStageResult:
    """Run or skip the historical backtest and research diagnostics."""
    equity_path = out_dir / "auto_backtest_equity.csv"
    holdings_bt_path = out_dir / "auto_backtest_holdings.csv"
    trades_path = out_dir / "auto_backtest_trades.csv"
    metrics_path = out_dir / "auto_backtest_metrics.json"
    backtest_runtime_config: dict[str, Any] = {**selected_config["backtest"], **selected_config["strategy"]}
    annual_state_router: AnnualStateRouterRuntime | None = None
    if args.skip_backtest:
        _stage(status, out_dir, "backtest", "skipped")
        result = BacktestResult(
            equity_curve=pd.Series(dtype=float, name="equity"),
            holdings=pd.DataFrame(),
            trades=pd.DataFrame(),
            metrics={"backtest_skipped": True},
        )
    else:
        if _annual_state_router_enabled(selected_config):
            _stage(status, out_dir, "backtest", "running", "building annual state router scores")
            annual_state_router = _build_annual_state_router_runtime(
                config=selected_config,
                prices=prices,
                start_date=args.start_date,
                end_date=end_date,
                out_dir=out_dir,
                artifacts=artifacts,
                status=status,
            )
            scores = annual_state_router.routed.scores
            bt_config = annual_state_router.backtest_config
        else:
            _stage(status, out_dir, "backtest", "running", "building strategy scores")
            scores = build_strategy_scores(factors, selected_config, price_df=prices)
            _stage(status, out_dir, "backtest", "running", "resampling signals")
            scores = resample_signals(scores, selected_config["strategy"].get("rebalance_freq", "daily"))
            _stage(status, out_dir, "backtest", "running", "preparing backtest config")
            bt_config = apply_defensive_timing_to_backtest_config(
                backtest_runtime_config,
                prices,
                selected_config,
            )
            bt_config = RiskPolicy(selected_config).apply_to_backtest_config(bt_config)
        backtest_runtime_config = bt_config
        _stage(status, out_dir, "backtest", "running", "running historical backtest")
        result = run_backtest(
            scores,
            prices,
            args.start_date,
            end_date,
            bt_config,
        )
        result.equity_curve.to_csv(equity_path, encoding="utf-8-sig")
        result.holdings.to_csv(holdings_bt_path, index=False, encoding="utf-8-sig")
        result.trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
        _stage(status, out_dir, "backtest", "complete")

    _write_json(metrics_path, result.metrics)
    artifacts.append(metrics_path)
    if not args.skip_backtest:
        artifacts.extend([equity_path, holdings_bt_path, trades_path])

    backtest_yearly_quality = build_yearly_breakdown(result, backtest_runtime_config) if not args.skip_backtest else pd.DataFrame()
    backtest_quality = assess_backtest_quality(result.metrics, selected_config.get("quality", {}), yearly=backtest_yearly_quality)
    backtest_quality_path = out_dir / "auto_backtest_quality.json"
    _write_json(backtest_quality_path, backtest_quality.to_dict())
    artifacts.append(backtest_quality_path)
    backtest_quality_gate = backtest_quality.is_acceptable or args.allow_low_quality

    research_diagnostics: dict[str, Any] = {"enabled": False, "issues": ["backtest_skipped"] if args.skip_backtest else []}
    research_tables: dict[str, pd.DataFrame] = {}
    research_files: dict[str, str] = {}
    if not args.skip_backtest:
        _stage(status, out_dir, "research_diagnostics", "running")
        research_diagnostics, research_tables = build_research_diagnostics(
            result.equity_curve,
            result.holdings,
            result.trades,
            prices,
            selected_config,
        )
        research_files = write_research_diagnostics(research_diagnostics, research_tables, out_dir)
        artifacts.extend(Path(path) for path in research_files.values())
        _stage(
            status,
            out_dir,
            "research_diagnostics",
            "complete",
            ",".join(map(str, research_diagnostics.get("issues", []))) or "ok",
        )
    else:
        _stage(status, out_dir, "research_diagnostics", "skipped")

    return BacktestStageResult(
        result=result,
        backtest_runtime_config=backtest_runtime_config,
        annual_state_router=annual_state_router,
        backtest_quality=backtest_quality,
        backtest_quality_gate=backtest_quality_gate,
        backtest_quality_path=backtest_quality_path,
        metrics_path=metrics_path,
        research_diagnostics=research_diagnostics,
        research_tables=research_tables,
        research_files=research_files,
    )


def _run_signal_stage(
    args: argparse.Namespace,
    selected_config: dict[str, Any],
    factors: pd.DataFrame,
    prices: pd.DataFrame,
    factor_file: str,
    end_date: str,
    out_dir: Path,
    status: dict[str, Any],
    artifacts: list[Path],
    *,
    data_health: Any,
    data_governance: Any,
    data_gate: bool,
    governance_gate: bool,
    parameter_quality: ParameterQualityReport,
    parameter_quality_gate: bool,
    backtest_quality: ParameterQualityReport,
    backtest_quality_gate: bool,
    result: BacktestResult,
    backtest_runtime_config: dict[str, Any],
    annual_state_router: AnnualStateRouterRuntime | None,
    selected_params: dict[str, Any],
    selected_params_status: str,
    validation: pd.DataFrame,
    research_diagnostics: dict[str, Any],
    research_tables: dict[str, pd.DataFrame],
) -> SignalStageResult:
    """Generate the signal, execution templates, and failure-analysis artifacts."""
    _stage(status, out_dir, "generate_signal", "running")
    previous, previous_source = read_signal_previous_holdings(selected_config)
    logger.info("Previous holdings source: %s (%d instruments)", previous_source, len(previous))
    signal_date_arg = _resolve_signal_date_arg(args.date, end_date)
    signal_config = selected_config
    signal_scores: pd.Series | None = None
    if annual_state_router is not None:
        signal_config = _annual_state_router_signal_config(selected_config, annual_state_router, signal_date_arg)
        signal_scores = annual_state_router.routed.scores
    signal_df, target_holdings = generate_signal(
        signal_date_arg,
        previous_holdings=previous,
        factor_file=factor_file,
        config=signal_config,
        factors=factors,
        scores=signal_scores,
        price_df=prices,
    )
    output_date = _signal_output_date(signal_df, signal_date_arg, factors=factors)
    intended = next_trade_date(output_date, price_df=prices) or next_business_day(
        output_date,
        config=selected_config,
        price_df=prices,
    )
    intended_text = str(pd.Timestamp(intended).date())
    account = load_account_state(selected_config)
    current_holdings = load_current_holdings(selected_config)
    account_issues = validate_account_inputs(account, current_holdings, selected_config)

    block_reasons: list[str] = []
    quality_warnings: list[str] = []
    if not data_health.is_healthy:
        quality_warnings.extend([f"data:{issue}" for issue in data_health.issues])
    if not data_governance.is_point_in_time_ready:
        quality_warnings.extend([f"governance:{issue}" for issue in data_governance.issues])
    if not parameter_quality.is_acceptable:
        quality_warnings.extend([f"params:{issue}" for issue in parameter_quality.issues])
    if not backtest_quality.is_acceptable:
        quality_warnings.extend([f"backtest:{issue}" for issue in backtest_quality.issues])
    if account_issues:
        quality_warnings.extend([f"account:{issue}" for issue in account_issues])
    if not data_gate:
        block_reasons.extend([f"data:{issue}" for issue in data_health.issues])
    if not governance_gate:
        block_reasons.extend([f"governance:{issue}" for issue in data_governance.issues])
    if not parameter_quality_gate:
        block_reasons.extend([f"params:{issue}" for issue in parameter_quality.issues])
    if not backtest_quality_gate:
        block_reasons.extend([f"backtest:{issue}" for issue in backtest_quality.issues])
    if account_issues:
        block_reasons.extend([f"account:{issue}" for issue in account_issues])
    allowed_with_warnings = not block_reasons and bool(quality_warnings) and args.force_official
    is_executable = not quality_warnings or allowed_with_warnings
    if not is_executable and not block_reasons:
        block_reasons = list(quality_warnings)
    if getattr(args, "candidate_only", False):
        is_executable = False
        if "candidate_only_requested" not in block_reasons:
            block_reasons.append("candidate_only_requested")

    failure_analysis: dict[str, Any] = {"enabled": False, "issues": ["backtest_skipped"] if args.skip_backtest else []}
    failure_files: dict[str, str] = {}
    if not args.skip_backtest:
        failure_analysis, failure_tables = build_failure_analysis_artifacts(
            selected_params=selected_params,
            selected_params_status=selected_params_status,
            parameter_quality=parameter_quality.to_dict(),
            backtest_quality=backtest_quality.to_dict(),
            backtest_metrics=result.metrics,
            block_reasons=block_reasons,
            quality_warnings=quality_warnings,
            validation=validation,
            backtest_result=result,
            backtest_config=backtest_runtime_config,
            research_diagnostics=research_diagnostics,
            research_tables=research_tables,
            start_date=args.start_date,
            end_date=end_date,
        )
        failure_files = write_failure_analysis_artifacts(failure_analysis, failure_tables, out_dir)
        artifacts.extend(Path(path) for path in failure_files.values())
        if failure_analysis.get("parameter_backtest_mismatch"):
            logger.warning("Parameter validation passed, but full-history backtest failed quality gates.")
        if not backtest_quality.is_acceptable:
            gaps = failure_analysis.get("backtest_threshold_gaps", {})
            logger.warning(
                "Backtest quality failed: annual_return=%s target=%s gap=%s; max_drawdown=%s limit=%s gap=%s.",
                gaps.get("annual_return"),
                gaps.get("min_backtest_annual_return"),
                gaps.get("annual_return_gap"),
                gaps.get("max_drawdown"),
                gaps.get("max_backtest_drawdown_limit"),
                gaps.get("max_drawdown_gap"),
            )
        drawdown_summary = failure_analysis.get("drawdown_summary", {})
        if drawdown_summary.get("trough_date"):
            logger.info(
                "Worst drawdown: peak=%s start=%s trough=%s recovery=%s max_drawdown=%s.",
                drawdown_summary.get("peak_date"),
                drawdown_summary.get("start_date"),
                drawdown_summary.get("trough_date"),
                drawdown_summary.get("recovery_date"),
                drawdown_summary.get("max_drawdown"),
            )
        logger.info("Failure analysis saved to %s", failure_files.get("failure_analysis"))

    if is_executable:
        signal_path, holdings_path = save_signal(signal_df, target_holdings, output_date, config=selected_config)
    else:
        signal_path, holdings_path = _save_candidate_signal(signal_df, target_holdings, output_date, out_dir)
    artifacts.extend([signal_path, holdings_path])

    orders = generate_manual_orders(
        signal_df,
        target_holdings,
        prices,
        signal_date=output_date,
        intended_trade_date=intended_text,
        config=selected_config,
        account=account,
        current_holdings=current_holdings,
        is_executable=is_executable,
        block_reasons=block_reasons,
    )
    orders_path = save_manual_orders(orders, output_date, out_dir, executable=is_executable)
    artifacts.append(orders_path)
    confirmation = generate_order_confirmation_template(orders, output_date, intended_text, block_reasons=block_reasons)
    fill_feedback = generate_fill_feedback_template(orders, output_date, intended_text)
    execution_files = save_execution_templates(
        confirmation,
        fill_feedback,
        output_date,
        selected_config,
        executable=is_executable,
    )
    artifacts.extend(Path(path) for path in execution_files.values())
    _stage(status, out_dir, "generate_signal", "complete", "executable" if is_executable else "blocked")

    fundamental_screen, fundamental_files = _maybe_build_fundamental_screen(selected_config, str(output_date), out_dir)
    artifacts.extend(Path(path) for path in fundamental_files.values())

    return SignalStageResult(
        signal_df=signal_df,
        target_holdings=target_holdings,
        output_date=str(output_date),
        intended_trade_date=intended_text,
        previous_holdings_source=previous_source,
        account=account,
        is_executable=is_executable,
        block_reasons=block_reasons,
        quality_warnings=quality_warnings,
        failure_analysis=failure_analysis,
        failure_files=failure_files,
        fundamental_screen=fundamental_screen,
        fundamental_files=fundamental_files,
        signal_path=signal_path,
        holdings_path=holdings_path,
        orders_path=orders_path,
        execution_files=execution_files,
    )


def _write_auto_report_stage(
    args: argparse.Namespace,
    target_resolution: Any,
    selected_config: dict[str, Any],
    selected_params: dict[str, Any],
    selected_params_status: str,
    parameter_quality: ParameterQualityReport,
    backtest_quality: ParameterQualityReport,
    data_health: Any,
    data_governance: Any,
    result: BacktestResult,
    research_diagnostics: dict[str, Any],
    annual_state_router: AnnualStateRouterRuntime | None,
    validation: pd.DataFrame,
    summary: pd.DataFrame,
    signal_stage: SignalStageResult,
    out_dir: Path,
    artifacts: list[Path],
    *,
    health_json: Path,
    governance_path: Path,
    quality_path: Path,
    metrics_path: Path,
    backtest_quality_path: Path,
    research_files: dict[str, str],
) -> ReportStageResult:
    """Write selected parameter, JSON, markdown, and optional archive artifacts."""
    selected_params_path = out_dir / "auto_selected_params.json"
    _write_json(selected_params_path, selected_params)
    artifacts.append(selected_params_path)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target_date_resolution": target_resolution.to_dict(),
        "selected_params": selected_params,
        "selected_params_status": selected_params_status,
        "strategy_mode": "annual_state_router" if annual_state_router is not None else "strategy_config",
        "annual_state_router": _annual_state_router_report(annual_state_router),
        "parameter_quality": parameter_quality.to_dict(),
        "backtest_quality": backtest_quality.to_dict(),
        "data_health": data_health.to_dict(),
        "data_governance": data_governance.to_dict(),
        "backtest_metrics": result.metrics,
        "research_diagnostics": research_diagnostics,
        "failure_analysis": signal_stage.failure_analysis,
        "fundamental_screen": signal_stage.fundamental_screen,
        "account": signal_stage.account.to_dict(),
        "signal_summary": signal_action_summary(signal_stage.signal_df),
        "signal_date": signal_stage.output_date,
        "intended_trade_date": signal_stage.intended_trade_date,
        "previous_holdings_source": signal_stage.previous_holdings_source,
        "is_executable": signal_stage.is_executable,
        "block_reasons": signal_stage.block_reasons,
        "quality_warnings": signal_stage.quality_warnings,
        "allow_low_quality": bool(args.allow_low_quality),
        "allow_unhealthy": bool(args.allow_unhealthy),
        "force_official": bool(args.force_official),
        "candidate_only": bool(getattr(args, "candidate_only", False)),
        "skip_optimize": bool(args.skip_optimize),
        "skip_backtest": bool(args.skip_backtest),
        "validation_windows": int(len(validation)),
        "validation_param_sets": int(len(summary)),
        "files": {
            "signal": str(signal_stage.signal_path),
            "holdings": str(signal_stage.holdings_path),
            "manual_orders": str(signal_stage.orders_path),
            **signal_stage.execution_files,
            "data_health": str(health_json),
            "data_governance": str(governance_path),
            "parameter_quality": str(quality_path),
            "backtest_metrics": str(metrics_path),
            "backtest_quality": str(backtest_quality_path),
            **research_files,
            **(annual_state_router.files if annual_state_router is not None else {}),
            **signal_stage.failure_files,
            **signal_stage.fundamental_files,
        },
    }
    report_path = out_dir / "auto_signal_report.json"
    _write_json(report_path, report)
    markdown_path = write_daily_signal_report(report, out_dir)
    artifacts.extend([report_path, markdown_path])

    if not args.no_archive:
        archive_dir = archive_run(
            artifacts,
            selected_config.get("reports", {}).get("history_dir", "outputs/history"),
            signal_stage.output_date,
        )
        report["files"]["archive_dir"] = str(archive_dir)
        _write_json(report_path, report)

    return ReportStageResult(report_path=report_path, markdown_path=markdown_path, report=report)


def _annual_state_router_report(runtime: AnnualStateRouterRuntime | None) -> dict[str, Any]:
    if runtime is None:
        return {"enabled": False}
    routes = runtime.routed.score_routes
    years = runtime.routed.year_routes
    latest_route = routes.iloc[-1].to_dict() if not routes.empty else {}
    return {
        "enabled": True,
        "score_rows": int(len(runtime.routed.scores)),
        "route_count": int(len(routes)),
        "year_route_count": int(len(years)),
        "latest_route": latest_route,
        "source_definitions": {name: definition.__dict__ for name, definition in runtime.source_definitions.items()},
        "files": runtime.files,
    }


def main() -> None:
    """函数说明：解析命令行参数并执行主流程。"""
    base_config = load_config()
    parser = argparse.ArgumentParser(description="Run data refresh, automatic walk-forward tuning, backtest and latest signal.")
    parser.add_argument("--skip-update", action="store_true", help="Skip Tushare data update.")
    parser.add_argument("--skip-convert", action="store_true", help="Skip raw-to-price conversion.")
    parser.add_argument("--skip-factor", action="store_true", help="Read the factor cache directly without cache validation or recomputation.")
    parser.add_argument("--force-factor", action="store_true", help="Recompute factors even when the existing cache matches the request.")
    parser.add_argument("--skip-adj-factor-meta", action="store_true", help="Skip refreshing adj-factor version metadata.")
    parser.add_argument("--skip-optimize", action="store_true", help="Use current config strategy instead of automatic tuning.")
    parser.add_argument("--skip-backtest", action="store_true", help="Generate signal without running the full historical backtest.")
    parser.add_argument("--allow-unhealthy", action="store_true", help="Continue even if data health checks fail.")
    parser.add_argument("--allow-low-quality", action="store_true", help="Continue and write candidate outputs even if quality gates fail.")
    parser.add_argument("--force-official", action="store_true", help="Write official outputs despite allowed quality warnings.")
    parser.add_argument(
        "--candidate-only",
        action="store_true",
        help="Hold outputs as candidate even when every gate passes; never write official signal/holdings or promote.",
    )
    parser.add_argument("--no-archive", action="store_true", help="Do not copy run artifacts into outputs/history.")
    parser.add_argument(
        "--promote-candidate",
        metavar="YYYY-MM-DD",
        help="Promote candidate_signal_DATE.csv and candidate_holdings_DATE.csv to official signal/latest holdings, then exit.",
    )
    parser.add_argument("--start-date", default=base_config["data"]["start_date"])
    parser.add_argument("--end-date", default=base_config["data"]["end_date"])
    parser.add_argument("--date", default="latest", help="Signal date, YYYY-MM-DD, or latest.")
    parser.add_argument("--chunk-size", type=int, default=base_config["data"].get("update_chunk_size", 20))
    parser.add_argument("--sleep-seconds", type=float, default=base_config["data"].get("update_sleep_seconds", 1))
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--include-existing", action="store_true", help="Refresh all existing raw files, even if already latest.")
    parser.add_argument("--factor-groups", help="Comma-separated factor groups. Defaults to the fast baseline grid.")
    parser.add_argument("--top-n", help="Comma-separated portfolio sizes. Defaults to the fast baseline grid.")
    parser.add_argument("--max-turnover", help="Comma-separated max turnover values. Defaults to the fast baseline grid.")
    parser.add_argument("--rank-buffer", help="Comma-separated rank buffer values. Defaults to the fast baseline grid.")
    parser.add_argument("--rebalance-freq", help="Comma-separated rebalance frequencies. Defaults to the fast baseline grid.")
    parser.add_argument("--max-weight-per-stock", help="Comma-separated per-stock caps, or none.")
    parser.add_argument("--stop-loss-pct", help="Comma-separated stop-loss percentages, or none.")
    parser.add_argument("--take-profit-pct", help="Comma-separated take-profit percentages, or none.")
    parser.add_argument("--circuit-breaker-drawdown", help="Comma-separated portfolio drawdown breakers, or none.")
    parser.add_argument("--circuit-breaker-cooldown-days", help="Comma-separated breaker cooldown sessions, or none.")
    parser.add_argument("--circuit-breaker-target-exposure", help="Comma-separated breaker target exposures, or none.")
    parser.add_argument("--target-vol", help="Comma-separated target volatility values, or none.")
    parser.add_argument("--max-industry-weight", help="Comma-separated max single-industry weights, or none.")
    parser.add_argument("--rebalance-drift-threshold", help="Comma-separated rebalance drift thresholds, or none.")
    parser.add_argument("--full-grid", action="store_true", help="Use the full default grid instead of the fast baseline grid.")
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--test-months", type=int, default=12)
    parser.add_argument("--step-months", type=int, default=12)
    parser.add_argument("--turnover-penalty", type=float, default=0.02)
    parser.add_argument("--cost-penalty", type=float, default=1.0)
    parser.add_argument("--target-annual-return", type=float, default=base_config.get("quality", {}).get("target_annual_return", 0.20))
    parser.add_argument("--min-annual-return", type=float, default=base_config.get("quality", {}).get("min_optimizer_annual_return", 0.18))
    parser.add_argument("--drawdown-limit", type=float, default=base_config.get("quality", {}).get("max_backtest_drawdown_limit", -0.20))
    parser.add_argument("--drawdown-penalty", type=float, default=4.0)
    parser.add_argument(
        "--optimize-timeout-seconds",
        type=float,
        default=base_config.get("quality", {}).get("optimizer_timeout_seconds"),
        help="Maximum seconds for walk-forward grid validation. Omit or use <=0 to disable.",
    )
    parser.add_argument(
        "--max-optimize-combinations",
        type=int,
        default=base_config.get("quality", {}).get("max_optimizer_combinations"),
        help="Maximum parameter combinations allowed per validation window. Omit to disable.",
    )
    args = parser.parse_args()
    if args.candidate_only and args.promote_candidate:
        parser.error("--candidate-only cannot be combined with --promote-candidate.")

    config = deepcopy(base_config)
    out_dir = resolve_path(config["outputs"].get("dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.promote_candidate:
        signal_path, holdings_path = _promote_candidate(args.promote_candidate, config, out_dir)
        logger.info("Promoted candidate signal to %s", signal_path)
        logger.info("Promoted candidate holdings to %s", holdings_path)
        return

    config["data"]["start_date"] = args.start_date
    target_resolution = resolve_target_date(args.end_date, config=config)
    end_date = target_resolution.target_date
    config["data"]["end_date"] = end_date
    status = _new_status(target_resolution.to_dict())
    _write_status(out_dir, status)
    artifacts: list[Path] = []

    try:
        logger.info(
            "Resolved target end date: %s (requested=%s, latest_trade_date=%s, cutoff=%s, reason=%s).",
            end_date,
            target_resolution.requested,
            target_resolution.latest_trade_date,
            target_resolution.cutoff_time,
            target_resolution.reason,
        )
        data_stage = _run_data_preparation_stage(args, config, end_date, out_dir, status, artifacts)
        factor_file = data_stage.factor_file
        factors = data_stage.factors
        prices = data_stage.prices
        data_health = data_stage.data_health
        data_governance = data_stage.data_governance
        data_gate = data_stage.data_gate
        governance_gate = data_stage.governance_gate
        health_json = data_stage.health_json
        governance_path = data_stage.governance_path

        optimization = _run_optimization_stage(args, config, factors, prices, end_date, out_dir, status, artifacts)
        selected_config = optimization.selected_config
        selected_params = optimization.selected_params
        selected_params_status = optimization.selected_params_status
        validation = optimization.validation
        summary = optimization.summary

        parameter_quality = (
            _annual_state_router_quality(config, config.get("quality", {}))
            or (
                assess_parameter_quality(summary, config.get("quality", {}))
                if not args.skip_optimize
                else _validated_strategy_quality(config, config.get("quality", {}))
                or _skipped_quality(config.get("quality", {}))
            )
        )
        quality_path = out_dir / "auto_parameter_quality.json"
        _write_json(quality_path, parameter_quality.to_dict())
        artifacts.append(quality_path)
        parameter_quality_gate = parameter_quality.is_acceptable or args.allow_low_quality

        logger.info("Selected strategy params: %s", selected_params)
        if _annual_state_router_enabled(selected_config) and len(factors) > 1:
            factors = factors.tail(1).copy()
            data_stage.factors = factors
            gc.collect()
            logger.info("Released the full factor frame before annual-state source construction; retained one date row.")
        backtest_stage = _run_backtest_stage(args, selected_config, factors, prices, end_date, out_dir, status, artifacts)
        result = backtest_stage.result
        backtest_runtime_config = backtest_stage.backtest_runtime_config
        annual_state_router = backtest_stage.annual_state_router
        backtest_quality = backtest_stage.backtest_quality
        backtest_quality_gate = backtest_stage.backtest_quality_gate
        backtest_quality_path = backtest_stage.backtest_quality_path
        metrics_path = backtest_stage.metrics_path
        research_diagnostics = backtest_stage.research_diagnostics
        research_tables = backtest_stage.research_tables
        research_files = backtest_stage.research_files

        signal_stage = _run_signal_stage(
            args,
            selected_config,
            factors,
            prices,
            factor_file,
            end_date,
            out_dir,
            status,
            artifacts,
            data_health=data_health,
            data_governance=data_governance,
            data_gate=data_gate,
            governance_gate=governance_gate,
            parameter_quality=parameter_quality,
            parameter_quality_gate=parameter_quality_gate,
            backtest_quality=backtest_quality,
            backtest_quality_gate=backtest_quality_gate,
            result=result,
            backtest_runtime_config=backtest_runtime_config,
            annual_state_router=annual_state_router,
            selected_params=selected_params,
            selected_params_status=selected_params_status,
            validation=validation,
            research_diagnostics=research_diagnostics,
            research_tables=research_tables,
        )

        report_stage = _write_auto_report_stage(
            args,
            target_resolution,
            selected_config,
            selected_params,
            selected_params_status,
            parameter_quality,
            backtest_quality,
            data_health,
            data_governance,
            result,
            research_diagnostics,
            annual_state_router,
            validation,
            summary,
            signal_stage,
            out_dir,
            artifacts,
            health_json=health_json,
            governance_path=governance_path,
            quality_path=quality_path,
            metrics_path=metrics_path,
            backtest_quality_path=backtest_quality_path,
            research_files=research_files,
        )

        status["status"] = "complete" if signal_stage.is_executable else "blocked"
        status["finished_at"] = datetime.now().isoformat(timespec="seconds")
        status["target_date_resolution"] = target_resolution.to_dict()
        status["is_executable"] = signal_stage.is_executable
        status["block_reasons"] = signal_stage.block_reasons
        status["quality_warnings"] = signal_stage.quality_warnings
        status["selected_params_status"] = selected_params_status
        status["strategy_mode"] = "annual_state_router" if annual_state_router is not None else "strategy_config"
        _write_status(out_dir, status)
        logger.info("Auto signal saved to %s", signal_stage.signal_path)
        logger.info("Manual orders saved to %s", signal_stage.orders_path)
        logger.info("Auto report saved to %s", report_stage.report_path)
    except Exception as exc:
        status["status"] = "failed"
        status["finished_at"] = datetime.now().isoformat(timespec="seconds")
        status["last_error"] = str(exc)
        _stage(status, out_dir, "run", "failed", str(exc))
        _write_status(out_dir, status)
        raise


def _new_status(target_date_resolution: dict[str, str] | None = None) -> dict[str, Any]:
    """函数说明：处理 new_status 的内部辅助逻辑。"""
    status: dict[str, Any] = {"status": "running", "started_at": datetime.now().isoformat(timespec="seconds"), "stages": []}
    if target_date_resolution is not None:
        status["target_date_resolution"] = target_date_resolution
    return status


def _update_result_status(result: Any) -> dict[str, Any]:
    """函数说明：更新 update_result_status 的内部辅助逻辑。"""
    if hasattr(result, "to_status_dict"):
        value = result.to_status_dict()
        return value if isinstance(value, dict) else {}
    status = getattr(result, "status", "")
    if status:
        return {
            "status": status,
            "failed_symbols": getattr(result, "failed_symbols", 0),
            "remaining_symbols": getattr(result, "remaining_symbols", 0),
            "last_error": getattr(result, "last_error", ""),
            "written_symbols": len(result) if hasattr(result, "__len__") else 0,
        }
    if isinstance(result, dict):
        return {"status": "complete", "written_symbols": len(result)}
    return {"status": "complete"}


def _update_status_message(info: dict[str, Any]) -> str:
    """函数说明：更新 update_status_message 的内部辅助逻辑。"""
    parts = [
        f"written={info.get('written_symbols', 0)}",
        f"failed={info.get('failed_symbols', 0)}",
        f"remaining={info.get('remaining_symbols', 0)}",
    ]
    if info.get("progress_path"):
        parts.append(f"progress={info['progress_path']}")
    if info.get("last_error"):
        parts.append(f"last_error={info['last_error']}")
    return "; ".join(parts)


def _can_reuse_conversion_outputs(update_info: dict[str, Any] | None, config: dict[str, Any], end_date: str) -> bool:
    """Return whether a no-change update may reuse current Qlib/price outputs."""
    if not update_info or str(update_info.get("status")) != "complete":
        return False
    if int(update_info.get("written_symbols", 0) or 0) != 0:
        return False

    target = pd.Timestamp(end_date).normalize()
    price_path = resolve_path(config.get("ic", {}).get("price_file", "data/prices/ohlcv_adjusted.parquet"))
    provider = resolve_path(config.get("qlib", {}).get("provider_uri", "data/qlib_data"))
    calendar_path = provider / "calendars" / "day.txt"
    instrument_path = provider / "instruments" / "all.txt"
    if not price_path.exists() or not calendar_path.exists() or not instrument_path.exists():
        return False
    try:
        price_index = pd.read_parquet(price_path, columns=[]).index
        if len(price_index) == 0 or pd.to_datetime(price_index).max().normalize() < target:
            return False
        calendar = pd.read_csv(calendar_path, header=None, usecols=[0]).iloc[:, 0]
        if calendar.empty or pd.to_datetime(calendar, errors="coerce").max().normalize() < target:
            return False
    except (OSError, ValueError, TypeError, IndexError):
        return False
    return True


def _resolve_signal_date_arg(value: str, target_end_date: str) -> str:
    """函数说明：解析 resolve_signal_date_arg 的内部辅助逻辑。"""
    return target_end_date if str(value).strip().lower() in {"auto", "latest_trade_date", "latest_trading_day"} else value


def _signal_output_date(signal_df: pd.DataFrame, signal_date_arg: str, factors: pd.DataFrame | None = None) -> str:
    """函数说明：处理 signal_output_date 的内部辅助逻辑。"""
    if not signal_df.empty and "date" in signal_df.columns:
        return str(signal_df["date"].iloc[0])
    signal_date = getattr(signal_df, "attrs", {}).get("signal_date")
    if signal_date:
        return str(signal_date)
    inferred = _infer_signal_output_date(signal_date_arg, factors)
    if inferred:
        return inferred
    return signal_date_arg


def _infer_signal_output_date(signal_date_arg: str, factors: pd.DataFrame | None) -> str | None:
    """函数说明：处理 infer_signal_output_date 的内部辅助逻辑。"""
    if factors is None or factors.empty or not isinstance(factors.index, pd.MultiIndex):
        return None
    date_level = factors.index.names[0] or 0
    dates = pd.DatetimeIndex(pd.to_datetime(factors.index.get_level_values(date_level)).normalize()).unique().sort_values()
    if dates.empty:
        return None
    arg = str(signal_date_arg).strip().lower()
    if arg in {"", "none", "latest"}:
        return str(pd.Timestamp(dates.max()).date())
    requested = pd.to_datetime(signal_date_arg, errors="coerce")
    if pd.isna(requested):
        return None
    eligible = dates[dates <= pd.Timestamp(requested).normalize()]
    if eligible.empty:
        return None
    return str(pd.Timestamp(eligible.max()).date())


def _stage(status: dict[str, Any], out_dir: Path, name: str, state: str, message: str = "") -> None:
    """函数说明：处理 stage 的内部辅助逻辑。"""
    status["stages"].append(
        {
            "name": name,
            "state": state,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "message": message,
        }
    )
    _write_status(out_dir, status)


def _maybe_build_fundamental_screen(config: dict[str, Any], output_date: str, out_dir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Build the optional fundamental screen artifact for the daily report."""

    screen_cfg = config.get("fundamental_screen", {})
    if not bool(screen_cfg.get("include_in_auto_report", False)):
        return {"enabled": False, "status": "disabled"}, {}
    try:
        result = build_fundamental_screen(config=config, as_of=output_date)
        csv_path, report_path = write_fundamental_screen_outputs(
            result,
            out_dir,
            top_n=int(screen_cfg.get("top_n", 30)),
        )
        files = {
            "fundamental_screen_csv": str(csv_path),
            "fundamental_screen_report": str(report_path),
        }
        summary = summarize_fundamental_screen_result(
            result,
            top_n=int(screen_cfg.get("auto_report_top_n", 10)),
            csv_path=csv_path,
            report_path=report_path,
        )
        return summary, files
    except Exception as exc:  # pragma: no cover - defensive non-blocking report hook
        logger.exception("Fundamental screen report skipped: %s", exc)
        return {"enabled": True, "status": "failed", "error": str(exc)}, {}


def _validation_progress_message(row: dict[str, object], completed: int) -> str:
    """函数说明：处理 validation_progress_message 的内部辅助逻辑。"""
    test_start = _date_message_value(row.get("test_start"))
    test_end = _date_message_value(row.get("test_end"))
    factor_group = row.get("factor_group", "")
    top_n = row.get("top_n", "")
    rebalance_freq = row.get("rebalance_freq", "")
    return f"{completed} results; latest={test_start}..{test_end} factor_group={factor_group} top_n={top_n} rebalance={rebalance_freq}"


def _date_message_value(value: object) -> str:
    """函数说明：处理 date_message_value 的内部辅助逻辑。"""
    if value is None or value == "":
        return ""
    try:
        return pd.Timestamp(value).date().isoformat()
    except (TypeError, ValueError):
        return str(value)


def _write_status(out_dir: Path, status: dict[str, Any]) -> None:
    """函数说明：写入 write_status 的内部辅助逻辑。"""
    _write_json(out_dir / "auto_run_status.json", status)


def _write_json(path: Path, value: Any) -> None:
    """函数说明：写入 write_json 的内部辅助逻辑。"""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _save_candidate_signal(signal_df: pd.DataFrame, holdings: list[str], signal_date: str, out_dir: Path) -> tuple[Path, Path]:
    """函数说明：保存 save_candidate_signal 的内部辅助逻辑。"""
    signal_path = out_dir / f"candidate_signal_{signal_date}.csv"
    holdings_path = out_dir / f"candidate_holdings_{signal_date}.csv"
    signal_df.to_csv(signal_path, index=False, encoding="utf-8-sig")
    pd.DataFrame({"instrument": holdings}).to_csv(holdings_path, index=False, encoding="utf-8-sig")
    return signal_path, holdings_path


def _promote_candidate(date_arg: str, config: dict, out_dir: Path) -> tuple[Path, Path]:
    """函数说明：处理 promote_candidate 的内部辅助逻辑。"""
    try:
        signal_date = str(pd.Timestamp(date_arg).date())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid --promote-candidate date: {date_arg!r}. Expected YYYY-MM-DD.") from exc
    signal_path = out_dir / f"candidate_signal_{signal_date}.csv"
    holdings_path = out_dir / f"candidate_holdings_{signal_date}.csv"
    if not signal_path.exists():
        raise FileNotFoundError(f"Candidate signal file not found: {signal_path}")
    if not holdings_path.exists():
        raise FileNotFoundError(f"Candidate holdings file not found: {holdings_path}")
    signal_df = pd.read_csv(signal_path)
    holdings_df = pd.read_csv(holdings_path)
    col = "instrument" if "instrument" in holdings_df.columns else "ticker"
    if col not in holdings_df.columns:
        raise ValueError(f"Candidate holdings file must contain instrument or ticker column: {holdings_path}")
    holdings = holdings_df[col].dropna().astype(str).tolist()
    return save_signal(signal_df, holdings, signal_date, config=config)


def _validated_strategy_quality(config: dict[str, Any], quality_config: dict) -> ParameterQualityReport | None:
    """函数说明：处理 validated_strategy_quality 的内部辅助逻辑。"""
    evidence_cfg = config.get("validated_strategy", {})
    if not isinstance(evidence_cfg, dict) or not bool(evidence_cfg.get("enabled", False)):
        return None

    summary_file = evidence_cfg.get("summary_file")
    candidate = str(evidence_cfg.get("candidate", "")).strip()
    if not summary_file:
        return _formal_candidate_quality_report(quality_config, ["validated_strategy_summary_file_missing"])

    summary_path = resolve_path(summary_file)
    if not summary_path.exists():
        return _formal_candidate_quality_report(
            quality_config,
            [f"validated_strategy_summary_file_not_found:{summary_path}"],
        )

    frame = pd.read_csv(summary_path)
    if frame.empty:
        return _formal_candidate_quality_report(quality_config, ["validated_strategy_summary_empty"])
    if "candidate" not in frame.columns:
        return _formal_candidate_quality_report(quality_config, ["validated_strategy_candidate_column_missing"])
    if candidate:
        rows = frame[frame["candidate"].astype(str) == candidate]
        if rows.empty:
            return _formal_candidate_quality_report(quality_config, [f"validated_strategy_candidate_not_found:{candidate}"])
    elif len(frame) == 1:
        rows = frame
    else:
        return _formal_candidate_quality_report(quality_config, ["validated_strategy_candidate_missing"])

    row = rows.iloc[-1]
    annual_return = _quality_number(row.get("annual_return"), 0.0)
    max_drawdown = _quality_number(row.get("max_drawdown"), 0.0)
    sharpe = _quality_number(row.get("sharpe"), 0.0)
    annual_turnover = _quality_number(row.get("annual_turnover"), 0.0)
    annual_trade_cost_ratio = _quality_number(row.get("annual_trade_cost_ratio"), 0.0)
    windows = 1
    positive_return_rate = 1.0 if annual_return > 0 else 0.0
    annual_return_mean = annual_return
    annual_return_min = annual_return
    max_drawdown_worst = max_drawdown

    years_path_value = row.get("years_path")
    if pd.notna(years_path_value) and str(years_path_value).strip():
        years_path = resolve_path(str(years_path_value))
        if years_path.exists():
            yearly = pd.read_csv(years_path)
            if {"annual_return", "max_drawdown"}.issubset(yearly.columns) and not yearly.empty:
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

    issues: list[str] = []
    if bool(evidence_cfg.get("require_is_acceptable", True)) and not _truthy(row.get("is_acceptable")):
        issues.append("validated_strategy_not_formally_acceptable")
    if windows < min_windows:
        issues.append(f"validation_windows_below_threshold:{windows}<{min_windows}")
    if positive_return_rate < min_positive:
        issues.append(f"positive_return_rate_below_threshold:{positive_return_rate:.4f}<{min_positive:.4f}")
    if annual_return < min_return:
        issues.append(f"validated_strategy_annual_return_below_threshold:{annual_return:.4f}<{min_return:.4f}")
    if annual_return_mean < min_return:
        issues.append(f"annual_return_mean_below_threshold:{annual_return_mean:.4f}<{min_return:.4f}")
    if sharpe < min_sharpe:
        issues.append(f"sharpe_mean_below_threshold:{sharpe:.4f}<{min_sharpe:.4f}")
    if max_drawdown < max_drawdown_limit:
        issues.append(f"validated_strategy_max_drawdown_worse_than_limit:{max_drawdown:.4f}<{max_drawdown_limit:.4f}")
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


def _formal_candidate_quality_report(quality_config: dict, issues: list[str]) -> ParameterQualityReport:
    """函数说明：处理 formal_candidate_quality_report 的内部辅助逻辑。"""
    min_windows = int(quality_config.get("min_validation_windows", 3))
    min_positive = float(quality_config.get("min_positive_return_rate", 0.5))
    min_return = float(quality_config.get("min_optimizer_annual_return", quality_config.get("target_annual_return", 0.20)))
    min_sharpe = float(quality_config.get("min_sharpe_mean", 0.0))
    max_drawdown = float(quality_config.get("max_drawdown_limit", -0.20))
    max_turnover = float(quality_config.get("max_annual_turnover", 20.0))
    max_cost = float(quality_config.get("max_annual_trade_cost_ratio", 0.2))
    return ParameterQualityReport(
        is_acceptable=False,
        issues=issues,
        windows=0,
        positive_return_rate=0.0,
        annual_return_mean=0.0,
        annual_return_min=0.0,
        sharpe_mean=0.0,
        max_drawdown_worst=0.0,
        annual_turnover_mean=0.0,
        annual_trade_cost_ratio_mean=0.0,
        min_validation_windows=min_windows,
        min_positive_return_rate=min_positive,
        min_optimizer_annual_return=min_return,
        min_sharpe_mean=min_sharpe,
        max_drawdown_limit=max_drawdown,
        max_annual_turnover=max_turnover,
        max_annual_trade_cost_ratio=max_cost,
    )


def _quality_number(value: Any, default: float) -> float:
    """函数说明：处理 quality_number 的内部辅助逻辑。"""
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return default
    return float(parsed)


def _truthy(value: Any) -> bool:
    """函数说明：处理 truthy 的内部辅助逻辑。"""
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _skipped_quality(quality_config: dict) -> ParameterQualityReport:
    """函数说明：处理 skipped_quality 的内部辅助逻辑。"""
    return ParameterQualityReport(
        is_acceptable=False,
        issues=["parameter_validation_skipped"],
        windows=0,
        positive_return_rate=0.0,
        annual_return_mean=0.0,
        annual_return_min=0.0,
        sharpe_mean=0.0,
        max_drawdown_worst=0.0,
        annual_turnover_mean=0.0,
        annual_trade_cost_ratio_mean=0.0,
        min_validation_windows=int(quality_config.get("min_validation_windows", 3)),
        min_positive_return_rate=float(quality_config.get("min_positive_return_rate", 0.5)),
        min_optimizer_annual_return=float(
            quality_config.get("min_optimizer_annual_return", quality_config.get("target_annual_return", 0.20))
        ),
        min_sharpe_mean=float(quality_config.get("min_sharpe_mean", 0.0)),
        max_drawdown_limit=float(quality_config.get("max_drawdown_limit", -0.20)),
        max_annual_turnover=float(quality_config.get("max_annual_turnover", 20.0)),
        max_annual_trade_cost_ratio=float(quality_config.get("max_annual_trade_cost_ratio", 0.2)),
    )


if __name__ == "__main__":
    main()
