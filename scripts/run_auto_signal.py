"""模块说明：提供 run_auto_signal 命令行入口。"""

from __future__ import annotations

import argparse
import gc
from copy import deepcopy
from datetime import datetime
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
from src.auto_signal.data_stage import DataStageServices, run_data_preparation_stage as _run_data_preparation_stage_impl
from src.auto_signal.optimization_stage import (
    OptimizationStageServices,
    run_optimization_stage as _run_optimization_stage_impl,
)
from src.auto_signal.backtest_stage import (
    BacktestStageServices,
    run_backtest_stage as _run_backtest_stage_impl,
)
from src.auto_signal.signal_stage import (
    SignalStageServices,
    build_fundamental_screen_stage as _maybe_build_fundamental_screen,
    infer_signal_output_date as _infer_signal_output_date,
    resolve_signal_date_arg as _resolve_signal_date_arg,
    run_signal_stage as _run_signal_stage_impl,
    save_candidate_signal as _save_candidate_signal,
    signal_output_date as _signal_output_date,
)
from src.auto_signal.report_stage import (
    ReportStageServices,
    annual_state_router_report as _annual_state_router_report,
    write_auto_report_stage as _write_auto_report_stage_impl,
)
from src.auto_signal.quality import (
    formal_candidate_quality_report as _formal_candidate_quality_report,
    quality_number as _quality_number,
    skipped_quality as _skipped_quality,
    truthy as _truthy,
    validated_strategy_quality as _validated_strategy_quality,
)
from src.auto_signal.router import (
    RouterServices,
    annual_state_router_cfg as _annual_state_router_cfg,
    annual_state_router_combo_issues as _annual_state_router_combo_issues,
    annual_state_router_enabled as _annual_state_router_enabled,
    annual_state_router_evidence_provenance_issues as _annual_state_router_evidence_provenance_issues_impl,
    annual_state_router_evidence_yearly as _annual_state_router_evidence_yearly,
    annual_state_router_quality as _annual_state_router_quality_impl,
    annual_state_router_selected_params as _annual_state_router_selected_params,
    annual_state_router_signal_config as _annual_state_router_signal_config,
    annual_state_router_source_definitions as _annual_state_router_source_definitions_impl,
    build_annual_state_router_runtime as _build_annual_state_router_runtime_impl,
    effective_router_score_date as _effective_router_score_date,
    optional_router_text as _optional_router_text,
    router_reason_set as _router_reason_set,
    router_route_for_date as _router_route_for_date,
)
from src.auto_signal.models import (
    AnnualStateRouterRuntime,
    BacktestStageResult,
    DataPreparationStageResult,
    OptimizationStageResult,
    ReportStageResult,
    SignalStageResult,
)
from src.auto_signal.status import (
    can_reuse_conversion_outputs as _can_reuse_conversion_outputs,
    new_status as _new_status,
    stage as _stage,
    update_result_status as _update_result_status,
    update_status_message as _update_status_message,
    validation_progress_message as _validation_progress_message,
    write_json as _write_json,
    write_status as _write_status,
)
from src.adj_factor_metadata import build_adj_factor_metadata, write_adj_factor_metadata
from src.annual_router import (
    ANNUAL_ROUTER_ENGINE_CONTRACT,
    RoutedScoreRun,
    ScoreSourceDefinition,
)
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
from src.market_regime import apply_defensive_timing_to_backtest_config
from src.optimizer import OptimizationTimeoutError, run_walk_forward_grid_validation
from src.reporting import archive_run, signal_action_summary, write_daily_signal_report
from src.research_diagnostics import build_research_diagnostics, write_research_diagnostics
from src.scoring import build_strategy_scores
from src.risk_policy import RiskPolicy
from src.signal_generator import generate_signal, read_signal_previous_holdings, save_signal
from src.strategy import resample_signals
from src.trading_calendar import next_business_day, next_trade_date, resolve_target_date
from scripts.run_annual_state_router_backtest import (
    build_score_sources,
    configured_source_definitions,
)
from scripts.run_fundamental_quality_backtest import month_end_signal_dates

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def _router_services() -> RouterServices:
    return RouterServices(
        configured_source_definitions=configured_source_definitions,
        build_score_sources=build_score_sources,
        month_end_signal_dates=month_end_signal_dates,
    )


def _annual_state_router_source_definitions(config: dict[str, Any]) -> dict[str, ScoreSourceDefinition]:
    return _annual_state_router_source_definitions_impl(config, services=_router_services())


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
    return _build_annual_state_router_runtime_impl(
        config=config,
        prices=prices,
        start_date=start_date,
        end_date=end_date,
        out_dir=out_dir,
        artifacts=artifacts,
        status=status,
        services=_router_services(),
    )


def _annual_state_router_quality(config: dict[str, Any], quality_config: dict) -> ParameterQualityReport | None:
    return _annual_state_router_quality_impl(config, quality_config, services=_router_services())


def _annual_state_router_evidence_provenance_issues(config: dict[str, Any], payload: Any) -> list[str]:
    return _annual_state_router_evidence_provenance_issues_impl(config, payload, services=_router_services())



def _run_data_preparation_stage(
    args: argparse.Namespace,
    config: dict[str, Any],
    end_date: str,
    out_dir: Path,
    status: dict[str, Any],
    artifacts: list[Path],
) -> DataPreparationStageResult:
    """Compatibility wrapper that wires script-level collaborators into the data stage."""
    return _run_data_preparation_stage_impl(
        args,
        config,
        end_date,
        out_dir,
        status,
        artifacts,
        services=DataStageServices(
            update_daily_data_resumable=update_daily_data_resumable,
            convert_to_qlib_format=convert_to_qlib_format,
            load_or_compute_factors=load_or_compute_factors,
            build_data_health_report=build_data_health_report,
            write_data_health_report=write_data_health_report,
            build_adj_factor_metadata=build_adj_factor_metadata,
            write_adj_factor_metadata=write_adj_factor_metadata,
            build_data_governance_report=build_data_governance_report,
            write_data_governance_report=write_data_governance_report,
            resolve_path=resolve_path,
            can_reuse_conversion_outputs=_can_reuse_conversion_outputs,
        ),
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
    """Compatibility wrapper that preserves script-level patch points."""
    return _run_optimization_stage_impl(
        args,
        config,
        factors,
        prices,
        end_date,
        out_dir,
        status,
        artifacts,
        services=OptimizationStageServices(
            annual_state_router_enabled=_annual_state_router_enabled,
            annual_state_router_selected_params=_annual_state_router_selected_params,
            run_walk_forward_grid_validation=run_walk_forward_grid_validation,
            summarize_parameter_validation=summarize_parameter_validation,
            select_stable_params=select_stable_params,
            apply_strategy_params=apply_strategy_params,
            apply_defensive_timing_to_backtest_config=apply_defensive_timing_to_backtest_config,
            risk_policy_factory=RiskPolicy,
        ),
    )


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
    """Compatibility wrapper that preserves script-level patch points."""
    return _run_backtest_stage_impl(
        args,
        selected_config,
        factors,
        prices,
        end_date,
        out_dir,
        status,
        artifacts,
        services=BacktestStageServices(
            annual_state_router_enabled=_annual_state_router_enabled,
            build_annual_state_router_runtime=_build_annual_state_router_runtime,
            build_strategy_scores=build_strategy_scores,
            resample_signals=resample_signals,
            apply_defensive_timing_to_backtest_config=apply_defensive_timing_to_backtest_config,
            risk_policy_factory=RiskPolicy,
            run_backtest=run_backtest,
            build_yearly_breakdown=build_yearly_breakdown,
            assess_backtest_quality=assess_backtest_quality,
            build_research_diagnostics=build_research_diagnostics,
            write_research_diagnostics=write_research_diagnostics,
        ),
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
    """Compatibility wrapper that preserves script-level patch points."""
    return _run_signal_stage_impl(
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
        services=SignalStageServices(
            annual_state_router_signal_config=_annual_state_router_signal_config,
            read_signal_previous_holdings=read_signal_previous_holdings,
            generate_signal=generate_signal,
            signal_output_date=_signal_output_date,
            next_trade_date=next_trade_date,
            next_business_day=next_business_day,
            load_account_state=load_account_state,
            load_current_holdings=load_current_holdings,
            validate_account_inputs=validate_account_inputs,
            build_failure_analysis_artifacts=build_failure_analysis_artifacts,
            write_failure_analysis_artifacts=write_failure_analysis_artifacts,
            save_signal=save_signal,
            save_candidate_signal=_save_candidate_signal,
            generate_manual_orders=generate_manual_orders,
            save_manual_orders=save_manual_orders,
            generate_order_confirmation_template=generate_order_confirmation_template,
            generate_fill_feedback_template=generate_fill_feedback_template,
            save_execution_templates=save_execution_templates,
            build_fundamental_screen=_maybe_build_fundamental_screen,
        ),
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
    """Compatibility wrapper for the importable report stage."""
    return _write_auto_report_stage_impl(
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
        services=ReportStageServices(
            annual_state_router_report=_annual_state_router_report,
            signal_action_summary=signal_action_summary,
            write_daily_signal_report=write_daily_signal_report,
            archive_run=archive_run,
        ),
    )


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


if __name__ == "__main__":
    main()
