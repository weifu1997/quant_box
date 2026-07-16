"""Auto-signal report rendering and optional archive stage."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.auto_signal.models import AnnualStateRouterRuntime, ReportStageResult, SignalStageResult
from src.auto_signal.status import write_json
from src.reporting import archive_run, signal_action_summary, write_daily_signal_report


def annual_state_router_report(runtime: AnnualStateRouterRuntime | None) -> dict[str, Any]:
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


@dataclass(frozen=True)
class ReportStageServices:
    annual_state_router_report: Callable[..., dict[str, Any]] = annual_state_router_report
    signal_action_summary: Callable[..., dict[str, Any]] = signal_action_summary
    write_daily_signal_report: Callable[..., Path] = write_daily_signal_report
    archive_run: Callable[..., Path] = archive_run


def write_auto_report_stage(
    args: Namespace,
    target_resolution: Any,
    selected_config: dict[str, Any],
    selected_params: dict[str, Any],
    selected_params_status: str,
    parameter_quality: Any,
    backtest_quality: Any,
    data_health: Any,
    data_governance: Any,
    result: Any,
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
    services: ReportStageServices | None = None,
) -> ReportStageResult:
    """Write selected parameter, JSON, markdown, and optional archive artifacts."""
    services = services or ReportStageServices()
    selected_params_path = out_dir / "auto_selected_params.json"
    write_json(selected_params_path, selected_params)
    artifacts.append(selected_params_path)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target_date_resolution": target_resolution.to_dict(),
        "selected_params": selected_params,
        "selected_params_status": selected_params_status,
        "strategy_mode": "annual_state_router" if annual_state_router is not None else "strategy_config",
        "annual_state_router": services.annual_state_router_report(annual_state_router),
        "parameter_quality": parameter_quality.to_dict(),
        "backtest_quality": backtest_quality.to_dict(),
        "data_health": data_health.to_dict(),
        "data_governance": data_governance.to_dict(),
        "backtest_metrics": result.metrics,
        "research_diagnostics": research_diagnostics,
        "failure_analysis": signal_stage.failure_analysis,
        "fundamental_screen": signal_stage.fundamental_screen,
        "account": signal_stage.account.to_dict(),
        "signal_summary": services.signal_action_summary(signal_stage.signal_df),
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
    write_json(report_path, report)
    markdown_path = services.write_daily_signal_report(report, out_dir)
    artifacts.extend([report_path, markdown_path])

    if not args.no_archive:
        archive_dir = services.archive_run(
            artifacts,
            selected_config.get("reports", {}).get("history_dir", "outputs/history"),
            signal_stage.output_date,
        )
        report["files"]["archive_dir"] = str(archive_dir)
        write_json(report_path, report)

    return ReportStageResult(report_path=report_path, markdown_path=markdown_path, report=report)
