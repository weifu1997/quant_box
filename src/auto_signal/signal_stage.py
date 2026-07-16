"""Signal, account, manual-order, and failure-analysis stage."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Any, Callable

import pandas as pd

from src.auto_signal.models import AnnualStateRouterRuntime, SignalStageResult
from src.auto_signal.status import stage
from src.failure_analysis import build_failure_analysis_artifacts, write_failure_analysis_artifacts
from src.fundamental_data import build_fundamental_screen, summarize_fundamental_screen_result, write_fundamental_screen_outputs
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
from src.signal_generator import generate_signal, read_signal_previous_holdings, save_signal
from src.trading_calendar import next_business_day, next_trade_date

logger = logging.getLogger(__name__)


def resolve_signal_date_arg(value: str, target_end_date: str) -> str:
    return target_end_date if str(value).strip().lower() in {"auto", "latest_trade_date", "latest_trading_day"} else value


def signal_output_date(signal_df: pd.DataFrame, signal_date_arg: str, factors: pd.DataFrame | None = None) -> str:
    if not signal_df.empty and "date" in signal_df.columns:
        return str(signal_df["date"].iloc[0])
    signal_date = getattr(signal_df, "attrs", {}).get("signal_date")
    if signal_date:
        return str(signal_date)
    inferred = infer_signal_output_date(signal_date_arg, factors)
    if inferred:
        return inferred
    return signal_date_arg


def infer_signal_output_date(signal_date_arg: str, factors: pd.DataFrame | None) -> str | None:
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


def save_candidate_signal(signal_df: pd.DataFrame, holdings: list[str], signal_date: str, out_dir: Path) -> tuple[Path, Path]:
    signal_path = out_dir / f"candidate_signal_{signal_date}.csv"
    holdings_path = out_dir / f"candidate_holdings_{signal_date}.csv"
    signal_df.to_csv(signal_path, index=False, encoding="utf-8-sig")
    pd.DataFrame({"instrument": holdings}).to_csv(holdings_path, index=False, encoding="utf-8-sig")
    return signal_path, holdings_path


def build_fundamental_screen_stage(config: dict[str, Any], output_date: str, out_dir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    screen_cfg = config.get("fundamental_screen", {})
    if not bool(screen_cfg.get("include_in_auto_report", False)):
        return {"enabled": False, "status": "disabled"}, {}
    try:
        result = build_fundamental_screen(config=config, as_of=output_date)
        csv_path, report_path = write_fundamental_screen_outputs(result, out_dir, top_n=int(screen_cfg.get("top_n", 30)))
        files = {"fundamental_screen_csv": str(csv_path), "fundamental_screen_report": str(report_path)}
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


@dataclass(frozen=True)
class SignalStageServices:
    annual_state_router_signal_config: Callable[..., dict[str, Any]]
    read_signal_previous_holdings: Callable[..., tuple[list[str], str]] = read_signal_previous_holdings
    generate_signal: Callable[..., tuple[pd.DataFrame, list[str]]] = generate_signal
    signal_output_date: Callable[..., str] = signal_output_date
    next_trade_date: Callable[..., Any] = next_trade_date
    next_business_day: Callable[..., Any] = next_business_day
    load_account_state: Callable[..., Any] = load_account_state
    load_current_holdings: Callable[..., Any] = load_current_holdings
    validate_account_inputs: Callable[..., list[str]] = validate_account_inputs
    build_failure_analysis_artifacts: Callable[..., tuple[dict[str, Any], dict[str, pd.DataFrame]]] = build_failure_analysis_artifacts
    write_failure_analysis_artifacts: Callable[..., dict[str, str]] = write_failure_analysis_artifacts
    save_signal: Callable[..., tuple[Path, Path]] = save_signal
    save_candidate_signal: Callable[..., tuple[Path, Path]] = save_candidate_signal
    generate_manual_orders: Callable[..., Any] = generate_manual_orders
    save_manual_orders: Callable[..., Path] = save_manual_orders
    generate_order_confirmation_template: Callable[..., Any] = generate_order_confirmation_template
    generate_fill_feedback_template: Callable[..., Any] = generate_fill_feedback_template
    save_execution_templates: Callable[..., dict[str, str]] = save_execution_templates
    build_fundamental_screen: Callable[..., tuple[dict[str, Any], dict[str, str]]] = build_fundamental_screen_stage


def run_signal_stage(
    args: Namespace,
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
    parameter_quality: Any,
    parameter_quality_gate: bool,
    backtest_quality: Any,
    backtest_quality_gate: bool,
    result: Any,
    backtest_runtime_config: dict[str, Any],
    annual_state_router: AnnualStateRouterRuntime | None,
    selected_params: dict[str, Any],
    selected_params_status: str,
    validation: pd.DataFrame,
    research_diagnostics: dict[str, Any],
    research_tables: dict[str, pd.DataFrame],
    services: SignalStageServices,
) -> SignalStageResult:
    stage(status, out_dir, "generate_signal", "running")
    previous, previous_source = services.read_signal_previous_holdings(selected_config)
    logger.info("Previous holdings source: %s (%d instruments)", previous_source, len(previous))
    signal_date_arg = resolve_signal_date_arg(args.date, end_date)
    signal_config = selected_config
    signal_scores: pd.Series | None = None
    if annual_state_router is not None:
        signal_config = services.annual_state_router_signal_config(selected_config, annual_state_router, signal_date_arg)
        signal_scores = annual_state_router.routed.scores
    signal_df, target_holdings = services.generate_signal(
        signal_date_arg,
        previous_holdings=previous,
        factor_file=factor_file,
        config=signal_config,
        factors=factors,
        scores=signal_scores,
        price_df=prices,
    )
    output_date = services.signal_output_date(signal_df, signal_date_arg, factors=factors)
    intended = services.next_trade_date(output_date, price_df=prices) or services.next_business_day(
        output_date,
        config=selected_config,
        price_df=prices,
    )
    intended_text = str(pd.Timestamp(intended).date())
    account = services.load_account_state(selected_config)
    current_holdings = services.load_current_holdings(selected_config)
    account_issues = services.validate_account_inputs(account, current_holdings, selected_config)

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
        failure_analysis, failure_tables = services.build_failure_analysis_artifacts(
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
        failure_files = services.write_failure_analysis_artifacts(failure_analysis, failure_tables, out_dir)
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
        signal_path, holdings_path = services.save_signal(signal_df, target_holdings, output_date, config=selected_config)
    else:
        signal_path, holdings_path = services.save_candidate_signal(signal_df, target_holdings, output_date, out_dir)
    artifacts.extend([signal_path, holdings_path])

    orders = services.generate_manual_orders(
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
    orders_path = services.save_manual_orders(orders, output_date, out_dir, executable=is_executable)
    artifacts.append(orders_path)
    confirmation = services.generate_order_confirmation_template(orders, output_date, intended_text, block_reasons=block_reasons)
    fill_feedback = services.generate_fill_feedback_template(orders, output_date, intended_text)
    execution_files = services.save_execution_templates(
        confirmation,
        fill_feedback,
        output_date,
        selected_config,
        executable=is_executable,
    )
    artifacts.extend(Path(path) for path in execution_files.values())
    stage(status, out_dir, "generate_signal", "complete", "executable" if is_executable else "blocked")

    fundamental_screen, fundamental_files = services.build_fundamental_screen(selected_config, str(output_date), out_dir)
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
