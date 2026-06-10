"""模块说明：提供 run_auto_signal 命令行入口。"""

from __future__ import annotations

import argparse
from copy import deepcopy
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
from src.market_regime import apply_defensive_timing_to_backtest_config
from src.optimizer import BASELINE_GRID, DEFAULT_GRID, run_walk_forward_grid_validation
from src.reporting import archive_run, signal_action_summary, write_daily_signal_report
from src.research_diagnostics import build_research_diagnostics, write_research_diagnostics
from src.scoring import build_strategy_scores
from src.selection_constraints import apply_selection_constraints_to_backtest_config
from src.signal_generator import generate_signal, read_previous_holdings, save_signal
from src.strategy import resample_signals
from src.trading_calendar import next_business_day, next_trade_date, resolve_target_date

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


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
    args = parser.parse_args()

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

        selected_config = config
        selected_params: dict[str, Any] = dict(config.get("strategy", {}))
        selected_params_status = "current_config"
        validation = pd.DataFrame()
        summary = pd.DataFrame()
        if not args.skip_optimize:
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

            base_bt_config = apply_defensive_timing_to_backtest_config({**config["backtest"], **config["strategy"]}, prices, config)
            base_bt_config = apply_selection_constraints_to_backtest_config(
                base_bt_config,
                config,
                force=_grid_has_enabled_value(grid, "max_industry_weight"),
            )
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
            )
            validation_path = out_dir / "auto_validation_windows.csv"
            validation.to_csv(validation_path, index=False, encoding="utf-8-sig")
            summary = summarize_parameter_validation(validation, param_columns=param_columns)
            summary_path = out_dir / "auto_parameter_summary.csv"
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
        else:
            _stage(status, out_dir, "optimize_params", "skipped")
            selected_params_status = "skipped"

        parameter_quality = (
            assess_parameter_quality(summary, config.get("quality", {}))
            if not args.skip_optimize
            else _validated_strategy_quality(config, config.get("quality", {}))
            or _skipped_quality(config.get("quality", {}))
        )
        quality_path = out_dir / "auto_parameter_quality.json"
        _write_json(quality_path, parameter_quality.to_dict())
        artifacts.append(quality_path)
        parameter_quality_gate = parameter_quality.is_acceptable or args.allow_low_quality

        logger.info("Selected strategy params: %s", selected_params)
        equity_path = out_dir / "auto_backtest_equity.csv"
        holdings_bt_path = out_dir / "auto_backtest_holdings.csv"
        trades_path = out_dir / "auto_backtest_trades.csv"
        metrics_path = out_dir / "auto_backtest_metrics.json"
        backtest_runtime_config: dict[str, Any] = {**selected_config["backtest"], **selected_config["strategy"]}
        if args.skip_backtest:
            _stage(status, out_dir, "backtest", "skipped")
            result = BacktestResult(
                equity_curve=pd.Series(dtype=float, name="equity"),
                holdings=pd.DataFrame(),
                trades=pd.DataFrame(),
                metrics={"backtest_skipped": True},
            )
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
            bt_config = apply_selection_constraints_to_backtest_config(bt_config, selected_config)
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
        backtest_yearly_quality = (
            build_yearly_breakdown(result, backtest_runtime_config)
            if not args.skip_backtest
            else pd.DataFrame()
        )
        backtest_quality = assess_backtest_quality(result.metrics, config.get("quality", {}), yearly=backtest_yearly_quality)
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

        _stage(status, out_dir, "generate_signal", "running")
        previous = read_previous_holdings(selected_config["outputs"]["holdings_file"])
        signal_date_arg = _resolve_signal_date_arg(args.date, end_date)
        signal_df, target_holdings = generate_signal(
            signal_date_arg,
            previous_holdings=previous,
            factor_file=factor_file,
            config=selected_config,
            factors=factors,
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
        block_reasons = []
        quality_warnings = []
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

        selected_params_path = out_dir / "auto_selected_params.json"
        _write_json(selected_params_path, selected_params)
        artifacts.append(selected_params_path)
        report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "target_date_resolution": target_resolution.to_dict(),
            "selected_params": selected_params,
            "selected_params_status": selected_params_status,
            "parameter_quality": parameter_quality.to_dict(),
            "backtest_quality": backtest_quality.to_dict(),
            "data_health": data_health.to_dict(),
            "data_governance": data_governance.to_dict(),
            "backtest_metrics": result.metrics,
            "research_diagnostics": research_diagnostics,
            "failure_analysis": failure_analysis,
            "fundamental_screen": fundamental_screen,
            "account": account.to_dict(),
            "signal_summary": signal_action_summary(signal_df),
            "signal_date": str(output_date),
            "intended_trade_date": intended_text,
            "is_executable": is_executable,
            "block_reasons": block_reasons,
            "quality_warnings": quality_warnings,
            "allow_low_quality": bool(args.allow_low_quality),
            "allow_unhealthy": bool(args.allow_unhealthy),
            "force_official": bool(args.force_official),
            "skip_optimize": bool(args.skip_optimize),
            "skip_backtest": bool(args.skip_backtest),
            "validation_windows": int(len(validation)),
            "validation_param_sets": int(len(summary)),
            "files": {
                "signal": str(signal_path),
                "holdings": str(holdings_path),
                "manual_orders": str(orders_path),
                **execution_files,
                "data_health": str(health_json),
                "data_governance": str(governance_path),
                "parameter_quality": str(quality_path),
                "backtest_metrics": str(metrics_path),
                "backtest_quality": str(backtest_quality_path),
                **research_files,
                **failure_files,
                **fundamental_files,
            },
        }
        report_path = out_dir / "auto_signal_report.json"
        _write_json(report_path, report)
        markdown_path = write_daily_signal_report(report, out_dir)
        artifacts.extend([report_path, markdown_path])

        if not args.no_archive:
            archive_dir = archive_run(artifacts, selected_config.get("reports", {}).get("history_dir", "outputs/history"), str(output_date))
            report["files"]["archive_dir"] = str(archive_dir)
            _write_json(report_path, report)

        status["status"] = "complete" if is_executable else "blocked"
        status["finished_at"] = datetime.now().isoformat(timespec="seconds")
        status["target_date_resolution"] = target_resolution.to_dict()
        status["is_executable"] = is_executable
        status["block_reasons"] = block_reasons
        _write_status(out_dir, status)
        logger.info("Auto signal saved to %s", signal_path)
        logger.info("Manual orders saved to %s", orders_path)
        logger.info("Auto report saved to %s", report_path)
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
        logger.warning("Fundamental screen report skipped: %s", exc)
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
    except Exception as exc:
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
