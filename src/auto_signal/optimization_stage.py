"""Automatic parameter validation and selection stage."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.auto_signal.models import OptimizationStageResult
from src.auto_signal.status import stage, validation_progress_message
from src.auto_tuning import (
    apply_strategy_params,
    select_stable_params,
    summarize_parameter_validation,
)
from src.market_regime import apply_defensive_timing_to_backtest_config
from src.optimizer import BASELINE_GRID, DEFAULT_GRID, OptimizationTimeoutError, run_walk_forward_grid_validation
from src.risk_policy import RiskPolicy

logger = logging.getLogger(__name__)


def _csv_values(value: str, cast: Callable[[str], Any]) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _csv_optional_values(value: str, cast: Callable[[str], Any]) -> list[Any]:
    values: list[Any] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower() in {"none", "null", "off"}:
            values.append(None)
        else:
            values.append(cast(item))
    return values


def _grid_values(value: str | None, defaults: list[Any], cast: Callable[[str], Any]) -> list[Any]:
    return list(defaults) if value is None else _csv_values(value, cast)


def _maybe_add_grid_values(grid: dict[str, list[Any]], key: str, value: str | None, cast: Callable[[str], Any]) -> None:
    if value is not None:
        grid[key] = _csv_optional_values(value, cast)


def _grid_has_enabled_value(grid: dict[str, list[Any]], key: str) -> bool:
    return any(value is not None for value in grid.get(key, []))


@dataclass(frozen=True)
class OptimizationStageServices:
    annual_state_router_enabled: Callable[[dict[str, Any]], bool]
    annual_state_router_selected_params: Callable[[dict[str, Any]], dict[str, Any]]
    run_walk_forward_grid_validation: Callable[..., pd.DataFrame] = run_walk_forward_grid_validation
    summarize_parameter_validation: Callable[..., pd.DataFrame] = summarize_parameter_validation
    select_stable_params: Callable[..., dict[str, Any]] = select_stable_params
    apply_strategy_params: Callable[..., dict[str, Any]] = apply_strategy_params
    apply_defensive_timing_to_backtest_config: Callable[..., dict[str, Any]] = apply_defensive_timing_to_backtest_config
    risk_policy_factory: Callable[[dict[str, Any]], RiskPolicy] = RiskPolicy


def run_optimization_stage(
    args: Namespace,
    config: dict[str, Any],
    factors: pd.DataFrame,
    prices: pd.DataFrame,
    end_date: str,
    out_dir: Path,
    status: dict[str, Any],
    artifacts: list[Path],
    *,
    services: OptimizationStageServices,
) -> OptimizationStageResult:
    """Run or skip automatic parameter validation and selection."""
    selected_config = config
    selected_params: dict[str, Any] = dict(config.get("strategy", {}))
    selected_params_status = "current_config"
    validation = pd.DataFrame()
    summary = pd.DataFrame()
    if services.annual_state_router_enabled(config):
        stage(status, out_dir, "optimize_params", "skipped", "annual_state_router_enabled")
        selected_params = services.annual_state_router_selected_params(config)
        return OptimizationStageResult(config, selected_params, "annual_state_router", validation, summary)
    if args.skip_optimize:
        stage(status, out_dir, "optimize_params", "skipped")
        return OptimizationStageResult(config, selected_params, "skipped", validation, summary)

    stage(status, out_dir, "optimize_params", "running")
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
    stage(status, out_dir, "optimize_params", "running", f"0 results; {total_combinations} combinations per validation window")

    def on_validation_result(row: dict[str, object], frame: pd.DataFrame) -> None:
        stage(status, out_dir, "optimize_params", "running", validation_progress_message(row, len(frame)))

    risk_policy = services.risk_policy_factory(config)
    base_bt_config = services.apply_defensive_timing_to_backtest_config(
        {**config["backtest"], **config["strategy"]}, prices, config
    )
    base_bt_config = risk_policy.apply_to_backtest_config(
        base_bt_config,
        force_industry_map=_grid_has_enabled_value(grid, "max_industry_weight"),
    )
    validation_path = out_dir / "auto_validation_windows.csv"
    summary_path = out_dir / "auto_parameter_summary.csv"
    try:
        validation = services.run_walk_forward_grid_validation(
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
        summary = services.summarize_parameter_validation(validation, param_columns=param_columns)
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        artifacts.extend([validation_path, summary_path])
        status["optimizer_timeout"] = {
            "message": str(exc),
            "completed_windows": exc.completed_windows,
            "completed_combinations": exc.completed_combinations,
            "validation_path": str(validation_path),
            "summary_path": str(summary_path),
        }
        stage(status, out_dir, "optimize_params", "timeout", str(exc))
        raise
    validation.to_csv(validation_path, index=False, encoding="utf-8-sig")
    summary = services.summarize_parameter_validation(validation, param_columns=param_columns)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    artifacts.extend([validation_path, summary_path])
    if not summary.empty:
        try:
            selected_params = services.select_stable_params(
                summary,
                config.get("quality", {}),
                param_columns=param_columns,
                strict=True,
            )
            selected_config = services.apply_strategy_params(config, selected_params)
            selected_params_status = "selected_acceptable_params"
        except ValueError as exc:
            if str(exc) != "no_acceptable_params":
                raise
            selected_params = {}
            selected_config = config
            selected_params_status = "no_acceptable_params"
    stage(status, out_dir, "optimize_params", "complete")
    return OptimizationStageResult(selected_config, selected_params, selected_params_status, validation, summary)
