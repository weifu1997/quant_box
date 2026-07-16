"""Typed result bundles shared by auto-signal stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.annual_router import RoutedScoreRun, ScoreSourceDefinition
from src.auto_tuning import ParameterQualityReport
from src.backtest import BacktestResult


@dataclass
class OptimizationStageResult:
    selected_config: dict[str, Any]
    selected_params: dict[str, Any]
    selected_params_status: str
    validation: pd.DataFrame
    summary: pd.DataFrame


@dataclass
class DataPreparationStageResult:
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
    routed: RoutedScoreRun
    source_definitions: dict[str, ScoreSourceDefinition]
    backtest_config: dict[str, Any]
    files: dict[str, str]


@dataclass
class SignalStageResult:
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
    report_path: Path
    markdown_path: Path
    report: dict[str, Any]
