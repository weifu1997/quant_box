"""模块说明：生成日度信号报告并归档运行产物。"""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import pandas as pd

from src.config_loader import resolve_path


def write_daily_signal_report(report: dict[str, Any], out_dir: str | Path) -> Path:
    """函数说明：写入 write_daily_signal_report 主要逻辑。"""
    output_dir = resolve_path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "daily_signal_report.md"
    path.write_text(_render_report(report), encoding="utf-8")
    return path


def archive_run(files: list[str | Path], history_dir: str | Path, signal_date: str) -> Path:
    """函数说明：归档 archive_run 主要逻辑。"""
    target = resolve_path(history_dir) / signal_date
    target.mkdir(parents=True, exist_ok=True)
    for item in files:
        path = Path(item)
        if not path.is_absolute():
            path = resolve_path(path)
        if path.exists() and path.is_file():
            shutil.copy2(path, target / path.name)
    return target


def _render_report(report: dict[str, Any]) -> str:
    """函数说明：处理 render_report 的内部辅助逻辑。"""
    selected_params = report.get("selected_params", {})
    data_health = report.get("data_health", {})
    param_quality = report.get("parameter_quality", {})
    backtest_quality = report.get("backtest_quality", {})
    data_governance = report.get("data_governance", {})
    metrics = report.get("backtest_metrics", {})
    research = report.get("research_diagnostics", {})
    failure_analysis = report.get("failure_analysis", {})
    fundamental = report.get("fundamental_screen", {})
    account = report.get("account", {})
    signal_summary = report.get("signal_summary", {})
    block_reasons = report.get("block_reasons", [])
    quality_warnings = report.get("quality_warnings", [])

    lines = [
        "# Daily Signal Report",
        "",
        f"- Signal date: {report.get('signal_date', '')}",
        f"- Intended trade date: {report.get('intended_trade_date', '')}",
        f"- Executable: {report.get('is_executable', False)}",
        f"- Skip optimize: {report.get('skip_optimize', False)}",
        f"- Skip backtest: {report.get('skip_backtest', False)}",
        f"- Allow low quality: {report.get('allow_low_quality', False)}",
        f"- Allow unhealthy: {report.get('allow_unhealthy', False)}",
        f"- Force official: {report.get('force_official', False)}",
    ]
    if block_reasons:
        lines.append(f"- Block reasons: {', '.join(map(str, block_reasons))}")
    if quality_warnings:
        lines.append(f"- Quality warnings: {', '.join(map(str, quality_warnings))}")
    lines.extend(
        [
            "",
            "## Data Health",
            "",
            f"- Healthy: {data_health.get('is_healthy', False)}",
            f"- Target symbols: {data_health.get('target_symbols', 0)}",
            f"- Raw coverage: {_pct(data_health.get('raw_target_coverage'))}",
            f"- Price coverage: {_pct(data_health.get('price_target_coverage'))}",
            f"- Factor coverage: {_pct(data_health.get('factor_target_coverage'))}",
            f"- Raw latest coverage: {_pct(data_health.get('raw_latest_target_coverage'))}",
            f"- Price latest coverage: {_pct(data_health.get('price_latest_target_coverage'))}",
            f"- Factor latest coverage: {_pct(data_health.get('factor_latest_target_coverage'))}",
            f"- Raw latest: {data_health.get('raw_latest_date', '')}",
            f"- Price latest: {data_health.get('price_latest_date', '')}",
            f"- Factor latest: {data_health.get('factor_latest_date', '')}",
            f"- Issues: {', '.join(data_health.get('issues', [])) or 'none'}",
            "",
            "## Data Governance",
            "",
            f"- Point-in-time ready: {data_governance.get('is_point_in_time_ready', False)}",
            f"- ST filter mode: {data_governance.get('st_filter_mode', '')}",
            f"- Universe rows: {data_governance.get('universe_rows', 0)}",
            f"- Delisted rows observed: {data_governance.get('delisted_rows', 0)}",
            f"- Index constituents available: {data_governance.get('index_constituents_available', False)}",
            f"- Index snapshot month coverage: {data_governance.get('index_constituents_observed_months', 0)}/{data_governance.get('index_constituents_expected_months', 0)}",
            f"- Historical universe enabled: {data_governance.get('historical_universe_enabled', False)}",
            f"- Historical universe available: {data_governance.get('historical_universe_available', False)}",
            f"- Historical universe source coverage: {_historical_universe_source_coverage_text(data_governance)}",
            f"- Daily basic date coverage: {data_governance.get('daily_basic_covered_dates', 0)}/{data_governance.get('daily_basic_expected_dates', 0)}",
            f"- Raw adj-factor sample: {data_governance.get('raw_adj_factor_files_with_column', 0)}/{data_governance.get('raw_adj_factor_sampled_files', 0)}",
            f"- Factor cache metadata: {data_governance.get('factor_cache_meta_available', False)}",
            f"- Adj-factor version metadata: {data_governance.get('adj_factor_meta_available', False)}",
            f"- Adj-factor coverage: {data_governance.get('adj_factor_meta_files_with_adj_factor', 0)}/{data_governance.get('adj_factor_meta_raw_file_count', 0)}",
            f"- Adj-factor missing symbols: {_symbol_preview(data_governance.get('adj_factor_meta_missing_symbols', []))}",
            f"- Adj-factor end date: {data_governance.get('adj_factor_meta_end_date', '')}",
            f"- Issues: {', '.join(data_governance.get('issues', [])) or 'none'}",
            f"- Warnings: {', '.join(data_governance.get('warnings', [])) or 'none'}",
            *_repair_action_lines(data_governance.get("repair_actions", [])),
            "",
            "## Parameter Quality",
            "",
            f"- Acceptable: {param_quality.get('is_acceptable', False)}",
            f"- Windows: {param_quality.get('windows', 0)}",
            f"- Positive return rate: {_pct(param_quality.get('positive_return_rate'))}",
            f"- Annual return mean: {_pct(param_quality.get('annual_return_mean'))}",
            f"- Annual return min: {_pct(param_quality.get('annual_return_min'))}",
            f"- Annual return target: {_pct(param_quality.get('min_optimizer_annual_return'))}",
            f"- Sharpe mean: {_num(param_quality.get('sharpe_mean'))}",
            f"- Worst drawdown: {_pct(param_quality.get('max_drawdown_worst'))}",
            f"- Issues: {', '.join(param_quality.get('issues', [])) or 'none'}",
            "",
            "## Backtest Quality",
            "",
            f"- Acceptable: {backtest_quality.get('is_acceptable', False)}",
            f"- Annual return: {_pct(backtest_quality.get('annual_return'))}",
            f"- Annual return target: {_pct(backtest_quality.get('min_backtest_annual_return'))}",
            f"- Max drawdown: {_pct(backtest_quality.get('max_drawdown'))}",
            f"- Max drawdown limit: {_pct(backtest_quality.get('max_backtest_drawdown_limit'))}",
            f"- Yearly min annual return: {_pct(backtest_quality.get('yearly_min_annual_return'))}",
            f"- Years below return target: {', '.join(map(str, backtest_quality.get('years_below_return_target', []))) or 'none'}",
            f"- Yearly worst drawdown: {_pct(backtest_quality.get('yearly_worst_max_drawdown'))}",
            f"- Years breaching drawdown limit: {', '.join(map(str, backtest_quality.get('years_breaching_drawdown_limit', []))) or 'none'}",
            f"- Calmar: {_num(backtest_quality.get('calmar'))}",
            f"- Issues: {', '.join(backtest_quality.get('issues', [])) or 'none'}",
            "",
            "## Failure Analysis",
            "",
            *_failure_analysis_lines(failure_analysis),
            "",
            "## Selected Params",
            "",
            *_mapping_lines(selected_params),
            "",
            "## Backtest Metrics",
            "",
            *_mapping_lines(metrics, max_items=12),
            "",
            "## Research Diagnostics",
            "",
            f"- Enabled: {research.get('enabled', False)}",
            *_mapping_lines(research.get("benchmark", {}), max_items=8),
            *_mapping_lines(_cost_summary(research), max_items=6),
            *_mapping_lines(_turnover_summary(research), max_items=6),
            *_mapping_lines(_exposure_summary(research), max_items=6),
            f"- Issues: {', '.join(research.get('issues', [])) or 'none'}",
            "",
            "## Fundamental Screen",
            "",
            *_fundamental_screen_lines(fundamental),
            "",
            "## Account",
            "",
            f"- Total asset: {_num(account.get('total_asset'))}",
            f"- Cash: {_num(account.get('cash'))}",
            f"- Holdings loaded: {account.get('holdings_loaded', False)}",
            "",
            "## Signal Summary",
            "",
            f"- BUY: {signal_summary.get('BUY', 0)}",
            f"- HOLD: {signal_summary.get('HOLD', 0)}",
            f"- SELL: {signal_summary.get('SELL', 0)}",
            "",
            "## Execution Loop",
            "",
            f"- Manual orders: {report.get('files', {}).get('manual_orders', '')}",
            f"- Order confirmation: {report.get('files', {}).get('order_confirmation', '')}",
            f"- Fill feedback: {report.get('files', {}).get('fill_feedback', '')}",
            "",
            "## Output Files",
            "",
            *_mapping_lines(report.get("files", {})),
            "",
        ]
    )
    return "\n".join(lines)


def signal_action_summary(signal_df: pd.DataFrame) -> dict[str, int]:
    """函数说明：处理 signal_action_summary 主要逻辑。"""
    if signal_df.empty or "action" not in signal_df.columns:
        return {"BUY": 0, "HOLD": 0, "SELL": 0}
    counts = signal_df["action"].astype(str).str.upper().value_counts()
    return {key: int(counts.get(key, 0)) for key in ["BUY", "HOLD", "SELL"]}


def _mapping_lines(values: dict[str, Any], max_items: int | None = None) -> list[str]:
    """函数说明：处理 mapping_lines 的内部辅助逻辑。"""
    items = list(values.items())
    if max_items is not None:
        items = items[:max_items]
    return [f"- {key}: {value}" for key, value in items]


def _repair_action_lines(actions: Any, max_items: int = 5) -> list[str]:
    """函数说明：处理 repair_action_lines 的内部辅助逻辑。"""
    if not isinstance(actions, list) or not actions:
        return ["- Repair actions: none"]
    lines = []
    for action in actions[:max_items]:
        if not isinstance(action, dict):
            continue
        component = action.get("component", "")
        reason = action.get("reason", "")
        commands = action.get("commands", [])
        command_text = " | ".join(str(command) for command in commands) if isinstance(commands, list) else str(commands)
        lines.append(f"- Repair action: {component} ({reason}) -> {command_text}")
    return lines or ["- Repair actions: none"]


def _historical_universe_source_coverage_text(data_governance: dict[str, Any]) -> str:
    coverage = data_governance.get("historical_universe_source_coverage", {})
    if not isinstance(coverage, dict) or not coverage:
        return "none"
    labels = []
    for source, summary in coverage.items():
        if not isinstance(summary, dict):
            continue
        observed = summary.get("observed_months", 0)
        expected = summary.get("expected_months", 0)
        labels.append(f"{source} {observed}/{expected}")
    return ", ".join(labels) if labels else "none"


def _symbol_preview(symbols: Any, max_items: int = 8) -> str:
    """函数说明：处理 symbol_preview 的内部辅助逻辑。"""
    if not isinstance(symbols, list) or not symbols:
        return "none"
    values = [str(symbol) for symbol in symbols]
    suffix = "" if len(values) <= max_items else f" (+{len(values) - max_items} more)"
    return ", ".join(values[:max_items]) + suffix


def _fundamental_screen_lines(fundamental: Any) -> list[str]:
    if not isinstance(fundamental, dict) or not fundamental:
        return ["- Enabled: False", "- Status: missing"]
    enabled = bool(fundamental.get("enabled", False))
    status = str(fundamental.get("status", ""))
    lines = [
        f"- Enabled: {enabled}",
        f"- Status: {status}",
    ]
    if not enabled or status == "failed":
        if fundamental.get("error"):
            lines.append(f"- Error: {fundamental.get('error')}")
        return lines
    lines.extend(
        [
            f"- As of date: {fundamental.get('as_of_date', '')}",
            f"- Covered rows: {fundamental.get('covered_rows', 0)}/{fundamental.get('rows', 0)}",
            f"- Fundamental coverage: {fundamental.get('fundamental_coverage', '')}",
            f"- Dividend coverage: {fundamental.get('dividend_coverage', '')}",
            f"- Passed: {fundamental.get('passed', 0)}",
            f"- Watch: {fundamental.get('watch', 0)}",
            f"- Top pass: {_fundamental_record_preview(fundamental.get('top_pass', []))}",
            f"- Top watch: {_fundamental_record_preview(fundamental.get('top_watch', []))}",
        ]
    )
    files = fundamental.get("files", {})
    if isinstance(files, dict) and files:
        lines.append(f"- Report: {files.get('report', '')}")
    return lines


def _fundamental_record_preview(records: Any, max_items: int = 5) -> str:
    if not isinstance(records, list) or not records:
        return "none"
    labels = []
    for record in records[:max_items]:
        if not isinstance(record, dict):
            continue
        symbol = str(record.get("ts_code", "") or "")
        name = str(record.get("name", "") or "")
        industry = str(record.get("industry", "") or "")
        score = record.get("total_score", "")
        reason = str(record.get("failed_reasons", "") or "")
        label = " ".join(part for part in [symbol, name, industry, f"score={score}", reason] if part)
        labels.append(label)
    suffix = "" if len(records) <= max_items else f" (+{len(records) - max_items} more)"
    return "; ".join(labels) + suffix if labels else "none"


def _cost_summary(research: dict[str, Any]) -> dict[str, Any]:
    """函数说明：处理 cost_summary 的内部辅助逻辑。"""
    costs = research.get("cost_attribution", {})
    if not isinstance(costs, dict):
        return {}
    keys = ["trade_count", "total_trade_cost", "cost_drag_on_initial_equity", "capacity_warning_count"]
    return {f"cost_{key}": costs.get(key) for key in keys if key in costs}


def _exposure_summary(research: dict[str, Any]) -> dict[str, Any]:
    """函数说明：处理 exposure_summary 的内部辅助逻辑。"""
    exposure = research.get("exposure", {})
    if not isinstance(exposure, dict):
        return {}
    keys = ["latest_date", "latest_position_count", "latest_top_position_weight", "latest_max_industry_weight"]
    return {f"exposure_{key}": exposure.get(key) for key in keys if key in exposure}


def _turnover_summary(research: dict[str, Any]) -> dict[str, Any]:
    """函数说明：处理 turnover_summary 的内部辅助逻辑。"""
    turnover = research.get("turnover_attribution", {})
    if not isinstance(turnover, dict):
        return {}
    keys = [
        "annual_turnover_estimate",
        "annual_turnover_without_rebalance_trims_estimate",
        "normal_rebalance_sell_count",
        "rebalance_trim_sell_count",
        "rebalance_exit_sell_count",
        "rebalance_trim_share_of_normal_sells",
        "rebalance_trim_notional",
        "rebalance_trim_trade_cost",
        "rebalance_trim_trade_cost_drag_on_initial_equity",
        "rebalance_trim_cost_share_of_total_trade_cost",
        "risk_exit_sell_count",
        "risk_exit_share_of_executable_sells",
        "risk_exit_trade_cost",
        "blocked_sell_count",
    ]
    return {f"turnover_{key}": turnover.get(key) for key in keys if key in turnover}


def _failure_analysis_lines(analysis: Any) -> list[str]:
    """Render the compact failure diagnosis section."""
    if not isinstance(analysis, dict) or not analysis.get("enabled", False):
        issues = analysis.get("issues", []) if isinstance(analysis, dict) else []
        return [f"- Enabled: False", f"- Issues: {', '.join(map(str, issues)) or 'none'}"]

    gaps = analysis.get("backtest_threshold_gaps", {})
    drawdown = analysis.get("drawdown_summary", {})
    scope = analysis.get("failure_scope_summary", {})
    return [
        f"- Primary failure area: {analysis.get('primary_failure_area', '')}",
        f"- Parameter/backtest mismatch: {analysis.get('parameter_backtest_mismatch', False)}; failure scope: {scope.get('primary_scope', '')}",
        f"- Backtest gaps: annual return {_pct(gaps.get('annual_return_gap'))}; max drawdown {_pct(gaps.get('max_drawdown_gap'))}",
        (
            "- Worst drawdown: "
            f"{drawdown.get('start_date') or drawdown.get('peak_date') or ''}"
            f" -> {drawdown.get('trough_date') or ''}"
            f" (recovery: {drawdown.get('recovery_date') or 'none'})"
        ),
        f"- Drawdown strategy / benchmark / active return: {_pct(drawdown.get('strategy_return_peak_to_trough'))} / {_pct(drawdown.get('benchmark_return_peak_to_trough'))} / {_pct(drawdown.get('active_return_peak_to_trough'))}",
        f"- Drawdown trades/cost/risk exits/blocked: {drawdown.get('trades_peak_to_trough', '')}/{_num(drawdown.get('trade_cost_peak_to_trough'))}/{drawdown.get('risk_exit_trades_peak_to_trough', '')}/{drawdown.get('blocked_trades_peak_to_trough', '')}",
        f"- Top drawdown contributors: {_drawdown_contributor_preview(drawdown)}",
        f"- Next check: {_next_failure_check(analysis)}",
    ]


def _drawdown_contributor_preview(drawdown: dict[str, Any]) -> str:
    """Preview the largest negative drawdown contributors."""
    records = drawdown.get("top_negative_instruments", [])
    if not isinstance(records, list) or not records:
        return "none"
    labels = []
    for record in records[:3]:
        if not isinstance(record, dict):
            continue
        instrument = record.get("instrument", "")
        contribution = _pct(record.get("gross_contribution"))
        labels.append(f"{instrument} {contribution}".strip())
    return ", ".join(labels) if labels else "none"


def _next_failure_check(analysis: dict[str, Any]) -> str:
    """Choose a short next-step hint for a failed automatic run."""
    scope = analysis.get("failure_scope_summary", {})
    primary_scope = scope.get("primary_scope") if isinstance(scope, dict) else None
    if primary_scope == "pre_validation_history":
        return "include pre-validation years in optimization gates"
    if primary_scope == "cross_window_full_history":
        return "review cross-window drawdown and full-history path risk"
    if primary_scope == "validation_window":
        return "tighten selected-parameter validation gates"
    if analysis.get("parameter_backtest_mismatch"):
        return "compare validation windows with pre-validation and full-history segments"
    primary = str(analysis.get("primary_failure_area", ""))
    if primary == "params":
        return "review parameter grid and validation quality thresholds"
    if primary == "backtest":
        return "review drawdown control and yearly return breakdown"
    if primary in {"data", "governance"}:
        return "repair data readiness before trusting performance diagnostics"
    if primary == "account":
        return "load current holdings before promoting candidate orders"
    return "review failure analysis artifacts"


def _pct(value: Any) -> str:
    """函数说明：处理 pct 的内部辅助逻辑。"""
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return ""


def _num(value: Any) -> str:
    """函数说明：处理 num 的内部辅助逻辑。"""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return ""
