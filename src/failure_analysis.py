"""模块说明：生成自动信号失败原因分析的 JSON 和表格产物。"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import numpy as np
import pandas as pd

from src.backtest import BacktestResult, calculate_metrics
from src.config_loader import resolve_path


PARAM_COLUMNS = [
    "factor_group",
    "top_n",
    "max_turnover",
    "rank_buffer",
    "rebalance_freq",
    "max_weight_per_stock",
    "stop_loss_pct",
    "take_profit_pct",
    "circuit_breaker_drawdown",
    "circuit_breaker_cooldown_days",
    "circuit_breaker_target_exposure",
    "target_vol",
    "max_industry_weight",
    "rebalance_drift_threshold",
]

METRIC_COLUMNS = [
    "annual_return",
    "max_drawdown",
    "sharpe",
    "calmar",
    "annual_turnover",
    "annual_trade_cost_ratio",
    "win_rate",
    "trade_cost",
]

SEGMENT_COLUMNS = [
    "row_type",
    "period_label",
    "start_date",
    "end_date",
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "failure_scope",
    "annual_return_failed",
    "max_drawdown_failed",
    "failure_driver",
    *PARAM_COLUMNS,
    *METRIC_COLUMNS,
    "optimization_score",
]

YEARLY_COLUMNS = [
    "year",
    "start_date",
    "end_date",
    "trading_days",
    *METRIC_COLUMNS,
]


def build_failure_analysis_artifacts(
    *,
    selected_params: dict[str, Any],
    selected_params_status: str,
    parameter_quality: dict[str, Any],
    backtest_quality: dict[str, Any],
    backtest_metrics: dict[str, Any],
    block_reasons: list[str],
    quality_warnings: list[str],
    validation: pd.DataFrame,
    backtest_result: BacktestResult,
    backtest_config: dict[str, Any],
    research_diagnostics: dict[str, Any],
    research_tables: dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """函数说明：构建解释自动运行失败原因的分析数据和表格。"""
    drawdown_summary = build_drawdown_failure_summary(research_diagnostics, research_tables)
    threshold_gaps = _backtest_threshold_gaps(backtest_quality)
    parameter_backtest_mismatch = bool(parameter_quality.get("is_acceptable")) and not bool(backtest_quality.get("is_acceptable"))
    primary_failure_area = _primary_failure_area(block_reasons, quality_warnings)
    validation_vs_backtest = build_validation_vs_backtest_table(
        validation,
        selected_params,
        backtest_result,
        backtest_config,
        backtest_quality,
        start_date,
        end_date,
    )
    yearly_breakdown = build_yearly_breakdown(backtest_result, backtest_config)
    failure_scope_summary = build_failure_scope_summary(validation_vs_backtest)

    analysis = {
        "enabled": True,
        "is_failure": bool(block_reasons or quality_warnings),
        "primary_failure_area": primary_failure_area,
        "parameter_backtest_mismatch": parameter_backtest_mismatch,
        "selected_params_status": selected_params_status,
        "selected_params": _json_safe(selected_params),
        "parameter_quality": _json_safe(parameter_quality),
        "backtest_quality": _json_safe(backtest_quality),
        "backtest_metrics": _json_safe(backtest_metrics),
        "backtest_threshold_gaps": threshold_gaps,
        "block_reasons": list(block_reasons),
        "quality_warnings": list(quality_warnings),
        "drawdown_summary": drawdown_summary,
        "failure_scope_summary": failure_scope_summary,
        "explanation": _failure_explanation(
            parameter_backtest_mismatch=parameter_backtest_mismatch,
            primary_failure_area=primary_failure_area,
            backtest_quality=backtest_quality,
            threshold_gaps=threshold_gaps,
            drawdown_summary=drawdown_summary,
            failure_scope_summary=failure_scope_summary,
        ),
    }

    tables = {
        "validation_vs_backtest": validation_vs_backtest,
        "yearly_breakdown": yearly_breakdown,
    }
    return analysis, tables


def write_failure_analysis_artifacts(
    analysis: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    out_dir: str | Path,
) -> dict[str, str]:
    """函数说明：写入失败分析 JSON 和 CSV 产物。"""
    output_dir = resolve_path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    analysis_path = output_dir / "auto_failure_analysis.json"
    analysis_path.write_text(json.dumps(_json_safe(analysis), indent=2, ensure_ascii=False), encoding="utf-8")
    paths["failure_analysis"] = str(analysis_path)

    drawdown_path = output_dir / "auto_drawdown_summary.json"
    drawdown_path.write_text(
        json.dumps(_json_safe(analysis.get("drawdown_summary", {})), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["drawdown_summary"] = str(drawdown_path)

    validation_path = output_dir / "auto_validation_vs_backtest.csv"
    _table_or_empty(tables.get("validation_vs_backtest"), SEGMENT_COLUMNS).to_csv(
        validation_path,
        index=False,
        encoding="utf-8-sig",
    )
    paths["validation_vs_backtest"] = str(validation_path)

    yearly_path = output_dir / "auto_backtest_yearly_breakdown.csv"
    _table_or_empty(tables.get("yearly_breakdown"), YEARLY_COLUMNS).to_csv(
        yearly_path,
        index=False,
        encoding="utf-8-sig",
    )
    paths["backtest_yearly_breakdown"] = str(yearly_path)
    return paths


def build_validation_vs_backtest_table(
    validation: pd.DataFrame,
    selected_params: dict[str, Any],
    backtest_result: BacktestResult,
    backtest_config: dict[str, Any],
    backtest_quality: dict[str, Any],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """函数说明：对比入选验证窗口与全历史回测分段。"""
    rows: list[dict[str, Any]] = []
    selected_validation = _selected_validation_rows(validation, selected_params or backtest_config)
    first_validation_start = _first_test_start(selected_validation if not selected_validation.empty else validation)

    for _, row in selected_validation.iterrows():
        rows.append(_validation_comparison_row(row, selected_params or backtest_config, backtest_quality))

    rows.extend(
        _backtest_segment_rows(
            backtest_result,
            backtest_config,
            selected_params or backtest_config,
            backtest_quality,
            start_date,
            end_date,
            first_validation_start,
        )
    )
    return pd.DataFrame(rows, columns=SEGMENT_COLUMNS)


def build_yearly_breakdown(backtest_result: BacktestResult, backtest_config: dict[str, Any]) -> pd.DataFrame:
    """函数说明：按自然年拆分回测表现。"""
    equity = _equity_series(backtest_result.equity_curve)
    if equity.empty:
        return pd.DataFrame(columns=YEARLY_COLUMNS)

    rows: list[dict[str, Any]] = []
    for year, group in equity.groupby(equity.index.year):
        segment = group.sort_index()
        metrics = _metrics_for_segment(segment, backtest_result.trades, backtest_config, segment.index.min(), segment.index.max())
        rows.append(
            {
                "year": int(year),
                "start_date": _date_text(segment.index.min()),
                "end_date": _date_text(segment.index.max()),
                "trading_days": int(len(segment)),
                **{key: metrics.get(key) for key in METRIC_COLUMNS},
            }
        )
    return pd.DataFrame(rows, columns=YEARLY_COLUMNS)


def build_drawdown_failure_summary(
    research_diagnostics: dict[str, Any],
    research_tables: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """函数说明：从研究诊断结果中提取最大回撤摘要。"""
    drawdown = research_diagnostics.get("drawdown", {})
    holding_attribution = research_diagnostics.get("holding_attribution", {})
    regime_trades = research_diagnostics.get("regime_trades", {})
    if not isinstance(drawdown, dict):
        drawdown = {}
    if not isinstance(holding_attribution, dict):
        holding_attribution = {}
    if not isinstance(regime_trades, dict):
        regime_trades = {}

    top_drawdowns = drawdown.get("top_drawdowns", [])
    worst_drawdown = top_drawdowns[0] if isinstance(top_drawdowns, list) and top_drawdowns and isinstance(top_drawdowns[0], dict) else {}
    peak_equity = _number_or_none(worst_drawdown.get("peak_equity"))
    trough_equity = _number_or_none(worst_drawdown.get("trough_equity"))
    recovery_equity = _number_or_none(worst_drawdown.get("recovery_equity"))
    strategy_return = _period_return(peak_equity, trough_equity)
    benchmark_return = _number_or_none(drawdown.get("max_drawdown_benchmark_return_peak_to_trough"))

    return {
        "enabled": bool(drawdown.get("enabled", False)),
        "max_drawdown": drawdown.get("max_drawdown"),
        "peak_date": drawdown.get("max_drawdown_peak_date"),
        "start_date": drawdown.get("max_drawdown_start_date"),
        "trough_date": drawdown.get("max_drawdown_trough_date"),
        "recovery_date": drawdown.get("max_drawdown_recovery_date"),
        "peak_equity": peak_equity,
        "trough_equity": trough_equity,
        "recovery_equity": recovery_equity,
        "strategy_return_peak_to_trough": strategy_return,
        "benchmark_return_peak_to_trough": benchmark_return,
        "active_return_peak_to_trough": _gap(strategy_return, benchmark_return),
        "days_to_trough": drawdown.get("max_drawdown_days_to_trough"),
        "days_to_recovery": drawdown.get("max_drawdown_days_to_recovery"),
        "trades_peak_to_trough": drawdown.get("trades_peak_to_trough"),
        "buy_trades_peak_to_trough": worst_drawdown.get("buy_trades_peak_to_trough"),
        "sell_trades_peak_to_trough": worst_drawdown.get("sell_trades_peak_to_trough"),
        "risk_exit_trades_peak_to_trough": drawdown.get("risk_exit_trades_peak_to_trough"),
        "blocked_trades_peak_to_trough": drawdown.get("blocked_trades_peak_to_trough"),
        "trade_cost_peak_to_trough": _sum_column(research_tables.get("max_drawdown_trades"), "trade_cost"),
        "top_negative_instruments": _top_negative_records(
            research_tables.get("max_drawdown_instrument_attribution"),
            "gross_contribution",
            10,
            fallback=holding_attribution.get("max_drawdown_top_negative_instruments", []),
        ),
        "top_negative_industries": _top_negative_records(
            research_tables.get("max_drawdown_industry_attribution"),
            "gross_contribution",
            10,
            fallback=holding_attribution.get("max_drawdown_top_negative_industries", []),
        ),
        "trade_costs_by_status_reason": _top_records(
            research_tables.get("max_drawdown_trade_costs_by_status_reason"),
            "trade_cost",
            10,
            ascending=False,
        ),
        "regime_trade_summary": _json_safe(regime_trades),
    }


def build_failure_scope_summary(validation_vs_backtest: pd.DataFrame) -> dict[str, Any]:
    """函数说明：归纳失败主要来自验证窗口、窗口前历史还是跨窗口全历史。"""
    if validation_vs_backtest.empty:
        return {
            "primary_scope": "unknown",
            "description": "No validation/backtest comparison rows were generated.",
            "failed_scopes": [],
            "scope_details": [],
        }
    frame = validation_vs_backtest.copy()
    annual_failed = _truthy_series(frame.get("annual_return_failed"), frame.index)
    drawdown_failed = _truthy_series(frame.get("max_drawdown_failed"), frame.index)
    frame["_failed"] = annual_failed | drawdown_failed

    scope_order = ["validation_window", "pre_validation_history", "validation_forward_history", "full_history"]
    scope_details = [_failure_scope_detail(frame, scope) for scope in scope_order if scope in set(frame["failure_scope"].astype(str))]
    failed_scopes = [detail["scope"] for detail in scope_details if detail["failed_rows"] > 0]

    if "validation_window" in failed_scopes:
        primary_scope = "validation_window"
    elif "pre_validation_history" in failed_scopes:
        primary_scope = "pre_validation_history"
    elif "validation_forward_history" in failed_scopes:
        primary_scope = "validation_forward_history"
    elif failed_scopes == ["full_history"]:
        primary_scope = "cross_window_full_history"
    elif "full_history" in failed_scopes:
        primary_scope = "full_history"
    else:
        primary_scope = "none"

    return {
        "primary_scope": primary_scope,
        "description": _failure_scope_description(primary_scope),
        "failed_scopes": failed_scopes,
        "scope_details": scope_details,
    }


def _selected_validation_rows(validation: pd.DataFrame, selected_params: dict[str, Any]) -> pd.DataFrame:
    """函数说明：筛选与入选参数匹配的验证窗口行。"""
    if validation.empty:
        return pd.DataFrame(columns=validation.columns)
    frame = validation.copy()
    mask = pd.Series(True, index=frame.index)
    comparable = False
    for column in PARAM_COLUMNS:
        if column not in frame.columns or column not in selected_params:
            continue
        comparable = True
        expected = selected_params.get(column)
        values = frame[column]
        if expected is None:
            mask &= values.isna()
        else:
            mask &= values.map(_comparable_value).eq(_comparable_value(expected))
    if not comparable:
        return frame
    return frame[mask].copy()


def _validation_comparison_row(row: pd.Series, selected_params: dict[str, Any], backtest_quality: dict[str, Any]) -> dict[str, Any]:
    """函数说明：构建单个验证窗口的对比行。"""
    test_start = row.get("test_start")
    test_end = row.get("test_end")
    metrics = {key: _number_or_none(row.get(key)) for key in METRIC_COLUMNS}
    failure_flags = _failure_flags(metrics, backtest_quality)
    result = {
        "row_type": "validation_window",
        "period_label": f"{_date_text(test_start)}..{_date_text(test_end)}",
        "start_date": _date_text(test_start),
        "end_date": _date_text(test_end),
        "train_start": _date_text(row.get("train_start")),
        "train_end": _date_text(row.get("train_end")),
        "test_start": _date_text(test_start),
        "test_end": _date_text(test_end),
        "failure_scope": "validation_window",
        **failure_flags,
        "optimization_score": _number_or_none(row.get("optimization_score")),
    }
    result.update(_param_values(row, selected_params))
    result.update(metrics)
    return result


def _backtest_segment_rows(
    backtest_result: BacktestResult,
    backtest_config: dict[str, Any],
    selected_params: dict[str, Any],
    backtest_quality: dict[str, Any],
    start_date: str,
    end_date: str,
    first_validation_start: pd.Timestamp | None,
) -> list[dict[str, Any]]:
    """函数说明：构建全历史和验证后区间的回测分段行。"""
    equity = _equity_series(backtest_result.equity_curve)
    if equity.empty:
        return []
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    segments: list[tuple[str, str, pd.Timestamp, pd.Timestamp]] = [("full_history", "full_history", start, end)]
    if first_validation_start is not None:
        pre_end = first_validation_start - pd.Timedelta(days=1)
        if pre_end >= start:
            segments.append(("pre_validation_history", "pre_validation_history", start, pre_end))
        if first_validation_start <= end:
            segments.append(("validation_forward_history", "validation_forward_history", first_validation_start, end))

    rows: list[dict[str, Any]] = []
    for label, scope, seg_start, seg_end in segments:
        segment = equity[(equity.index >= seg_start) & (equity.index <= seg_end)]
        if segment.empty:
            continue
        metrics = (
            dict(backtest_result.metrics)
            if label == "full_history"
            else _metrics_for_segment(segment, backtest_result.trades, backtest_config, seg_start, seg_end)
        )
        row = {
            "row_type": "backtest_segment",
            "period_label": label,
            "start_date": _date_text(seg_start),
            "end_date": _date_text(seg_end),
            "failure_scope": scope,
            **_failure_flags(metrics, backtest_quality),
            "optimization_score": None,
        }
        row.update(_param_values(pd.Series(dtype=object), selected_params))
        row.update({key: _number_or_none(metrics.get(key)) for key in METRIC_COLUMNS})
        rows.append(row)
    return rows


def _metrics_for_segment(
    equity: pd.Series,
    trades: pd.DataFrame,
    backtest_config: dict[str, Any],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict[str, Any]:
    """函数说明：计算指定净值区间的回测指标。"""
    segment = _equity_series(equity)
    if len(segment) < 2:
        return {key: None for key in METRIC_COLUMNS}
    segment_config = dict(backtest_config)
    segment_config["initial_capital"] = float(segment.iloc[0])
    return calculate_metrics(segment, _trades_between(trades, start_date, end_date), segment_config)


def _trades_between(trades: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """函数说明：截取指定日期区间内的成交记录。"""
    if trades.empty or "date" not in trades.columns:
        return pd.DataFrame()
    frame = trades.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    return frame[(frame["date"] >= pd.Timestamp(start_date).normalize()) & (frame["date"] <= pd.Timestamp(end_date).normalize())].copy()


def _equity_series(equity_curve: pd.Series | pd.DataFrame) -> pd.Series:
    """函数说明：规范化回测净值为按日期索引的序列。"""
    if isinstance(equity_curve, pd.DataFrame):
        if equity_curve.empty:
            return pd.Series(dtype=float, name="equity")
        series = equity_curve["equity"] if "equity" in equity_curve.columns else equity_curve.iloc[:, 0]
    else:
        series = equity_curve
    result = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    result.index = pd.to_datetime(result.index).normalize()
    return result[~result.index.duplicated(keep="last")].sort_index().rename("equity")


def _first_test_start(validation: pd.DataFrame) -> pd.Timestamp | None:
    """函数说明：读取验证结果中最早的测试开始日期。"""
    if validation.empty or "test_start" not in validation.columns:
        return None
    dates = pd.to_datetime(validation["test_start"], errors="coerce").dropna()
    if dates.empty:
        return None
    return pd.Timestamp(dates.min()).normalize()


def _param_values(row: pd.Series, selected_params: dict[str, Any]) -> dict[str, Any]:
    """函数说明：合并验证行和入选参数中的参数取值。"""
    values: dict[str, Any] = {}
    for column in PARAM_COLUMNS:
        if column in row.index and not pd.isna(row.get(column)):
            values[column] = row.get(column)
        else:
            values[column] = selected_params.get(column)
    return values


def _backtest_threshold_gaps(backtest_quality: dict[str, Any]) -> dict[str, Any]:
    """函数说明：计算回测指标与质量门槛之间的差距。"""
    annual_return = _number_or_none(backtest_quality.get("annual_return"))
    min_return = _number_or_none(backtest_quality.get("min_backtest_annual_return"))
    max_drawdown = _number_or_none(backtest_quality.get("max_drawdown"))
    drawdown_limit = _number_or_none(backtest_quality.get("max_backtest_drawdown_limit"))
    return {
        "annual_return": annual_return,
        "min_backtest_annual_return": min_return,
        "annual_return_gap": _gap(annual_return, min_return),
        "max_drawdown": max_drawdown,
        "max_backtest_drawdown_limit": drawdown_limit,
        "max_drawdown_gap": _gap(max_drawdown, drawdown_limit),
    }


def _failure_flags(metrics: dict[str, Any], backtest_quality: dict[str, Any]) -> dict[str, Any]:
    """函数说明：标记收益率和回撤是否触发失败条件。"""
    annual_return = _number_or_none(metrics.get("annual_return"))
    min_return = _number_or_none(backtest_quality.get("min_backtest_annual_return"))
    max_drawdown = _number_or_none(metrics.get("max_drawdown"))
    drawdown_limit = _number_or_none(backtest_quality.get("max_backtest_drawdown_limit"))
    annual_failed = annual_return is not None and min_return is not None and annual_return < min_return
    drawdown_failed = max_drawdown is not None and drawdown_limit is not None and max_drawdown < drawdown_limit
    drivers = []
    if annual_failed:
        drivers.append("annual_return")
    if drawdown_failed:
        drivers.append("max_drawdown")
    return {
        "annual_return_failed": annual_failed,
        "max_drawdown_failed": drawdown_failed,
        "failure_driver": ",".join(drivers) if drivers else "none",
    }


def _failure_scope_detail(frame: pd.DataFrame, scope: str) -> dict[str, Any]:
    """函数说明：汇总单个失败范围的行数、失败驱动和最差指标。"""
    rows = frame[frame["failure_scope"].astype(str) == scope]
    failed_rows = rows[rows["_failed"]]
    drivers: list[str] = []
    if "failure_driver" in failed_rows.columns:
        for value in failed_rows["failure_driver"].dropna().astype(str):
            drivers.extend([part for part in value.split(",") if part and part != "none"])
    return {
        "scope": scope,
        "rows": int(len(rows)),
        "failed_rows": int(len(failed_rows)),
        "failure_drivers": sorted(set(drivers)),
        "worst_annual_return": _series_min(rows.get("annual_return")),
        "worst_max_drawdown": _series_min(rows.get("max_drawdown")),
    }


def _failure_scope_description(scope: str) -> str:
    """函数说明：给失败范围生成简短说明。"""
    descriptions = {
        "validation_window": "At least one selected validation window failed a quality gate.",
        "pre_validation_history": "The failure is mainly before the first validation window.",
        "validation_forward_history": "The failure is in the validation-forward full-history segment.",
        "cross_window_full_history": "Standalone windows pass, but the compounded full-history path fails.",
        "full_history": "The full-history backtest fails a quality gate.",
        "none": "No validation/backtest comparison scope failed.",
        "unknown": "No validation/backtest comparison rows were generated.",
    }
    return descriptions.get(scope, descriptions["unknown"])


def _truthy_series(series: Any, index: pd.Index) -> pd.Series:
    """函数说明：把 bool/字符串标志统一转换为布尔序列。"""
    if series is None:
        return pd.Series(False, index=index)
    values = pd.Series(series, index=index)
    return values.map(_truthy_value).fillna(False).astype(bool)


def _truthy_value(value: Any) -> bool:
    """函数说明：判断表格标志值是否表示真。"""
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _series_min(series: Any) -> float | None:
    """函数说明：读取数值列中的最小值。"""
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.min())


def _failure_explanation(
    *,
    parameter_backtest_mismatch: bool,
    primary_failure_area: str,
    backtest_quality: dict[str, Any],
    threshold_gaps: dict[str, Any],
    drawdown_summary: dict[str, Any],
    failure_scope_summary: dict[str, Any],
) -> list[str]:
    """函数说明：生成失败原因的简短文字解释。"""
    lines: list[str] = []
    if parameter_backtest_mismatch:
        lines.append("Parameter validation passed, but the full-history backtest failed quality gates.")
    else:
        lines.append(f"Primary failure area: {primary_failure_area}.")
    if failure_scope_summary.get("primary_scope"):
        lines.append(
            "Failure scope: "
            f"{failure_scope_summary.get('primary_scope')} "
            f"({failure_scope_summary.get('description')})."
        )
    if threshold_gaps.get("annual_return_gap") is not None:
        lines.append(
            "Backtest annual return gap: "
            f"{threshold_gaps['annual_return_gap']:.6f} "
            f"({backtest_quality.get('annual_return')} vs {backtest_quality.get('min_backtest_annual_return')})."
        )
    if threshold_gaps.get("max_drawdown_gap") is not None:
        lines.append(
            "Backtest max drawdown gap: "
            f"{threshold_gaps['max_drawdown_gap']:.6f} "
            f"({backtest_quality.get('max_drawdown')} vs {backtest_quality.get('max_backtest_drawdown_limit')})."
        )
    if drawdown_summary.get("trough_date"):
        lines.append(
            "Worst drawdown ran from "
            f"{drawdown_summary.get('start_date') or drawdown_summary.get('peak_date')} "
            f"to {drawdown_summary.get('trough_date')}."
        )
    return lines


def _primary_failure_area(block_reasons: list[str], quality_warnings: list[str]) -> str:
    """函数说明：从阻断原因和警告中识别主要失败区域。"""
    reasons = [*block_reasons, *quality_warnings]
    for prefix in ["backtest:", "params:", "data:", "governance:", "account:"]:
        if any(str(reason).startswith(prefix) for reason in reasons):
            return prefix.rstrip(":")
    return "none"


def _top_negative_records(
    table: pd.DataFrame | None,
    column: str,
    limit: int,
    fallback: Any = None,
) -> list[dict[str, Any]]:
    """函数说明：读取负贡献最大的记录并在缺失时使用兜底数据。"""
    records = _top_records(table, column, limit, ascending=True)
    if records:
        return records
    if isinstance(fallback, list):
        return [_json_safe(record) for record in fallback[:limit] if isinstance(record, dict)]
    return []


def _top_records(table: pd.DataFrame | None, column: str, limit: int, ascending: bool) -> list[dict[str, Any]]:
    """函数说明：按指定列选取排序后的前若干条记录。"""
    if table is None or table.empty or column not in table.columns:
        return []
    return _json_safe(table.sort_values(column, ascending=ascending).head(limit).to_dict(orient="records"))


def _table_or_empty(table: pd.DataFrame | None, columns: list[str]) -> pd.DataFrame:
    """函数说明：返回包含指定列的表格或空表。"""
    if table is None or table.empty:
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in table.columns:
            table[column] = None
    return table[columns]


def _gap(value: float | None, threshold: float | None) -> float | None:
    """函数说明：计算指标值相对门槛的差距。"""
    if value is None or threshold is None:
        return None
    return float(value - threshold)


def _period_return(start_value: float | None, end_value: float | None) -> float | None:
    """函数说明：计算区间起止净值收益率。"""
    if start_value is None or end_value is None or start_value == 0:
        return None
    return float(end_value / start_value - 1.0)


def _sum_column(table: pd.DataFrame | None, column: str) -> float | None:
    """函数说明：汇总表格中指定数值列。"""
    if table is None or table.empty or column not in table.columns:
        return None
    return float(pd.to_numeric(table[column], errors="coerce").fillna(0.0).sum())


def _number_or_none(value: Any) -> float | None:
    """函数说明：将可解析数值转换为浮点数，否则返回空值。"""
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _comparable_value(value: Any) -> Any:
    """函数说明：将参数值转换为可稳定比较的形式。"""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (float, np.floating)):
        return round(float(value), 10)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return str(value)


def _date_text(value: Any) -> str | None:
    """函数说明：将日期值格式化为 YYYY-MM-DD 文本。"""
    if value is None or pd.isna(value):
        return None
    return str(pd.Timestamp(value).date())


def _json_safe(value: Any) -> Any:
    """函数说明：将 pandas 和 numpy 值转换为 JSON 友好的对象。"""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value
