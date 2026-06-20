"""Review style routing, risk exposure, and trading constraints after diagnostics pass."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config_loader import resolve_path


REPORT_BASENAME = "optimization_review"


def build_optimization_review(artifact_dir: str | Path = "outputs") -> dict[str, Any]:
    """Build an optimization-stage review from trusted diagnostic artifacts."""
    root = resolve_path(artifact_dir)
    quant = _read_json(root / "quant_diagnostic_report.json")
    baseline_metrics = _read_json(root / "backtest_metrics.json")
    auto_metrics = _read_json(root / "auto_backtest_metrics.json")
    auto_quality = _read_json(root / "auto_backtest_quality.json")
    auto_status = _read_json(root / "auto_run_status.json")
    research = _read_json(root / "auto_research_diagnostics.json")

    year_routes = _read_csv(root / "auto_annual_state_router_year_routes.csv")
    score_routes = _read_csv(root / "auto_annual_state_router_score_routes.csv")
    yearly = _read_csv(root / "auto_backtest_yearly_breakdown.csv")

    diagnostics_ready = bool(quant.get("optimization_ready"))
    auto_quality_acceptable = bool(auto_quality.get("is_acceptable")) if auto_quality else False
    auto_ready = bool(auto_status.get("is_executable")) and auto_quality_acceptable
    performance = _performance_comparison(baseline_metrics, auto_metrics)
    style = _style_summary(year_routes, score_routes)
    risk = _risk_summary(research)
    trading = _trading_summary(research, yearly, score_routes)
    recommendations = _recommendations(
        diagnostics_ready=diagnostics_ready,
        auto_ready=auto_ready,
        performance=performance,
        risk=risk,
        trading=trading,
    )

    return {
        "artifact_dir": str(root),
        "status": "ready" if diagnostics_ready and auto_ready else "review",
        "diagnostics_ready": diagnostics_ready,
        "strategy_mode": auto_status.get("strategy_mode", ""),
        "auto_signal_executable": bool(auto_status.get("is_executable")),
        "auto_backtest_acceptable": auto_quality_acceptable,
        "performance": performance,
        "style_recognition": style,
        "risk_exposure": risk,
        "trading_constraints": trading,
        "recommendations": recommendations,
    }


def write_optimization_review(report: dict[str, Any], out_dir: str | Path = "outputs") -> dict[str, str]:
    """Write optimization review JSON and Markdown reports."""
    output_dir = resolve_path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{REPORT_BASENAME}.json"
    md_path = output_dir / f"{REPORT_BASENAME}.md"
    json_path.write_text(json.dumps(_json_safe(report), indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_optimization_review_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def render_optimization_review_markdown(report: dict[str, Any]) -> str:
    """Render a concise human-readable optimization review."""
    perf = report.get("performance", {})
    style = report.get("style_recognition", {})
    risk = report.get("risk_exposure", {})
    trading = report.get("trading_constraints", {})
    lines = [
        "# Optimization Review",
        "",
        f"- Status: {report.get('status', '')}",
        f"- Diagnostics ready: {report.get('diagnostics_ready', False)}",
        f"- Strategy mode: {report.get('strategy_mode', '')}",
        f"- Auto signal executable: {report.get('auto_signal_executable', False)}",
        "",
        "## Performance",
        "",
        f"- Baseline annual return: {_pct(perf.get('baseline_annual_return'))}",
        f"- Optimized annual return: {_pct(perf.get('optimized_annual_return'))}",
        f"- Annual return delta: {_pct(perf.get('annual_return_delta'))}",
        f"- Baseline max drawdown: {_pct(perf.get('baseline_max_drawdown'))}",
        f"- Optimized max drawdown: {_pct(perf.get('optimized_max_drawdown'))}",
        f"- Drawdown improvement: {_pct(perf.get('drawdown_improvement'))}",
        "",
        "## Style Recognition",
        "",
        f"- Latest route: {style.get('latest_source', '')} ({style.get('latest_reason', '')})",
        f"- Latest exposure: {_num(style.get('latest_exposure'))}",
        f"- Source counts: {style.get('source_counts', {})}",
        f"- Reason counts: {style.get('reason_counts', {})}",
        "",
        "## Risk Exposure",
        "",
        f"- Latest positions: {risk.get('latest_position_count')}",
        f"- Latest top industry weight: {_pct(risk.get('latest_top_industry_weight'))}",
        f"- Latest top position weight: {_pct(risk.get('latest_top_position_weight'))}",
        f"- Market-cap buckets: {risk.get('market_cap_bucket_weights', {})}",
        f"- Risk flags: {', '.join(risk.get('flags', [])) or 'none'}",
        "",
        "## Trading Constraints",
        "",
        f"- Annual turnover estimate: {_num(trading.get('annual_turnover_estimate'))}",
        f"- Annual trade cost ratio max: {_pct(trading.get('annual_trade_cost_ratio_max'))}",
        f"- Cost drag on initial equity: {_pct(trading.get('cost_drag_on_initial_equity'))}",
        f"- Boosted route rows: {trading.get('boosted_route_rows')}",
        f"- Trading flags: {', '.join(trading.get('flags', [])) or 'none'}",
        "",
        "## Recommendations",
        "",
    ]
    lines.extend([f"- {item}" for item in report.get("recommendations", [])])
    lines.append("")
    return "\n".join(lines)


def _performance_comparison(baseline: dict[str, Any], optimized: dict[str, Any]) -> dict[str, Any]:
    baseline_annual = _number_or_none(baseline.get("annual_return"))
    optimized_annual = _number_or_none(optimized.get("annual_return"))
    baseline_drawdown = _number_or_none(baseline.get("max_drawdown"))
    optimized_drawdown = _number_or_none(optimized.get("max_drawdown"))
    return {
        "baseline_annual_return": baseline_annual,
        "optimized_annual_return": optimized_annual,
        "annual_return_delta": _delta(optimized_annual, baseline_annual),
        "baseline_max_drawdown": baseline_drawdown,
        "optimized_max_drawdown": optimized_drawdown,
        "drawdown_improvement": _delta(optimized_drawdown, baseline_drawdown),
        "baseline_sharpe": _number_or_none(baseline.get("sharpe")),
        "optimized_sharpe": _number_or_none(optimized.get("sharpe")),
        "baseline_calmar": _number_or_none(baseline.get("calmar")),
        "optimized_calmar": _number_or_none(optimized.get("calmar")),
    }


def _style_summary(year_routes: pd.DataFrame, score_routes: pd.DataFrame) -> dict[str, Any]:
    latest = year_routes.tail(1).iloc[0].to_dict() if not year_routes.empty else {}
    source_counts = _value_counts(score_routes, "source")
    reason_counts = _value_counts(score_routes, "reason")
    exposure = pd.to_numeric(score_routes.get("exposure"), errors="coerce") if "exposure" in score_routes else pd.Series(dtype=float)
    return {
        "latest_year": _int_or_none(latest.get("year")),
        "latest_source": latest.get("source", ""),
        "latest_reason": latest.get("reason", ""),
        "latest_exposure": _number_or_none(latest.get("exposure")),
        "source_counts": source_counts,
        "reason_counts": reason_counts,
        "min_exposure": _series_min(exposure),
        "max_exposure": _series_max(exposure),
        "reduced_exposure_rows": int((exposure < 1.0).sum()) if not exposure.empty else 0,
    }


def _risk_summary(research: dict[str, Any]) -> dict[str, Any]:
    exposure = research.get("exposure", {}) if isinstance(research, dict) else {}
    buckets = exposure.get("market_cap_buckets", []) if isinstance(exposure, dict) else []
    bucket_weights = {
        str(row.get("bucket")): _number_or_none(row.get("weight"))
        for row in buckets
        if isinstance(row, dict) and row.get("bucket") is not None
    }
    top_industry = _number_or_none(exposure.get("latest_max_industry_weight")) if isinstance(exposure, dict) else None
    top_position = _number_or_none(exposure.get("latest_top_position_weight")) if isinstance(exposure, dict) else None
    position_count = _int_or_none(exposure.get("latest_position_count")) if isinstance(exposure, dict) else None
    flags: list[str] = []
    if position_count is not None and position_count < 5:
        flags.append(f"low_position_count:{position_count}<5")
    if top_industry is not None and top_industry > 0.35:
        flags.append(f"high_industry_concentration:{top_industry:.4f}>0.35")
    if bucket_weights.get("small", 0.0) and float(bucket_weights.get("small") or 0.0) > 0.80:
        flags.append(f"small_cap_concentration:{bucket_weights['small']:.4f}>0.80")
    return {
        "latest_position_count": position_count,
        "latest_top_industry_weight": top_industry,
        "latest_top_position_weight": top_position,
        "market_cap_bucket_weights": bucket_weights,
        "flags": flags,
    }


def _trading_summary(research: dict[str, Any], yearly: pd.DataFrame, score_routes: pd.DataFrame) -> dict[str, Any]:
    turnover = research.get("turnover_attribution", {}) if isinstance(research, dict) else {}
    costs = research.get("cost_attribution", {}) if isinstance(research, dict) else {}
    annual_cost = pd.to_numeric(yearly.get("annual_trade_cost_ratio"), errors="coerce") if "annual_trade_cost_ratio" in yearly else pd.Series(dtype=float)
    max_turnover = pd.to_numeric(score_routes.get("max_turnover"), errors="coerce") if "max_turnover" in score_routes else pd.Series(dtype=float)
    rank_buffer = pd.to_numeric(score_routes.get("rank_buffer"), errors="coerce") if "rank_buffer" in score_routes else pd.Series(dtype=float)
    boosted = int(((max_turnover > 1) | (rank_buffer < 10)).sum()) if not score_routes.empty else 0
    flags: list[str] = []
    annual_cost_max = _series_max(annual_cost)
    if annual_cost_max is not None and annual_cost_max > 0.20:
        flags.append(f"annual_trade_cost_ratio_above_target:{annual_cost_max:.4f}>0.20")
    trim_share = _number_or_none(turnover.get("rebalance_trim_cost_share_of_total_trade_cost")) if isinstance(turnover, dict) else None
    if trim_share is not None and trim_share > 0.20:
        flags.append(f"rebalance_trim_cost_share_high:{trim_share:.4f}>0.20")
    return {
        "annual_turnover_estimate": _number_or_none(turnover.get("annual_turnover_estimate")) if isinstance(turnover, dict) else None,
        "annual_turnover_without_rebalance_trims_estimate": _number_or_none(turnover.get("annual_turnover_without_rebalance_trims_estimate")) if isinstance(turnover, dict) else None,
        "annual_trade_cost_ratio_max": annual_cost_max,
        "cost_drag_on_initial_equity": _number_or_none(costs.get("cost_drag_on_initial_equity")) if isinstance(costs, dict) else None,
        "boosted_route_rows": boosted,
        "max_turnover_values": sorted({int(value) for value in max_turnover.dropna().tolist()}) if not max_turnover.empty else [],
        "rank_buffer_values": sorted({int(value) for value in rank_buffer.dropna().tolist()}) if not rank_buffer.empty else [],
        "flags": flags,
    }


def _recommendations(
    *,
    diagnostics_ready: bool,
    auto_ready: bool,
    performance: dict[str, Any],
    risk: dict[str, Any],
    trading: dict[str, Any],
) -> list[str]:
    items: list[str] = []
    if diagnostics_ready and auto_ready and _positive(performance.get("annual_return_delta")):
        items.append("Use annual_state_router as the optimized style-routing path; it improves annual return versus the baseline and passes quality gates.")
    if _positive(performance.get("drawdown_improvement")):
        items.append("Keep routed exposure controls enabled; optimized max drawdown is better than the baseline.")
    if risk.get("flags"):
        items.append("Do not loosen concentration controls yet; review risk flags before increasing top_n or exposure.")
    else:
        items.append("Current concentration evidence has no hard flags.")
    if trading.get("flags"):
        items.append("Do not increase turnover until trading-cost flags are resolved.")
    else:
        items.append("Current turnover and trade-cost evidence stays inside configured gates.")
    return items


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    return {str(key): int(value) for key, value in frame[column].astype(str).value_counts().to_dict().items()}


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


def _positive(value: Any) -> bool:
    number = _number_or_none(value)
    return number is not None and number > 0


def _number_or_none(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    number = _number_or_none(value)
    return int(number) if number is not None else None


def _series_min(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.min()) if not clean.empty else None


def _series_max(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.max()) if not clean.empty else None


def _pct(value: Any) -> str:
    number = _number_or_none(value)
    return "" if number is None else f"{number:.2%}"


def _num(value: Any) -> str:
    number = _number_or_none(value)
    return "" if number is None else f"{number:.6f}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value
