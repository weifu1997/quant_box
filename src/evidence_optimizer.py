"""Build evidence-backed style, risk, and trading optimization plans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config_loader import resolve_path


REPORT_BASENAME = "evidence_optimization_plan"
REQUIRED_GRID_COLUMNS = {
    "key",
    "annual_return",
    "max_drawdown",
    "annual_trade_cost_ratio",
    "full_goal",
}


def build_evidence_optimization_plan(
    artifact_dir: str | Path = "outputs",
    *,
    grid_glob: str = "*router_grid*.csv",
    max_industry_weight_target: float = 0.35,
    annual_trade_cost_ratio_target: float = 0.20,
) -> dict[str, Any]:
    """Build the next optimization plan from trusted diagnostics and router grids."""
    root = resolve_path(artifact_dir)
    review = _read_json(root / "optimization_review.json")
    selected_params = _read_json(root / "auto_selected_params.json")
    candidates = _load_grid_candidates(root, grid_glob)
    selected = _select_candidate(candidates, annual_trade_cost_ratio_target=annual_trade_cost_ratio_target)
    risk_constraints = _risk_constraints(
        review.get("risk_exposure", {}),
        max_industry_weight_target=max_industry_weight_target,
    )
    trading_constraints = _trading_constraints(
        review.get("trading_constraints", {}),
        selected,
        annual_trade_cost_ratio_target=annual_trade_cost_ratio_target,
    )
    route_parameters = _route_parameters(selected_params, selected)
    status = "ready" if review.get("status") == "ready" and selected else "review"
    caveats = _caveats(review, candidates, selected)
    return {
        "artifact_dir": str(root),
        "status": status,
        "style_recognition": {
            "strategy_mode": selected_params.get("strategy_mode") or review.get("strategy_mode", ""),
            "selected_route_parameters": route_parameters,
            "candidate": selected,
        },
        "risk_exposure": risk_constraints,
        "trading_constraints": trading_constraints,
        "next_commands": _next_commands(
            selected,
            max_industry_weight=risk_constraints.get("max_industry_weight"),
            artifact_dir=root,
        ),
        "caveats": caveats,
    }


def write_evidence_optimization_plan(report: dict[str, Any], out_dir: str | Path = "outputs") -> dict[str, str]:
    """Write machine-readable and human-readable optimization-plan artifacts."""
    output_dir = resolve_path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{REPORT_BASENAME}.json"
    md_path = output_dir / f"{REPORT_BASENAME}.md"
    json_path.write_text(json.dumps(_json_safe(report), indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_evidence_optimization_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def render_evidence_optimization_markdown(report: dict[str, Any]) -> str:
    """Render a concise optimization plan."""
    style = report.get("style_recognition", {})
    candidate = style.get("candidate", {})
    risk = report.get("risk_exposure", {})
    trading = report.get("trading_constraints", {})
    lines = [
        "# Evidence Optimization Plan",
        "",
        f"- Status: {report.get('status', '')}",
        f"- Strategy mode: {style.get('strategy_mode', '')}",
        f"- Selected candidate: {candidate.get('source_file', '')}",
        f"- Candidate annual return: {_pct(candidate.get('annual_return'))}",
        f"- Candidate max drawdown: {_pct(candidate.get('max_drawdown'))}",
        f"- Candidate annual trade cost ratio: {_pct(candidate.get('annual_trade_cost_ratio'))}",
        "",
        "## Risk Exposure",
        "",
        f"- Max industry weight: {_pct(risk.get('max_industry_weight'))}",
        f"- Target minimum positions: {risk.get('target_min_positions')}",
        f"- Small-cap concentration action: {risk.get('small_cap_action', '')}",
        "",
        "## Trading Constraints",
        "",
        f"- Annual trade cost ratio target: {_pct(trading.get('annual_trade_cost_ratio_target'))}",
        f"- Turnover action: {trading.get('turnover_action', '')}",
        f"- Candidate turnover mode: {trading.get('candidate_turnover_mode', '')}",
        f"- Candidate boost reasons: {trading.get('candidate_turnover_boost_reasons', '')}",
        "",
        "## Next Commands",
        "",
    ]
    lines.extend([f"- `{command}`" for command in report.get("next_commands", [])])
    caveats = report.get("caveats", [])
    if caveats:
        lines.extend(["", "## Caveats", ""])
        lines.extend([f"- {caveat}" for caveat in caveats])
    lines.append("")
    return "\n".join(lines)


def _load_grid_candidates(root: Path, grid_glob: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for path in sorted(root.glob(grid_glob)):
        frame = _read_csv(path)
        if frame.empty or not REQUIRED_GRID_COLUMNS.issubset(frame.columns):
            continue
        for _, row in frame.iterrows():
            item = {str(key): _scalar(value) for key, value in row.to_dict().items()}
            item["source_file"] = str(path)
            item["full_goal"] = _boolish(item.get("full_goal"))
            item["annual_return"] = _number_or_none(item.get("annual_return"))
            item["max_drawdown"] = _number_or_none(item.get("max_drawdown"))
            item["annual_trade_cost_ratio"] = _number_or_none(item.get("annual_trade_cost_ratio"))
            candidates.append(item)
    return candidates


def _select_candidate(candidates: list[dict[str, Any]], *, annual_trade_cost_ratio_target: float) -> dict[str, Any]:
    eligible = [
        candidate
        for candidate in candidates
        if candidate.get("full_goal")
        and _number_or_none(candidate.get("annual_trade_cost_ratio")) is not None
        and float(candidate["annual_trade_cost_ratio"]) <= float(annual_trade_cost_ratio_target)
    ]
    if not eligible:
        return {}
    return max(
        eligible,
        key=lambda item: (
            float(item.get("annual_return") or 0.0),
            float(item.get("max_drawdown") or -1.0),
            -float(item.get("annual_trade_cost_ratio") or 0.0),
        ),
    )


def _risk_constraints(risk: dict[str, Any], *, max_industry_weight_target: float) -> dict[str, Any]:
    flags = [str(flag) for flag in risk.get("flags", [])] if isinstance(risk, dict) else []
    low_positions = any(flag.startswith("low_position_count:") for flag in flags)
    high_industry = any(flag.startswith("high_industry_concentration:") for flag in flags)
    small_cap = any(flag.startswith("small_cap_concentration:") for flag in flags)
    return {
        "max_industry_weight": float(max_industry_weight_target) if high_industry else None,
        "target_min_positions": 5 if low_positions else None,
        "small_cap_action": "reduce_small_cap_concentration" if small_cap else "monitor",
        "source_flags": flags,
    }


def _trading_constraints(
    trading: dict[str, Any],
    selected: dict[str, Any],
    *,
    annual_trade_cost_ratio_target: float,
) -> dict[str, Any]:
    flags = [str(flag) for flag in trading.get("flags", [])] if isinstance(trading, dict) else []
    high_cost = any(flag.startswith("annual_trade_cost_ratio_above_target:") for flag in flags)
    return {
        "annual_trade_cost_ratio_target": float(annual_trade_cost_ratio_target),
        "turnover_action": "do_not_increase_turnover" if high_cost else "keep_current_turnover_gate",
        "candidate_turnover_mode": selected.get("turnover_mode", ""),
        "candidate_turnover_boost_reasons": selected.get("turnover_boost_reasons", ""),
        "candidate_turnover_boost_max_turnover": _int_or_none(selected.get("turnover_boost_max_turnover")),
        "candidate_turnover_boost_rank_buffer": _int_or_none(selected.get("turnover_boost_rank_buffer")),
        "source_flags": flags,
    }


def _route_parameters(selected_params: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "initial_source",
        "missing_ret252_exposure",
        "flat_negative_exposure",
        "moderate_positive_source",
        "moderate_positive_ret252_min",
        "moderate_low_source",
        "moderate_low_ret252_min",
        "moderate_low_ret252_max",
        "moderate_low_exposure",
        "strong_trailing_exposure",
        "turnover_mode",
        "turnover_boost_reasons",
        "turnover_boost_max_turnover",
        "turnover_boost_rank_buffer",
        "full_turnover_on_route_change",
        "use_defensive_timing",
        "include_expanded_sources",
    ]
    result = {key: selected_params.get(key) for key in keys if key in selected_params}
    for key in keys:
        if key in selected and selected.get(key) not in {None, ""}:
            result[key] = selected[key]
    return result


def _next_commands(selected: dict[str, Any], *, max_industry_weight: Any, artifact_dir: Path) -> list[str]:
    if not selected:
        return []
    command = [
        r".\.venv\Scripts\python.exe",
        r"scripts\run_annual_state_router_grid.py",
        "--cache-dir",
        r"outputs\router_score_cache",
        "--output",
        r"outputs\evidence_optimized_router_grid.csv",
        "--write-hit-prefix",
        r"outputs\evidence_optimized_router_hit",
    ]
    mapping = {
        "missing_ret252_exposure": "--missing-ret252-exposures",
        "strong_trailing_exposure": "--strong-trailing-exposures",
        "moderate_positive_source": "--moderate-positive-sources",
        "moderate_positive_ret252_min": "--moderate-positive-ret252-mins",
        "moderate_low_source": "--moderate-low-sources",
        "moderate_low_ret252_min": "--moderate-low-ret252-mins",
        "moderate_low_ret252_max": "--moderate-low-ret252-maxs",
        "moderate_low_exposure": "--moderate-low-exposures",
        "turnover_mode": "--turnover-modes",
        "turnover_boost_reasons": "--turnover-boost-reason-sets",
        "turnover_boost_max_turnover": "--turnover-boost-max-turnovers",
        "turnover_boost_rank_buffer": "--turnover-boost-rank-buffers",
    }
    for key, flag in mapping.items():
        value = selected.get(key)
        if value not in {None, ""}:
            command.extend([flag, str(value)])
    command.extend(["--equity-overlay-sideways-exposures", "none"])
    command.extend(["--equity-overlay-bear-exposures", "none"])
    command.extend(["--defensive-bear-exposures", "none"])
    if max_industry_weight is not None:
        command.extend(["--max-industry-weights", str(max_industry_weight)])
    return [" ".join(command)]


def _caveats(review: dict[str, Any], candidates: list[dict[str, Any]], selected: dict[str, Any]) -> list[str]:
    caveats: list[str] = []
    if review.get("status") != "ready":
        caveats.append("Optimization review is not ready; run diagnostics and optimization review first.")
    if not candidates:
        caveats.append("No router grid rows with the required columns were found.")
    elif not selected:
        caveats.append("No full-goal router grid candidate stayed within the annual trade-cost target.")
    caveats.append("Risk constraints are generated as research overlays; they are not written into config/settings.yaml.")
    return caveats


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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


def _scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _pct(value: Any) -> str:
    number = _number_or_none(value)
    return "" if number is None else f"{number:.2%}"


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
