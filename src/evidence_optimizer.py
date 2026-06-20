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
    risk_constraints["overlay_validation"] = _risk_overlay_validation(
        candidates,
        max_industry_weight=risk_constraints.get("max_industry_weight"),
        annual_trade_cost_ratio_target=annual_trade_cost_ratio_target,
    )
    risk_constraints["risk_exit_refill_validation"] = (
        _boolean_overlay_validation(
            candidates,
            field="rebalance_after_risk_exit",
            annual_trade_cost_ratio_target=annual_trade_cost_ratio_target,
        )
        if risk_constraints.get("target_min_positions")
        else {"status": "not_required", "candidate": {}}
    )
    trading_constraints = _trading_constraints(
        review.get("trading_constraints", {}),
        selected,
        annual_trade_cost_ratio_target=annual_trade_cost_ratio_target,
    )
    route_parameters = _route_parameters(selected_params, selected)
    candidate_evidence = _candidate_evidence(
        candidates,
        selected,
        annual_trade_cost_ratio_target=annual_trade_cost_ratio_target,
    )
    decisions = _optimization_decisions(selected, risk_constraints, trading_constraints)
    status = "ready" if review.get("status") == "ready" and selected and _risk_overlay_ready(risk_constraints) else "review"
    caveats = _caveats(review, candidates, selected, risk_constraints)
    return {
        "artifact_dir": str(root),
        "status": status,
        "style_recognition": {
            "status": "ready" if selected else "review",
            "strategy_mode": selected_params.get("strategy_mode") or review.get("strategy_mode", ""),
            "selected_route_parameters": route_parameters,
            "candidate": selected,
        },
        "candidate_evidence": candidate_evidence,
        "risk_exposure": risk_constraints,
        "trading_constraints": trading_constraints,
        "optimization_decisions": decisions,
        "next_commands": _next_commands(
            selected,
            max_industry_weight=risk_constraints.get("max_industry_weight"),
            include_risk_exit_refill=bool(risk_constraints.get("target_min_positions")),
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
    evidence = report.get("candidate_evidence", {})
    risk = report.get("risk_exposure", {})
    trading = report.get("trading_constraints", {})
    lines = [
        "# Evidence Optimization Plan",
        "",
        f"- Status: {report.get('status', '')}",
        f"- Style status: {style.get('status', '')}",
        f"- Risk status: {risk.get('status', '')}",
        f"- Trading status: {trading.get('status', '')}",
        f"- Strategy mode: {style.get('strategy_mode', '')}",
        f"- Selected candidate: {candidate.get('source_file', '')}",
        f"- Candidate annual return: {_pct(candidate.get('annual_return'))}",
        f"- Candidate max drawdown: {_pct(candidate.get('max_drawdown'))}",
        f"- Candidate annual trade cost ratio: {_pct(candidate.get('annual_trade_cost_ratio'))}",
        "",
        "## Candidate Evidence",
        "",
        f"- Evaluated grid rows: {evidence.get('evaluated_grid_rows', 0)}",
        f"- Full-goal rows: {evidence.get('full_goal_rows', 0)}",
        f"- Cost-eligible full-goal rows: {evidence.get('cost_eligible_full_goal_rows', 0)}",
        f"- Best rejected candidate reason: {evidence.get('best_rejected_candidate', {}).get('reject_reason', '')}",
        "",
        "## Risk Exposure",
        "",
        f"- Max industry weight: {_pct(risk.get('max_industry_weight'))}",
        f"- Target minimum positions: {risk.get('target_min_positions')}",
        f"- Small-cap concentration action: {risk.get('small_cap_action', '')}",
        f"- Risk overlay validation: {risk.get('overlay_validation', {}).get('status', '')}",
        f"- Risk-exit refill validation: {risk.get('risk_exit_refill_validation', {}).get('status', '')}",
        "",
        "## Trading Constraints",
        "",
        f"- Annual trade cost ratio target: {_pct(trading.get('annual_trade_cost_ratio_target'))}",
        f"- Turnover action: {trading.get('turnover_action', '')}",
        f"- Candidate turnover mode: {trading.get('candidate_turnover_mode', '')}",
        f"- Candidate boost reasons: {trading.get('candidate_turnover_boost_reasons', '')}",
        "",
        "## Optimization Decisions",
        "",
    ]
    lines.extend([f"- {decision}" for decision in report.get("optimization_decisions", [])])
    lines.extend(
        [
            "",
            "## Next Commands",
            "",
        ]
    )
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


def _candidate_evidence(
    candidates: list[dict[str, Any]],
    selected: dict[str, Any],
    *,
    annual_trade_cost_ratio_target: float,
) -> dict[str, Any]:
    full_goal = [candidate for candidate in candidates if candidate.get("full_goal")]
    cost_eligible = [
        candidate
        for candidate in full_goal
        if _number_or_none(candidate.get("annual_trade_cost_ratio")) is not None
        and float(candidate["annual_trade_cost_ratio"]) <= float(annual_trade_cost_ratio_target)
    ]
    rejected = [_rejected_candidate(candidate, annual_trade_cost_ratio_target) for candidate in candidates]
    rejected = [candidate for candidate in rejected if candidate.get("reject_reason")]
    return {
        "evaluated_grid_rows": len(candidates),
        "full_goal_rows": len(full_goal),
        "cost_eligible_full_goal_rows": len(cost_eligible),
        "selected_key": selected.get("key", ""),
        "best_rejected_candidate": _best_rejected_candidate(rejected),
    }


def _rejected_candidate(candidate: dict[str, Any], annual_trade_cost_ratio_target: float) -> dict[str, Any]:
    reason = ""
    cost = _number_or_none(candidate.get("annual_trade_cost_ratio"))
    if candidate.get("full_goal") and cost is not None and cost > float(annual_trade_cost_ratio_target):
        reason = "trade_cost_above_target"
    elif not candidate.get("full_goal"):
        failed_years = str(candidate.get("failed_years") or "").strip()
        reason = f"full_gate_failed:{failed_years}" if failed_years else "full_gate_failed"
    if not reason:
        return {}
    result = _compact_candidate(candidate)
    result["reject_reason"] = reason
    return result


def _best_rejected_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        return {}
    return max(
        candidates,
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
    result = {
        "max_industry_weight": float(max_industry_weight_target) if high_industry else None,
        "target_min_positions": 5 if low_positions else None,
        "small_cap_action": "reduce_small_cap_concentration" if small_cap else "monitor",
        "source_flags": flags,
    }
    result["status"] = "review" if low_positions or high_industry or small_cap else "ready"
    return result


def _risk_overlay_validation(
    candidates: list[dict[str, Any]],
    *,
    max_industry_weight: Any,
    annual_trade_cost_ratio_target: float,
) -> dict[str, Any]:
    target = _number_or_none(max_industry_weight)
    if target is None:
        return {"status": "not_required", "candidate": {}}
    tested = [
        candidate
        for candidate in candidates
        if _number_or_none(candidate.get("max_industry_weight")) is not None
        and abs(float(candidate["max_industry_weight"]) - target) <= 1e-12
    ]
    if not tested:
        return {"status": "not_tested", "candidate": {}}
    best = max(
        tested,
        key=lambda item: (
            bool(item.get("full_goal")),
            float(item.get("annual_return") or 0.0),
            float(item.get("max_drawdown") or -1.0),
            -float(item.get("annual_trade_cost_ratio") or 0.0),
        ),
    )
    passes = bool(best.get("full_goal")) and (
        _number_or_none(best.get("annual_trade_cost_ratio")) is not None
        and float(best["annual_trade_cost_ratio"]) <= float(annual_trade_cost_ratio_target)
    )
    return {"status": "pass" if passes else "fail", "candidate": _compact_candidate(best)}


def _risk_overlay_ready(risk_constraints: dict[str, Any]) -> bool:
    for key in ("overlay_validation", "risk_exit_refill_validation"):
        validation = risk_constraints.get(key, {})
        status = validation.get("status") if isinstance(validation, dict) else ""
        if status == "fail":
            return False
    return True


def _boolean_overlay_validation(
    candidates: list[dict[str, Any]],
    *,
    field: str,
    annual_trade_cost_ratio_target: float,
) -> dict[str, Any]:
    tested = [candidate for candidate in candidates if _boolish(candidate.get(field))]
    if not tested:
        return {"status": "not_tested", "candidate": {}}
    best = max(
        tested,
        key=lambda item: (
            bool(item.get("full_goal")),
            float(item.get("annual_return") or 0.0),
            float(item.get("max_drawdown") or -1.0),
            -float(item.get("annual_trade_cost_ratio") or 0.0),
        ),
    )
    passes = bool(best.get("full_goal")) and (
        _number_or_none(best.get("annual_trade_cost_ratio")) is not None
        and float(best["annual_trade_cost_ratio"]) <= float(annual_trade_cost_ratio_target)
    )
    return {"status": "pass" if passes else "fail", "candidate": _compact_candidate(best)}


def _trading_constraints(
    trading: dict[str, Any],
    selected: dict[str, Any],
    *,
    annual_trade_cost_ratio_target: float,
) -> dict[str, Any]:
    flags = [str(flag) for flag in trading.get("flags", [])] if isinstance(trading, dict) else []
    high_cost = any(flag.startswith("annual_trade_cost_ratio_above_target:") for flag in flags)
    return {
        "status": "ready" if selected else "review",
        "annual_trade_cost_ratio_target": float(annual_trade_cost_ratio_target),
        "turnover_action": "do_not_increase_turnover" if high_cost else "keep_current_turnover_gate",
        "candidate_turnover_mode": selected.get("turnover_mode", ""),
        "candidate_turnover_boost_reasons": selected.get("turnover_boost_reasons", ""),
        "candidate_turnover_boost_max_turnover": _int_or_none(selected.get("turnover_boost_max_turnover")),
        "candidate_turnover_boost_rank_buffer": _int_or_none(selected.get("turnover_boost_rank_buffer")),
        "source_flags": flags,
    }


def _optimization_decisions(
    selected: dict[str, Any],
    risk_constraints: dict[str, Any],
    trading_constraints: dict[str, Any],
) -> list[str]:
    decisions: list[str] = []
    if selected:
        decisions.append(
            "Adopt the selected annual-state-router style candidate for research follow-up; it passes the full yearly, drawdown, turnover, and cost gates."
        )
    overlay = risk_constraints.get("overlay_validation", {})
    if isinstance(overlay, dict) and overlay.get("status") == "fail":
        decisions.append("Reject the tested hard max-industry-weight overlay for now; it failed the full router gate.")
    elif isinstance(overlay, dict) and overlay.get("status") == "pass":
        decisions.append("Keep the tested max-industry-weight overlay as an eligible risk-control candidate.")
    refill = risk_constraints.get("risk_exit_refill_validation", {})
    if isinstance(refill, dict) and refill.get("status") == "fail":
        decisions.append("Reject same-day refill after stop-loss/take-profit exits; it increased failures or costs in tested evidence.")
    elif isinstance(refill, dict) and refill.get("status") == "pass":
        decisions.append("Keep same-day risk-exit refill as an eligible risk-control candidate.")
    if trading_constraints.get("turnover_action") == "do_not_increase_turnover":
        decisions.append("Do not loosen turnover; preserve the selected candidate's rank10 turnover and boost limits.")
    if risk_constraints.get("small_cap_action") == "reduce_small_cap_concentration":
        decisions.append("Search for natural diversification through route source, threshold, and exposure changes before writing risk limits into config.")
    return decisions


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


def _next_commands(
    selected: dict[str, Any],
    *,
    max_industry_weight: Any,
    include_risk_exit_refill: bool,
    artifact_dir: Path,
) -> list[str]:
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
        command.extend(["--max-industry-weights", f"none,{max_industry_weight}"])
    if include_risk_exit_refill:
        command.extend(["--rebalance-after-risk-exit-options", "false,true"])
    return [" ".join(command)]


def _caveats(
    review: dict[str, Any],
    candidates: list[dict[str, Any]],
    selected: dict[str, Any],
    risk_constraints: dict[str, Any],
) -> list[str]:
    caveats: list[str] = []
    if review.get("status") != "ready":
        caveats.append("Optimization review is not ready; run diagnostics and optimization review first.")
    if not candidates:
        caveats.append("No router grid rows with the required columns were found.")
    elif not selected:
        caveats.append("No full-goal router grid candidate stayed within the annual trade-cost target.")
    validation = risk_constraints.get("overlay_validation", {})
    if isinstance(validation, dict) and validation.get("status") == "fail":
        caveats.append("The tested max-industry-weight overlay did not pass the full router gate; continue risk search before adopting it.")
    refill_validation = risk_constraints.get("risk_exit_refill_validation", {})
    if isinstance(refill_validation, dict) and refill_validation.get("status") == "fail":
        caveats.append("The tested risk-exit refill overlay did not pass the full router gate; do not auto-refill after risk exits yet.")
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
    except pd.errors.ParserError:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


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


def _compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "key",
        "source_file",
        "annual_return",
        "max_drawdown",
        "annual_trade_cost_ratio",
        "failed_years",
        "full_goal",
        "moderate_low_source",
        "moderate_low_ret252_min",
        "moderate_low_exposure",
        "max_industry_weight",
        "rebalance_after_risk_exit",
        "turnover_mode",
        "turnover_boost_reasons",
    ]
    return {key: candidate.get(key) for key in keys if key in candidate}


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
