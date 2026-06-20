"""Build a five-layer diagnostic report from existing quant workflow artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config_loader import resolve_path


STATUS_PASS = "pass"
STATUS_WARN = "warn"
STATUS_FAIL = "fail"

BACKTEST_REQUIRED = ("backtest_equity.csv", "backtest_holdings.csv", "backtest_trades.csv", "backtest_metrics.json")
REPRODUCIBILITY_FILES = (
    "backtest_equity.csv",
    "backtest_holdings.csv",
    "backtest_trades.csv",
    "backtest_metrics.json",
)
COST_COLUMNS = ("commission_cost", "tax_cost", "transfer_fee_cost", "slippage_cost")
EXECUTED_STATUSES = {"filled", "partial", "risk_exit"}
REPORT_BASENAME = "quant_diagnostic_report"


@dataclass(frozen=True)
class DiagnosticCheck:
    """A single pass/warn/fail diagnostic finding."""

    layer: str
    name: str
    status: str
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "name": self.name,
            "status": self.status,
            "summary": self.summary,
            "evidence": _json_safe(self.evidence),
            "caveats": list(self.caveats),
        }


def build_quant_diagnostic_report(
    artifact_dir: str | Path = "outputs",
    *,
    compare_dir: str | Path | None = None,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Build the ordered diagnostic report from existing output artifacts."""
    root = resolve_path(artifact_dir)
    compare_root = resolve_path(compare_dir) if compare_dir is not None else None
    checks = [
        *_backtest_engine_checks(root, compare_root, tolerance=tolerance),
        *_data_checks(root),
        *_factor_checks(root),
        *_portfolio_checks(root),
    ]
    checks.extend(_optimization_checks(checks))
    layers = _summarize_layers(checks)
    return {
        "artifact_dir": str(root),
        "compare_dir": str(compare_root) if compare_root is not None else "",
        "optimization_ready": _layer_required_checks_pass(layers, ["backtest_engine", "data", "factor", "portfolio"]),
        "layers": layers,
        "checks": [check.to_dict() for check in checks],
    }


def write_quant_diagnostic_report(report: dict[str, Any], out_dir: str | Path = "outputs") -> dict[str, str]:
    """Write machine-readable and human-readable diagnostic artifacts."""
    output_dir = resolve_path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{REPORT_BASENAME}.json"
    md_path = output_dir / f"{REPORT_BASENAME}.md"
    json_path.write_text(json.dumps(_json_safe(report), indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_quant_diagnostic_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def render_quant_diagnostic_markdown(report: dict[str, Any]) -> str:
    """Render a compact five-layer diagnostic report."""
    lines = [
        "# Quant Diagnostic Report",
        "",
        f"- Artifact dir: {report.get('artifact_dir', '')}",
        f"- Compare dir: {report.get('compare_dir', '') or 'not provided'}",
        f"- Optimization ready: {report.get('optimization_ready', False)}",
        "",
    ]
    checks = report.get("checks", [])
    for layer, title in [
        ("backtest_engine", "1. Backtest Engine"),
        ("data", "2. Data"),
        ("factor", "3. Factor"),
        ("portfolio", "4. Portfolio"),
        ("optimization", "5. Optimization Readiness"),
    ]:
        layer_checks = [check for check in checks if check.get("layer") == layer]
        summary = report.get("layers", {}).get(layer, {})
        lines.extend(
            [
                f"## {title}",
                "",
                f"- Status: {summary.get('status', STATUS_WARN)}",
                f"- Passed/failed/warned: {summary.get('pass', 0)}/{summary.get('fail', 0)}/{summary.get('warn', 0)}",
                "",
            ]
        )
        for check in layer_checks:
            lines.append(f"- [{str(check.get('status', '')).upper()}] {check.get('name', '')}: {check.get('summary', '')}")
            caveats = check.get("caveats", [])
            if caveats:
                lines.append(f"  Caveat: {'; '.join(map(str, caveats[:3]))}")
        lines.append("")
    return "\n".join(lines)


def _backtest_engine_checks(root: Path, compare_root: Path | None, *, tolerance: float) -> list[DiagnosticCheck]:
    checks: list[DiagnosticCheck] = []
    missing = [name for name in BACKTEST_REQUIRED if not (root / name).exists()]
    if missing:
        checks.append(
            DiagnosticCheck(
                "backtest_engine",
                "required_artifacts",
                STATUS_FAIL,
                "Backtest engine artifacts are missing.",
                {"missing": missing},
                ["Run scripts/run_backtest.py or scripts/run_auto_signal.py before trusting performance diagnostics."],
            )
        )
        return checks
    checks.append(
        DiagnosticCheck(
            "backtest_engine",
            "required_artifacts",
            STATUS_PASS,
            "Backtest equity, holdings, trades, and metrics artifacts are present.",
        )
    )
    checks.append(_reproducibility_check(root, compare_root))
    checks.append(_cost_invariant_check(root, tolerance=tolerance))
    checks.append(_cash_equity_invariant_check(root, tolerance=tolerance))
    checks.append(_holding_roll_forward_check(root))
    return checks


def _reproducibility_check(root: Path, compare_root: Path | None) -> DiagnosticCheck:
    if compare_root is None:
        return DiagnosticCheck(
            "backtest_engine",
            "same_input_reproducibility",
            STATUS_WARN,
            "No compare directory was provided, so same-input reproducibility was not checked.",
            caveats=["Pass --compare-dir with a second run's artifact directory."],
        )
    mismatched: list[str] = []
    missing: list[str] = []
    for name in REPRODUCIBILITY_FILES:
        left = root / name
        right = compare_root / name
        if not left.exists() or not right.exists():
            missing.append(name)
            continue
        if _file_digest(left) != _file_digest(right):
            mismatched.append(name)
    if mismatched:
        return DiagnosticCheck(
            "backtest_engine",
            "same_input_reproducibility",
            STATUS_FAIL,
            "Repeated run artifacts differ.",
            {"mismatched": mismatched, "missing": missing},
        )
    if missing:
        return DiagnosticCheck(
            "backtest_engine",
            "same_input_reproducibility",
            STATUS_WARN,
            "Comparable artifacts were incomplete, but all available files matched.",
            {"missing": missing},
        )
    return DiagnosticCheck(
        "backtest_engine",
        "same_input_reproducibility",
        STATUS_PASS,
        "Repeated run artifacts match exactly.",
    )


def _cost_invariant_check(root: Path, *, tolerance: float) -> DiagnosticCheck:
    trades = _read_csv(root / "backtest_trades.csv")
    metrics = _read_json(root / "backtest_metrics.json")
    missing_columns = [column for column in COST_COLUMNS if column not in trades.columns]
    if missing_columns:
        return DiagnosticCheck(
            "backtest_engine",
            "trade_cost_invariant",
            STATUS_FAIL,
            "Trade cost columns are missing from the trades artifact.",
            {"missing_columns": missing_columns},
        )
    trade_cost = float(sum(pd.to_numeric(trades[column], errors="coerce").fillna(0.0).sum() for column in COST_COLUMNS))
    metric_cost = _number_or_none(metrics.get("trade_cost"))
    if metric_cost is None:
        return DiagnosticCheck(
            "backtest_engine",
            "trade_cost_invariant",
            STATUS_WARN,
            "Trade costs can be summed from trades, but metrics.trade_cost is missing.",
            {"trade_cost_sum": trade_cost},
        )
    diff = abs(trade_cost - metric_cost)
    return DiagnosticCheck(
        "backtest_engine",
        "trade_cost_invariant",
        STATUS_PASS if diff <= tolerance else STATUS_FAIL,
        "Trade cost metric matches the sum of commission, tax, transfer fee, and slippage costs."
        if diff <= tolerance
        else "Trade cost metric does not match the sum of cost columns.",
        {"trade_cost_sum": trade_cost, "metric_trade_cost": metric_cost, "diff": diff},
    )


def _cash_equity_invariant_check(root: Path, *, tolerance: float) -> DiagnosticCheck:
    equity = _read_equity(root / "backtest_equity.csv")
    holdings = _read_csv(root / "backtest_holdings.csv")
    if equity.empty:
        return DiagnosticCheck("backtest_engine", "cash_equity_invariant", STATUS_FAIL, "Equity curve is empty.")
    if holdings.empty:
        return DiagnosticCheck(
            "backtest_engine",
            "cash_equity_invariant",
            STATUS_WARN,
            "Holdings are empty; cash/equity invariant can only confirm the equity curve exists.",
        )
    required = {"date", "value"}
    if not required.issubset(holdings.columns):
        return DiagnosticCheck(
            "backtest_engine",
            "cash_equity_invariant",
            STATUS_FAIL,
            "Holdings artifact is missing date or value columns.",
            {"missing_columns": sorted(required - set(holdings.columns))},
        )
    values = holdings.copy()
    values["date"] = pd.to_datetime(values["date"], errors="coerce").dt.normalize()
    values["value"] = pd.to_numeric(values["value"], errors="coerce").fillna(0.0)
    holding_value = values.groupby("date")["value"].sum()
    aligned = pd.concat([equity.rename("equity"), holding_value.rename("holding_value")], axis=1).fillna({"holding_value": 0.0})
    aligned["implied_cash"] = aligned["equity"] - aligned["holding_value"]
    min_cash = _number_or_none(aligned["implied_cash"].min())
    status = STATUS_PASS if min_cash is not None and min_cash >= -abs(tolerance) else STATUS_FAIL
    return DiagnosticCheck(
        "backtest_engine",
        "cash_equity_invariant",
        status,
        "Implied cash is non-negative across holding dates." if status == STATUS_PASS else "Holdings value exceeds equity on at least one date.",
        {"min_implied_cash": min_cash, "date_count": int(len(aligned))},
    )


def _holding_roll_forward_check(root: Path) -> DiagnosticCheck:
    trades = _read_csv(root / "backtest_trades.csv")
    holdings = _read_csv(root / "backtest_holdings.csv")
    if holdings.empty:
        return DiagnosticCheck(
            "backtest_engine",
            "holding_roll_forward",
            STATUS_WARN,
            "Holdings are empty; roll-forward could not be checked.",
        )
    required_holdings = {"instrument", "shares"}
    required_trades = {"instrument", "shares", "side", "status"}
    if not required_holdings.issubset(holdings.columns) or not trades.empty and not required_trades.issubset(trades.columns):
        return DiagnosticCheck(
            "backtest_engine",
            "holding_roll_forward",
            STATUS_FAIL,
            "Holdings or trades artifact is missing roll-forward columns.",
        )
    expected = _expected_final_shares(trades)
    actual = holdings.copy()
    if "date" in actual.columns:
        actual["_date"] = pd.to_datetime(actual["date"], errors="coerce")
        latest = actual["_date"].max()
        actual = actual[actual["_date"] == latest]
    actual_shares = pd.to_numeric(actual["shares"], errors="coerce").fillna(0.0).groupby(actual["instrument"].astype(str)).sum()
    combined = pd.concat([expected.rename("expected"), actual_shares.rename("actual")], axis=1).fillna(0.0)
    diff = (combined["expected"] - combined["actual"]).abs()
    mismatches = combined[diff > 0.0]
    status = STATUS_PASS if mismatches.empty else STATUS_FAIL
    return DiagnosticCheck(
        "backtest_engine",
        "holding_roll_forward",
        status,
        "Final holdings match executed buy/sell roll-forward from a zero initial position."
        if status == STATUS_PASS
        else "Final holdings do not match executed buy/sell roll-forward.",
        {"mismatched_instruments": mismatches.head(10).to_dict(orient="index"), "instrument_count": int(len(combined))},
    )


def _data_checks(root: Path) -> list[DiagnosticCheck]:
    health = _read_json(root / "data_health_report.json")
    governance = _read_json(root / "data_governance_report.json")
    checks = [
        _json_gate_check(
            "data",
            "data_health",
            health,
            "is_healthy",
            "Data health report is healthy.",
            "Data health report has blocking issues.",
            "Data health report is missing.",
        ),
        _json_gate_check(
            "data",
            "point_in_time_governance",
            governance,
            "is_point_in_time_ready",
            "Point-in-time governance report is ready.",
            "Point-in-time governance report has blocking issues.",
            "Data governance report is missing.",
        ),
    ]
    checks.append(_data_evidence_check(governance, "adjustment_readiness", ["adj_factor_meta_available"], "Adjustment metadata evidence is present."))
    checks.append(_data_evidence_check(governance, "st_suspension_limit_readiness", ["st_filter_mode"], "ST/suspension/limit-rule evidence is present."))
    checks.append(
        _data_evidence_check(
            governance,
            "survivorship_readiness",
            ["historical_universe_available", "index_constituents_available"],
            "Historical universe and index constituent evidence are present.",
        )
    )
    return checks


def _factor_checks(root: Path) -> list[DiagnosticCheck]:
    checks: list[DiagnosticCheck] = []
    ic_summary = _read_optional_csv(root / "factor_ic_summary.csv")
    if ic_summary is None:
        checks.append(
            DiagnosticCheck(
                "factor",
                "ic_summary",
                STATUS_WARN,
                "Factor IC summary is missing.",
                caveats=["Run scripts/run_optimize.py or another IC-producing workflow to create outputs/factor_ic_summary.csv."],
            )
        )
    else:
        required = {"mean_ic", "ic_ir", "positive_ratio", "count"}
        missing = sorted(required - set(ic_summary.columns))
        status = STATUS_FAIL if missing else STATUS_PASS
        checks.append(
            DiagnosticCheck(
                "factor",
                "ic_summary",
                status,
                "Factor IC summary has the expected columns." if status == STATUS_PASS else "Factor IC summary is missing expected columns.",
                {"rows": int(len(ic_summary)), "missing_columns": missing},
            )
        )
    checks.append(_artifact_presence_check(root, "factor", "yearly_ic_stability", ["factor_ic_yearly.csv"], "Yearly IC stability evidence is present."))
    checks.append(_artifact_presence_check(root, "factor", "group_return_spread", ["factor_group_returns.csv", "factor_quantile_returns.csv"], "Factor group-return evidence is present."))
    research = _read_json(root / "auto_research_diagnostics.json")
    exposure = research.get("exposure", {}) if isinstance(research, dict) else {}
    exposure_ready = isinstance(exposure, dict) and bool(exposure.get("enabled"))
    checks.append(
        DiagnosticCheck(
            "factor",
            "industry_market_cap_exposure",
            STATUS_PASS if exposure_ready else STATUS_WARN,
            "Industry and market-cap exposure evidence is present."
            if exposure_ready
            else "Industry and market-cap exposure evidence is missing.",
            {"exposure": exposure if exposure_ready else {}},
        )
    )
    return checks


def _portfolio_checks(root: Path) -> list[DiagnosticCheck]:
    checks = [
        _artifact_presence_check(root, "portfolio", "yearly_failures", ["backtest_yearly.csv", "auto_backtest_yearly_breakdown.csv"], "Yearly failure evidence is present."),
        _research_section_check(root, "turnover", "turnover_attribution", "Turnover attribution evidence is present."),
        _research_section_check(root, "cost_drag", "cost_attribution", "Cost attribution evidence is present."),
        _research_section_check(root, "drawdown_source", "drawdown", "Drawdown source evidence is present."),
        _research_section_check(root, "concentration_exposure", "exposure", "Concentration and exposure evidence is present."),
    ]
    failure = _read_json(root / "auto_failure_analysis.json")
    checks.append(
        DiagnosticCheck(
            "portfolio",
            "failure_scope",
            STATUS_PASS if bool(failure.get("enabled")) else STATUS_WARN,
            "Failure-scope analysis evidence is present." if bool(failure.get("enabled")) else "Failure-scope analysis evidence is missing.",
            {"primary_failure_area": failure.get("primary_failure_area"), "failure_scope_summary": failure.get("failure_scope_summary")},
        )
    )
    return checks


def _optimization_checks(checks: list[DiagnosticCheck]) -> list[DiagnosticCheck]:
    layers = _summarize_layers(checks)
    ready = _layer_required_checks_pass(layers, ["backtest_engine", "data", "factor", "portfolio"])
    blockers = [layer for layer in ["backtest_engine", "data", "factor", "portfolio"] if layers.get(layer, {}).get("status") != STATUS_PASS]
    return [
        DiagnosticCheck(
            "optimization",
            "optimization_gate",
            STATUS_PASS if ready else STATUS_FAIL,
            "Optimization can be considered after the first four diagnostic layers passed."
            if ready
            else "Optimization is blocked until earlier diagnostic layers pass.",
            {"blocking_layers": blockers},
        )
    ]


def _json_gate_check(
    layer: str,
    name: str,
    payload: dict[str, Any],
    flag: str,
    pass_summary: str,
    fail_summary: str,
    missing_summary: str,
) -> DiagnosticCheck:
    if not payload:
        return DiagnosticCheck(layer, name, STATUS_WARN, missing_summary)
    status = STATUS_PASS if bool(payload.get(flag)) else STATUS_FAIL
    return DiagnosticCheck(
        layer,
        name,
        status,
        pass_summary if status == STATUS_PASS else fail_summary,
        {"issues": payload.get("issues", []), "warnings": payload.get("warnings", [])},
    )


def _data_evidence_check(payload: dict[str, Any], name: str, keys: list[str], summary: str) -> DiagnosticCheck:
    if not payload:
        return DiagnosticCheck("data", name, STATUS_WARN, "Data governance evidence is missing.")
    evidence = {key: payload.get(key) for key in keys}
    ready = all(_truthy_evidence(value) for value in evidence.values())
    return DiagnosticCheck(
        "data",
        name,
        STATUS_PASS if ready else STATUS_WARN,
        summary if ready else f"{name} evidence is incomplete.",
        evidence,
    )


def _artifact_presence_check(root: Path, layer: str, name: str, filenames: list[str], summary: str) -> DiagnosticCheck:
    present = [filename for filename in filenames if (root / filename).exists()]
    return DiagnosticCheck(
        layer,
        name,
        STATUS_PASS if present else STATUS_WARN,
        summary if present else f"{name} evidence is missing.",
        {"present": present, "expected_any": filenames},
    )


def _research_section_check(root: Path, name: str, section: str, summary: str) -> DiagnosticCheck:
    research = _read_json(root / "auto_research_diagnostics.json")
    section_data = research.get(section, {}) if isinstance(research, dict) else {}
    ready = isinstance(section_data, dict) and bool(section_data.get("enabled", section_data))
    return DiagnosticCheck(
        "portfolio",
        name,
        STATUS_PASS if ready else STATUS_WARN,
        summary if ready else f"{name} evidence is missing.",
        {section: section_data if ready else {}},
    )


def _summarize_layers(checks: list[DiagnosticCheck]) -> dict[str, dict[str, Any]]:
    layers: dict[str, dict[str, Any]] = {}
    for check in checks:
        summary = layers.setdefault(check.layer, {"pass": 0, "warn": 0, "fail": 0, "status": STATUS_PASS})
        summary[check.status] = int(summary.get(check.status, 0)) + 1
    for summary in layers.values():
        if summary.get("fail", 0):
            summary["status"] = STATUS_FAIL
        elif summary.get("warn", 0):
            summary["status"] = STATUS_WARN
        else:
            summary["status"] = STATUS_PASS
    return layers


def _layer_required_checks_pass(layers: dict[str, dict[str, Any]], required_layers: list[str]) -> bool:
    return all(layers.get(layer, {}).get("status") == STATUS_PASS for layer in required_layers)


def _expected_final_shares(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    frame = trades.copy()
    status = frame["status"].astype(str).str.strip().str.lower()
    frame = frame[status.isin(EXECUTED_STATUSES)].copy()
    if frame.empty:
        return pd.Series(dtype=float)
    side = frame["side"].astype(str).str.strip().str.upper()
    shares = pd.to_numeric(frame["shares"], errors="coerce").fillna(0.0)
    signed = np.where(side == "BUY", shares, np.where(side == "SELL", -shares, 0.0))
    return pd.Series(signed, index=frame["instrument"].astype(str)).groupby(level=0).sum()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _read_equity(path: Path) -> pd.Series:
    if not path.exists():
        return pd.Series(dtype=float, name="equity")
    frame = pd.read_csv(path, index_col=0)
    if frame.empty:
        return pd.Series(dtype=float, name="equity")
    column = "equity" if "equity" in frame.columns else frame.columns[0]
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    series.index = pd.to_datetime(series.index, errors="coerce").normalize()
    return series[~series.index.isna()].sort_index().rename("equity")


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number_or_none(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _truthy_evidence(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return bool(value)


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
