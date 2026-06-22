"""Read-only dashboard view model for latest auto-signal artifacts."""

from __future__ import annotations

from collections.abc import Mapping
import csv
import json
from pathlib import Path
from typing import Any

from src.config_loader import load_config, resolve_path


DASHBOARD_VERSION = 1
CSV_PREVIEW_LIMIT = 50


def build_dashboard_snapshot(out_dir: str | Path | None = None, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build a compact latest-run dashboard model from local output artifacts."""
    output_dir = _output_dir(out_dir, config)
    report_artifact = _read_json_artifact(output_dir / "auto_signal_report.json")
    status_artifact = _read_json_artifact(output_dir / "auto_run_status.json")

    report = report_artifact.get("data") if report_artifact.get("valid") else None
    status = status_artifact.get("data") if status_artifact.get("valid") else None
    if not isinstance(report, Mapping):
        report = None
    if not isinstance(status, Mapping):
        status = None

    artifacts = _build_artifacts(output_dir, report)
    manual_orders_artifact = next((item for item in artifacts if item["id"] == "manual_orders"), None)
    orders = _build_orders(manual_orders_artifact)
    block_reasons = _string_list((report or status or {}).get("block_reasons"))
    quality_warnings = _string_list((report or status or {}).get("quality_warnings"))

    errors = [
        item["error"]
        for item in [report_artifact, status_artifact]
        if item.get("exists") and not item.get("valid") and item.get("error")
    ]
    return {
        "version": DASHBOARD_VERSION,
        "output_dir": str(output_dir),
        "readiness": _build_readiness(report, report_artifact, block_reasons),
        "latest_run": _build_latest_run(report, status),
        "gates": _build_gates(report, status, output_dir),
        "block_reasons": block_reasons,
        "quality_warnings": quality_warnings,
        "signal_summary": _mapping_value((report or {}).get("signal_summary")),
        "orders": orders,
        "artifacts": artifacts,
        "report": _build_report_section(output_dir, artifacts, report, status),
        "errors": errors,
    }


def resolve_dashboard_artifact(artifact_id: str, out_dir: str | Path | None = None) -> Path | None:
    """Resolve a downloadable dashboard artifact by id, constrained to the output directory."""
    snapshot = build_dashboard_snapshot(out_dir=out_dir)
    output_dir = Path(snapshot["output_dir"]).resolve()
    for artifact in snapshot["artifacts"]:
        if artifact.get("id") != artifact_id or not artifact.get("downloadable"):
            continue
        path = Path(str(artifact.get("path", ""))).resolve()
        if _is_relative_to(path, output_dir):
            return path
    return None


def _output_dir(out_dir: str | Path | None, config: Mapping[str, Any] | None) -> Path:
    if out_dir is not None:
        return resolve_path(out_dir)
    cfg = dict(config) if config is not None else load_config()
    return resolve_path(_mapping_value(cfg.get("outputs")).get("dir", "outputs"))


def _build_readiness(
    report: Mapping[str, Any] | None,
    report_artifact: Mapping[str, Any],
    block_reasons: list[str],
) -> dict[str, Any]:
    if not report_artifact.get("exists"):
        return {
            "status": "missing",
            "label": "Missing latest report",
            "summary": "Run the auto-signal workflow to create outputs/auto_signal_report.json.",
            "is_executable": None,
        }
    if not report_artifact.get("valid") or report is None:
        return {
            "status": "error",
            "label": "Report unreadable",
            "summary": str(report_artifact.get("error") or "auto_signal_report.json is malformed."),
            "is_executable": None,
        }
    if bool(report.get("is_executable")):
        return {
            "status": "ready",
            "label": "Ready for manual review",
            "summary": "All required gates allowed official review for the latest signal.",
            "is_executable": True,
        }

    non_hold_reasons = [reason for reason in block_reasons if reason != "candidate_only_requested"]
    if block_reasons and not non_hold_reasons:
        return {
            "status": "candidate_only",
            "label": "Candidate-only hold",
            "summary": "The run intentionally produced candidate artifacts only.",
            "is_executable": False,
        }
    return {
        "status": "blocked",
        "label": "Blocked",
        "summary": f"{len(block_reasons)} blocker(s) require attention before manual trading review.",
        "is_executable": False,
    }


def _build_latest_run(report: Mapping[str, Any] | None, status: Mapping[str, Any] | None) -> dict[str, Any]:
    source = report or status or {}
    target_resolution = _mapping_value(source.get("target_date_resolution"))
    return {
        "generated_at": source.get("generated_at") or source.get("completed_at") or source.get("started_at"),
        "status": source.get("status") or (status or {}).get("status"),
        "strategy_mode": source.get("strategy_mode") or (status or {}).get("strategy_mode"),
        "signal_date": source.get("signal_date"),
        "intended_trade_date": source.get("intended_trade_date"),
        "requested_date": target_resolution.get("requested"),
        "target_date": target_resolution.get("target_date"),
        "latest_trade_date": target_resolution.get("latest_trade_date"),
        "latest_stage": _latest_stage(status),
    }


def _build_gates(
    report: Mapping[str, Any] | None,
    status: Mapping[str, Any] | None,
    output_dir: Path,
) -> list[dict[str, Any]]:
    report = report or {}
    status = status or {}
    block_reasons = _string_list(report.get("block_reasons") or status.get("block_reasons"))
    data_health = _mapping_value(report.get("data_health")) or _read_json_data(output_dir / "data_health_report.json")
    data_governance = _mapping_value(report.get("data_governance")) or _read_json_data(output_dir / "data_governance_report.json")
    parameter_quality = _mapping_value(report.get("parameter_quality")) or _read_json_data(output_dir / "auto_parameter_quality.json")
    backtest_quality = _mapping_value(report.get("backtest_quality")) or _read_json_data(output_dir / "auto_backtest_quality.json")
    account = _mapping_value(report.get("account"))

    return [
        _health_gate("data_health", "Data health", data_health, "is_healthy"),
        _governance_gate(data_governance),
        _quality_gate("parameter_quality", "Parameter quality", parameter_quality),
        _quality_gate("backtest_quality", "Backtest quality", backtest_quality),
        _account_gate(account, block_reasons),
        _candidate_gate(report, block_reasons),
    ]


def _health_gate(gate_id: str, label: str, payload: Mapping[str, Any], bool_field: str) -> dict[str, Any]:
    if not payload:
        return _gate(gate_id, label, "missing", "No artifact found.", [])
    issues = _string_list(payload.get("issues"))
    status = "pass" if bool(payload.get(bool_field)) else "fail"
    summary = "Healthy." if status == "pass" else _issue_summary(issues, "Gate failed.")
    details = {
        "raw_latest_coverage": payload.get("raw_latest_target_coverage"),
        "price_latest_coverage": payload.get("price_latest_target_coverage"),
        "factor_latest_coverage": payload.get("factor_latest_target_coverage"),
        "requested_end_date": payload.get("requested_end_date"),
    }
    return _gate(gate_id, label, status, summary, issues, details)


def _governance_gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not payload:
        return _gate("data_governance", "Data governance", "missing", "No artifact found.", [])
    issues = _string_list(payload.get("issues"))
    warnings = _string_list(payload.get("warnings"))
    if issues or payload.get("is_point_in_time_ready") is False:
        status = "fail"
        summary = _issue_summary(issues, "Point-in-time governance failed.")
    elif warnings:
        status = "warn"
        summary = _issue_summary(warnings, "Ready with warnings.")
    else:
        status = "pass"
        summary = "Point-in-time inputs are ready."
    return _gate("data_governance", "Data governance", status, summary, issues or warnings)


def _quality_gate(gate_id: str, label: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if not payload:
        return _gate(gate_id, label, "missing", "No artifact found.", [])
    issues = _string_list(payload.get("issues"))
    status = "pass" if bool(payload.get("is_acceptable")) else "fail"
    details = {
        "annual_return": payload.get("annual_return") or payload.get("annual_return_mean"),
        "max_drawdown": payload.get("max_drawdown") or payload.get("max_drawdown_worst"),
        "windows": payload.get("windows"),
    }
    return _gate(gate_id, label, status, "Acceptable." if status == "pass" else _issue_summary(issues, "Not acceptable."), issues, details)


def _account_gate(account: Mapping[str, Any], block_reasons: list[str]) -> dict[str, Any]:
    account_reasons = [reason for reason in block_reasons if reason.startswith("account:")]
    if account_reasons:
        return _gate("account", "Account and holdings", "fail", _issue_summary(account_reasons, "Account gate failed."), account_reasons)
    if not account:
        return _gate("account", "Account and holdings", "missing", "No account summary found.", [])
    if account.get("holdings_loaded") is False:
        return _gate("account", "Account and holdings", "warn", "Account loaded, but holdings were not loaded.", [])
    return _gate("account", "Account and holdings", "pass", "Account and holdings summary loaded.", [])


def _candidate_gate(report: Mapping[str, Any], block_reasons: list[str]) -> dict[str, Any]:
    if report.get("candidate_only") or "candidate_only_requested" in block_reasons:
        return _gate("candidate_only", "Candidate-only mode", "hold", "Candidate-only output was requested.", [])
    return _gate("candidate_only", "Candidate-only mode", "pass", "Candidate-only hold is not active.", [])


def _gate(
    gate_id: str,
    label: str,
    status: str,
    summary: str,
    issues: list[str],
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": gate_id,
        "label": label,
        "status": status,
        "summary": summary,
        "issues": issues,
        "details": dict(details or {}),
    }


def _build_orders(artifact: Mapping[str, Any] | None) -> dict[str, Any]:
    if not artifact or not artifact.get("path"):
        return {
            "path": "",
            "exists": False,
            "valid": False,
            "columns": [],
            "rows": [],
            "total_rows": 0,
            "preview_limit": CSV_PREVIEW_LIMIT,
            "action_counts": {},
            "actionable_count": 0,
            "error": "manual_orders artifact was not referenced by the latest report.",
        }
    preview = _read_csv_preview(Path(str(artifact["path"])), CSV_PREVIEW_LIMIT)
    action_counts: dict[str, int] = {}
    actionable_count = 0
    for row in preview["rows"]:
        action = str(row.get("action", "")).strip() or "UNKNOWN"
        action_counts[action] = action_counts.get(action, 0) + 1
        if _truthy(row.get("is_order_actionable")):
            actionable_count += 1
    preview["action_counts"] = action_counts
    preview["actionable_count"] = actionable_count
    return preview


def _build_artifacts(output_dir: Path, report: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    report_files = _mapping_value((report or {}).get("files"))
    candidates: list[tuple[str, str, str, Path]] = [
        ("auto_signal_report", "Auto signal JSON report", "json", output_dir / "auto_signal_report.json"),
        ("auto_run_status", "Auto run status", "json", output_dir / "auto_run_status.json"),
        ("daily_report", "Daily Markdown report", "markdown", output_dir / "daily_signal_report.md"),
    ]
    for artifact_id, label, kind in [
        ("signal", "Signal CSV", "csv"),
        ("holdings", "Holdings CSV", "csv"),
        ("manual_orders", "Manual orders CSV", "csv"),
        ("order_confirmation", "Order confirmation CSV", "csv"),
        ("fill_feedback", "Fill feedback CSV", "csv"),
        ("data_health", "Data health JSON", "json"),
        ("data_governance", "Data governance JSON", "json"),
        ("parameter_quality", "Parameter quality JSON", "json"),
        ("backtest_quality", "Backtest quality JSON", "json"),
        ("fundamental_screen_report", "Fundamental screen report", "markdown"),
    ]:
        value = report_files.get(artifact_id)
        if value:
            candidates.append((artifact_id, label, kind, resolve_path(str(value))))

    seen: set[str] = set()
    artifacts: list[dict[str, Any]] = []
    output_root = output_dir.resolve()
    for artifact_id, label, kind, path in candidates:
        if artifact_id in seen:
            continue
        seen.add(artifact_id)
        resolved = path.resolve()
        exists = resolved.exists()
        artifacts.append(
            {
                "id": artifact_id,
                "label": label,
                "kind": kind,
                "path": str(resolved),
                "exists": exists,
                "downloadable": exists and _is_relative_to(resolved, output_root),
            }
        )
    return artifacts


def _build_report_section(
    output_dir: Path,
    artifacts: list[Mapping[str, Any]],
    report: Mapping[str, Any] | None,
    status: Mapping[str, Any] | None,
) -> dict[str, Any]:
    daily_report = next((item for item in artifacts if item["id"] == "daily_report"), None)
    source = report or status or {}
    return {
        "mode": "structured_summary_with_artifact_link",
        "daily_markdown": daily_report or {
            "id": "daily_report",
            "label": "Daily Markdown report",
            "kind": "markdown",
            "path": str((output_dir / "daily_signal_report.md").resolve()),
            "exists": False,
            "downloadable": False,
        },
        "summary": [
            {"label": "Strategy mode", "value": source.get("strategy_mode") or ""},
            {"label": "Signal date", "value": source.get("signal_date") or ""},
            {"label": "Intended trade date", "value": source.get("intended_trade_date") or ""},
        ],
    }


def _read_json_artifact(path: Path) -> dict[str, Any]:
    artifact = {"path": str(path.resolve()), "exists": path.exists(), "valid": False, "error": None, "data": None}
    if not path.exists():
        artifact["error"] = f"File not found: {path}"
        return artifact
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        artifact["error"] = f"Failed to read JSON {path}: {exc}"
        return artifact
    if not isinstance(data, Mapping):
        artifact["error"] = f"JSON artifact must be an object: {path}"
        return artifact
    artifact["valid"] = True
    artifact["data"] = dict(data)
    return artifact


def _read_json_data(path: Path) -> dict[str, Any]:
    artifact = _read_json_artifact(path)
    data = artifact.get("data")
    return dict(data) if artifact.get("valid") and isinstance(data, Mapping) else {}


def _read_csv_preview(path: Path, limit: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "exists": path.exists(),
        "valid": False,
        "columns": [],
        "rows": [],
        "total_rows": 0,
        "preview_limit": limit,
        "error": None,
    }
    if not path.exists():
        result["error"] = f"File not found: {path}"
        return result
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            result["columns"] = list(reader.fieldnames or [])
            rows: list[dict[str, str]] = []
            total = 0
            for row in reader:
                if total < limit:
                    rows.append({key: "" if value is None else str(value) for key, value in row.items()})
                total += 1
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        result["error"] = f"Failed to read CSV {path}: {exc}"
        return result
    result["valid"] = True
    result["rows"] = rows
    result["total_rows"] = total
    return result


def _latest_stage(status: Mapping[str, Any] | None) -> dict[str, Any] | None:
    stages = _list_value((status or {}).get("stages"))
    for stage in reversed(stages):
        if isinstance(stage, Mapping):
            return dict(stage)
    return None


def _issue_summary(issues: list[str], fallback: str) -> str:
    if not issues:
        return fallback
    first = issues[0]
    suffix = f" (+{len(issues) - 1} more)" if len(issues) > 1 else ""
    return first + suffix


def _mapping_value(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True

