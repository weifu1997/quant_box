"""Read-only dashboard view model for latest auto-signal artifacts."""

from __future__ import annotations

from collections.abc import Mapping
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re
from typing import Any

from src.common import normalize_instrument
from src.config_loader import load_config, resolve_path
from src.dashboard_stock import load_instrument_name_map
from src.manual_orders import load_account_state, load_current_holdings, validate_account_inputs
from src.trading_calendar import resolve_target_date


DASHBOARD_VERSION = 1
CSV_PREVIEW_LIMIT = 50


def build_dashboard_snapshot(out_dir: str | Path | None = None, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build a compact latest-run dashboard model from local output artifacts."""
    output_dir = _output_dir(out_dir, config)
    report_artifact = _read_json_artifact(output_dir / "auto_signal_report.json")
    status_artifact = _read_json_artifact(output_dir / "auto_run_status.json")
    governance_artifact = _read_json_artifact(output_dir / "data_governance_report.json")

    report = report_artifact.get("data") if report_artifact.get("valid") else None
    status = status_artifact.get("data") if status_artifact.get("valid") else None
    if not isinstance(report, Mapping):
        report = None
    if not isinstance(status, Mapping):
        status = None

    artifacts = _build_artifacts(output_dir, report)
    manual_orders_artifact = next((item for item in artifacts if item["id"] == "manual_orders"), None)
    orders = _build_orders(manual_orders_artifact, config=config)
    governance_context = _post_run_governance_context(report, _artifact_data(governance_artifact))
    block_reasons = _filter_resolved_governance_reasons(
        _string_list((report or status or {}).get("block_reasons")),
        governance_context,
    )
    quality_warnings = _filter_resolved_governance_reasons(
        _string_list((report or status or {}).get("quality_warnings")),
        governance_context,
    )
    freshness_notes = _freshness_notes(governance_context)

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
        "gates": _build_gates(report, status, output_dir, governance_artifact, governance_context),
        "block_reasons": block_reasons,
        "blocker_actions": _build_blocker_actions(block_reasons, freshness_notes, report, artifacts),
        "quality_warnings": quality_warnings,
        "freshness_notes": freshness_notes,
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


def build_dashboard_precheck(out_dir: str | Path | None = None, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build a read-only pre-run check model before starting auto-signal."""
    cfg = dict(config) if config is not None else load_config()
    output_dir = _output_dir(out_dir, cfg)
    target_item, target_resolution = _precheck_target_date(cfg)
    health = _read_json_data(output_dir / "data_health_report.json")
    governance = _read_json_data(output_dir / "data_governance_report.json")
    data_update_progress = _read_json_data(output_dir / "data_update_progress.json")
    configured_min_factor_coverage = _optional_float(_mapping_value(cfg.get("quality")).get("min_factor_coverage"))
    items = [
        target_item,
        _precheck_data_health(health, target_resolution),
        _precheck_governance(governance, target_resolution),
        _precheck_factor_freshness(
            health,
            governance,
            target_resolution,
            data_update_progress,
            configured_min_factor_coverage=configured_min_factor_coverage,
        ),
        _precheck_account(cfg),
    ]
    fail_count = sum(1 for item in items if item["status"] == "fail")
    missing_count = sum(1 for item in items if item["status"] == "missing")
    warn_count = sum(1 for item in items if item["status"] == "warn")
    if fail_count:
        status = "fail"
        summary = f"发现 {fail_count} 项阻塞，建议先修复后再重跑自动信号。"
    elif missing_count or warn_count:
        status = "warn"
        summary = f"有 {missing_count + warn_count} 项需要确认；可以运行，但不建议直接依赖正式输出。"
    else:
        status = "pass"
        summary = "运行前检查通过，可以重跑自动信号。"
    return {
        "version": DASHBOARD_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "summary": summary,
        "can_run_normal": status == "pass",
        "target_date_resolution": target_resolution,
        "items": items,
    }


def _output_dir(out_dir: str | Path | None, config: Mapping[str, Any] | None) -> Path:
    if out_dir is not None:
        return resolve_path(out_dir)
    cfg = dict(config) if config is not None else load_config()
    return resolve_path(_mapping_value(cfg.get("outputs")).get("dir", "outputs"))


def _precheck_target_date(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        resolution = resolve_target_date(config.get("data", {}).get("end_date"), config=dict(config)).to_dict()
    except Exception as exc:
        return (
            _precheck_item(
                "target_date",
                "目标交易日",
                "fail",
                f"无法解析自动信号目标日期：{exc}",
                [f"target_date_resolution_failed:{exc}"],
            ),
            {},
        )
    warnings = _string_list(resolution.get("calendar_warnings"))
    status = "warn" if warnings else "pass"
    summary = (
        f"目标日期 {resolution.get('target_date')}，日历来源 {resolution.get('calendar_source')}。"
        if status == "pass"
        else f"目标日期 {resolution.get('target_date')} 已解析，但日历存在降级：{warnings[0]}"
    )
    return (
        _precheck_item(
            "target_date",
            "目标交易日",
            status,
            summary,
            warnings,
            {
                "target_date": resolution.get("target_date"),
                "latest_trade_date": resolution.get("latest_trade_date"),
                "calendar_source": resolution.get("calendar_source"),
                "reason": resolution.get("reason"),
            },
        ),
        resolution,
    )


def _precheck_data_health(health: Mapping[str, Any], target_resolution: Mapping[str, Any]) -> dict[str, Any]:
    if not health:
        return _precheck_item("data_health", "数据健康证据", "missing", "未找到 data_health_report.json，无法提前确认数据健康。")
    issues = _string_list(health.get("issues"))
    if issues or not bool(health.get("is_healthy")):
        summary = _issue_summary(issues, "数据健康检查未通过。")
        return _precheck_item("data_health", "数据健康证据", "fail", summary, issues, _data_health_details(health))
    stale = _artifact_target_stale(health.get("requested_end_date"), target_resolution.get("target_date"))
    if stale:
        return _precheck_item(
            "data_health",
            "数据健康证据",
            "warn",
            "数据健康报告的目标日期不是当前目标日期，建议重跑后再确认。",
            [stale],
            _data_health_details(health),
        )
    return _precheck_item("data_health", "数据健康证据", "pass", "原始数据、价格面板和因子覆盖最近一次检查通过。", [], _data_health_details(health))


def _precheck_governance(governance: Mapping[str, Any], target_resolution: Mapping[str, Any]) -> dict[str, Any]:
    if not governance:
        return _precheck_item("data_governance", "点时治理证据", "missing", "未找到 data_governance_report.json，无法提前确认点时数据。")
    issues = _string_list(governance.get("issues"))
    warnings = _string_list(governance.get("warnings"))
    stale = _artifact_target_stale(governance.get("factor_cache_meta_end_date"), target_resolution.get("target_date"))
    if issues or governance.get("is_point_in_time_ready") is False:
        status = "fail"
        summary = _issue_summary(issues, "点时治理检查未通过。")
    elif stale:
        status = "warn"
        summary = "点时治理报告的因子截止日期早于当前目标日期。"
        warnings = [stale, *warnings]
    elif warnings:
        status = "warn"
        summary = _issue_summary(warnings, "点时治理有提示。")
    else:
        status = "pass"
        summary = "daily_basic、ST 日历和点时治理证据可用。"
    return _precheck_item("data_governance", "点时治理证据", status, summary, issues or warnings, _governance_precheck_details(governance))


def _precheck_factor_freshness(
    health: Mapping[str, Any],
    governance: Mapping[str, Any],
    target_resolution: Mapping[str, Any],
    data_update_progress: Mapping[str, Any] | None = None,
    *,
    configured_min_factor_coverage: float | None = None,
) -> dict[str, Any]:
    target_date = str(target_resolution.get("target_date") or "")
    if health:
        issues = [issue for issue in _string_list(health.get("issues")) if issue.startswith("factor_")]
        coverage = _optional_float(health.get("factor_latest_target_coverage"))
        evidence_threshold = _optional_float(health.get("min_factor_coverage"))
        threshold = configured_min_factor_coverage
        if threshold is None:
            threshold = 1.0 if evidence_threshold is None else evidence_threshold
        target_symbols = _optional_int(health.get("target_symbols"))
        current_symbols = _optional_int(health.get("factor_latest_target_symbols"))
        progress = _mapping_value(data_update_progress)
        progress_is_current = bool(target_date) and str(progress.get("target_end_date") or "") == target_date
        progress_status = str(progress.get("status") or "") if progress_is_current else ""
        confirmed_symbols = (
            _optional_int(progress.get("confirmed_no_new_data_symbols"))
            if progress_is_current and progress_status == "complete"
            else None
        )
        unconfirmed_symbols = _optional_int(progress.get("remaining_unconfirmed_symbols")) if progress_is_current else None
        details = {
            "factor_latest_date": health.get("factor_latest_date"),
            "factor_latest_coverage": coverage,
            "factor_latest_target_symbols": current_symbols,
            "target_symbols": target_symbols,
            "min_factor_coverage": threshold,
            "evidence_min_factor_coverage": evidence_threshold,
            "requested_end_date": health.get("requested_end_date"),
            "data_update_progress_status": progress_status or None,
            "data_update_progress_target_date": progress.get("target_end_date") if progress_is_current else None,
            "confirmed_no_new_data_symbols": confirmed_symbols,
            "remaining_unconfirmed_symbols": unconfirmed_symbols,
        }
        if issues:
            return _precheck_item("factor_freshness", "因子新鲜度", "fail", _issue_summary(issues, "因子新鲜度不足。"), issues, details)
        if coverage is not None and coverage < threshold:
            coverage_issue = f"factor_latest_coverage_below_threshold:{coverage:.4f}<{threshold:.4f}"
            return _precheck_item(
                "factor_freshness",
                "因子新鲜度",
                "fail",
                f"因子目标日覆盖率 {coverage:.2%}，低于 {threshold:.2%} 门槛。",
                [coverage_issue],
                details,
            )
        if unconfirmed_symbols is not None and unconfirmed_symbols > 0:
            unconfirmed_issue = f"factor_symbols_unconfirmed:{unconfirmed_symbols}"
            return _precheck_item(
                "factor_freshness",
                "因子新鲜度",
                "warn",
                f"仍有 {unconfirmed_symbols} 只股票未确认是否存在目标日行情，建议先完成行情更新。",
                [unconfirmed_issue],
                details,
            )
        if coverage is not None:
            summary = _factor_coverage_summary(
                coverage,
                threshold,
                current_symbols=current_symbols,
                target_symbols=target_symbols,
                confirmed_symbols=confirmed_symbols,
                unconfirmed_symbols=unconfirmed_symbols,
            )
            return _precheck_item("factor_freshness", "因子新鲜度", "pass", summary, [], details)
        stale = _artifact_target_stale(health.get("factor_latest_date"), target_date)
        if stale:
            return _precheck_item("factor_freshness", "因子新鲜度", "warn", "因子最新日期早于当前目标日期。", [stale], details)
        return _precheck_item("factor_freshness", "因子新鲜度", "pass", "因子缓存覆盖最近一次目标日期。", [], details)
    if governance:
        meta_end = governance.get("factor_cache_meta_end_date")
        details = {
            "factor_cache_meta_end_date": meta_end,
            "factor_cache_meta_available": governance.get("factor_cache_meta_available"),
        }
        if governance.get("factor_cache_meta_available") is False:
            return _precheck_item("factor_freshness", "因子新鲜度", "missing", "因子缓存元数据缺失，无法提前确认因子新鲜度。", [], details)
        stale = _artifact_target_stale(meta_end, target_date)
        status = "warn" if stale else "pass"
        summary = "因子元数据可用。" if status == "pass" else "因子元数据截止日期早于当前目标日期。"
        return _precheck_item("factor_freshness", "因子新鲜度", status, summary, [stale] if stale else [], details)
    return _precheck_item("factor_freshness", "因子新鲜度", "missing", "缺少数据健康或点时治理报告，无法提前确认因子新鲜度。")


def _precheck_account(config: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(config)
    try:
        account = load_account_state(cfg)
        holdings = load_current_holdings(cfg)
        issues = validate_account_inputs(account, holdings, cfg)
    except Exception as exc:
        return _precheck_item("account", "账户与持仓", "fail", f"账户或持仓读取失败：{exc}", [f"account_precheck_failed:{exc}"])
    details = {
        "account_file": account.source_file,
        "holdings_file": account.holdings_file,
        "holdings_loaded": account.holdings_loaded,
        "holdings_rows": int(len(holdings)),
        "total_asset": account.total_asset,
        "cash": account.cash,
    }
    status = "pass" if not issues else "fail"
    summary = "账户与当前持仓输入可用。" if status == "pass" else _issue_summary(issues, "账户与当前持仓输入未通过。")
    return _precheck_item("account", "账户与持仓", status, summary, issues, details)


def _precheck_item(
    item_id: str,
    label: str,
    status: str,
    summary: str,
    issues: list[str] | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    action = _precheck_action(item_id, issues or [])
    return {
        "id": item_id,
        "label": label,
        "status": status,
        "summary": summary,
        "issues": list(issues or []),
        "details": dict(details or {}),
        "action": action,
    }


def _precheck_action(item_id: str, issues: list[str]) -> dict[str, Any] | None:
    if item_id == "data_governance" and any(issue.startswith("daily_basic_") for issue in issues):
        return {"label": "修复 daily_basic", "action": "repair_point_in_time"}
    if item_id in {"data_health", "factor_freshness"} and issues:
        return {"label": "重跑自动信号", "action": "run_auto_signal", "mode": "normal"}
    return None


def _artifact_target_stale(value: Any, target_date: Any) -> str:
    if not value or not target_date:
        return ""
    try:
        observed = str(Path(str(value)).name) if isinstance(value, Path) else str(value)
        if datetime.fromisoformat(observed) < datetime.fromisoformat(str(target_date)):
            return f"artifact_before_target:{observed}<{target_date}"
    except ValueError:
        return ""
    return ""


def _data_health_details(health: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "requested_end_date": health.get("requested_end_date"),
        "raw_latest_date": health.get("raw_latest_date"),
        "price_latest_date": health.get("price_latest_date"),
        "factor_latest_date": health.get("factor_latest_date"),
        "factor_latest_target_symbols": health.get("factor_latest_target_symbols"),
        "target_symbols": health.get("target_symbols"),
        "min_factor_coverage": health.get("min_factor_coverage"),
        "raw_latest_coverage": health.get("raw_latest_target_coverage"),
        "price_latest_coverage": health.get("price_latest_target_coverage"),
        "factor_latest_coverage": health.get("factor_latest_target_coverage"),
    }


def _governance_precheck_details(governance: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "generated_at": governance.get("generated_at"),
        "daily_basic_end_date": governance.get("daily_basic_end_date"),
        "daily_basic_coverage": governance.get("daily_basic_date_coverage"),
        "st_calendar_end_date": governance.get("st_calendar_end_date"),
        "st_calendar_coverage_end_date": governance.get("st_calendar_coverage_end_date"),
        "factor_cache_meta_end_date": governance.get("factor_cache_meta_end_date"),
    }


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
    governance_artifact: Mapping[str, Any] | None = None,
    governance_context: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    report = report or {}
    status = status or {}
    block_reasons = _string_list(report.get("block_reasons") or status.get("block_reasons"))
    data_health = _mapping_value(report.get("data_health")) or _read_json_data(output_dir / "data_health_report.json")
    data_governance, governance_details = _dashboard_governance_payload(
        report,
        governance_artifact,
        governance_context,
        output_dir,
    )
    parameter_quality = _mapping_value(report.get("parameter_quality")) or _read_json_data(output_dir / "auto_parameter_quality.json")
    backtest_quality = _mapping_value(report.get("backtest_quality")) or _read_json_data(output_dir / "auto_backtest_quality.json")
    account = _mapping_value(report.get("account"))

    return [
        _health_gate("data_health", "Data health", data_health, "is_healthy"),
        _governance_gate(data_governance, governance_details),
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


def _governance_gate(payload: Mapping[str, Any], details: Mapping[str, Any] | None = None) -> dict[str, Any]:
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
    return _gate("data_governance", "Data governance", status, summary, issues or warnings, details)


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


def _dashboard_governance_payload(
    report: Mapping[str, Any],
    governance_artifact: Mapping[str, Any] | None,
    governance_context: Mapping[str, Any] | None,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    embedded = _mapping_value(report.get("data_governance"))
    current = _artifact_data(governance_artifact)
    if governance_context and current:
        return current, _governance_details(report, governance_context)
    if embedded:
        return embedded, {}
    if current:
        return current, {}
    return _read_json_data(output_dir / "data_governance_report.json"), {}


def _post_run_governance_context(
    report: Mapping[str, Any] | None,
    current_governance: Mapping[str, Any],
) -> dict[str, Any]:
    if not report or not current_governance:
        return {}
    embedded = _mapping_value(report.get("data_governance"))
    if not embedded or not _is_newer_payload(current_governance, report):
        return {}
    current_issues = set(_string_list(current_governance.get("issues")))
    embedded_issues = set(_string_list(embedded.get("issues")))
    resolved_issues = sorted(issue for issue in embedded_issues if issue not in current_issues)
    return {
        "current": dict(current_governance),
        "supersedes_auto_report": True,
        "resolved_issues": resolved_issues,
        "current_generated_at": _timestamp_text(current_governance),
        "auto_report_generated_at": _timestamp_text(report),
    }


def _filter_resolved_governance_reasons(
    reasons: list[str],
    governance_context: Mapping[str, Any] | None,
) -> list[str]:
    resolved = set(_string_list((governance_context or {}).get("resolved_issues")))
    if not resolved:
        return reasons
    filtered: list[str] = []
    for reason in reasons:
        issue = _governance_reason_issue(reason)
        if issue and issue in resolved:
            continue
        filtered.append(reason)
    return filtered


def _freshness_notes(governance_context: Mapping[str, Any] | None) -> list[str]:
    if governance_context and _list_value(governance_context.get("resolved_issues")):
        return ["data_governance_repaired_after_auto_report"]
    return []


def _build_blocker_actions(
    block_reasons: list[str],
    freshness_notes: list[str],
    report: Mapping[str, Any] | None = None,
    artifacts: list[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    items = [_blocker_action(reason, report=report, artifacts=artifacts) for reason in block_reasons]
    for note in freshness_notes:
        items.append(_blocker_action(note, source="freshness_note", report=report, artifacts=artifacts))
    return items


def _blocker_action(
    reason: str,
    source: str = "block_reason",
    *,
    report: Mapping[str, Any] | None = None,
    artifacts: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    issue = _normalized_reason(reason)
    title = "需要人工处理"
    detail = "该阻塞项没有安全的一键修复动作，请查看原始报告和日志。"
    severity = "danger"
    action: dict[str, Any] | None = None
    report_artifact: dict[str, Any] | None = None

    if issue.startswith("daily_basic_"):
        title = "补齐 daily_basic 点时数据"
        detail = "运行点时数据修复，只补 daily_basic 缺口；完成后需要重跑自动信号刷新最终结论。"
        severity = "warning"
        action = {"label": "修复 daily_basic", "action": "repair_point_in_time"}
    elif issue in {"data_governance_repaired_after_auto_report"}:
        title = "重跑自动信号刷新结论"
        detail = "最新点时治理报告已经修复旧报告里的缺口，需要重跑自动信号让复核结论同步。"
        severity = "info"
        action = {"label": "重跑自动信号", "action": "run_auto_signal", "mode": "normal"}
    elif issue.startswith("factor_latest_") or issue.startswith("factor_coverage_"):
        title = "刷新因子并重跑信号"
        detail = "因子缓存或因子覆盖未达到目标日期；重跑自动信号会按当前门槛重新计算并再次检查。"
        action = {"label": "重跑自动信号", "action": "run_auto_signal", "mode": "normal"}
    elif issue == "candidate_only_requested":
        title = "退出候选输出模式"
        detail = "本次运行只生成候选产物；如需进入人工交易复核，请用正常门槛输出重跑。"
        severity = "hold"
        action = {"label": "正常门槛重跑", "action": "run_auto_signal", "mode": "normal"}
    elif issue.startswith("st_calendar_"):
        title = "补齐 ST 日历"
        detail = "运行完整点时治理更新，补齐 ST 日历、指数成分和 daily_basic；完成后需要重跑自动信号。"
        severity = "warning"
        action = {"label": "更新点时治理数据", "action": "update_point_in_time_all"}
    elif issue.startswith("index_constituents_"):
        title = "补齐指数/股票池点时数据"
        detail = "运行完整点时治理更新，补齐沪深300、中证500和中证1000指数成分窗口。"
        severity = "warning"
        action = {"label": "更新指数成分", "action": "update_point_in_time_all"}
    elif issue.startswith("historical_universe_"):
        title = "重建历史股票池"
        detail = "使用最新点时指数成分重建沪深300、中证500和中证1000历史股票池。"
        severity = "warning"
        action = {"label": "构建历史股票池", "action": "build_historical_universe"}
    elif issue.startswith("account_") or reason.startswith("account:"):
        title = "检查账户与持仓输入"
        detail = "账户或持仓输入未通过检查，需要先修正本地账户/持仓文件。"
    elif reason.startswith("params:") or reason.startswith("backtest:"):
        title = "复核策略质量门槛"
        detail = _quality_blocker_detail(reason, report)
        artifact_id = "parameter_quality" if reason.startswith("params:") else "backtest_quality"
        report_artifact = _blocker_report_artifact(artifacts, artifact_id)

    return {
        "id": f"{source}:{reason}",
        "source": source,
        "reason": reason,
        "issue": issue,
        "title": title,
        "detail": detail,
        "severity": severity,
        "action": action,
        "report_artifact": report_artifact,
    }


def _quality_blocker_detail(reason: str, report: Mapping[str, Any] | None) -> str:
    if reason.startswith("params:"):
        issue = _normalized_reason(reason)
        if issue == "annual_state_router_evidence_engine_contract_mismatch":
            return (
                "正式年度路由证据由旧版或不兼容的回测引擎生成，无法证明当前策略；"
                "请使用当前规范日历和选择调度重新验证，全部通过后再更新正式证据。"
            )
        return "参数验证未达到门槛；请查看参数质量报告，修正证据或策略参数后重新验证。"

    quality = _mapping_value((report or {}).get("backtest_quality"))
    issue = _normalized_reason(reason)
    yearly_match = re.fullmatch(
        r"backtest_yearly_annual_return_below_threshold:(\d{4})=(-?\d+(?:\.\d+)?)<(-?\d+(?:\.\d+)?)",
        issue,
    )
    if yearly_match:
        year, observed_text, threshold_text = yearly_match.groups()
        observed = float(observed_text)
        threshold = float(threshold_text)
        annual_return = _optional_float(quality.get("annual_return"))
        drawdown = _optional_float(quality.get("max_drawdown"))
        context: list[str] = []
        if annual_return is not None:
            context.append(f"全历史年化收益 {annual_return:.2%}")
        if drawdown is not None:
            context.append(f"最大回撤 {drawdown:.2%}")
        prefix = "、".join(context)
        if prefix:
            prefix += "；"
        return (
            f"{prefix}{year} 年分段年化收益 {observed:.2%}，低于 {threshold:.2%} 门槛，"
            "因此只保留候选产物，不生成正式交易信号。"
        )
    return "参数或回测质量未达到门槛；请查看质量报告和失败分析后调整策略证据。"


def _blocker_report_artifact(
    artifacts: list[Mapping[str, Any]] | None,
    artifact_id: str,
) -> dict[str, Any]:
    artifact = next((item for item in artifacts or [] if item.get("id") == artifact_id), None)
    if artifact is None:
        return {
            "id": artifact_id,
            "label": "Backtest quality JSON" if artifact_id == "backtest_quality" else "Parameter quality JSON",
            "kind": "json",
            "exists": False,
            "downloadable": False,
        }
    return {
        key: artifact.get(key)
        for key in ("id", "label", "kind", "exists", "downloadable")
    }


def _normalized_reason(reason: str) -> str:
    text = str(reason).strip()
    for prefix in ("data:", "governance:", "data_governance:", "params:", "param:", "backtest:", "account:"):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def _governance_details(report: Mapping[str, Any], governance_context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source": "data_governance_report",
        "supersedes_auto_report": bool(governance_context.get("supersedes_auto_report")),
        "source_generated_at": governance_context.get("current_generated_at"),
        "auto_report_generated_at": governance_context.get("auto_report_generated_at") or _timestamp_text(report),
        "resolved_auto_report_issues": _string_list(governance_context.get("resolved_issues")),
    }


def _governance_reason_issue(reason: str) -> str | None:
    text = str(reason)
    for prefix in ("governance:", "data_governance:"):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return None


def _is_newer_payload(current: Mapping[str, Any], reference: Mapping[str, Any]) -> bool:
    current_timestamp = _timestamp_text(current)
    reference_timestamp = _timestamp_text(reference)
    return bool(current_timestamp and reference_timestamp and current_timestamp > reference_timestamp)


def _timestamp_text(payload: Mapping[str, Any]) -> str:
    return str(payload.get("generated_at") or payload.get("completed_at") or payload.get("started_at") or "")


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


def _build_orders(artifact: Mapping[str, Any] | None, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
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
    _enrich_order_names(preview, config=config)
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


def _enrich_order_names(preview: dict[str, Any], config: Mapping[str, Any] | None = None) -> None:
    if not preview.get("valid") or "instrument" not in preview.get("columns", []):
        return
    name_map = load_instrument_name_map(config)
    if not name_map:
        return
    columns = list(preview.get("columns", []))
    if "name" not in columns:
        insert_at = columns.index("instrument") + 1
        columns.insert(insert_at, "name")
        preview["columns"] = columns
    for row in preview.get("rows", []):
        if not isinstance(row, dict):
            continue
        existing = str(row.get("name", "")).strip()
        if existing:
            continue
        instrument = normalize_instrument(row.get("instrument", ""))
        row["name"] = name_map.get(instrument, "")


def _build_artifacts(output_dir: Path, report: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    report_files = _mapping_value((report or {}).get("files"))
    candidates: list[tuple[str, str, str, Path]] = [
        ("auto_signal_report", "Auto signal JSON report", "json", output_dir / "auto_signal_report.json"),
        ("auto_run_status", "Auto run status", "json", output_dir / "auto_run_status.json"),
        ("daily_report", "Daily Markdown report", "markdown", output_dir / "daily_signal_report.md"),
    ]
    for artifact_id, label, kind, default_name in [
        ("signal", "Signal CSV", "csv", None),
        ("holdings", "Holdings CSV", "csv", None),
        ("manual_orders", "Manual orders CSV", "csv", None),
        ("order_confirmation", "Order confirmation CSV", "csv", None),
        ("fill_feedback", "Fill feedback CSV", "csv", None),
        ("data_health", "Data health JSON", "json", None),
        ("data_governance", "Data governance JSON", "json", None),
        ("parameter_quality", "Parameter quality JSON", "json", "auto_parameter_quality.json"),
        ("backtest_quality", "Backtest quality JSON", "json", "auto_backtest_quality.json"),
        ("fundamental_screen_report", "Fundamental screen report", "markdown", None),
    ]:
        value = report_files.get(artifact_id)
        if value:
            candidates.append((artifact_id, label, kind, resolve_path(str(value))))
        elif default_name:
            candidates.append((artifact_id, label, kind, output_dir / default_name))

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
    return _artifact_data(artifact)


def _artifact_data(artifact: Mapping[str, Any] | None) -> dict[str, Any]:
    if not artifact or not artifact.get("valid"):
        return {}
    data = artifact.get("data")
    return dict(data) if isinstance(data, Mapping) else {}


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


def _factor_coverage_summary(
    coverage: float,
    threshold: float,
    *,
    current_symbols: int | None,
    target_symbols: int | None,
    confirmed_symbols: int | None,
    unconfirmed_symbols: int | None,
) -> str:
    if current_symbols is None or target_symbols is None or target_symbols <= 0:
        return f"因子目标日覆盖率 {coverage:.2%}，达到 {threshold:.2%} 门槛。"
    missing_symbols = max(target_symbols - current_symbols, 0)
    summary = f"因子目标日覆盖 {current_symbols}/{target_symbols}（{coverage:.2%}），达到 {threshold:.2%} 门槛"
    if missing_symbols <= 0:
        return summary + "。"
    if unconfirmed_symbols == 0 and confirmed_symbols is not None and confirmed_symbols >= missing_symbols:
        return summary + f"；其余 {missing_symbols} 只已确认停牌或无新行情。"
    return summary + f"；其余 {missing_symbols} 只未产生目标日因子行。"


def _optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _optional_int(value: Any) -> int | None:
    parsed = _optional_float(value)
    if parsed is None or parsed < 0 or not parsed.is_integer():
        return None
    return int(parsed)


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
