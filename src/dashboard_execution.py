"""Controlled fill-feedback editing and holdings application for the local dashboard."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd

from src.config_loader import load_config, resolve_path
from src.manual_orders import apply_fill_feedback, load_current_holdings, validate_fill_feedback


EDITABLE_FILL_FIELDS = (
    "fill_status",
    "actual_trade_date",
    "executed_shares",
    "executed_price",
    "commission_cost",
    "tax_cost",
    "transfer_fee_cost",
    "slippage_note",
    "broker_order_id",
    "fill_note",
)


def build_execution_workspace(config: dict | None = None) -> dict[str, Any]:
    """Return the latest official fill template and current holdings for Web editing."""
    cfg = config or load_config()
    fill_path = _latest_official_fill_path(cfg)
    holdings = load_current_holdings(cfg)
    if fill_path is None:
        return {
            "version": 1,
            "status": "missing",
            "message": "尚未找到可回填的正式成交模板。请先生成通过质量门槛的正式交易单。",
            "source_id": None,
            "source_path": None,
            "rows": [],
            "holdings": _records(holdings),
            "editable_fields": list(EDITABLE_FILL_FIELDS),
        }

    try:
        fills = pd.read_csv(fill_path)
    except Exception as exc:
        return {
            "version": 1,
            "status": "error",
            "message": f"成交回填模板读取失败：{exc}",
            "source_id": fill_path.name,
            "source_path": str(fill_path),
            "rows": [],
            "holdings": _records(holdings),
            "editable_fields": list(EDITABLE_FILL_FIELDS),
        }

    fills = fills.reset_index(drop=True)
    rows = _records(fills)
    for row_id, row in enumerate(rows):
        row["row_id"] = row_id
    issues = validate_fill_feedback(holdings, fills)
    pending_count = _status_series(fills).eq("PENDING").sum()
    return {
        "version": 1,
        "status": "ready" if not issues else "needs_input",
        "message": "成交回填已通过校验，可以更新持仓。" if not issues else "请完成成交状态和实际成交数量后再更新持仓。",
        "source_id": fill_path.name,
        "source_path": str(fill_path),
        "signal_date": _first_value(fills, "signal_date"),
        "intended_trade_date": _first_value(fills, "intended_trade_date"),
        "rows": rows,
        "holdings": _records(holdings),
        "editable_fields": list(EDITABLE_FILL_FIELDS),
        "issues": issues,
        "pending_count": int(pending_count),
    }


def preview_execution_feedback(payload: dict[str, Any], config: dict | None = None) -> dict[str, Any]:
    """Validate Web-edited fills and return the resulting holdings without writing files."""
    cfg = config or load_config()
    fill_path, fills = _submitted_fills(payload, cfg)
    current = load_current_holdings(cfg)
    issues = validate_fill_feedback(current, fills)
    updated = current.copy()
    if not issues:
        updated = apply_fill_feedback(current, fills)
    return {
        "valid": not issues,
        "issues": issues,
        "source_id": fill_path.name,
        "current_holdings": _records(current),
        "updated_holdings": _records(updated),
        "summary": _fill_summary(fills),
    }


def apply_execution_feedback(payload: dict[str, Any], config: dict | None = None) -> dict[str, Any]:
    """Validate, persist the latest official fill template, and update current holdings."""
    if payload.get("confirm") is not True:
        raise ValueError("confirm must be true before holdings can be updated")
    cfg = config or load_config()
    fill_path, fills = _submitted_fills(payload, cfg)
    current = load_current_holdings(cfg)
    issues = validate_fill_feedback(current, fills)
    if issues:
        raise ValueError("Invalid fill feedback: " + ",".join(issues[:10]))
    updated = apply_fill_feedback(current, fills)
    holdings_path = resolve_path(cfg.get("account", {}).get("current_holdings_file", "config/current_holdings.csv"))

    _write_csv_atomic(fill_path, fills)
    _write_csv_atomic(holdings_path, updated)

    audit = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_id": fill_path.name,
        "fill_file": str(fill_path),
        "holdings_output": str(holdings_path),
        "current_positions": int(len(current)),
        "updated_positions": int(len(updated)),
        **_fill_summary(fills),
    }
    output_dir = resolve_path(cfg.get("outputs", {}).get("dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / f"fill_apply_audit_{_first_value(fills, 'signal_date') or datetime.now().strftime('%Y-%m-%d')}.json"
    _write_text_atomic(audit_path, json.dumps(audit, indent=2, ensure_ascii=False))
    return {
        "status": "applied",
        "message": "成交回填已保存，真实持仓已更新。",
        "source_id": fill_path.name,
        "holdings_path": str(holdings_path),
        "audit_path": str(audit_path),
        "holdings": _records(updated),
        "summary": _fill_summary(fills),
    }


def _submitted_fills(payload: dict[str, Any], config: dict) -> tuple[Path, pd.DataFrame]:
    fill_path = _latest_official_fill_path(config)
    if fill_path is None:
        raise FileNotFoundError("No official fill feedback template is available")
    source_id = str(payload.get("source_id") or "")
    if source_id != fill_path.name:
        raise ValueError("The fill feedback template changed; refresh the execution workspace")
    submitted_rows = payload.get("rows")
    if not isinstance(submitted_rows, list):
        raise ValueError("rows must be a list")
    original = pd.read_csv(fill_path).reset_index(drop=True)
    if len(submitted_rows) != len(original):
        raise ValueError("rows must include every fill feedback row")

    by_id: dict[int, dict[str, Any]] = {}
    for row in submitted_rows:
        if not isinstance(row, dict):
            raise ValueError("each fill feedback row must be an object")
        try:
            row_id = int(row.get("row_id"))
        except (TypeError, ValueError) as exc:
            raise ValueError("each fill feedback row requires a valid row_id") from exc
        if row_id in by_id or row_id < 0 or row_id >= len(original):
            raise ValueError("fill feedback row_id is duplicate or out of range")
        by_id[row_id] = row
    if set(by_id) != set(range(len(original))):
        raise ValueError("rows must include every fill feedback row exactly once")

    merged = original.copy()
    for row_id, row in by_id.items():
        for field in EDITABLE_FILL_FIELDS:
            if field in row:
                merged.at[row_id, field] = row[field]
    return fill_path, merged


def _latest_official_fill_path(config: dict) -> Path | None:
    output_root = resolve_path(config.get("outputs", {}).get("dir", "outputs"))
    configured = config.get("manual_orders", {}).get("fill_feedback_dir")
    fill_dir = resolve_path(configured) if configured else output_root / "fill_feedback"
    if not fill_dir.exists():
        return None
    paths = [
        path
        for path in fill_dir.glob("fill_feedback_*.csv")
        if path.is_file() and not path.name.startswith("fill_feedback_candidate_")
    ]
    return max(paths, key=lambda path: (path.stat().st_mtime_ns, path.name)) if paths else None


def _fill_summary(fills: pd.DataFrame) -> dict[str, Any]:
    status = _status_series(fills)
    applied = status.isin({"FILLED", "PARTIAL"})
    executed = pd.to_numeric(fills.get("executed_shares", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    return {
        "fill_rows": int(len(fills)),
        "applied_fill_rows": int(applied.sum()),
        "executed_shares": float(executed[applied].sum()),
        "fill_status_counts": {str(key): int(value) for key, value in status.value_counts().to_dict().items()},
    }


def _status_series(frame: pd.DataFrame) -> pd.Series:
    return frame.get("fill_status", pd.Series(dtype=object)).fillna("").astype(str).str.strip().str.upper()


def _first_value(frame: pd.DataFrame, column: str) -> str | None:
    if column not in frame.columns:
        return None
    values = frame[column].dropna().astype(str).str.strip()
    values = values[values != ""]
    return values.iloc[0] if not values.empty else None


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict(orient="records")


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", suffix=".csv", prefix=f".{path.stem}-", dir=path.parent, delete=False, encoding="utf-8-sig", newline="") as handle:
        temp_path = Path(handle.name)
    try:
        frame.to_csv(temp_path, index=False, encoding="utf-8-sig")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", suffix=path.suffix, prefix=f".{path.stem}-", dir=path.parent, delete=False, encoding="utf-8") as handle:
        temp_path = Path(handle.name)
        handle.write(content)
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
