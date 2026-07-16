"""Status and small artifact helpers shared by auto-signal stages."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config_loader import resolve_path


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_status(out_dir: Path, status: dict[str, Any]) -> None:
    write_json(out_dir / "auto_run_status.json", status)


def new_status(target_date_resolution: dict[str, str] | None = None) -> dict[str, Any]:
    status: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "stages": [],
    }
    if target_date_resolution is not None:
        status["target_date_resolution"] = target_date_resolution
    return status


def stage(status: dict[str, Any], out_dir: Path, name: str, state: str, message: str = "") -> None:
    status["stages"].append(
        {
            "name": name,
            "state": state,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "message": message,
        }
    )
    write_status(out_dir, status)


def update_result_status(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_status_dict"):
        value = result.to_status_dict()
        return value if isinstance(value, dict) else {}
    status = getattr(result, "status", "")
    if status:
        return {
            "status": status,
            "failed_symbols": getattr(result, "failed_symbols", 0),
            "remaining_symbols": getattr(result, "remaining_symbols", 0),
            "last_error": getattr(result, "last_error", ""),
            "written_symbols": len(result) if hasattr(result, "__len__") else 0,
        }
    if isinstance(result, dict):
        return {"status": "complete", "written_symbols": len(result)}
    return {"status": "complete"}


def update_status_message(info: dict[str, Any]) -> str:
    parts = [
        f"written={info.get('written_symbols', 0)}",
        f"failed={info.get('failed_symbols', 0)}",
        f"remaining={info.get('remaining_symbols', 0)}",
    ]
    if info.get("progress_path"):
        parts.append(f"progress={info['progress_path']}")
    if info.get("last_error"):
        parts.append(f"last_error={info['last_error']}")
    return "; ".join(parts)


def can_reuse_conversion_outputs(
    update_info: dict[str, Any] | None,
    config: dict[str, Any],
    end_date: str,
) -> bool:
    """Return whether a no-change update may reuse current Qlib/price outputs."""
    if not update_info or str(update_info.get("status")) != "complete":
        return False
    if int(update_info.get("written_symbols", 0) or 0) != 0:
        return False

    target = pd.Timestamp(end_date).normalize()
    price_path = resolve_path(config.get("ic", {}).get("price_file", "data/prices/ohlcv_adjusted.parquet"))
    provider = resolve_path(config.get("qlib", {}).get("provider_uri", "data/qlib_data"))
    calendar_path = provider / "calendars" / "day.txt"
    instrument_path = provider / "instruments" / "all.txt"
    if not price_path.exists() or not calendar_path.exists() or not instrument_path.exists():
        return False
    try:
        price_index = pd.read_parquet(price_path, columns=[]).index
        if len(price_index) == 0 or pd.to_datetime(price_index).max().normalize() < target:
            return False
        calendar = pd.read_csv(calendar_path, header=None, usecols=[0]).iloc[:, 0]
        if calendar.empty or pd.to_datetime(calendar, errors="coerce").max().normalize() < target:
            return False
    except (OSError, ValueError, TypeError, IndexError):
        return False
    return True


def validation_progress_message(row: dict[str, object], completed: int) -> str:
    test_start = _date_message_value(row.get("test_start"))
    test_end = _date_message_value(row.get("test_end"))
    return (
        f"{completed} results; latest={test_start}..{test_end} "
        f"factor_group={row.get('factor_group', '')} top_n={row.get('top_n', '')} "
        f"rebalance={row.get('rebalance_freq', '')}"
    )


def _date_message_value(value: object) -> str:
    if value is None or value == "":
        return ""
    try:
        return pd.Timestamp(value).date().isoformat()
    except (TypeError, ValueError):
        return str(value)
