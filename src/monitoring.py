"""Monitoring helpers for exported run status files."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def build_auto_status_metrics(status: Mapping[str, Any]) -> str:
    """Build Prometheus text metrics from auto_run_status.json content."""
    lines = [
        "# HELP quant_box_auto_run_status Last automatic signal run status.",
        "# TYPE quant_box_auto_run_status gauge",
        f'quant_box_auto_run_status{{status="{_label(status.get("status", "unknown"))}"}} 1',
        "# HELP quant_box_auto_run_block_reasons_total Number of blocking reasons in the last run.",
        "# TYPE quant_box_auto_run_block_reasons_total gauge",
        f"quant_box_auto_run_block_reasons_total {_number(len(_list_value(status.get('block_reasons'))))}",
    ]
    if "is_executable" in status:
        lines.extend(
            [
                "# HELP quant_box_auto_run_is_executable Whether the last signal was executable.",
                "# TYPE quant_box_auto_run_is_executable gauge",
                f"quant_box_auto_run_is_executable {_number(1 if bool(status.get('is_executable')) else 0)}",
            ]
        )

    stage_counts: dict[str, int] = {}
    latest_stage_state: dict[str, str] = {}
    for stage in _list_value(status.get("stages")):
        if not isinstance(stage, Mapping):
            continue
        name = str(stage.get("name", "")).strip()
        state = str(stage.get("state", "unknown")).strip() or "unknown"
        if not name:
            continue
        stage_counts[name] = stage_counts.get(name, 0) + 1
        latest_stage_state[name] = state
    if latest_stage_state:
        lines.extend(
            [
                "# HELP quant_box_auto_run_stage_state Latest state per auto-signal stage.",
                "# TYPE quant_box_auto_run_stage_state gauge",
            ]
        )
        for name in sorted(latest_stage_state):
            lines.append(
                f'quant_box_auto_run_stage_state{{stage="{_label(name)}",state="{_label(latest_stage_state[name])}"}} 1'
            )
        lines.extend(
            [
                "# HELP quant_box_auto_run_stage_updates_total Number of status updates written per stage.",
                "# TYPE quant_box_auto_run_stage_updates_total gauge",
            ]
        )
        for name in sorted(stage_counts):
            lines.append(f'quant_box_auto_run_stage_updates_total{{stage="{_label(name)}"}} {_number(stage_counts[name])}')

    optimizer_timeout = status.get("optimizer_timeout")
    if isinstance(optimizer_timeout, Mapping):
        lines.extend(
            [
                "# HELP quant_box_optimizer_timeout_completed_combinations Completed optimizer combinations before timeout.",
                "# TYPE quant_box_optimizer_timeout_completed_combinations gauge",
                "quant_box_optimizer_timeout_completed_combinations "
                f"{_number(optimizer_timeout.get('completed_combinations', 0))}",
                "# HELP quant_box_optimizer_timeout_completed_windows Completed optimizer windows before timeout.",
                "# TYPE quant_box_optimizer_timeout_completed_windows gauge",
                f"quant_box_optimizer_timeout_completed_windows {_number(optimizer_timeout.get('completed_windows', 0))}",
            ]
        )

    return "\n".join(lines) + "\n"


def write_auto_status_metrics(status: Mapping[str, Any], output_path: str | Path) -> Path:
    """Write Prometheus text metrics and return the resolved output path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_auto_status_metrics(status), encoding="utf-8")
    return path


def _list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _label(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _number(value: Any) -> str:
    try:
        return str(float(value))
    except (TypeError, ValueError):
        return "0.0"
