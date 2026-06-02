from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import pandas as pd

from src.config_loader import resolve_path


def write_daily_signal_report(report: dict[str, Any], out_dir: str | Path) -> Path:
    output_dir = resolve_path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "daily_signal_report.md"
    path.write_text(_render_report(report), encoding="utf-8")
    return path


def archive_run(files: list[str | Path], history_dir: str | Path, signal_date: str) -> Path:
    target = resolve_path(history_dir) / signal_date
    target.mkdir(parents=True, exist_ok=True)
    for item in files:
        path = Path(item)
        if not path.is_absolute():
            path = resolve_path(path)
        if path.exists() and path.is_file():
            shutil.copy2(path, target / path.name)
    return target


def _render_report(report: dict[str, Any]) -> str:
    selected_params = report.get("selected_params", {})
    data_health = report.get("data_health", {})
    param_quality = report.get("parameter_quality", {})
    metrics = report.get("backtest_metrics", {})
    account = report.get("account", {})
    signal_summary = report.get("signal_summary", {})
    block_reasons = report.get("block_reasons", [])

    lines = [
        "# Daily Signal Report",
        "",
        f"- Signal date: {report.get('signal_date', '')}",
        f"- Intended trade date: {report.get('intended_trade_date', '')}",
        f"- Executable: {report.get('is_executable', False)}",
    ]
    if block_reasons:
        lines.append(f"- Block reasons: {', '.join(map(str, block_reasons))}")
    lines.extend(
        [
            "",
            "## Data Health",
            "",
            f"- Healthy: {data_health.get('is_healthy', False)}",
            f"- Target symbols: {data_health.get('target_symbols', 0)}",
            f"- Raw coverage: {_pct(data_health.get('raw_target_coverage'))}",
            f"- Price coverage: {_pct(data_health.get('price_target_coverage'))}",
            f"- Factor coverage: {_pct(data_health.get('factor_target_coverage'))}",
            f"- Raw latest: {data_health.get('raw_latest_date', '')}",
            f"- Price latest: {data_health.get('price_latest_date', '')}",
            f"- Factor latest: {data_health.get('factor_latest_date', '')}",
            f"- Issues: {', '.join(data_health.get('issues', [])) or 'none'}",
            "",
            "## Parameter Quality",
            "",
            f"- Acceptable: {param_quality.get('is_acceptable', False)}",
            f"- Windows: {param_quality.get('windows', 0)}",
            f"- Positive return rate: {_pct(param_quality.get('positive_return_rate'))}",
            f"- Sharpe mean: {_num(param_quality.get('sharpe_mean'))}",
            f"- Worst drawdown: {_pct(param_quality.get('max_drawdown_worst'))}",
            f"- Issues: {', '.join(param_quality.get('issues', [])) or 'none'}",
            "",
            "## Selected Params",
            "",
            *_mapping_lines(selected_params),
            "",
            "## Backtest Metrics",
            "",
            *_mapping_lines(metrics, max_items=12),
            "",
            "## Account",
            "",
            f"- Total asset: {_num(account.get('total_asset'))}",
            f"- Cash: {_num(account.get('cash'))}",
            f"- Holdings loaded: {account.get('holdings_loaded', False)}",
            "",
            "## Signal Summary",
            "",
            f"- BUY: {signal_summary.get('BUY', 0)}",
            f"- HOLD: {signal_summary.get('HOLD', 0)}",
            f"- SELL: {signal_summary.get('SELL', 0)}",
            "",
            "## Output Files",
            "",
            *_mapping_lines(report.get("files", {})),
            "",
        ]
    )
    return "\n".join(lines)


def signal_action_summary(signal_df: pd.DataFrame) -> dict[str, int]:
    if signal_df.empty or "action" not in signal_df.columns:
        return {"BUY": 0, "HOLD": 0, "SELL": 0}
    counts = signal_df["action"].astype(str).str.upper().value_counts()
    return {key: int(counts.get(key, 0)) for key in ["BUY", "HOLD", "SELL"]}


def _mapping_lines(values: dict[str, Any], max_items: int | None = None) -> list[str]:
    items = list(values.items())
    if max_items is not None:
        items = items[:max_items]
    return [f"- {key}: {value}" for key, value in items]


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return ""


def _num(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return ""
