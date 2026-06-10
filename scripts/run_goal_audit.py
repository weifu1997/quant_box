"""Audit the active annual return and drawdown goal for a backtest output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._shared import yearly_stats
from src.config_loader import load_config, resolve_path


DEFAULT_EQUITY_FILE = "outputs/backtest_equity.csv"
DEFAULT_METRICS_FILE = "outputs/backtest_metrics.json"
DEFAULT_OUTPUT_PREFIX = "outputs/goal_audit_current"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit yearly 20/20 goal progress from backtest artifacts.")
    parser.add_argument("--equity-file", default=DEFAULT_EQUITY_FILE)
    parser.add_argument("--metrics-file", default=DEFAULT_METRICS_FILE)
    parser.add_argument("--years-file", default="", help="Optional yearly CSV to audit instead of recomputing from equity.")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--return-target", type=float, default=None)
    parser.add_argument("--drawdown-limit", type=float, default=None)
    args = parser.parse_args()

    config = load_config()
    return_target, drawdown_limit = goal_thresholds(
        config,
        return_target=args.return_target,
        drawdown_limit=args.drawdown_limit,
    )
    metrics = load_metrics(resolve_path(args.metrics_file))
    if args.years_file:
        yearly = load_yearly(resolve_path(args.years_file))
    else:
        yearly = yearly_stats(load_equity(resolve_path(args.equity_file)), {**config.get("backtest", {}), **config.get("strategy", {})})
    audited_yearly, summary = audit_yearly_goal(yearly, return_target=return_target, drawdown_limit=drawdown_limit)

    outputs = write_audit_outputs(
        output_prefix=resolve_path(args.output_prefix),
        yearly=audited_yearly,
        summary=summary,
        metrics=metrics,
    )
    print(
        "Goal audit: "
        f"status={'PASS' if summary['is_goal_met'] else 'FAIL'}; "
        f"return_pass={summary['year_return_pass_count']}/{summary['year_count']}; "
        f"drawdown_pass={summary['year_drawdown_pass_count']}/{summary['year_count']}"
    )
    print(f"Wrote markdown: {outputs['markdown']}")


def goal_thresholds(
    config: dict[str, Any],
    *,
    return_target: float | None = None,
    drawdown_limit: float | None = None,
) -> tuple[float, float]:
    quality = config.get("quality", {})
    if return_target is None:
        return_target = float(
            quality.get("min_backtest_annual_return", quality.get("target_annual_return", 0.20))
        )
    if drawdown_limit is None:
        drawdown_limit = float(
            quality.get("max_backtest_drawdown_limit", quality.get("max_drawdown_limit", -0.20))
        )
    return float(return_target), float(drawdown_limit)


def load_equity(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Equity file not found: {path}")
    frame = pd.read_csv(path, index_col=0)
    if frame.empty:
        return pd.Series(dtype=float, name="equity")
    if "equity" in frame.columns:
        series = frame["equity"]
    else:
        numeric_columns = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
        if not numeric_columns:
            raise ValueError(f"No numeric equity column found in {path}")
        series = frame[numeric_columns[0]]
    series.index = pd.to_datetime(series.index)
    return pd.to_numeric(series, errors="coerce").dropna().sort_index().rename("equity")


def load_yearly(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Yearly file not found: {path}")
    return pd.read_csv(path)


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def audit_yearly_goal(
    yearly: pd.DataFrame,
    *,
    return_target: float,
    drawdown_limit: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    audited = yearly.copy()
    for column in ["annual_return", "max_drawdown"]:
        if column not in audited.columns:
            audited[column] = pd.Series(dtype=float)
        audited[column] = pd.to_numeric(audited[column], errors="coerce")
    if "year" not in audited.columns:
        audited["year"] = pd.Series(dtype="Int64")
    audited["year"] = pd.to_numeric(audited["year"], errors="coerce").astype("Int64")

    audited["return_target"] = return_target
    audited["drawdown_limit"] = drawdown_limit
    audited["annual_return_pass"] = audited["annual_return"] >= return_target
    audited["drawdown_pass"] = audited["max_drawdown"] >= drawdown_limit
    audited["goal_pass"] = audited["annual_return_pass"] & audited["drawdown_pass"]
    audited["annual_return_gap"] = audited["annual_return"] - return_target
    audited["drawdown_buffer"] = audited["max_drawdown"] - drawdown_limit

    years_below_return = _years(audited.loc[~audited["annual_return_pass"], "year"])
    years_breaching_drawdown = _years(audited.loc[~audited["drawdown_pass"], "year"])
    failed_years = _years(audited.loc[~audited["goal_pass"], "year"])
    year_count = int(len(audited))
    return_pass_count = int(audited["annual_return_pass"].sum()) if year_count else 0
    drawdown_pass_count = int(audited["drawdown_pass"].sum()) if year_count else 0
    summary = {
        "return_target": return_target,
        "drawdown_limit": drawdown_limit,
        "year_count": year_count,
        "year_return_pass_count": return_pass_count,
        "year_drawdown_pass_count": drawdown_pass_count,
        "is_goal_met": bool(year_count and return_pass_count == year_count and drawdown_pass_count == year_count),
        "years_below_return_target": years_below_return,
        "years_breaching_drawdown_limit": years_breaching_drawdown,
        "failed_years": failed_years,
        "min_yearly_annual_return": _min_or_none(audited["annual_return"]),
        "worst_yearly_drawdown": _min_or_none(audited["max_drawdown"]),
        "worst_return_year": _year_at_min(audited, "annual_return"),
        "worst_drawdown_year": _year_at_min(audited, "max_drawdown"),
    }
    return audited, summary


def write_audit_outputs(
    *,
    output_prefix: Path,
    yearly: pd.DataFrame,
    summary: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, str]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    years_path = Path(str(output_prefix) + "_years.csv")
    json_path = Path(str(output_prefix) + ".json")
    markdown_path = Path(str(output_prefix) + ".md")

    yearly.to_csv(years_path, index=False, encoding="utf-8-sig")
    payload = {"summary": _json_safe(summary), "metrics": _json_safe(metrics), "years_path": str(years_path)}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(render_markdown(summary=summary, metrics=metrics, yearly=yearly), encoding="utf-8")
    return {"years": str(years_path), "json": str(json_path), "markdown": str(markdown_path)}


def render_markdown(*, summary: dict[str, Any], metrics: dict[str, Any], yearly: pd.DataFrame) -> str:
    failed = yearly.loc[~yearly["goal_pass"]].copy() if "goal_pass" in yearly.columns else pd.DataFrame()
    failed = failed.sort_values(["annual_return_gap", "drawdown_buffer"], ascending=[True, True])
    lines = [
        "# Goal Audit",
        "",
        f"- Status: {'PASS' if summary.get('is_goal_met') else 'FAIL'}",
        f"- Annual return target: {_pct(summary.get('return_target'))}",
        f"- Drawdown limit: {_pct(summary.get('drawdown_limit'))}",
        f"- Yearly return pass count: {summary.get('year_return_pass_count')}/{summary.get('year_count')}",
        f"- Yearly drawdown pass count: {summary.get('year_drawdown_pass_count')}/{summary.get('year_count')}",
        "",
        "## Failed Years",
        "",
    ]
    if metrics:
        lines[5:5] = [
            f"- Full-period annual return: {_pct(metrics.get('annual_return'))}",
            f"- Full-period max drawdown: {_pct(metrics.get('max_drawdown'))}",
        ]
    if failed.empty:
        lines.append("No failed years.")
    else:
        lines.extend(_markdown_table(failed, ["year", "annual_return", "annual_return_gap", "max_drawdown", "drawdown_buffer"]))
    lines.extend(
        [
            "",
            "## Failure Summary",
            "",
            f"- Years below return target: {summary.get('years_below_return_target', [])}",
            f"- Years breaching drawdown limit: {summary.get('years_breaching_drawdown_limit', [])}",
            f"- Worst return year: {summary.get('worst_return_year')}",
            f"- Worst drawdown year: {summary.get('worst_drawdown_year')}",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame[columns].iterrows():
        values = []
        for column in columns:
            value = row.get(column)
            if column == "year" and pd.notna(value):
                values.append(str(int(value)))
            elif column in {"annual_return", "annual_return_gap", "max_drawdown", "drawdown_buffer"}:
                values.append(_pct(value))
            else:
                values.append("" if pd.isna(value) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _pct(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return ""
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return ""


def _years(values: pd.Series) -> list[int]:
    return [int(value) for value in values.dropna().astype(int).to_list()]


def _min_or_none(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.min()) if not clean.empty else None


def _year_at_min(frame: pd.DataFrame, column: str) -> int | None:
    clean = frame.dropna(subset=[column, "year"])
    if clean.empty:
        return None
    return int(clean.loc[clean[column].idxmin(), "year"])


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    main()
