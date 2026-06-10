"""Scan yearly backtest artifacts for calendar-year goal bottlenecks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_goal_audit import goal_thresholds
from src.config_loader import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan *_years.csv artifacts for yearly 20/20 bottlenecks.")
    parser.add_argument("--glob", default="outputs/*_years.csv")
    parser.add_argument("--output", default="outputs/goal_bottleneck_scan.csv")
    parser.add_argument("--return-target", type=float, default=None)
    parser.add_argument("--drawdown-limit", type=float, default=None)
    args = parser.parse_args()

    config = load_config()
    return_target, drawdown_limit = goal_thresholds(
        config,
        return_target=args.return_target,
        drawdown_limit=args.drawdown_limit,
    )
    rows = load_yearly_artifacts(args.glob)
    summary = build_bottleneck_summary(rows, return_target=return_target, drawdown_limit=drawdown_limit)
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Scanned yearly rows={len(rows)} files={rows['candidate_file'].nunique() if not rows.empty else 0}")
    print(f"Wrote bottleneck scan: {output}")
    if not summary.empty:
        print(summary.to_string(index=False))


def load_yearly_artifacts(pattern: str) -> pd.DataFrame:
    paths = sorted(Path().glob(pattern))
    rows: list[pd.DataFrame] = []
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError):
            continue
        if not {"year", "annual_return", "max_drawdown"}.issubset(frame.columns):
            continue
        data = frame[["year", "annual_return", "max_drawdown"]].copy()
        data["candidate_file"] = str(path)
        rows.append(data)
    if not rows:
        return pd.DataFrame(columns=["year", "annual_return", "max_drawdown", "candidate_file"])
    result = pd.concat(rows, ignore_index=True)
    result["year"] = pd.to_numeric(result["year"], errors="coerce").astype("Int64")
    result["annual_return"] = pd.to_numeric(result["annual_return"], errors="coerce")
    result["max_drawdown"] = pd.to_numeric(result["max_drawdown"], errors="coerce")
    return result.dropna(subset=["year"]).reset_index(drop=True)


def build_bottleneck_summary(
    yearly_rows: pd.DataFrame,
    *,
    return_target: float,
    drawdown_limit: float,
) -> pd.DataFrame:
    if yearly_rows.empty:
        return pd.DataFrame(
            columns=[
                "year",
                "candidate_rows",
                "best_annual_return",
                "best_annual_return_file",
                "best_return_row_drawdown",
                "return_pass_count",
                "drawdown_pass_count",
                "both_pass_count",
                "return_target_gap",
            ]
        )
    rows: list[dict[str, Any]] = []
    for year, group in yearly_rows.groupby("year", sort=True):
        best_idx = group["annual_return"].idxmax()
        best = group.loc[best_idx]
        return_pass = group["annual_return"] >= return_target
        drawdown_pass = group["max_drawdown"] >= drawdown_limit
        rows.append(
            {
                "year": int(year),
                "candidate_rows": int(len(group)),
                "best_annual_return": float(best["annual_return"]),
                "best_annual_return_file": str(best["candidate_file"]),
                "best_return_row_drawdown": float(best["max_drawdown"]),
                "return_pass_count": int(return_pass.sum()),
                "drawdown_pass_count": int(drawdown_pass.sum()),
                "both_pass_count": int((return_pass & drawdown_pass).sum()),
                "return_target_gap": max(0.0, float(return_target) - float(best["annual_return"])),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
