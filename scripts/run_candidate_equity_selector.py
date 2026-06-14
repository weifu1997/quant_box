"""Run a rolling, non-lookahead selector over candidate equity artifacts."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._shared import dated_output_path, yearly_stats
from scripts.run_goal_audit import audit_yearly_goal, goal_thresholds, write_audit_outputs
from src.config_loader import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe a trailing-performance selector over existing candidate equity curves."
    )
    parser.add_argument("--equity-glob", action="append", default=["outputs/*_equity.csv"])
    parser.add_argument("--exclude", default="backtest_equity,auto_backtest")
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--lookback-days", default="63,126,252")
    parser.add_argument("--top-k", default="1,3,5")
    parser.add_argument("--drawdown-penalty", default="0,0.5,1.0")
    parser.add_argument("--min-periods", type=int, default=60)
    parser.add_argument("--rebalance-freq", choices=["monthly", "weekly"], default="monthly")
    parser.add_argument("--output-prefix", default=dated_output_path("candidate_equity_selector", suffix=""))
    args = parser.parse_args()

    config = load_config()
    output_prefix = resolve_path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    equities = load_candidate_equities(
        args.equity_glob,
        excludes=_csv_strings(args.exclude),
        max_candidates=args.max_candidates,
    )
    if equities.empty:
        raise ValueError("No candidate equity curves matched the requested glob(s).")

    rows: list[dict[str, Any]] = []
    runs: dict[tuple[int, int, float], SelectorRun] = {}
    return_target, drawdown_limit = goal_thresholds(config)
    for lookback_days in _csv_values(args.lookback_days, int):
        for top_k in _csv_values(args.top_k, int):
            for drawdown_penalty in _csv_values(args.drawdown_penalty, float):
                run = run_equity_selector(
                    equities,
                    lookback_days=lookback_days,
                    top_k=top_k,
                    min_periods=args.min_periods,
                    drawdown_penalty=drawdown_penalty,
                    rebalance_freq=args.rebalance_freq,
                )
                yearly = yearly_stats(run.equity, config.get("backtest", {}))
                audited_yearly, audit_summary = audit_yearly_goal(
                    yearly,
                    return_target=return_target,
                    drawdown_limit=drawdown_limit,
                )
                metrics = equity_metrics(run.equity, annual_days=int(config.get("backtest", {}).get("annual_trading_days", 252)))
                rows.append(
                    {
                        "lookback_days": lookback_days,
                        "top_k": top_k,
                        "drawdown_penalty": drawdown_penalty,
                        "candidate_count": int(equities.shape[1]),
                        **metrics,
                        **_audit_row(audit_summary),
                    }
                )
                runs[(lookback_days, top_k, drawdown_penalty)] = SelectorRun(
                    equity=run.equity,
                    selections=run.selections,
                    audited_yearly=audited_yearly,
                )

    summary = _sort_summary(pd.DataFrame(rows))
    summary_path = Path(str(output_prefix) + "_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    best = summary.iloc[0].to_dict()
    best_key = (int(best["lookback_days"]), int(best["top_k"]), float(best["drawdown_penalty"]))
    best_run = runs[best_key]
    best_run.equity.to_csv(Path(str(output_prefix) + "_best_equity.csv"), encoding="utf-8-sig")
    best_run.selections.to_csv(Path(str(output_prefix) + "_best_selections.csv"), index=False, encoding="utf-8-sig")
    best_run.audited_yearly.to_csv(Path(str(output_prefix) + "_best_years.csv"), index=False, encoding="utf-8-sig")
    payload = {
        "best": _json_safe(best),
        "summary_path": str(summary_path),
        "candidate_count": int(equities.shape[1]),
        "equity_glob": args.equity_glob,
        "excludes": _csv_strings(args.exclude),
        "note": "Scores use only equity rows strictly before each rebalance date.",
    }
    Path(str(output_prefix) + "_best_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_audit_outputs(
        output_prefix=Path(str(output_prefix) + "_best_audit"),
        yearly=best_run.audited_yearly,
        summary={
            "return_target": return_target,
            "drawdown_limit": drawdown_limit,
            "year_count": int(best["year_count"]),
            "year_return_pass_count": int(best["year_return_pass_count"]),
            "year_drawdown_pass_count": int(best["year_drawdown_pass_count"]),
            "is_goal_met": bool(best["is_goal_met"]),
            "years_below_return_target": json.loads(str(best["years_below_return_target"])),
            "years_breaching_drawdown_limit": json.loads(str(best["years_breaching_drawdown_limit"])),
            "failed_years": json.loads(str(best["failed_years"])),
            "min_yearly_annual_return": float(best["min_yearly_annual_return"]),
            "worst_yearly_drawdown": float(best["worst_yearly_drawdown"]),
            "worst_return_year": int(best["worst_return_year"]) if pd.notna(best["worst_return_year"]) else None,
            "worst_drawdown_year": int(best["worst_drawdown_year"]) if pd.notna(best["worst_drawdown_year"]) else None,
        },
        metrics=equity_metrics(best_run.equity, annual_days=int(config.get("backtest", {}).get("annual_trading_days", 252))),
    )
    print(f"Loaded candidate equities: {equities.shape[1]} columns x {equities.shape[0]} rows")
    print(f"Best selector summary saved to: {summary_path}")
    print(summary.head(10).to_string(index=False))


class SelectorRun:
    def __init__(self, equity: pd.Series, selections: pd.DataFrame, audited_yearly: pd.DataFrame | None = None) -> None:
        self.equity = equity
        self.selections = selections
        self.audited_yearly = audited_yearly if audited_yearly is not None else pd.DataFrame()


def load_candidate_equities(
    patterns: list[str],
    *,
    excludes: list[str] | None = None,
    max_candidates: int = 0,
) -> pd.DataFrame:
    paths = _matched_paths(patterns, excludes or [])
    if max_candidates > 0:
        paths = paths[:max_candidates]
    columns: list[pd.Series] = []
    names: set[str] = set()
    for path in paths:
        equity = read_equity_curve(path)
        if equity.empty:
            continue
        name = _unique_name(_candidate_name(path), names)
        names.add(name)
        columns.append(_normalize_equity(equity).rename(name))
    if not columns:
        return pd.DataFrame()
    return pd.concat(columns, axis=1).sort_index().ffill()


def read_equity_curve(path: str | Path) -> pd.Series:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Equity file not found: {source}")
    frame = pd.read_csv(source, index_col=0)
    if frame.empty:
        return pd.Series(dtype=float, name=source.stem)
    if "equity" in frame.columns:
        series = frame["equity"]
    else:
        numeric_columns = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
        if not numeric_columns:
            raise ValueError(f"No numeric equity column found in {source}")
        series = frame[numeric_columns[0]]
    series.index = pd.to_datetime(series.index, errors="coerce")
    series = pd.to_numeric(series, errors="coerce")
    return series.dropna().sort_index().rename(source.stem)


def run_equity_selector(
    equities: pd.DataFrame,
    *,
    lookback_days: int,
    top_k: int,
    min_periods: int,
    drawdown_penalty: float = 0.0,
    rebalance_freq: str = "monthly",
) -> SelectorRun:
    if equities.empty:
        raise ValueError("equities must not be empty")
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if min_periods <= 0:
        raise ValueError("min_periods must be positive")
    equity_frame = equities.sort_index().astype(float).ffill()
    returns = equity_frame.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rebalance_dates = set(rebalance_schedule(equity_frame.index, rebalance_freq))
    current_weights = _equal_weights([column for column in equity_frame.columns])
    values: list[float] = []
    selection_rows: list[dict[str, Any]] = []
    current_value = 1.0
    for date, daily_returns in returns.iterrows():
        if pd.Timestamp(date) in rebalance_dates:
            scores = trailing_candidate_scores(
                equity_frame,
                pd.Timestamp(date),
                lookback_days=lookback_days,
                min_periods=min_periods,
                drawdown_penalty=drawdown_penalty,
            )
            selected = scores.dropna().sort_values(ascending=False).head(top_k).index.to_list()
            if selected:
                current_weights = _equal_weights(selected)
                reason = "trailing_score"
            else:
                available = daily_returns.dropna().index.to_list()
                current_weights = _equal_weights(available)
                reason = "fallback_equal"
            selection_rows.append(
                {
                    "date": pd.Timestamp(date).date().isoformat(),
                    "reason": reason,
                    "selected_count": int(len(current_weights)),
                    "selected_candidates": "|".join(current_weights.index.astype(str)),
                }
            )
        aligned = daily_returns.reindex(current_weights.index).fillna(0.0)
        current_value *= 1.0 + float((aligned * current_weights).sum())
        values.append(current_value)
    equity = pd.Series(values, index=equity_frame.index, name="equity")
    return SelectorRun(equity=equity, selections=pd.DataFrame(selection_rows))


def trailing_candidate_scores(
    equities: pd.DataFrame,
    as_of_date: pd.Timestamp,
    *,
    lookback_days: int,
    min_periods: int,
    drawdown_penalty: float = 0.0,
) -> pd.Series:
    history = equities.loc[equities.index < pd.Timestamp(as_of_date)].tail(lookback_days + 1)
    scores: dict[str, float] = {}
    for column in history.columns:
        series = history[column].dropna()
        if len(series) < min_periods:
            scores[str(column)] = np.nan
            continue
        total_return = float(series.iloc[-1] / series.iloc[0] - 1.0) if series.iloc[0] else np.nan
        drawdown = float((series / series.cummax() - 1.0).min())
        scores[str(column)] = total_return + float(drawdown_penalty) * drawdown
    return pd.Series(scores, dtype=float)


def rebalance_schedule(index: pd.DatetimeIndex, freq: str) -> list[pd.Timestamp]:
    dates = pd.DatetimeIndex(index).dropna().sort_values()
    if dates.empty:
        return []
    period_freq = "W-FRI" if str(freq).lower() == "weekly" else "M"
    frame = pd.DataFrame({"date": dates})
    return [pd.Timestamp(value) for value in frame.groupby(dates.to_period(period_freq))["date"].max().to_list()]


def equity_metrics(equity: pd.Series, *, annual_days: int = 252) -> dict[str, float]:
    series = equity.dropna().sort_index().astype(float)
    if series.empty:
        return {"total_return": 0.0, "annual_return": 0.0, "max_drawdown": 0.0, "annual_volatility": 0.0, "sharpe": 0.0}
    total_return = float(series.iloc[-1] / series.iloc[0] - 1.0) if series.iloc[0] else 0.0
    years = max((series.index.max() - series.index.min()).days / 365.25, 1 / max(annual_days, 1))
    annual_return = float((1.0 + total_return) ** (1.0 / years) - 1.0) if total_return > -1 else -1.0
    returns = series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    annual_volatility = float(returns.std(ddof=0) * np.sqrt(max(annual_days, 1))) if not returns.empty else 0.0
    drawdown = series / series.cummax() - 1.0
    sharpe = annual_return / annual_volatility if annual_volatility else 0.0
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
    }


def _matched_paths(patterns: list[str], excludes: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        resolved_pattern = str(resolve_path(pattern))
        paths.extend(Path(value) for value in glob.glob(resolved_pattern))
    unique = sorted({path.resolve() for path in paths})
    lowered_excludes = [value.lower() for value in excludes]
    return [path for path in unique if not any(token in path.name.lower() for token in lowered_excludes)]


def _candidate_name(path: Path) -> str:
    name = path.stem
    return name[: -len("_equity")] if name.endswith("_equity") else name


def _unique_name(name: str, used: set[str]) -> str:
    if name not in used:
        return name
    idx = 2
    while f"{name}_{idx}" in used:
        idx += 1
    return f"{name}_{idx}"


def _normalize_equity(equity: pd.Series) -> pd.Series:
    series = equity.dropna().astype(float)
    if series.empty or not series.iloc[0]:
        return series
    return series / float(series.iloc[0])


def _equal_weights(columns: list[str]) -> pd.Series:
    clean = [str(column) for column in columns if str(column)]
    if not clean:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(clean), index=clean, dtype=float)


def _csv_values(value: str, cast):
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def _csv_strings(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _audit_row(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "year_count": int(summary.get("year_count", 0) or 0),
        "year_return_pass_count": int(summary.get("year_return_pass_count", 0) or 0),
        "year_drawdown_pass_count": int(summary.get("year_drawdown_pass_count", 0) or 0),
        "is_goal_met": bool(summary.get("is_goal_met", False)),
        "years_below_return_target": json.dumps(summary.get("years_below_return_target", []), ensure_ascii=False),
        "years_breaching_drawdown_limit": json.dumps(summary.get("years_breaching_drawdown_limit", []), ensure_ascii=False),
        "failed_years": json.dumps(summary.get("failed_years", []), ensure_ascii=False),
        "min_yearly_annual_return": summary.get("min_yearly_annual_return"),
        "worst_yearly_drawdown": summary.get("worst_yearly_drawdown"),
        "worst_return_year": summary.get("worst_return_year"),
        "worst_drawdown_year": summary.get("worst_drawdown_year"),
    }


def _sort_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    return summary.sort_values(
        ["is_goal_met", "year_return_pass_count", "year_drawdown_pass_count", "annual_return", "max_drawdown"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


if __name__ == "__main__":
    main()
