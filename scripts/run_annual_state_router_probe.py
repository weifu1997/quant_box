"""Probe an annual market-state router over existing candidate equity curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._shared import dated_output_path, yearly_stats
from scripts.run_candidate_equity_selector import equity_metrics, read_equity_curve
from scripts.run_goal_audit import audit_yearly_goal, goal_thresholds, write_audit_outputs
from src.config_loader import load_config, resolve_path
from src.market_regime import _benchmark_close


DEFAULT_SOURCE_FILES: dict[str, str] = {
    "db_size": "outputs/codex_goal_fast_promoted_summary_20260610_db_circ_mv_inv_q80_top7_take035_equity.csv",
    "quality": "outputs/fundamental_quality_top5_full_20260611_equity.csv",
    "selector": "outputs/selector_weight_lb63_top5_posprop_top5_formal_20260611_equity.csv",
    "industry": "outputs/codex_goal_industry_momentum_summary_20260611_ind_roc120_q65_top10_take035_equity.csv",
    "beta": "outputs/codex_goal_fast_promoted_summary_20260610_beta60_q65_top5_take035_equity.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe a fixed annual market-state router over existing strategy equity curves."
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Source mapping in name=equity_csv form. Defaults to the current research candidate set.",
    )
    parser.add_argument("--initial-source", default="beta")
    parser.add_argument("--missing-ret252-exposure", type=float, default=0.65)
    parser.add_argument("--flat-negative-exposure", type=float, default=0.90)
    parser.add_argument("--output-prefix", default=dated_output_path("annual_state_router_probe", suffix=""))
    args = parser.parse_args()

    config = load_config()
    output_prefix = resolve_path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    source_files = parse_source_args(args.source) if args.source else dict(DEFAULT_SOURCE_FILES)
    source_returns = load_source_returns(source_files)
    prices = pd.read_parquet(resolve_path(config.get("ic", {}).get("price_file", "data/prices/ohlcv_adjusted.parquet")))
    benchmark = _benchmark_close(prices, config, config.get("market_regime", {})).dropna().sort_index()
    if benchmark.empty:
        raise ValueError("Benchmark close series is empty; cannot route by annual market state.")

    router = run_annual_state_router(
        source_returns=source_returns,
        benchmark=benchmark,
        initial_source=args.initial_source,
        missing_ret252_exposure=args.missing_ret252_exposure,
        flat_negative_exposure=args.flat_negative_exposure,
    )
    yearly = yearly_stats(router.equity, config.get("backtest", {}))
    return_target, drawdown_limit = goal_thresholds(config)
    audited_yearly, audit_summary = audit_yearly_goal(
        yearly,
        return_target=return_target,
        drawdown_limit=drawdown_limit,
    )
    metrics = equity_metrics(
        router.equity,
        annual_days=int(config.get("backtest", {}).get("annual_trading_days", 252)),
    )
    full_gate = {
        "annual_return_pass": bool(metrics["annual_return"] >= return_target),
        "max_drawdown_pass": bool(metrics["max_drawdown"] >= drawdown_limit),
        "is_full_goal_met": bool(
            metrics["annual_return"] >= return_target
            and metrics["max_drawdown"] >= drawdown_limit
            and audit_summary["is_goal_met"]
        ),
    }

    router.equity.to_csv(Path(str(output_prefix) + "_equity.csv"), encoding="utf-8-sig")
    router.routes.to_csv(Path(str(output_prefix) + "_routes.csv"), index=False, encoding="utf-8-sig")
    audited_yearly.to_csv(Path(str(output_prefix) + "_years.csv"), index=False, encoding="utf-8-sig")
    payload = {
        "metrics": metrics,
        "audit": audit_summary,
        "full_gate": full_gate,
        "source_files": {name: str(resolve_path(path)) for name, path in source_files.items()},
        "initial_source": args.initial_source,
        "missing_ret252_exposure": args.missing_ret252_exposure,
        "flat_negative_exposure": args.flat_negative_exposure,
        "note": (
            "Research probe: annual source routing uses benchmark data strictly before the first trading day "
            "of each calendar year. Rule thresholds are exploratory and selected in-sample; do not promote "
            "official signals from this artifact without separate validation."
        ),
    }
    Path(str(output_prefix) + "_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    write_audit_outputs(
        output_prefix=Path(str(output_prefix) + "_audit"),
        yearly=audited_yearly,
        summary=audit_summary,
        metrics=metrics,
    )
    print(
        f"annual_state_router annual={metrics['annual_return']:.4f} "
        f"dd={metrics['max_drawdown']:.4f} "
        f"yearly={audit_summary['year_return_pass_count']}/{audit_summary['year_drawdown_pass_count']} "
        f"full_goal={full_gate['is_full_goal_met']}"
    )
    print(f"wrote prefix: {output_prefix}")


class AnnualRouterRun:
    def __init__(self, equity: pd.Series, routes: pd.DataFrame) -> None:
        self.equity = equity
        self.routes = routes


def parse_source_args(values: list[str]) -> dict[str, str]:
    sources: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Source must use name=path form: {raw!r}")
        name, path = raw.split("=", 1)
        clean_name = name.strip()
        clean_path = path.strip()
        if not clean_name or not clean_path:
            raise ValueError(f"Source must include both name and path: {raw!r}")
        sources[clean_name] = clean_path
    return sources


def load_source_returns(source_files: dict[str, str]) -> pd.DataFrame:
    if not source_files:
        raise ValueError("At least one source equity curve is required.")
    columns: list[pd.Series] = []
    for name, raw_path in source_files.items():
        path = resolve_path(raw_path)
        equity = read_equity_curve(path)
        if equity.empty:
            raise ValueError(f"Source equity curve is empty: {path}")
        columns.append(equity.sort_index().pct_change(fill_method=None).rename(str(name)))
    return pd.concat(columns, axis=1).sort_index().fillna(0.0)


def run_annual_state_router(
    *,
    source_returns: pd.DataFrame,
    benchmark: pd.Series,
    initial_source: str,
    missing_ret252_exposure: float,
    flat_negative_exposure: float,
) -> AnnualRouterRun:
    if source_returns.empty:
        raise ValueError("source_returns must not be empty")
    if initial_source not in source_returns.columns:
        raise ValueError(f"initial_source is not in source returns: {initial_source}")
    benchmark = normalize_benchmark(benchmark)
    benchmark_returns = benchmark.pct_change(fill_method=None)
    current_value = 1.0
    current_source = initial_source
    current_exposure = 1.0
    values: list[float] = []
    route_rows: list[dict[str, Any]] = []
    for raw_date, daily_returns in source_returns.sort_index().iterrows():
        date = pd.Timestamp(raw_date).normalize()
        if not route_rows or int(date.year) != int(route_rows[-1]["year"]):
            decision = route_for_date(
                benchmark=benchmark,
                benchmark_returns=benchmark_returns,
                date=date,
                initial_source=initial_source,
                missing_ret252_exposure=missing_ret252_exposure,
                flat_negative_exposure=flat_negative_exposure,
            )
            if decision["source"] not in source_returns.columns:
                raise ValueError(f"Routed source is not in source returns: {decision['source']}")
            current_source = str(decision["source"])
            current_exposure = float(decision["exposure"])
            route_rows.append(decision)
        current_value *= 1.0 + float(daily_returns[current_source]) * current_exposure
        values.append(current_value)
    equity = pd.Series(values, index=pd.to_datetime(source_returns.index).normalize(), name="equity")
    return AnnualRouterRun(equity=equity, routes=pd.DataFrame(route_rows))


def route_for_date(
    *,
    benchmark: pd.Series,
    benchmark_returns: pd.Series,
    date: pd.Timestamp,
    initial_source: str,
    missing_ret252_exposure: float,
    flat_negative_exposure: float,
) -> dict[str, Any]:
    target = pd.Timestamp(date).normalize()
    history = benchmark.loc[benchmark.index < target]
    if history.empty:
        return route_row(target, initial_source, "insufficient_history", np.nan, np.nan, np.nan, 1.0)
    ret126 = trailing_return(history, 126)
    ret252 = trailing_return(history, 252)
    vol252 = trailing_volatility(benchmark_returns.loc[: history.index[-1]], 252)
    source, reason = route_source(ret126=ret126, ret252=ret252, vol252=vol252)
    exposure = 1.0
    if reason == "ret252_missing":
        exposure = float(missing_ret252_exposure)
    elif reason == "flat_with_negative_half_year":
        exposure = float(flat_negative_exposure)
    return route_row(target, source, reason, ret126, ret252, vol252, exposure)


def route_source(*, ret126: float, ret252: float, vol252: float) -> tuple[str, str]:
    if pd.isna(ret252):
        return "db_size", "ret252_missing"
    if float(ret252) < -0.05:
        if not pd.isna(vol252) and float(vol252) >= 0.25:
            return "quality", "negative_high_vol"
        return "industry", "negative_moderate_vol"
    if float(ret252) < 0.05 and not pd.isna(ret126) and float(ret126) < 0.0:
        return "beta", "flat_with_negative_half_year"
    if float(ret252) >= 0.30:
        return "selector", "strong_trailing_market"
    if float(ret252) < 0.16 and not pd.isna(vol252) and float(vol252) < 0.14:
        return "selector", "low_vol_moderate_uptrend"
    return "beta", "default_beta"


def route_row(
    date: pd.Timestamp,
    source: str,
    reason: str,
    ret126: float,
    ret252: float,
    vol252: float,
    exposure: float,
) -> dict[str, Any]:
    return {
        "year": int(pd.Timestamp(date).year),
        "decision_date": pd.Timestamp(date).date().isoformat(),
        "source": source,
        "reason": reason,
        "exposure": float(exposure),
        "ret126": optional_float(ret126),
        "ret252": optional_float(ret252),
        "vol252": optional_float(vol252),
    }


def normalize_benchmark(benchmark: pd.Series) -> pd.Series:
    series = pd.to_numeric(benchmark, errors="coerce").dropna().sort_index()
    series.index = pd.to_datetime(series.index).normalize()
    return series[~series.index.duplicated(keep="last")]


def trailing_return(history: pd.Series, days: int) -> float:
    window = history.tail(int(days) + 1)
    if len(window) <= int(days) or not float(window.iloc[0]):
        return float("nan")
    return float(window.iloc[-1] / window.iloc[0] - 1.0)


def trailing_volatility(returns: pd.Series, days: int) -> float:
    window = pd.to_numeric(returns, errors="coerce").dropna().tail(int(days))
    if window.empty:
        return float("nan")
    return float(window.std(ddof=0) * np.sqrt(252))


def optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


if __name__ == "__main__":
    main()
