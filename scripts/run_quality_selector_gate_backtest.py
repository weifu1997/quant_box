"""Run a market-regime gate between fundamental quality and selector scores."""

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
from scripts.run_fundamental_quality_backtest import (
    _load_optional_stock_basic,
    build_fundamental_quality_score_panel,
    month_end_signal_dates,
)
from scripts.run_goal_audit import audit_yearly_goal, goal_thresholds, write_audit_outputs
from scripts.run_selector_weight_backtest import apply_selector_directions, selector_weights_from_frame
from src.backtest import run_backtest
from src.config_loader import load_config, resolve_path
from src.factor_calculator import load_or_compute_factors
from src.fundamental_data import normalize_dividend_frame, normalize_fina_indicator_frame
from src.market_regime import _benchmark_close, apply_defensive_timing_to_backtest_config, detect_market_regime, regime_for_date
from src.risk_policy import RiskPolicy
from src.scoring import _apply_liquidity_filter
from src.strategy import composite_factor, resample_signals
from src.trading_calendar import resolve_target_date_value


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest a fixed market-regime gate between quality and selector scores.")
    parser.add_argument("--selector-file", required=True)
    parser.add_argument("--factor-file", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--max-turnover", type=int, default=1)
    parser.add_argument("--rank-buffer", type=int, default=20)
    parser.add_argument("--gate-mode", choices=["market_regime", "annual_volatility", "annual_market_state"], default="market_regime")
    parser.add_argument("--quality-regimes", default="bull,sideways")
    parser.add_argument("--selector-regimes", default="bear")
    parser.add_argument("--market-ma-window", type=int)
    parser.add_argument("--market-momentum-window", type=int)
    parser.add_argument("--market-bear-drawdown", type=float)
    parser.add_argument("--market-drawdown-window", type=int)
    parser.add_argument("--market-lag-days", type=int)
    parser.add_argument("--annual-vol-threshold", type=float, default=0.18)
    parser.add_argument("--annual-vol-window", type=int, default=252)
    parser.add_argument("--annual-vol-min-periods", type=int, default=120)
    parser.add_argument("--annual-vol-lag-days", type=int, default=1)
    parser.add_argument("--annual-vol-selector-if", choices=["below", "above"], default="below")
    parser.add_argument("--annual-state-vol-min", type=float, default=0.20)
    parser.add_argument("--annual-state-momentum-window", type=int, default=126)
    parser.add_argument("--annual-state-quality-momentum-max", type=float, default=0.08)
    parser.add_argument("--annual-state-quality-ret252-min", type=float, default=0.0)
    parser.add_argument("--annual-guard-drawdown", type=float)
    parser.add_argument("--annual-guard-target-exposure", type=float, default=0.0)
    parser.add_argument("--annual-guard-release-drawdown", type=float)
    parser.add_argument("--no-defensive-timing", action="store_true")
    args = parser.parse_args()

    config = load_config()
    start_date = args.start_date or config["data"]["start_date"]
    end_date = resolve_target_date_value(args.end_date or config["data"]["end_date"], config=config)
    output_prefix = resolve_path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    prices = pd.read_parquet(resolve_path(config.get("ic", {}).get("price_file", "data/prices/ohlcv_adjusted.parquet")))
    signal_dates = month_end_signal_dates(prices.index, start_date=start_date, end_date=end_date)
    quality_scores = build_quality_scores(config, signal_dates)
    selector_scores = build_selector_scores(
        config=config,
        prices=prices,
        selector_file=args.selector_file,
        factor_file=args.factor_file,
        start_date=start_date,
        end_date=end_date,
    )
    normalized_quality = normalize_score_panel(quality_scores)
    normalized_selector = normalize_score_panel(selector_scores)
    regime_config: dict[str, Any] | None = None
    annual_volatility = pd.Series(dtype=float, name="annual_volatility")
    annual_state = pd.DataFrame()
    if args.gate_mode == "annual_volatility":
        annual_volatility = benchmark_annual_volatility(
            prices=prices,
            config=config,
            window=args.annual_vol_window,
            min_periods=args.annual_vol_min_periods,
            lag_days=args.annual_vol_lag_days,
        )
        scores, gate_rows = annual_volatility_gated_scores(
            quality_scores=normalized_quality,
            selector_scores=normalized_selector,
            annual_volatility=annual_volatility,
            threshold=args.annual_vol_threshold,
            selector_if=args.annual_vol_selector_if,
        )
    elif args.gate_mode == "annual_market_state":
        annual_state = benchmark_annual_state(
            prices=prices,
            config=config,
            volatility_window=args.annual_vol_window,
            volatility_min_periods=args.annual_vol_min_periods,
            momentum_window=args.annual_state_momentum_window,
            lag_days=args.annual_vol_lag_days,
        )
        annual_volatility = annual_state.get("annual_volatility", pd.Series(dtype=float, name="annual_volatility"))
        scores, gate_rows = annual_market_state_gated_scores(
            quality_scores=normalized_quality,
            selector_scores=normalized_selector,
            annual_state=annual_state,
            volatility_min=args.annual_state_vol_min,
            quality_momentum_max=args.annual_state_quality_momentum_max,
            quality_ret252_min=args.annual_state_quality_ret252_min,
        )
    else:
        regime_config = market_regime_config(config, args)
        regimes = detect_market_regime(prices, regime_config)
        scores, gate_rows = gated_scores(
            quality_scores=normalized_quality,
            selector_scores=normalized_selector,
            regimes=regimes,
            quality_regimes=parse_regimes(args.quality_regimes),
            selector_regimes=parse_regimes(args.selector_regimes),
        )
    if scores.empty:
        raise ValueError("gated score panel is empty")

    bt_config = {
        **config.get("backtest", {}),
        **config.get("strategy", {}),
        "top_n": args.top_n,
        "max_turnover": args.max_turnover,
        "rank_buffer": args.rank_buffer,
        "rebalance_freq": "monthly",
    }
    if args.annual_guard_drawdown is not None:
        guard = {
            "enabled": True,
            "drawdown": abs(float(args.annual_guard_drawdown)),
            "target_exposure": args.annual_guard_target_exposure,
        }
        if args.annual_guard_release_drawdown is not None:
            guard["release_drawdown"] = abs(float(args.annual_guard_release_drawdown))
        bt_config["annual_drawdown_guard"] = guard
    timing_config = dict(config)
    if args.no_defensive_timing:
        timing_config.setdefault("defensive_timing", {})["enabled"] = False
    bt_config = apply_defensive_timing_to_backtest_config(bt_config, prices, timing_config)
    bt_config = RiskPolicy(config).apply_to_backtest_config(bt_config)
    result = run_backtest(scores, prices, start_date, end_date, bt_config)
    yearly = yearly_stats(result.equity_curve, bt_config)
    return_target, drawdown_limit = goal_thresholds(config)
    audited_yearly, audit_summary = audit_yearly_goal(
        yearly,
        return_target=return_target,
        drawdown_limit=drawdown_limit,
    )

    result.equity_curve.to_csv(Path(str(output_prefix) + "_equity.csv"), encoding="utf-8-sig")
    result.holdings.to_csv(Path(str(output_prefix) + "_holdings.csv"), index=False, encoding="utf-8-sig")
    result.trades.to_csv(Path(str(output_prefix) + "_trades.csv"), index=False, encoding="utf-8-sig")
    audited_yearly.to_csv(Path(str(output_prefix) + "_years.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(gate_rows).to_csv(Path(str(output_prefix) + "_gate.csv"), index=False, encoding="utf-8-sig")
    payload = {
        "metrics": result.metrics,
        "audit": audit_summary,
        "selector_file": str(resolve_path(args.selector_file)),
        "factor_file": str(resolve_path(args.factor_file)),
        "gate_mode": args.gate_mode,
        "quality_regimes": sorted(parse_regimes(args.quality_regimes)),
        "selector_regimes": sorted(parse_regimes(args.selector_regimes)),
        "gate_counts": pd.DataFrame(gate_rows)["source"].value_counts().to_dict() if gate_rows else {},
        "market_regime": regime_config.get("market_regime", {}) if regime_config else {},
        "annual_volatility_gate": {
            "threshold": args.annual_vol_threshold,
            "window": args.annual_vol_window,
            "min_periods": args.annual_vol_min_periods,
            "lag_days": args.annual_vol_lag_days,
            "selector_if": args.annual_vol_selector_if,
            "observations": int(annual_volatility.notna().sum()) if not annual_volatility.empty else 0,
        },
        "annual_market_state_gate": {
            "volatility_min": args.annual_state_vol_min,
            "momentum_window": args.annual_state_momentum_window,
            "quality_momentum_max": args.annual_state_quality_momentum_max,
            "quality_ret252_min": args.annual_state_quality_ret252_min,
            "observations": int(len(annual_state.dropna(how="all"))) if not annual_state.empty else 0,
        },
    }
    Path(str(output_prefix) + "_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    write_audit_outputs(
        output_prefix=Path(str(output_prefix) + "_audit"),
        yearly=audited_yearly,
        summary=audit_summary,
        metrics=result.metrics,
    )
    print(
        f"quality_selector_gate annual={result.metrics.get('annual_return', 0.0):.4f} "
        f"dd={result.metrics.get('max_drawdown', 0.0):.4f} "
        f"yearly={audit_summary['year_return_pass_count']}/{audit_summary['year_drawdown_pass_count']} "
        f"goal={audit_summary['is_goal_met']}"
    )
    print(f"wrote prefix: {output_prefix}")


def build_quality_scores(config: dict[str, Any], signal_dates: list[pd.Timestamp]) -> pd.Series:
    daily_basic = pd.read_parquet(resolve_path(config["data"]["daily_basic_file"]))
    fina_indicator = normalize_fina_indicator_frame(
        pd.read_parquet(resolve_path(config["fundamentals"]["fina_indicator_file"]))
    )
    dividend = normalize_dividend_frame(pd.read_parquet(resolve_path(config["fundamentals"]["dividend_file"])))
    stock_basic = _load_optional_stock_basic(config)
    covered_symbols = sorted(
        set(fina_indicator["ts_code"].dropna().astype(str))
        | set(dividend["ts_code"].dropna().astype(str))
    )
    daily_basic = daily_basic[daily_basic["ts_code"].astype(str).str.upper().isin(covered_symbols)].copy()
    scores, _diagnostics = build_fundamental_quality_score_panel(
        config=config,
        signal_dates=signal_dates,
        daily_basic=daily_basic,
        fina_indicator=fina_indicator,
        dividend=dividend,
        stock_basic=stock_basic,
        min_total_score=4.0,
        statuses={"PASS", "WATCH"},
    )
    return scores


def build_selector_scores(
    *,
    config: dict[str, Any],
    prices: pd.DataFrame,
    selector_file: str,
    factor_file: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    selector = pd.read_csv(resolve_path(selector_file))
    weights = selector_weights_from_frame(selector)
    columns = sorted({column for series in weights.values() for column in series.index})
    factors = load_or_compute_factors(start_date, end_date, cache_file=factor_file, columns=columns)
    signed_factors = apply_selector_directions(factors, weights)
    scores = composite_factor(
        signed_factors,
        method="ic_weighted",
        factor_weights_dynamic=weights,
        min_obs=int(config.get("strategy", {}).get("min_cross_section_obs", 5)),
    )
    scores = _apply_liquidity_filter(scores, prices, config.get("liquidity_filter", {}))
    return resample_signals(scores, "monthly")


def normalize_score_panel(scores: pd.Series) -> pd.Series:
    if scores.empty:
        return pd.Series(dtype=float, name="score")
    parts: list[pd.Series] = []
    for date, daily_scores in scores.groupby(level=0, sort=True):
        daily = daily_scores.droplevel(0).astype(float).dropna()
        if daily.empty:
            continue
        std = daily.std(ddof=0)
        normalized = (daily - daily.mean()) / (std if std else 1.0)
        normalized.index = pd.MultiIndex.from_product(
            [[pd.Timestamp(date).normalize()], normalized.index.astype(str)],
            names=["date", "instrument"],
        )
        parts.append(normalized.rename("score"))
    return pd.concat(parts).sort_index().rename("score") if parts else pd.Series(dtype=float, name="score")


def gated_scores(
    *,
    quality_scores: pd.Series,
    selector_scores: pd.Series,
    regimes: pd.Series,
    quality_regimes: set[str],
    selector_regimes: set[str],
) -> tuple[pd.Series, list[dict[str, Any]]]:
    dates = sorted(
        set(pd.to_datetime(quality_scores.index.get_level_values(0)).normalize())
        | set(pd.to_datetime(selector_scores.index.get_level_values(0)).normalize())
    )
    parts: list[pd.Series] = []
    rows: list[dict[str, Any]] = []
    for date in dates:
        regime = regime_for_date(regimes, date)
        source = "quality" if regime in quality_regimes else "selector" if regime in selector_regimes else "quality"
        source_scores = quality_scores if source == "quality" else selector_scores
        fallback_scores = selector_scores if source == "quality" else quality_scores
        daily = daily_score_for_date(source_scores, date)
        fallback_used = False
        if daily.empty:
            daily = daily_score_for_date(fallback_scores, date)
            fallback_used = True
        if daily.empty:
            continue
        daily.index = pd.MultiIndex.from_product([[date], daily.index.astype(str)], names=["date", "instrument"])
        parts.append(daily.rename("score"))
        rows.append(
            {
                "date": pd.Timestamp(date).date().isoformat(),
                "regime": regime,
                "source": source,
                "fallback_used": fallback_used,
                "scores": int(daily.notna().sum()),
            }
        )
    result = pd.concat(parts).sort_index().rename("score") if parts else pd.Series(dtype=float, name="score")
    return result, rows


def annual_volatility_gated_scores(
    *,
    quality_scores: pd.Series,
    selector_scores: pd.Series,
    annual_volatility: pd.Series,
    threshold: float,
    selector_if: str,
) -> tuple[pd.Series, list[dict[str, Any]]]:
    dates = sorted(
        set(pd.to_datetime(quality_scores.index.get_level_values(0)).normalize())
        | set(pd.to_datetime(selector_scores.index.get_level_values(0)).normalize())
    )
    source_by_year = annual_volatility_source_by_year(
        dates=dates,
        annual_volatility=annual_volatility,
        threshold=threshold,
        selector_if=selector_if,
    )
    parts: list[pd.Series] = []
    rows: list[dict[str, Any]] = []
    for date in dates:
        decision = source_by_year[pd.Timestamp(date).year]
        source = str(decision["source"])
        source_scores = quality_scores if source == "quality" else selector_scores
        fallback_scores = selector_scores if source == "quality" else quality_scores
        daily = daily_score_for_date(source_scores, date)
        fallback_used = False
        if daily.empty:
            daily = daily_score_for_date(fallback_scores, date)
            fallback_used = True
        if daily.empty:
            continue
        daily.index = pd.MultiIndex.from_product([[date], daily.index.astype(str)], names=["date", "instrument"])
        parts.append(daily.rename("score"))
        rows.append(
            {
                "date": pd.Timestamp(date).date().isoformat(),
                "year": int(pd.Timestamp(date).year),
                "source": source,
                "fallback_used": fallback_used,
                "scores": int(daily.notna().sum()),
                "annual_volatility": decision["annual_volatility"],
                "volatility_date": decision["volatility_date"],
                "threshold": float(threshold),
                "selector_if": selector_if,
            }
        )
    result = pd.concat(parts).sort_index().rename("score") if parts else pd.Series(dtype=float, name="score")
    return result, rows


def annual_market_state_gated_scores(
    *,
    quality_scores: pd.Series,
    selector_scores: pd.Series,
    annual_state: pd.DataFrame,
    volatility_min: float,
    quality_momentum_max: float,
    quality_ret252_min: float,
) -> tuple[pd.Series, list[dict[str, Any]]]:
    dates = sorted(
        set(pd.to_datetime(quality_scores.index.get_level_values(0)).normalize())
        | set(pd.to_datetime(selector_scores.index.get_level_values(0)).normalize())
    )
    source_by_year = annual_market_state_source_by_year(
        dates=dates,
        annual_state=annual_state,
        volatility_min=volatility_min,
        quality_momentum_max=quality_momentum_max,
        quality_ret252_min=quality_ret252_min,
    )
    parts: list[pd.Series] = []
    rows: list[dict[str, Any]] = []
    for date in dates:
        decision = source_by_year[pd.Timestamp(date).year]
        source = str(decision["source"])
        source_scores = quality_scores if source == "quality" else selector_scores
        fallback_scores = selector_scores if source == "quality" else quality_scores
        daily = daily_score_for_date(source_scores, date)
        fallback_used = False
        if daily.empty:
            daily = daily_score_for_date(fallback_scores, date)
            fallback_used = True
        if daily.empty:
            continue
        daily.index = pd.MultiIndex.from_product([[date], daily.index.astype(str)], names=["date", "instrument"])
        parts.append(daily.rename("score"))
        rows.append(
            {
                "date": pd.Timestamp(date).date().isoformat(),
                "year": int(pd.Timestamp(date).year),
                "source": source,
                "fallback_used": fallback_used,
                "scores": int(daily.notna().sum()),
                "state_date": decision["state_date"],
                "annual_volatility": decision["annual_volatility"],
                "ret252": decision["ret252"],
                "momentum": decision["momentum"],
                "volatility_min": float(volatility_min),
                "quality_momentum_max": float(quality_momentum_max),
                "quality_ret252_min": float(quality_ret252_min),
            }
        )
    result = pd.concat(parts).sort_index().rename("score") if parts else pd.Series(dtype=float, name="score")
    return result, rows


def annual_market_state_source_by_year(
    *,
    dates: list[pd.Timestamp],
    annual_state: pd.DataFrame,
    volatility_min: float,
    quality_momentum_max: float,
    quality_ret252_min: float,
) -> dict[int, dict[str, Any]]:
    state = normalize_annual_state_frame(annual_state)
    decisions: dict[int, dict[str, Any]] = {}
    for raw_date in sorted(pd.Timestamp(date).normalize() for date in dates):
        year = int(raw_date.year)
        if year in decisions:
            continue
        row, state_date = annual_state_for_date(state, raw_date)
        values = {} if row is None else row.to_dict()
        source = annual_market_state_source(
            values.get("annual_volatility"),
            values.get("momentum"),
            values.get("ret252"),
            volatility_min=volatility_min,
            quality_momentum_max=quality_momentum_max,
            quality_ret252_min=quality_ret252_min,
        )
        decisions[year] = {
            "year": year,
            "source": source,
            "state_date": state_date.date().isoformat() if state_date is not None else None,
            "annual_volatility": _optional_float(values.get("annual_volatility")),
            "ret252": _optional_float(values.get("ret252")),
            "momentum": _optional_float(values.get("momentum")),
            "decision_date": raw_date.date().isoformat(),
        }
    return decisions


def annual_market_state_source(
    annual_volatility: float | None,
    momentum: float | None,
    ret252: float | None,
    *,
    volatility_min: float,
    quality_momentum_max: float,
    quality_ret252_min: float,
) -> str:
    if annual_volatility is None or momentum is None or ret252 is None:
        return "quality"
    if pd.isna(annual_volatility) or pd.isna(momentum) or pd.isna(ret252):
        return "quality"
    use_quality = (
        float(annual_volatility) >= float(volatility_min)
        and float(momentum) <= float(quality_momentum_max)
        and float(ret252) >= float(quality_ret252_min)
    )
    return "quality" if use_quality else "selector"


def annual_state_for_date(annual_state: pd.DataFrame, date: pd.Timestamp) -> tuple[pd.Series | None, pd.Timestamp | None]:
    if annual_state.empty:
        return None, None
    target = pd.Timestamp(date).normalize()
    eligible = annual_state.loc[annual_state.index <= target].dropna(how="all")
    if eligible.empty:
        return None, None
    return eligible.iloc[-1], pd.Timestamp(eligible.index[-1])


def normalize_annual_state_frame(annual_state: pd.DataFrame) -> pd.DataFrame:
    if annual_state.empty:
        return pd.DataFrame()
    result = annual_state.copy()
    result.index = pd.to_datetime(result.index).normalize()
    result = result.sort_index()
    result = result[~result.index.duplicated(keep="last")]
    for column in result.columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def annual_volatility_source_by_year(
    *,
    dates: list[pd.Timestamp],
    annual_volatility: pd.Series,
    threshold: float,
    selector_if: str,
) -> dict[int, dict[str, Any]]:
    if selector_if not in {"below", "above"}:
        raise ValueError("selector_if must be 'below' or 'above'")
    normalized_volatility = annual_volatility.copy()
    if not normalized_volatility.empty:
        normalized_volatility.index = pd.to_datetime(normalized_volatility.index).normalize()
        normalized_volatility = pd.to_numeric(normalized_volatility, errors="coerce").sort_index()
        normalized_volatility = normalized_volatility[~normalized_volatility.index.duplicated(keep="last")]
    decisions: dict[int, dict[str, Any]] = {}
    for raw_date in sorted(pd.Timestamp(date).normalize() for date in dates):
        year = int(raw_date.year)
        if year in decisions:
            continue
        value, value_date = annual_volatility_for_date(normalized_volatility, raw_date)
        source = annual_volatility_source(value, threshold=threshold, selector_if=selector_if)
        decisions[year] = {
            "year": year,
            "source": source,
            "annual_volatility": value,
            "volatility_date": value_date.date().isoformat() if value_date is not None else None,
            "threshold": float(threshold),
            "selector_if": selector_if,
            "decision_date": raw_date.date().isoformat(),
        }
    return decisions


def annual_volatility_source(value: float | None, *, threshold: float, selector_if: str) -> str:
    if value is None or pd.isna(value):
        return "quality"
    volatility = float(value)
    if selector_if == "below":
        return "selector" if volatility <= float(threshold) else "quality"
    if selector_if == "above":
        return "selector" if volatility >= float(threshold) else "quality"
    raise ValueError("selector_if must be 'below' or 'above'")


def annual_volatility_for_date(annual_volatility: pd.Series, date: pd.Timestamp) -> tuple[float | None, pd.Timestamp | None]:
    if annual_volatility.empty:
        return None, None
    target = pd.Timestamp(date).normalize()
    eligible = annual_volatility.loc[annual_volatility.index <= target].dropna()
    if eligible.empty:
        return None, None
    return float(eligible.iloc[-1]), pd.Timestamp(eligible.index[-1])


def benchmark_annual_volatility(
    *,
    prices: pd.DataFrame,
    config: dict[str, Any],
    window: int,
    min_periods: int,
    lag_days: int,
) -> pd.Series:
    benchmark = _benchmark_close(prices, config, config.get("market_regime", {}))
    if benchmark.empty:
        return pd.Series(dtype=float, name="annual_volatility")
    benchmark = pd.to_numeric(benchmark, errors="coerce").dropna().sort_index()
    benchmark.index = pd.to_datetime(benchmark.index).normalize()
    benchmark = benchmark[~benchmark.index.duplicated(keep="last")]
    safe_window = max(1, int(window))
    safe_min_periods = max(1, min(int(min_periods), safe_window))
    volatility = benchmark.pct_change(fill_method=None).rolling(safe_window, min_periods=safe_min_periods).std(ddof=0)
    volatility = volatility * (252 ** 0.5)
    if lag_days > 0:
        volatility = volatility.shift(int(lag_days))
    return volatility.rename("annual_volatility")


def benchmark_annual_state(
    *,
    prices: pd.DataFrame,
    config: dict[str, Any],
    volatility_window: int,
    volatility_min_periods: int,
    momentum_window: int,
    lag_days: int,
) -> pd.DataFrame:
    benchmark = _benchmark_close(prices, config, config.get("market_regime", {}))
    if benchmark.empty:
        return pd.DataFrame(columns=["annual_volatility", "ret252", "momentum"])
    benchmark = pd.to_numeric(benchmark, errors="coerce").dropna().sort_index()
    benchmark.index = pd.to_datetime(benchmark.index).normalize()
    benchmark = benchmark[~benchmark.index.duplicated(keep="last")]
    safe_vol_window = max(1, int(volatility_window))
    safe_min_periods = max(1, min(int(volatility_min_periods), safe_vol_window))
    safe_momentum_window = max(1, int(momentum_window))
    state = pd.DataFrame(index=benchmark.index)
    state["annual_volatility"] = (
        benchmark.pct_change(fill_method=None)
        .rolling(safe_vol_window, min_periods=safe_min_periods)
        .std(ddof=0)
        * (252 ** 0.5)
    )
    state["ret252"] = benchmark.pct_change(252)
    state["momentum"] = benchmark.pct_change(safe_momentum_window)
    if lag_days > 0:
        state = state.shift(int(lag_days))
    return state


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def daily_score_for_date(scores: pd.Series, date: pd.Timestamp) -> pd.Series:
    if scores.empty:
        return pd.Series(dtype=float)
    key = pd.Timestamp(date).normalize()
    if key not in set(pd.to_datetime(scores.index.get_level_values(0)).normalize()):
        return pd.Series(dtype=float)
    try:
        return scores.xs(key, level=0, drop_level=True).dropna()
    except KeyError:
        return pd.Series(dtype=float)


def market_regime_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    result = dict(config)
    market_cfg = {**config.get("market_regime", {}), "enabled": True}
    if args.market_ma_window is not None:
        market_cfg["ma_window"] = args.market_ma_window
    if args.market_momentum_window is not None:
        market_cfg["momentum_window"] = args.market_momentum_window
    if args.market_bear_drawdown is not None:
        market_cfg["bear_drawdown_threshold"] = args.market_bear_drawdown
    if args.market_drawdown_window is not None:
        market_cfg["drawdown_window"] = args.market_drawdown_window
    if args.market_lag_days is not None:
        market_cfg["lag_days"] = args.market_lag_days
    result["market_regime"] = market_cfg
    return result


def parse_regimes(value: str) -> set[str]:
    return {item.strip().lower() for item in str(value or "").split(",") if item.strip()}


if __name__ == "__main__":
    main()
