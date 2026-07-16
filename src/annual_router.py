"""Reusable annual market-state routing and schedule construction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from src.risk_policy import RiskPolicy


ANNUAL_ROUTER_ENGINE_CONTRACT = {
    "version": 2,
    "signal_calendar": "canonical_month_end",
    "score_lookup": "latest_on_or_before",
    "selection_schedule": True,
}


@dataclass(frozen=True)
class ScoreSourceDefinition:
    name: str
    kind: str
    factor_group: str = ""
    factor_file: str = ""
    selector_file: str = ""
    top_n: int = 5
    max_turnover: int = 1
    rank_buffer: int = 20
    liquidity_quantile: float | None = None


@dataclass(frozen=True)
class RoutedScoreRun:
    scores: pd.Series
    score_routes: pd.DataFrame
    year_routes: pd.DataFrame


class AnnualRouterRun:
    def __init__(self, equity: pd.Series, routes: pd.DataFrame) -> None:
        self.equity = equity
        self.routes = routes


def run_annual_state_router(
    *,
    source_returns: pd.DataFrame,
    benchmark: pd.Series,
    initial_source: str,
    missing_ret252_exposure: float,
    flat_negative_exposure: float,
) -> AnnualRouterRun:
    """Route among source return streams once per calendar year."""
    if source_returns.empty:
        raise ValueError("source_returns must not be empty")
    if initial_source not in source_returns.columns:
        raise ValueError(f"initial_source is not in source returns: {initial_source}")
    normalized_benchmark = normalize_benchmark(benchmark)
    benchmark_returns = normalized_benchmark.pct_change(fill_method=None)
    current_value = 1.0
    current_source = initial_source
    current_exposure = 1.0
    values: list[float] = []
    route_rows: list[dict[str, Any]] = []
    for raw_date, daily_returns in source_returns.sort_index().iterrows():
        date = pd.Timestamp(raw_date).normalize()
        if not route_rows or int(date.year) != int(route_rows[-1]["year"]):
            decision = route_for_date(
                benchmark=normalized_benchmark,
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
    """Choose the route using only benchmark history before the decision date."""
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
    """Map trailing benchmark state to the configured score-source family."""
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


def run_annual_state_score_router(
    *,
    score_sources: dict[str, pd.Series],
    source_definitions: dict[str, ScoreSourceDefinition],
    price_dates: pd.DatetimeIndex,
    benchmark: pd.Series,
    initial_source: str,
    missing_ret252_exposure: float,
    flat_negative_exposure: float,
    signal_dates: list[pd.Timestamp] | None = None,
    fallback_source: str | None = None,
    moderate_positive_source: str | None = None,
    moderate_positive_ret252_min: float = 0.20,
    moderate_positive_exposure: float = 1.0,
    moderate_low_source: str | None = None,
    moderate_low_ret252_min: float = 0.18,
    moderate_low_ret252_max: float = 0.20,
    moderate_low_exposure: float = 1.0,
    moderate_lower_source: str | None = None,
    moderate_lower_ret252_min: float = 0.16,
    moderate_lower_ret252_max: float = 0.18,
    moderate_lower_exposure: float = 1.0,
    strong_trailing_exposure: float = 1.0,
    turnover_boost_reasons: set[str] | None = None,
    turnover_boost_max_turnover: int = 2,
    turnover_boost_rank_buffer: int = 10,
) -> RoutedScoreRun:
    if initial_source not in score_sources:
        raise ValueError(f"initial_source is not in score sources: {initial_source}")
    dates = (
        sorted({pd.Timestamp(date).normalize() for date in signal_dates})
        if signal_dates is not None
        else routed_signal_dates(score_sources)
    )
    normalized_price_dates = pd.DatetimeIndex(pd.to_datetime(price_dates).normalize()).unique().sort_values()
    signal_trade_dates = signal_trade_date_map(dates, normalized_price_dates)
    year_routes = annual_route_decisions(
        years=sorted({int(trade_date.year) for trade_date in signal_trade_dates.values()}),
        price_dates=normalized_price_dates,
        benchmark=benchmark,
        initial_source=initial_source,
        missing_ret252_exposure=missing_ret252_exposure,
        flat_negative_exposure=flat_negative_exposure,
        moderate_positive_source=moderate_positive_source,
        moderate_positive_ret252_min=moderate_positive_ret252_min,
        moderate_positive_exposure=moderate_positive_exposure,
        moderate_low_source=moderate_low_source,
        moderate_low_ret252_min=moderate_low_ret252_min,
        moderate_low_ret252_max=moderate_low_ret252_max,
        moderate_low_exposure=moderate_low_exposure,
        moderate_lower_source=moderate_lower_source,
        moderate_lower_ret252_min=moderate_lower_ret252_min,
        moderate_lower_ret252_max=moderate_lower_ret252_max,
        moderate_lower_exposure=moderate_lower_exposure,
        strong_trailing_exposure=strong_trailing_exposure,
    )
    route_by_year = {int(row["year"]): row for row in year_routes}
    boost_reasons = set(turnover_boost_reasons or set())
    parts: list[pd.Series] = []
    rows: list[dict[str, Any]] = []
    for date in dates:
        trade_date = signal_trade_dates.get(date)
        if trade_date is None:
            continue
        route_year = int(trade_date.year)
        decision = route_by_year[route_year]
        source = str(decision["source"])
        if source not in score_sources:
            raise ValueError(f"Routed source is not in score sources: {source}")
        actual_source = source
        daily = latest_score_on_or_before(score_sources[source], date)
        fallback_used = False
        if daily.empty and fallback_source:
            if fallback_source not in score_sources:
                raise ValueError(f"fallback_source is not in score sources: {fallback_source}")
            actual_source = fallback_source
            daily = latest_score_on_or_before(score_sources[fallback_source], date)
            fallback_used = True
        if daily.empty:
            raise ValueError(f"No scores for source={source} date={date.date()}.")
        daily.index = pd.MultiIndex.from_product([[date], daily.index.astype(str)], names=["date", "instrument"])
        parts.append(daily.rename("score"))
        definition = source_definitions[actual_source]
        top_n = int(definition.top_n)
        max_turnover = int(definition.max_turnover)
        rank_buffer = int(definition.rank_buffer)
        if str(decision["reason"]) in boost_reasons:
            max_turnover = min(top_n, max(1, int(turnover_boost_max_turnover)))
            rank_buffer = max(0, int(turnover_boost_rank_buffer))
        rows.append(
            {
                "date": date.date().isoformat(),
                "trade_date": trade_date.date().isoformat(),
                "signal_year": int(date.year),
                "year": route_year,
                "source": actual_source,
                "routed_source": source,
                "reason": decision["reason"],
                "fallback_used": fallback_used,
                "scores": int(daily.notna().sum()),
                "top_n": top_n,
                "max_turnover": max_turnover,
                "rank_buffer": rank_buffer,
                "exposure": float(decision["exposure"]),
            }
        )
    scores = pd.concat(parts).sort_index().rename("score") if parts else pd.Series(dtype=float, name="score")
    return RoutedScoreRun(scores=scores, score_routes=pd.DataFrame(rows), year_routes=pd.DataFrame(route_by_year.values()))


def latest_score_on_or_before(scores: pd.Series, date: pd.Timestamp) -> pd.Series:
    """Read the newest point-in-time score cross-section available by a signal date."""
    if scores.empty or not isinstance(scores.index, pd.MultiIndex):
        return pd.Series(dtype=float, name=scores.name)
    target = pd.Timestamp(date).normalize()
    dates = pd.DatetimeIndex(pd.to_datetime(scores.index.get_level_values(0)).normalize())
    eligible = dates[dates <= target]
    if eligible.empty:
        return pd.Series(dtype=float, name=scores.name)
    key = pd.Timestamp(eligible.max()).normalize()
    try:
        return scores.xs(key, level=0, drop_level=True).dropna()
    except KeyError:
        return pd.Series(dtype=float, name=scores.name)


def routed_signal_dates(score_sources: dict[str, pd.Series]) -> list[pd.Timestamp]:
    dates: set[pd.Timestamp] = set()
    for source, scores in score_sources.items():
        if scores.empty:
            raise ValueError(f"Score source is empty: {source}")
        if not isinstance(scores.index, pd.MultiIndex):
            raise ValueError(f"Score source must use MultiIndex date/instrument: {source}")
        dates.update(pd.Timestamp(value).normalize() for value in pd.to_datetime(scores.index.get_level_values(0)).unique())
    return sorted(dates)


def signal_trade_date_map(
    signal_dates: list[pd.Timestamp],
    price_dates: pd.DatetimeIndex,
) -> dict[pd.Timestamp, pd.Timestamp]:
    normalized_price_dates = pd.DatetimeIndex(pd.to_datetime(price_dates).normalize()).unique().sort_values()
    result: dict[pd.Timestamp, pd.Timestamp] = {}
    for raw_date in sorted(pd.Timestamp(date).normalize() for date in signal_dates):
        pos = normalized_price_dates.searchsorted(raw_date, side="right")
        if pos >= len(normalized_price_dates):
            continue
        result[raw_date] = pd.Timestamp(normalized_price_dates[pos]).normalize()
    return result


def annual_route_decisions(
    *,
    years: list[int],
    price_dates: pd.DatetimeIndex,
    benchmark: pd.Series,
    initial_source: str,
    missing_ret252_exposure: float,
    flat_negative_exposure: float,
    moderate_positive_source: str | None = None,
    moderate_positive_ret252_min: float = 0.20,
    moderate_positive_exposure: float = 1.0,
    moderate_low_source: str | None = None,
    moderate_low_ret252_min: float = 0.18,
    moderate_low_ret252_max: float = 0.20,
    moderate_low_exposure: float = 1.0,
    moderate_lower_source: str | None = None,
    moderate_lower_ret252_min: float = 0.16,
    moderate_lower_ret252_max: float = 0.18,
    moderate_lower_exposure: float = 1.0,
    strong_trailing_exposure: float = 1.0,
) -> list[dict[str, Any]]:
    normalized_benchmark = pd.to_numeric(benchmark, errors="coerce").dropna().sort_index()
    normalized_benchmark.index = pd.to_datetime(normalized_benchmark.index).normalize()
    benchmark_returns = normalized_benchmark.pct_change(fill_method=None)
    normalized_price_dates = pd.DatetimeIndex(pd.to_datetime(price_dates).normalize()).unique().sort_values()
    rows: list[dict[str, Any]] = []
    for year in sorted(set(int(year) for year in years)):
        year_dates = normalized_price_dates[normalized_price_dates.year == year]
        if year_dates.empty:
            raise ValueError(f"No price dates found for route year {year}.")
        route = route_for_date(
            benchmark=normalized_benchmark,
            benchmark_returns=benchmark_returns,
            date=pd.Timestamp(year_dates[0]).normalize(),
            initial_source=initial_source,
            missing_ret252_exposure=missing_ret252_exposure,
            flat_negative_exposure=flat_negative_exposure,
        )
        rows.append(
            adjust_route_decision(
                route,
                moderate_positive_source=moderate_positive_source,
                moderate_positive_ret252_min=moderate_positive_ret252_min,
                moderate_positive_exposure=moderate_positive_exposure,
                moderate_low_source=moderate_low_source,
                moderate_low_ret252_min=moderate_low_ret252_min,
                moderate_low_ret252_max=moderate_low_ret252_max,
                moderate_low_exposure=moderate_low_exposure,
                moderate_lower_source=moderate_lower_source,
                moderate_lower_ret252_min=moderate_lower_ret252_min,
                moderate_lower_ret252_max=moderate_lower_ret252_max,
                moderate_lower_exposure=moderate_lower_exposure,
                strong_trailing_exposure=strong_trailing_exposure,
            )
        )
    return rows


def adjust_route_decision(
    route: dict[str, Any],
    *,
    moderate_positive_source: str | None,
    moderate_positive_ret252_min: float,
    moderate_positive_exposure: float = 1.0,
    moderate_low_source: str | None = None,
    moderate_low_ret252_min: float = 0.18,
    moderate_low_ret252_max: float = 0.20,
    moderate_low_exposure: float = 1.0,
    moderate_lower_source: str | None = None,
    moderate_lower_ret252_min: float = 0.16,
    moderate_lower_ret252_max: float = 0.18,
    moderate_lower_exposure: float = 1.0,
    strong_trailing_exposure: float = 1.0,
) -> dict[str, Any]:
    result = dict(route)
    if result.get("reason") == "strong_trailing_market":
        result["exposure"] = float(result.get("exposure", 1.0)) * float(strong_trailing_exposure)
    ret252 = pd.to_numeric(pd.Series([result.get("ret252")]), errors="coerce").iloc[0]
    if (
        moderate_lower_source
        and result.get("reason") == "default_beta"
        and pd.notna(ret252)
        and float(ret252) >= float(moderate_lower_ret252_min)
        and float(ret252) < float(moderate_lower_ret252_max)
    ):
        result["source"] = moderate_lower_source
        result["reason"] = f"moderate_lower_{moderate_lower_source}"
        result["exposure"] = float(result.get("exposure", 1.0)) * float(moderate_lower_exposure)
        return result
    if (
        moderate_low_source
        and result.get("reason") == "default_beta"
        and pd.notna(ret252)
        and float(ret252) >= float(moderate_low_ret252_min)
        and float(ret252) < float(moderate_low_ret252_max)
    ):
        result["source"] = moderate_low_source
        result["reason"] = f"moderate_low_{moderate_low_source}"
        result["exposure"] = float(result.get("exposure", 1.0)) * float(moderate_low_exposure)
        return result
    if (
        moderate_positive_source
        and result.get("reason") == "default_beta"
        and pd.notna(ret252)
        and float(ret252) >= float(moderate_positive_ret252_min)
    ):
        result["source"] = moderate_positive_source
        result["reason"] = f"moderate_positive_{moderate_positive_source}"
        result["exposure"] = float(result.get("exposure", 1.0)) * float(moderate_positive_exposure)
    return result


def definitions_for_turnover_mode(
    definitions: dict[str, ScoreSourceDefinition],
    mode: str,
) -> dict[str, ScoreSourceDefinition]:
    mode = str(mode or "default").strip().lower()
    if mode == "default":
        return dict(definitions)
    result = dict(definitions)
    if mode == "turnover2":
        for name, definition in list(result.items()):
            result[name] = replace(
                definition,
                max_turnover=min(int(definition.top_n), max(2, int(definition.max_turnover))),
                rank_buffer=min(int(definition.rank_buffer), 10),
            )
        return result
    if mode == "rank10":
        for name, definition in list(result.items()):
            result[name] = replace(definition, rank_buffer=min(int(definition.rank_buffer), 10))
        return result
    if mode == "full":
        for name, definition in list(result.items()):
            result[name] = replace(definition, max_turnover=int(definition.top_n), rank_buffer=0)
        return result
    raise ValueError(f"Unsupported turnover mode: {mode}")


def routed_backtest_config(
    *,
    config: dict[str, Any],
    prices: pd.DataFrame,
    routed: RoutedScoreRun,
    source_definitions: dict[str, ScoreSourceDefinition],
    full_turnover_on_route_change: bool,
    use_defensive_timing: bool,
    disable_equity_overlay: bool = False,
) -> dict[str, Any]:
    max_top_n = max(int(definition.top_n) for definition in source_definitions.values())
    max_turnover = max(int(definition.max_turnover) for definition in source_definitions.values())
    max_rank_buffer = max(int(definition.rank_buffer) for definition in source_definitions.values())
    bt_config = {
        **config.get("backtest", {}),
        **config.get("strategy", {}),
        "top_n": max_top_n,
        "max_turnover": max_turnover,
        "rank_buffer": max_rank_buffer,
        "rebalance_freq": "monthly",
        "selection_schedule": selection_schedule_from_routes(
            routed.score_routes,
            full_turnover_on_route_change=full_turnover_on_route_change,
        ),
        "exposure_schedule": exposure_schedule_from_year_routes(routed.year_routes),
    }
    router_cfg = config.get("annual_state_router", {}) if isinstance(config.get("annual_state_router", {}), dict) else {}
    strategy_cfg = config.get("strategy", {}) if isinstance(config.get("strategy", {}), dict) else {}
    min_position_reasons = parse_reason_list(
        strategy_cfg.get("risk_exit_min_positions_reasons")
        or router_cfg.get("risk_exit_min_positions_reasons")
        or []
    )
    min_positions = strategy_cfg.get("risk_exit_min_positions", router_cfg.get("risk_exit_min_positions"))
    if min_positions is not None and min_position_reasons:
        bt_config["risk_exit_min_positions"] = 0
        bt_config["risk_exit_min_positions_schedule"] = risk_exit_min_positions_schedule_from_routes(
            routed.score_routes,
            min_positions=int(min_positions),
            reasons=min_position_reasons,
        )
    if use_defensive_timing:
        from src.market_regime import apply_defensive_timing_to_backtest_config

        bt_config = apply_defensive_timing_to_backtest_config(bt_config, prices, config)
    if disable_equity_overlay and isinstance(bt_config.get("equity_overlay"), dict):
        bt_config["equity_overlay"] = {**bt_config["equity_overlay"], "enabled": False}
    return RiskPolicy(config).apply_to_backtest_config(bt_config)


def parse_reason_list(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    normalized = str(value or "").replace("+", ",")
    return {item.strip() for item in normalized.split(",") if item.strip()}


def selection_schedule_from_routes(
    routes: pd.DataFrame,
    *,
    full_turnover_on_route_change: bool,
) -> dict[str, dict[str, int]]:
    if routes.empty:
        return {}
    result: dict[str, dict[str, int]] = {}
    previous_source: str | None = None
    for _, row in routes.sort_values("date").iterrows():
        source = str(row["source"])
        top_n = int(row["top_n"])
        max_turnover = int(row["max_turnover"])
        rank_buffer = int(row["rank_buffer"])
        if full_turnover_on_route_change and previous_source is not None and source != previous_source:
            max_turnover = top_n
            rank_buffer = 0
        result[str(row["date"])] = {
            "top_n": top_n,
            "max_turnover": max_turnover,
            "rank_buffer": rank_buffer,
        }
        previous_source = source
    return result


def exposure_schedule_from_year_routes(year_routes: pd.DataFrame) -> dict[str, float]:
    if year_routes.empty:
        return {}
    return {
        str(row["decision_date"]): float(row["exposure"])
        for _, row in year_routes.sort_values("decision_date").iterrows()
    }


def risk_exit_min_positions_schedule_from_routes(
    routes: pd.DataFrame,
    *,
    min_positions: int,
    reasons: set[str],
) -> dict[str, int]:
    if routes.empty or min_positions <= 0 or not reasons:
        return {}
    result: dict[str, int] = {}
    for _, row in routes.sort_values("date").iterrows():
        if str(row.get("reason", "")) in reasons:
            result[str(row["date"])] = int(min_positions)
    return result
