from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from itertools import product
import logging
from typing import Iterable

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.factor_ic import calculate_factor_ic, calculate_rolling_ic, make_ic_weights, make_rolling_ic_weights, summarize_ic
from src.scoring import build_strategy_scores
from src.strategy import composite_factor, resample_signals


logger = logging.getLogger(__name__)


STRATEGY_GRID_KEYS = ("factor_group", "top_n", "max_turnover", "rank_buffer", "rebalance_freq")
RISK_GRID_KEYS = (
    "max_weight_per_stock",
    "stop_loss_pct",
    "take_profit_pct",
    "circuit_breaker_drawdown",
    "circuit_breaker_cooldown_days",
    "circuit_breaker_target_exposure",
    "target_vol",
    "max_industry_weight",
    "rebalance_drift_threshold",
)

DEFAULT_GRID = {
    "factor_group": ["ic_weighted", "momentum"],
    "top_n": [5, 7, 10],
    "max_turnover": [1],
    "rank_buffer": [20, 30],
    "rebalance_freq": ["weekly", "monthly"],
}

BASELINE_GRID = {
    "factor_group": ["momentum", "factor:LOW0"],
    "top_n": [7, 10, 20],
    "max_turnover": [1],
    "rank_buffer": [30],
    "rebalance_freq": ["monthly"],
    "rebalance_drift_threshold": [0.0, 0.02, 0.05],
}


def run_parameter_grid(
    factor_df: pd.DataFrame,
    price_df: pd.DataFrame,
    base_config: dict,
    start_date: str,
    end_date: str,
    grid: dict[str, Iterable] | None = None,
    ic_weights: pd.Series | None = None,
    use_rolling_ic: bool = False,
    ic_horizon: int = 1,
    ic_method: str = "spearman",
    ic_min_obs: int = 20,
    ic_window: int = 252,
    ic_min_periods: int = 60,
    ic_min_abs: float = 0.02,
    ic_corr_threshold: float = 0.7,
    ic_top_k: int = 30,
    ic_weight_smoothing: float = 0.0,
    ic_max_weight_turnover: float | None = None,
    turnover_penalty: float = 0.02,
    cost_penalty: float = 1.0,
    target_annual_return: float | None = 0.20,
    min_annual_return: float | None = 0.18,
    drawdown_limit: float | None = -0.20,
    drawdown_penalty: float = 2.0,
    annual_return_weight: float = 0.5,
    calmar_weight: float = 0.25,
    scoring_config: dict | None = None,
    on_result: Callable[[dict[str, object], pd.DataFrame], None] | None = None,
) -> pd.DataFrame:
    grid = grid or DEFAULT_GRID
    dynamic_weights = None
    if "ic_weighted" in set(grid.get("factor_group", [])) and use_rolling_ic:
        rolling_ic = calculate_rolling_ic(
            factor_df,
            price_df,
            horizon=ic_horizon,
            method=ic_method,
            window=ic_window,
            min_periods=ic_min_periods,
            min_obs=ic_min_obs,
        )
        dynamic_weights = make_rolling_ic_weights(
            rolling_ic,
            top_k=ic_top_k,
            min_abs_ic=ic_min_abs,
            min_periods=ic_min_periods,
            correlation_threshold=ic_corr_threshold,
            weight_smoothing=ic_weight_smoothing,
            max_weight_turnover=ic_max_weight_turnover,
        )
    elif "ic_weighted" in set(grid.get("factor_group", [])) and ic_weights is None:
        ic_df = calculate_factor_ic(factor_df, price_df, horizon=ic_horizon, method=ic_method, min_obs=ic_min_obs)
        ic_weights = make_ic_weights(summarize_ic(ic_df), top_k=ic_top_k, min_abs_ic=ic_min_abs)

    score_cache: dict[tuple[str, str], pd.Series] = {}
    rows: list[dict[str, object]] = []
    keys = list(grid)
    for values in product(*(grid[key] for key in keys)):
        params = dict(zip(keys, values))
        factor_group = str(params["factor_group"])
        rebalance_freq = str(params.get("rebalance_freq", "daily"))
        cache_key = (factor_group, rebalance_freq)
        if cache_key not in score_cache:
            logger.info("Building scores for factor_group=%s rebalance_freq=%s.", factor_group, rebalance_freq)
            weights = ic_weights if factor_group == "ic_weighted" else None
            dynamic = dynamic_weights if factor_group == "ic_weighted" else None
            if factor_group == "ic_weighted":
                scores = composite_factor(factor_df, method=factor_group, factor_weights=weights, factor_weights_dynamic=dynamic)
            else:
                scores = build_strategy_scores(factor_df, _scoring_config(scoring_config, params), price_df=price_df)
            score_cache[cache_key] = resample_signals(scores, rebalance_freq)

        bt_config = {**base_config, **params}
        logger.info(
            "Running optimization combo: factor_group=%s top_n=%s max_turnover=%s rank_buffer=%s rebalance_freq=%s.",
            factor_group,
            params.get("top_n"),
            params.get("max_turnover"),
            params.get("rank_buffer"),
            rebalance_freq,
        )
        result = run_backtest(score_cache[cache_key], price_df, start_date, end_date, bt_config)
        row = {
            **params,
            **result.metrics,
            "optimization_score": _optimization_score(
                result.metrics,
                turnover_penalty=turnover_penalty,
                cost_penalty=cost_penalty,
                target_annual_return=target_annual_return,
                min_annual_return=min_annual_return,
                drawdown_limit=drawdown_limit,
                drawdown_penalty=drawdown_penalty,
                annual_return_weight=annual_return_weight,
                calmar_weight=calmar_weight,
            ),
        }
        rows.append(row)
        if on_result is not None:
            on_result(row, _sorted_results(rows))

    return _sorted_results(rows)


def run_walk_forward_optimization(
    factor_df: pd.DataFrame,
    price_df: pd.DataFrame,
    base_config: dict,
    start_date: str,
    end_date: str,
    grid: dict[str, Iterable] | None = None,
    train_years: int = 3,
    test_months: int = 12,
    step_months: int = 6,
    use_rolling_ic: bool = True,
    ic_horizon: int = 1,
    ic_method: str = "spearman",
    ic_min_obs: int = 20,
    ic_window: int = 252,
    ic_min_periods: int = 60,
    ic_min_abs: float = 0.02,
    ic_corr_threshold: float = 0.7,
    ic_top_k: int = 30,
    ic_weight_smoothing: float = 0.0,
    ic_max_weight_turnover: float | None = None,
    turnover_penalty: float = 0.02,
    cost_penalty: float = 1.0,
    target_annual_return: float | None = 0.20,
    min_annual_return: float | None = 0.18,
    drawdown_limit: float | None = -0.20,
    drawdown_penalty: float = 2.0,
    annual_return_weight: float = 0.5,
    calmar_weight: float = 0.25,
    on_result: Callable[[dict[str, object], pd.DataFrame], None] | None = None,
) -> pd.DataFrame:
    price_df = _normalize_window_price_frame(price_df)
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    rows: list[dict[str, object]] = []
    train_start = start

    while True:
        train_end = train_start + pd.DateOffset(years=train_years) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = min(test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1), end)
        if test_start > end or train_end >= end:
            break

        train_factors = _slice_factor_dates(factor_df, train_start, train_end)
        test_factors = _slice_factor_dates(factor_df, test_start, test_end)
        train_prices = price_df.loc[(price_df.index >= train_start) & (price_df.index <= train_end)]
        test_prices = price_df.loc[(price_df.index >= test_start) & (price_df.index <= test_end)]
        if train_factors.empty or test_factors.empty or train_prices.empty or test_prices.empty:
            train_start += pd.DateOffset(months=step_months)
            continue

        train_results = run_parameter_grid(
            train_factors,
            train_prices,
            base_config=base_config,
            start_date=train_start.strftime("%Y-%m-%d"),
            end_date=train_end.strftime("%Y-%m-%d"),
            grid=grid,
            use_rolling_ic=use_rolling_ic,
            ic_horizon=ic_horizon,
            ic_method=ic_method,
            ic_min_obs=ic_min_obs,
            ic_window=ic_window,
            ic_min_periods=ic_min_periods,
            ic_min_abs=ic_min_abs,
            ic_corr_threshold=ic_corr_threshold,
            ic_top_k=ic_top_k,
            ic_weight_smoothing=ic_weight_smoothing,
            ic_max_weight_turnover=ic_max_weight_turnover,
            turnover_penalty=turnover_penalty,
            cost_penalty=cost_penalty,
            target_annual_return=target_annual_return,
            min_annual_return=min_annual_return,
            drawdown_limit=drawdown_limit,
            drawdown_penalty=drawdown_penalty,
            annual_return_weight=annual_return_weight,
            calmar_weight=calmar_weight,
        )
        if train_results.empty:
            train_start += pd.DateOffset(months=step_months)
            continue

        params = train_results.iloc[0][list(grid or DEFAULT_GRID)].to_dict()
        factor_group = str(params["factor_group"])
        rebalance_freq = str(params.get("rebalance_freq", "daily"))
        weights = None
        dynamic_weights = None
        if factor_group == "ic_weighted":
            if use_rolling_ic:
                rolling_ic = calculate_rolling_ic(
                    train_factors,
                    train_prices,
                    horizon=ic_horizon,
                    method=ic_method,
                    window=ic_window,
                    min_periods=ic_min_periods,
                    min_obs=ic_min_obs,
                )
                train_dynamic_weights = make_rolling_ic_weights(
                    rolling_ic,
                    top_k=ic_top_k,
                    min_abs_ic=ic_min_abs,
                    min_periods=ic_min_periods,
                    correlation_threshold=ic_corr_threshold,
                    weight_smoothing=ic_weight_smoothing,
                    max_weight_turnover=ic_max_weight_turnover,
                )
                last_weights = _last_dynamic_weights(train_dynamic_weights)
                dynamic_weights = {pd.Timestamp(date).normalize(): last_weights for date in pd.to_datetime(test_factors.index.get_level_values(0).unique())}
                score_source = test_factors
            else:
                ic_df = calculate_factor_ic(train_factors, train_prices, horizon=ic_horizon, method=ic_method, min_obs=ic_min_obs)
                weights = make_ic_weights(summarize_ic(ic_df), top_k=ic_top_k, min_abs_ic=ic_min_abs)
                score_source = test_factors
        else:
            score_source = test_factors
        if factor_group == "ic_weighted" and use_rolling_ic:
            scores = composite_factor(score_source, method=factor_group, factor_weights_dynamic=dynamic_weights)
        elif factor_group == "ic_weighted":
            scores = composite_factor(score_source, method=factor_group, factor_weights=weights)
        else:
            scores = build_strategy_scores(score_source, {"strategy": {"factor_group": factor_group}}, price_df=test_prices)
        scores = resample_signals(scores, rebalance_freq)
        scores = _slice_score_dates(scores, test_start, test_end)
        result = run_backtest(
            scores,
            test_prices,
            test_start.strftime("%Y-%m-%d"),
            test_end.strftime("%Y-%m-%d"),
            {**base_config, **params},
        )
        row = {
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            **params,
            **result.metrics,
            "optimization_score": _optimization_score(
                result.metrics,
                turnover_penalty=turnover_penalty,
                cost_penalty=cost_penalty,
                target_annual_return=target_annual_return,
                min_annual_return=min_annual_return,
                drawdown_limit=drawdown_limit,
                drawdown_penalty=drawdown_penalty,
                annual_return_weight=annual_return_weight,
                calmar_weight=calmar_weight,
            ),
        }
        rows.append(row)
        if on_result is not None:
            on_result(row, pd.DataFrame(rows))
        train_start += pd.DateOffset(months=step_months)

    return pd.DataFrame(rows)


def run_walk_forward_grid_validation(
    factor_df: pd.DataFrame,
    price_df: pd.DataFrame,
    base_config: dict,
    start_date: str,
    end_date: str,
    grid: dict[str, Iterable] | None = None,
    train_years: int = 3,
    test_months: int = 12,
    step_months: int = 6,
    use_rolling_ic: bool = True,
    ic_horizon: int = 1,
    ic_method: str = "spearman",
    ic_min_obs: int = 20,
    ic_window: int = 252,
    ic_min_periods: int = 60,
    ic_min_abs: float = 0.02,
    ic_corr_threshold: float = 0.7,
    ic_top_k: int = 30,
    ic_weight_smoothing: float = 0.0,
    ic_max_weight_turnover: float | None = None,
    turnover_penalty: float = 0.02,
    cost_penalty: float = 1.0,
    target_annual_return: float | None = 0.20,
    min_annual_return: float | None = 0.18,
    drawdown_limit: float | None = -0.20,
    drawdown_penalty: float = 2.0,
    annual_return_weight: float = 0.5,
    calmar_weight: float = 0.25,
    scoring_config: dict | None = None,
    on_result: Callable[[dict[str, object], pd.DataFrame], None] | None = None,
) -> pd.DataFrame:
    """Evaluate every parameter combination on rolling out-of-sample windows."""
    price_df = _normalize_window_price_frame(price_df)
    grid = grid or DEFAULT_GRID
    keys = list(grid)
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    rows: list[dict[str, object]] = []
    train_start = start

    while True:
        train_end = train_start + pd.DateOffset(years=train_years) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = min(test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1), end)
        if test_start > end or train_end >= end:
            break

        train_factors = _slice_factor_dates(factor_df, train_start, train_end)
        test_factors = _slice_factor_dates(factor_df, test_start, test_end)
        scoring_factors = _slice_factor_dates(factor_df, train_start, test_end)
        train_prices = price_df.loc[(price_df.index >= train_start) & (price_df.index <= train_end)]
        test_prices = price_df.loc[(price_df.index >= test_start) & (price_df.index <= test_end)]
        scoring_prices = price_df.loc[(price_df.index >= train_start) & (price_df.index <= test_end)]
        if train_factors.empty or test_factors.empty or scoring_factors.empty or train_prices.empty or test_prices.empty or scoring_prices.empty:
            train_start += pd.DateOffset(months=step_months)
            continue

        static_ic_weights = None
        dynamic_ic_weights = None
        if "ic_weighted" in set(grid.get("factor_group", [])):
            logger.info("Preparing IC weights for validation window %s to %s.", test_start.date(), test_end.date())
            if use_rolling_ic:
                rolling_ic = calculate_rolling_ic(
                    train_factors,
                    train_prices,
                    horizon=ic_horizon,
                    method=ic_method,
                    window=ic_window,
                    min_periods=ic_min_periods,
                    min_obs=ic_min_obs,
                )
                train_dynamic_weights = make_rolling_ic_weights(
                    rolling_ic,
                    top_k=ic_top_k,
                    min_abs_ic=ic_min_abs,
                    min_periods=ic_min_periods,
                    correlation_threshold=ic_corr_threshold,
                    weight_smoothing=ic_weight_smoothing,
                    max_weight_turnover=ic_max_weight_turnover,
                )
                last_weights = _last_dynamic_weights(train_dynamic_weights)
                dynamic_ic_weights = {
                    pd.Timestamp(date).normalize(): last_weights
                    for date in pd.to_datetime(test_factors.index.get_level_values(0).unique())
                }
            else:
                ic_df = calculate_factor_ic(train_factors, train_prices, horizon=ic_horizon, method=ic_method, min_obs=ic_min_obs)
                static_ic_weights = make_ic_weights(summarize_ic(ic_df), top_k=ic_top_k, min_abs_ic=ic_min_abs)

        score_cache: dict[tuple[str, str], pd.Series] = {}
        for values in product(*(grid[key] for key in keys)):
            params = dict(zip(keys, values))
            factor_group = str(params["factor_group"])
            rebalance_freq = str(params.get("rebalance_freq", "daily"))
            cache_key = (factor_group, rebalance_freq)
            if cache_key not in score_cache:
                logger.info(
                    "Building validation scores for window %s to %s: factor_group=%s rebalance_freq=%s.",
                    test_start.date(),
                    test_end.date(),
                    factor_group,
                    rebalance_freq,
                )
                if factor_group == "ic_weighted":
                    scores = composite_factor(
                        test_factors,
                        method=factor_group,
                        factor_weights=static_ic_weights,
                        factor_weights_dynamic=dynamic_ic_weights,
                    )
                else:
                    scores = build_strategy_scores(scoring_factors, _scoring_config(scoring_config, params), price_df=scoring_prices)
                    scores = _slice_score_dates(scores, test_start, test_end)
                score_cache[cache_key] = resample_signals(scores, rebalance_freq)

            result = run_backtest(
                score_cache[cache_key],
                test_prices,
                test_start.strftime("%Y-%m-%d"),
                test_end.strftime("%Y-%m-%d"),
                {**base_config, **params},
            )
            row = {
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                **params,
                **result.metrics,
                "optimization_score": _optimization_score(
                    result.metrics,
                    turnover_penalty=turnover_penalty,
                    cost_penalty=cost_penalty,
                    target_annual_return=target_annual_return,
                    min_annual_return=min_annual_return,
                    drawdown_limit=drawdown_limit,
                    drawdown_penalty=drawdown_penalty,
                    annual_return_weight=annual_return_weight,
                    calmar_weight=calmar_weight,
                ),
            }
            rows.append(row)
            if on_result is not None:
                on_result(row, pd.DataFrame(rows))
        train_start += pd.DateOffset(months=step_months)

    return pd.DataFrame(rows)


def _slice_factor_dates(factor_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dates = pd.to_datetime(factor_df.index.get_level_values(0)).normalize()
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    return factor_df[(dates >= start) & (dates <= end)]


def _slice_score_dates(score_panel: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    dates = pd.to_datetime(score_panel.index.get_level_values(0)).normalize()
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    return score_panel[(dates >= start) & (dates <= end)]


def _normalize_window_price_frame(price_df: pd.DataFrame) -> pd.DataFrame:
    prices = price_df.copy()
    raw_dates = pd.DatetimeIndex(pd.to_datetime(prices.index, errors="coerce"))
    valid_dates = ~raw_dates.isna()
    prices = prices.loc[valid_dates].copy()
    raw_dates = raw_dates[valid_dates]
    if not prices.empty:
        order = np.argsort(raw_dates.to_numpy(), kind="mergesort")
        prices = prices.iloc[order].copy()
        raw_dates = raw_dates[order]
    prices.index = raw_dates.normalize()
    if prices.index.has_duplicates:
        prices = prices.loc[~prices.index.duplicated(keep="last")]
    return prices.sort_index()


def _scoring_config(config: dict | None, params: dict[str, object]) -> dict:
    result = deepcopy(config) if config is not None else {"strategy": {}}
    result.setdefault("strategy", {})
    for key in STRATEGY_GRID_KEYS:
        if key in params:
            result["strategy"][key] = params[key]
    return result


def _last_dynamic_weights(weights_by_date: dict[pd.Timestamp, pd.Series]) -> pd.Series:
    if not weights_by_date:
        return pd.Series(dtype=float)
    last_date = max(weights_by_date)
    return weights_by_date[last_date]


def _optimization_score(
    metrics: dict,
    turnover_penalty: float = 0.02,
    cost_penalty: float = 1.0,
    target_annual_return: float | None = 0.20,
    min_annual_return: float | None = 0.18,
    drawdown_limit: float | None = -0.20,
    drawdown_penalty: float = 2.0,
    annual_return_weight: float = 0.5,
    calmar_weight: float = 0.25,
) -> float:
    sharpe = _metric_float(metrics, "sharpe")
    annual_turnover = _metric_float(metrics, "annual_turnover")
    annual_trade_cost_ratio = _metric_float(metrics, "annual_trade_cost_ratio")
    score = sharpe - turnover_penalty * annual_turnover - cost_penalty * annual_trade_cost_ratio

    if _has_metric(metrics, "annual_return"):
        annual_return = _metric_float(metrics, "annual_return")
        if target_annual_return is not None and target_annual_return > 0:
            score += annual_return_weight * min(max(annual_return, 0.0), target_annual_return) / target_annual_return
        if min_annual_return is not None and annual_return < min_annual_return:
            score -= float(min_annual_return - annual_return)

    if _has_metric(metrics, "max_drawdown") and drawdown_limit is not None:
        max_drawdown = _metric_float(metrics, "max_drawdown")
        if max_drawdown < drawdown_limit:
            score -= drawdown_penalty * abs(max_drawdown - drawdown_limit)

    if _has_metric(metrics, "calmar"):
        score += calmar_weight * _metric_float(metrics, "calmar")
    return float(score)


def _sorted_results(rows: list[dict[str, object]]) -> pd.DataFrame:
    result_df = pd.DataFrame(rows)
    if result_df.empty:
        return result_df
    sort_columns = [column for column in ["optimization_score", "sharpe", "annual_return", "max_drawdown"] if column in result_df.columns]
    if not sort_columns:
        return result_df
    ascending = [column == "max_drawdown" for column in sort_columns]
    return result_df.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)


def _metric_float(metrics: dict, key: str) -> float:
    value = pd.to_numeric(metrics.get(key, 0.0), errors="coerce")
    if pd.isna(value):
        return 0.0
    return float(value)


def _has_metric(metrics: dict, key: str) -> bool:
    if key not in metrics:
        return False
    value = pd.to_numeric(metrics.get(key), errors="coerce")
    return not pd.isna(value)
