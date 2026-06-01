from __future__ import annotations

from itertools import product
from typing import Iterable

import pandas as pd

from src.backtest import run_backtest
from src.factor_ic import calculate_factor_ic, calculate_rolling_ic, make_ic_weights, make_rolling_ic_weights, summarize_ic
from src.strategy import composite_factor, resample_signals


DEFAULT_GRID = {
    "factor_group": ["momentum", "volatility", "all", "ic_weighted"],
    "top_n": [7, 10, 15],
    "max_turnover": [1, 2],
    "rank_buffer": [0, 5, 10],
    "rebalance_freq": ["daily", "weekly"],
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
    ic_window: int = 252,
    ic_min_periods: int = 60,
    ic_min_abs: float = 0.02,
    ic_corr_threshold: float = 0.7,
    ic_top_k: int = 30,
) -> pd.DataFrame:
    grid = grid or DEFAULT_GRID
    dynamic_weights = None
    if "ic_weighted" in set(grid.get("factor_group", [])) and use_rolling_ic:
        rolling_ic = calculate_rolling_ic(factor_df, price_df, window=ic_window, min_periods=ic_min_periods)
        dynamic_weights = make_rolling_ic_weights(
            rolling_ic,
            top_k=ic_top_k,
            min_abs_ic=ic_min_abs,
            min_periods=ic_min_periods,
            correlation_threshold=ic_corr_threshold,
        )
    elif "ic_weighted" in set(grid.get("factor_group", [])) and ic_weights is None:
        ic_df = calculate_factor_ic(factor_df, price_df)
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
            weights = ic_weights if factor_group == "ic_weighted" else None
            dynamic = dynamic_weights if factor_group == "ic_weighted" else None
            scores = composite_factor(factor_df, method=factor_group, factor_weights=weights, factor_weights_dynamic=dynamic)
            score_cache[cache_key] = resample_signals(scores, rebalance_freq)

        bt_config = {**base_config, **params}
        result = run_backtest(score_cache[cache_key], price_df, start_date, end_date, bt_config)
        rows.append({**params, **result.metrics})

    result_df = pd.DataFrame(rows)
    return result_df.sort_values(["sharpe", "annual_return", "max_drawdown"], ascending=[False, False, False])


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
    ic_window: int = 252,
    ic_min_periods: int = 60,
    ic_min_abs: float = 0.02,
    ic_corr_threshold: float = 0.7,
    ic_top_k: int = 30,
) -> pd.DataFrame:
    price_df = price_df.copy()
    price_df.index = pd.to_datetime(price_df.index)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
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
            ic_window=ic_window,
            ic_min_periods=ic_min_periods,
            ic_min_abs=ic_min_abs,
            ic_corr_threshold=ic_corr_threshold,
            ic_top_k=ic_top_k,
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
                rolling_ic = calculate_rolling_ic(train_factors, train_prices, window=ic_window, min_periods=ic_min_periods)
                train_dynamic_weights = make_rolling_ic_weights(
                    rolling_ic,
                    top_k=ic_top_k,
                    min_abs_ic=ic_min_abs,
                    min_periods=ic_min_periods,
                    correlation_threshold=ic_corr_threshold,
                )
                last_weights = _last_dynamic_weights(train_dynamic_weights)
                dynamic_weights = {pd.Timestamp(date).normalize(): last_weights for date in pd.to_datetime(test_factors.index.get_level_values(0).unique())}
                score_source = test_factors
            else:
                ic_df = calculate_factor_ic(train_factors, train_prices)
                weights = make_ic_weights(summarize_ic(ic_df), top_k=ic_top_k, min_abs_ic=ic_min_abs)
                score_source = test_factors
        else:
            score_source = test_factors
        scores = composite_factor(score_source, method=factor_group, factor_weights=weights, factor_weights_dynamic=dynamic_weights)
        scores = resample_signals(scores, rebalance_freq)
        scores = _slice_score_dates(scores, test_start, test_end)
        result = run_backtest(
            scores,
            test_prices,
            test_start.strftime("%Y-%m-%d"),
            test_end.strftime("%Y-%m-%d"),
            {**base_config, **params},
        )
        rows.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                **params,
                **result.metrics,
            }
        )
        train_start += pd.DateOffset(months=step_months)

    return pd.DataFrame(rows)


def _slice_factor_dates(factor_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dates = pd.to_datetime(factor_df.index.get_level_values(0))
    return factor_df[(dates >= start) & (dates <= end)]


def _slice_score_dates(score_panel: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    dates = pd.to_datetime(score_panel.index.get_level_values(0))
    return score_panel[(dates >= start) & (dates <= end)]


def _last_dynamic_weights(weights_by_date: dict[pd.Timestamp, pd.Series]) -> pd.Series:
    if not weights_by_date:
        return pd.Series(dtype=float)
    last_date = max(weights_by_date)
    return weights_by_date[last_date]
