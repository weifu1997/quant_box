from __future__ import annotations

from itertools import product
from typing import Iterable

import pandas as pd

from src.backtest import run_backtest
from src.factor_ic import calculate_factor_ic, make_ic_weights, summarize_ic
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
) -> pd.DataFrame:
    grid = grid or DEFAULT_GRID
    if "ic_weighted" in set(grid.get("factor_group", [])) and ic_weights is None:
        ic_df = calculate_factor_ic(factor_df, price_df)
        ic_weights = make_ic_weights(summarize_ic(ic_df))

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
            scores = composite_factor(factor_df, method=factor_group, factor_weights=weights)
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
        )
        if train_results.empty:
            train_start += pd.DateOffset(months=step_months)
            continue

        params = train_results.iloc[0][list(grid or DEFAULT_GRID)].to_dict()
        factor_group = str(params["factor_group"])
        rebalance_freq = str(params.get("rebalance_freq", "daily"))
        weights = None
        if factor_group == "ic_weighted":
            ic_df = calculate_factor_ic(train_factors, train_prices)
            weights = make_ic_weights(summarize_ic(ic_df))
        scores = composite_factor(test_factors, method=factor_group, factor_weights=weights)
        scores = resample_signals(scores, rebalance_freq)
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
