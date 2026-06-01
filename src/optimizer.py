from __future__ import annotations

from itertools import product
from typing import Iterable

import pandas as pd

from src.backtest import run_backtest
from src.factor_ic import calculate_factor_ic, make_ic_weights, summarize_ic
from src.strategy import composite_factor


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


def resample_signals(score_panel: pd.Series, rebalance_freq: str) -> pd.Series:
    if rebalance_freq == "daily":
        return score_panel
    if not isinstance(score_panel.index, pd.MultiIndex):
        raise ValueError("score_panel must use MultiIndex: datetime/instrument.")

    dates = pd.Index(pd.to_datetime(score_panel.index.get_level_values(0).unique())).sort_values()
    date_series = pd.Series(dates, index=dates)
    if rebalance_freq == "weekly":
        keep_dates = set(date_series.groupby(date_series.dt.to_period("W")).last())
    elif rebalance_freq == "monthly":
        keep_dates = set(date_series.groupby(date_series.dt.to_period("M")).last())
    else:
        raise ValueError(f"Unsupported rebalance_freq: {rebalance_freq}")
    return score_panel[score_panel.index.get_level_values(0).isin(keep_dates)]
