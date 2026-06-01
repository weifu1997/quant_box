from __future__ import annotations

import pandas as pd
import numpy as np


def make_forward_returns(price_df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    prices = price_df.copy()
    prices.index = pd.to_datetime(prices.index)
    prices.columns = prices.columns.astype(str)
    forward = prices.shift(-horizon) / prices - 1
    stacked = forward.stack(future_stack=True).rename("forward_return")
    stacked.index = stacked.index.set_names(["datetime", "instrument"])
    return stacked.dropna()


def calculate_factor_ic(
    factor_df: pd.DataFrame,
    price_df: pd.DataFrame,
    horizon: int = 1,
    method: str = "spearman",
    min_obs: int = 20,
) -> pd.DataFrame:
    if not isinstance(factor_df.index, pd.MultiIndex):
        raise ValueError("factor_df must use MultiIndex: datetime/instrument.")

    returns = make_forward_returns(price_df, horizon=horizon)
    factors = factor_df.select_dtypes("number")
    aligned = factors.join(returns, how="inner").dropna(subset=["forward_return"])
    if aligned.empty:
        raise ValueError("No overlapping factor and forward-return data.")

    date_level = aligned.index.names[0] or 0
    result = {}
    for col in factors.columns:
        series = aligned[[col, "forward_return"]].dropna()
        if series.empty:
            continue
        result[col] = series.groupby(level=date_level).apply(
            lambda group: _safe_corr(group[col], group["forward_return"], method=method, min_obs=min_obs)
        )
    return pd.DataFrame(result).sort_index()


def summarize_ic(ic_df: pd.DataFrame) -> pd.DataFrame:
    mean_ic = ic_df.mean()
    std_ic = ic_df.std(ddof=0)
    summary = pd.DataFrame(
        {
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "ic_ir": mean_ic / std_ic.replace(0, np.nan),
            "positive_ratio": (ic_df > 0).mean(),
            "count": ic_df.count(),
        }
    )
    summary = summary.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return summary.sort_values("ic_ir", ascending=False)


def make_ic_weights(
    ic_summary: pd.DataFrame,
    top_k: int = 30,
    min_abs_ic: float = 0.0,
) -> pd.Series:
    scores = ic_summary["ic_ir"].copy()
    if min_abs_ic > 0:
        scores = scores[ic_summary["mean_ic"].abs() >= min_abs_ic]
    scores = scores.reindex(scores.abs().sort_values(ascending=False).index).head(top_k)
    return scores.fillna(0)


def _safe_corr(x: pd.Series, y: pd.Series, method: str, min_obs: int) -> float:
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < min_obs:
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method))
