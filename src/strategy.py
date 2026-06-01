from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


FACTOR_GROUP_KEYWORDS = {
    "momentum": ("roc", "mom", "rsi", "bias"),
    "volatility": ("std", "var", "volatility"),
    "volume": ("volume", "vol", "vwap", "amount"),
}


def composite_factor(
    factor_df: pd.DataFrame,
    method: str = "momentum",
    factor_weights: pd.Series | dict[str, float] | None = None,
) -> pd.Series:
    if factor_df.empty:
        raise ValueError("factor_df is empty.")

    numeric = factor_df.select_dtypes("number")
    if numeric.empty:
        raise ValueError("factor_df has no numeric factor columns.")

    clean = _cross_sectional_zscore(numeric)
    method = method.lower()
    if method == "ic_weighted":
        if factor_weights is None:
            raise ValueError("factor_weights is required when method='ic_weighted'.")
        weights = pd.Series(factor_weights, dtype=float)
        common = [col for col in clean.columns if col in weights.index and weights.loc[col] != 0]
        if not common:
            raise ValueError("No overlapping non-zero factor weights for ic_weighted scoring.")
        selected = clean[common]
        aligned_weights = weights.loc[common]
        score = selected.mul(aligned_weights, axis=1).sum(axis=1) / aligned_weights.abs().sum()
        return score.rename("score")

    if method == "all":
        selected = clean
    else:
        keywords = FACTOR_GROUP_KEYWORDS.get(method, (method,))
        selected_cols = [col for col in clean.columns if any(key in str(col).lower() for key in keywords)]
        if not selected_cols:
            selected_cols = list(clean.columns)
        selected = clean[selected_cols]
    return selected.mean(axis=1).rename("score")


def select_stocks(
    score_series: pd.Series,
    top_n: int = 7,
    previous_holdings: Iterable[str] | None = None,
    max_turnover: int = 1,
    rank_buffer: int = 0,
) -> list[str]:
    scores = score_series.dropna().sort_values(ascending=False)
    ranked = [str(code) for code in scores.index.tolist()]
    if top_n <= 0 or not ranked:
        return []
    if previous_holdings is None:
        return ranked[:top_n]

    previous = [str(code) for code in previous_holdings]
    previous_set = set(previous)
    allowed_new = max(0, min(max_turnover, top_n))
    rank_map = {code: rank for rank, code in enumerate(ranked)}
    buffer_limit = top_n + max(0, rank_buffer)

    keep = [code for code in previous if rank_map.get(code, float("inf")) < buffer_limit]
    keep = sorted(keep, key=lambda code: rank_map.get(code, float("inf")))[:top_n]

    min_keep = max(0, min(len(previous), top_n - allowed_new))
    if len(keep) < min_keep:
        fallback = [code for code in previous if code in rank_map and code not in keep]
        fallback = sorted(fallback, key=lambda code: rank_map.get(code, float("inf")))
        keep.extend(fallback[: min_keep - len(keep)])

    slots = top_n - len(keep)
    additions = [code for code in ranked if code not in keep and code not in previous_set][: min(slots, allowed_new)]
    holdings = keep + additions

    if len(holdings) < top_n:
        holdings.extend([code for code in ranked if code not in holdings][: top_n - len(holdings)])

    new_count = len(set(holdings) - previous_set)
    if new_count > allowed_new:
        protected = [code for code in previous if code in ranked and code not in holdings]
        protected = sorted(protected, key=lambda code: rank_map.get(code, float("inf")))
        while len(set(holdings) - previous_set) > allowed_new and protected:
            for idx in range(len(holdings) - 1, -1, -1):
                if holdings[idx] not in previous_set:
                    holdings[idx] = protected.pop(0)
                    break

    return sorted(holdings[:top_n], key=lambda code: rank_map.get(code, float("inf")))


def generate_holdings_by_day(
    score_panel: pd.Series,
    top_n: int = 7,
    max_turnover: int = 1,
    rank_buffer: int = 0,
) -> pd.DataFrame:
    if not isinstance(score_panel.index, pd.MultiIndex):
        raise ValueError("score_panel must use MultiIndex: date/instrument.")

    rows: list[dict[str, object]] = []
    previous: list[str] | None = None
    date_level = score_panel.index.names[0] or 0
    for date, daily_scores in score_panel.groupby(level=date_level):
        scores = daily_scores.droplevel(date_level)
        holdings = select_stocks(
            scores,
            top_n=top_n,
            previous_holdings=previous,
            max_turnover=max_turnover,
            rank_buffer=rank_buffer,
        )
        rows.extend({"date": pd.Timestamp(date), "instrument": code, "weight": 1 / len(holdings)} for code in holdings)
        previous = holdings
    return pd.DataFrame(rows)


def _cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.MultiIndex):
        std = df.std(ddof=0).replace(0, pd.NA)
        return ((df - df.mean()) / std).fillna(0)

    date_level = df.index.names[0] or 0
    centered = df.groupby(level=date_level).transform(lambda s: s - s.mean())
    scaled = centered.groupby(level=date_level).transform(_scale_nonzero)
    return scaled.astype(float).where(np.isfinite(scaled.astype(float)), 0).fillna(0)


def _scale_nonzero(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    return s / std
