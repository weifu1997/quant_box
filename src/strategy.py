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
    factor_weights_dynamic: dict[pd.Timestamp, pd.Series] | None = None,
    factor_directions: pd.Series | dict[str, float] | None = None,
) -> pd.Series:
    if factor_df.empty:
        raise ValueError("factor_df is empty.")

    numeric = factor_df.select_dtypes("number")
    if numeric.empty:
        raise ValueError("factor_df has no numeric factor columns.")

    clean = _cross_sectional_zscore(numeric)
    if factor_directions is not None:
        directions = pd.Series(factor_directions, dtype=float)
        common = [col for col in clean.columns if col in directions.index]
        clean = clean.copy()
        clean[common] = clean[common].mul(directions.loc[common], axis=1)
    method = method.lower()
    if method == "ic_weighted":
        if factor_weights_dynamic is not None:
            return _dynamic_ic_weighted_score(clean, factor_weights_dynamic, factor_weights)
        if factor_weights is None:
            raise ValueError("factor_weights is required when method='ic_weighted'.")
        weights = pd.Series(factor_weights, dtype=float)
        common = [col for col in clean.columns if col in weights.index and weights.loc[col] != 0]
        if not common:
            raise ValueError("No overlapping non-zero factor weights for ic_weighted scoring.")
        selected = clean[common]
        aligned_weights = weights.loc[common]
        score = selected.mul(aligned_weights, axis=1).sum(axis=1, min_count=len(common)) / aligned_weights.abs().sum()
        return score.rename("score")

    if method == "all":
        selected = clean
    else:
        keywords = FACTOR_GROUP_KEYWORDS.get(method, (method,))
        selected_cols = [col for col in clean.columns if any(key in str(col).lower() for key in keywords)]
        if not selected_cols:
            selected_cols = list(clean.columns)
        selected = clean[selected_cols]
    return _row_mean_with_min_count(selected).rename("score")


def _dynamic_ic_weighted_score(
    clean: pd.DataFrame,
    factor_weights_dynamic: dict[pd.Timestamp, pd.Series],
    fallback_weights: pd.Series | dict[str, float] | None = None,
) -> pd.Series:
    if not isinstance(clean.index, pd.MultiIndex):
        raise ValueError("dynamic IC weights require MultiIndex factor data.")

    dynamic = {pd.Timestamp(date).normalize(): pd.Series(weights, dtype=float) for date, weights in factor_weights_dynamic.items()}
    fallback = pd.Series(fallback_weights, dtype=float) if fallback_weights is not None else None
    date_level = clean.index.names[0] or 0
    score_parts: list[pd.Series] = []
    prior_weights: list[pd.Series] = []

    for date, daily in clean.groupby(level=date_level, sort=True):
        key = pd.Timestamp(date).normalize()
        weights = dynamic.get(key)
        if weights is None and prior_weights:
            weights = _average_prior_weights(prior_weights)
        if weights is None and fallback is not None:
            weights = fallback
        if weights is None or weights.empty:
            score_parts.append(pd.Series(np.nan, index=daily.index, name="score"))
            continue

        common = [col for col in daily.columns if col in weights.index and weights.loc[col] != 0]
        if not common:
            score_parts.append(pd.Series(np.nan, index=daily.index, name="score"))
            continue
        aligned_weights = weights.loc[common]
        score = daily[common].mul(aligned_weights, axis=1).sum(axis=1, min_count=len(common)) / aligned_weights.abs().sum()
        score_parts.append(score.rename("score"))
        if key in dynamic:
            prior_weights.append(dynamic[key])

    if not score_parts:
        return pd.Series(dtype=float, name="score")
    return pd.concat(score_parts).sort_index().rename("score")


def _row_mean_with_min_count(df: pd.DataFrame) -> pd.Series:
    min_count = max(1, int(np.ceil(df.shape[1] / 2)))
    means = df.mean(axis=1, skipna=True)
    return means.where(df.count(axis=1) >= min_count)


def _average_prior_weights(prior_weights: list[pd.Series]) -> pd.Series:
    aligned = pd.concat(prior_weights, axis=1).fillna(0.0)
    recency = pd.Series(np.arange(1, aligned.shape[1] + 1, dtype=float), index=aligned.columns)
    averaged = aligned.mul(recency, axis=1).sum(axis=1) / recency.sum()
    denom = averaged.abs().sum()
    return averaged / denom if denom > 0 else averaged


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

    keep = _pick_keeps(previous, rank_map, top_n, allowed_new, rank_buffer)
    additions = _pick_additions(ranked, keep, previous_set, top_n, allowed_new)
    holdings = _dedupe_preserve(keep + additions)

    if len(holdings) < top_n:
        holdings.extend([code for code in ranked if code not in holdings][: top_n - len(holdings)])
        holdings = _dedupe_preserve(holdings)

    holdings = _enforce_turnover_cap(holdings, previous, previous_set, rank_map, allowed_new)
    return sorted(holdings[:top_n], key=lambda code: rank_map.get(code, float("inf")))


def _pick_keeps(
    previous: list[str],
    rank_map: dict[str, int],
    top_n: int,
    allowed_new: int,
    rank_buffer: int,
) -> list[str]:
    buffer_limit = top_n + max(0, rank_buffer)
    keep = [code for code in previous if rank_map.get(code, float("inf")) < buffer_limit]
    keep = sorted(_dedupe_preserve(keep), key=lambda code: rank_map.get(code, float("inf")))[:top_n]

    min_keep = max(0, min(len(set(previous)), top_n - allowed_new))
    if len(keep) < min_keep:
        fallback = [code for code in previous if code in rank_map and code not in keep]
        fallback = sorted(_dedupe_preserve(fallback), key=lambda code: rank_map.get(code, float("inf")))
        keep.extend(fallback[: min_keep - len(keep)])
    return _dedupe_preserve(keep)


def _pick_additions(
    ranked: list[str],
    keep: list[str],
    previous_set: set[str],
    top_n: int,
    allowed_new: int,
) -> list[str]:
    slots = max(top_n - len(keep), 0)
    return [code for code in ranked if code not in keep and code not in previous_set][: min(slots, allowed_new)]


def _enforce_turnover_cap(
    holdings: list[str],
    previous: list[str],
    previous_set: set[str],
    rank_map: dict[str, int],
    allowed_new: int,
) -> list[str]:
    protected = [code for code in previous if code in rank_map and code not in holdings]
    protected = sorted(_dedupe_preserve(protected), key=lambda code: rank_map.get(code, float("inf")))
    holdings = list(holdings)
    while len(set(holdings) - previous_set) > allowed_new and protected:
        for idx in range(len(holdings) - 1, -1, -1):
            if holdings[idx] not in previous_set:
                holdings[idx] = protected.pop(0)
                holdings = _dedupe_preserve(holdings)
                break
        else:
            break
    return holdings


def _dedupe_preserve(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


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


def resample_signals(score_panel: pd.Series, rebalance_freq: str) -> pd.Series:
    if rebalance_freq == "daily":
        return score_panel
    if not isinstance(score_panel.index, pd.MultiIndex):
        raise ValueError("score_panel must use MultiIndex: datetime/instrument.")

    dates = pd.Index(pd.to_datetime(score_panel.index.get_level_values(0).unique())).sort_values()
    date_series = pd.Series(dates, index=dates)
    if rebalance_freq == "weekly":
        keep_dates = set(date_series.resample("W-FRI").last().dropna())
    elif rebalance_freq == "monthly":
        keep_dates = set(date_series.resample("M").last().dropna())
    else:
        raise ValueError(f"Unsupported rebalance_freq: {rebalance_freq}")
    return score_panel[score_panel.index.get_level_values(0).isin(keep_dates)]


def _cross_sectional_zscore(df: pd.DataFrame, min_obs: int = 5) -> pd.DataFrame:
    if not isinstance(df.index, pd.MultiIndex):
        if len(df) < min_obs:
            return pd.DataFrame(np.nan, index=df.index, columns=df.columns)
        std = df.std(ddof=0).replace(0, pd.NA)
        return ((df - df.mean()) / std).replace([np.inf, -np.inf], np.nan)

    date_level = df.index.names[0] or 0
    grouped = df.groupby(level=date_level)
    counts = grouped.transform("count")
    means = grouped.transform("mean")
    mean_squares = df.pow(2).groupby(level=date_level).transform("mean")
    stds = np.sqrt((mean_squares - means.pow(2)).clip(lower=0)).replace(0, np.nan)
    scaled = (df - means) / stds
    scaled = scaled.where(counts >= min_obs)
    return scaled.astype(float).replace([np.inf, -np.inf], np.nan)
