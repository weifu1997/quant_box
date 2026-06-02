from __future__ import annotations

import pandas as pd
import numpy as np


def make_forward_returns(price_df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    prices = _close_prices(price_df)
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

    factor_cols = list(factors.columns)
    date_level = aligned.index.names[0] or 0
    corr = aligned[[*factor_cols, "forward_return"]].groupby(level=date_level).corr(method=method)
    ic = corr["forward_return"].unstack(level=-1).reindex(columns=factor_cols)

    pair_counts = aligned[factor_cols].notna().groupby(level=date_level).sum()
    ic = ic.where(pair_counts.reindex_like(ic).ge(min_obs))
    return ic.sort_index()


def calculate_rolling_ic(
    factor_df: pd.DataFrame,
    price_df: pd.DataFrame,
    horizon: int = 1,
    method: str = "spearman",
    window: int = 252,
    min_periods: int = 60,
    min_obs: int = 20,
) -> pd.DataFrame:
    daily_ic = calculate_factor_ic(factor_df, price_df, horizon=horizon, method=method, min_obs=min_obs)
    rolling_ic = daily_ic.shift(1).rolling(window=window, min_periods=min_periods).mean()
    rolling_ic.attrs["daily_ic"] = daily_ic
    rolling_ic.attrs["window"] = window
    rolling_ic.attrs["min_periods"] = min_periods
    return rolling_ic


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
    min_abs_ic: float = 0.02,
) -> pd.Series:
    scores = ic_summary["ic_ir"].copy()
    if min_abs_ic > 0:
        scores = scores[ic_summary["mean_ic"].abs() >= min_abs_ic]
    scores = scores.reindex(scores.abs().sort_values(ascending=False).index).head(top_k)
    return scores.fillna(0)


def make_rolling_ic_weights(
    rolling_ic_df: pd.DataFrame,
    top_k: int = 30,
    min_abs_ic: float = 0.02,
    min_periods: int = 60,
    correlation_threshold: float = 0.7,
    weight_smoothing: float = 0.0,
    max_weight_turnover: float | None = None,
) -> dict[pd.Timestamp, pd.Series]:
    if rolling_ic_df.empty:
        return {}

    daily_ic = rolling_ic_df.attrs.get("daily_ic")
    if not isinstance(daily_ic, pd.DataFrame):
        raise ValueError("rolling_ic_df must be produced by calculate_rolling_ic and include attrs['daily_ic'].")
    window = int(rolling_ic_df.attrs.get("window", 252))
    source = daily_ic.shift(1)
    rolling_std = source.rolling(window=window, min_periods=min_periods).std(ddof=0)
    rolling_count = source.rolling(window=window, min_periods=min_periods).count()

    weights_by_date: dict[pd.Timestamp, pd.Series] = {}
    previous_weights: pd.Series | None = None
    for date, mean_ic in rolling_ic_df.iterrows():
        count = rolling_count.loc[date].reindex(mean_ic.index).fillna(0)
        std_ic = rolling_std.loc[date].reindex(mean_ic.index)
        ic_ir = mean_ic / std_ic.replace(0, np.nan)
        valid = mean_ic.abs().ge(min_abs_ic) & count.ge(min_periods) & ic_ir.notna()
        if not valid.any():
            continue

        candidates = ic_ir[valid]
        history = source.loc[source.index <= date, candidates.index].tail(window).dropna(how="all")
        cluster_map = cluster_correlated_factors(history, threshold=correlation_threshold)
        keep = [factor for factor in candidates.index if factor in cluster_map]
        scores = candidates.loc[keep].reindex(candidates.loc[keep].abs().sort_values(ascending=False).index).head(top_k)
        denom = scores.abs().sum()
        if denom > 0:
            raw_weights = scores.fillna(0) / denom
            stable_weights = _stabilize_weights(
                raw_weights,
                previous_weights,
                weight_smoothing=weight_smoothing,
                max_weight_turnover=max_weight_turnover,
            )
            weights_by_date[pd.Timestamp(date).normalize()] = stable_weights
            previous_weights = stable_weights
    return weights_by_date


def cluster_correlated_factors(rolling_ic_df: pd.DataFrame, threshold: float = 0.7) -> dict[str, list[str]]:
    clean = rolling_ic_df.dropna(axis=1, how="all")
    clean = clean.loc[:, clean.std(ddof=0).fillna(0) > 0]
    if clean.empty:
        return {}

    corr = clean.corr().abs().fillna(0)
    factors = list(corr.columns)
    visited: set[str] = set()
    clusters: list[list[str]] = []
    for factor in factors:
        if factor in visited:
            continue
        queue = [factor]
        visited.add(factor)
        cluster: list[str] = []
        while queue:
            current = queue.pop(0)
            cluster.append(current)
            neighbors = corr.index[(corr.loc[current] > threshold) & (corr.index != current)].tolist()
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        clusters.append(cluster)

    summary = summarize_ic(clean)
    result: dict[str, list[str]] = {}
    for cluster in clusters:
        ranked = summary.reindex(cluster)["ic_ir"].abs().sort_values(ascending=False)
        representative = str(ranked.index[0]) if not ranked.empty else cluster[0]
        result[representative] = [factor for factor in cluster if factor != representative]
    return result


def _safe_corr(x: pd.Series, y: pd.Series, method: str, min_obs: int) -> float:
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < min_obs:
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method))


def _stabilize_weights(
    weights: pd.Series,
    previous: pd.Series | None,
    weight_smoothing: float = 0.0,
    max_weight_turnover: float | None = None,
) -> pd.Series:
    current = weights.astype(float)
    if previous is None or previous.empty:
        return _normalize_abs_weights(current)

    prior = previous.astype(float)
    index = current.index.union(prior.index)
    current = current.reindex(index, fill_value=0.0)
    prior = prior.reindex(index, fill_value=0.0)

    smoothing = max(0.0, min(float(weight_smoothing), 1.0))
    if smoothing > 0:
        current = smoothing * prior + (1.0 - smoothing) * current

    if max_weight_turnover is not None:
        max_turnover = max(0.0, float(max_weight_turnover))
        delta = current - prior
        turnover = float(delta.abs().sum())
        if turnover > max_turnover and turnover > 0:
            current = prior + delta * (max_turnover / turnover)

    return _normalize_abs_weights(current)


def _normalize_abs_weights(weights: pd.Series) -> pd.Series:
    clean = weights.replace([np.inf, -np.inf], np.nan).dropna()
    denom = clean.abs().sum()
    if denom <= 0:
        return clean
    return clean[clean != 0] / denom


def _close_prices(price_df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(price_df.columns, pd.MultiIndex):
        field_level = 0
        fields = price_df.columns.get_level_values(field_level).astype(str).str.lower()
        if "close" not in set(fields):
            raise ValueError("price_df MultiIndex columns must include a close field.")
        close = price_df.loc[:, fields == "close"].copy()
        close.columns = close.columns.get_level_values(1).astype(str)
        return close
    return price_df.copy()
