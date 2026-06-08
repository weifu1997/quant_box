from __future__ import annotations

import pandas as pd
import numpy as np

from src.common import normalize_instrument as _normalize_instrument


def make_forward_returns(price_df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    prices = _close_prices(price_df)
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
    factor_cols = list(factors.columns)
    date_level = factors.index.names[0] or 0
    factor_dates = pd.DatetimeIndex(pd.to_datetime(factors.index.get_level_values(date_level)).normalize())
    method_name = method.lower()

    rows: list[pd.Series] = []
    return_dates = set(pd.to_datetime(returns.index.get_level_values(0)).normalize())
    for date_key in pd.DatetimeIndex(factor_dates.dropna().unique()).sort_values():
        if date_key not in return_dates:
            continue
        daily_factors = factors[factor_dates == date_key]
        try:
            daily_returns = returns.xs(date_key, level=0, drop_level=True)
        except KeyError:
            continue
        daily = _normalize_daily_factor_frame(daily_factors, date_level)
        daily_returns = _normalize_instrument_series(daily_returns)
        aligned = daily.join(daily_returns, how="inner").dropna(subset=["forward_return"])
        if aligned.empty:
            continue
        if method_name in {"pearson", "spearman"}:
            row = _daily_target_factor_corr(aligned[factor_cols], aligned["forward_return"], method_name, min_obs)
        else:
            row = pd.Series(
                {
                    factor: _safe_corr(aligned[factor], aligned["forward_return"], method_name, min_obs)
                    for factor in factor_cols
                },
                dtype=float,
            )
        row.name = date_key
        rows.append(row.reindex(factor_cols))

    if not rows:
        raise ValueError("No overlapping factor and forward-return data.")
    return pd.DataFrame(rows).reindex(columns=factor_cols).sort_index()


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
    realized_lag = max(1, int(horizon))
    rolling_ic = daily_ic.shift(realized_lag).rolling(window=window, min_periods=min_periods).mean()
    rolling_ic.attrs["daily_ic"] = daily_ic
    rolling_ic.attrs["window"] = window
    rolling_ic.attrs["min_periods"] = min_periods
    rolling_ic.attrs["horizon"] = realized_lag
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
    correlation_rebalance_sessions: int = 1,
    weight_smoothing: float = 0.0,
    max_weight_turnover: float | None = None,
) -> dict[pd.Timestamp, pd.Series]:
    if rolling_ic_df.empty:
        return {}

    daily_ic = rolling_ic_df.attrs.get("daily_ic")
    if not isinstance(daily_ic, pd.DataFrame):
        raise ValueError("rolling_ic_df must be produced by calculate_rolling_ic and include attrs['daily_ic'].")
    window = int(rolling_ic_df.attrs.get("window", 252))
    horizon = max(1, int(rolling_ic_df.attrs.get("horizon", 1)))
    source = daily_ic.shift(horizon)
    rolling_std = source.rolling(window=window, min_periods=min_periods).std(ddof=0)
    rolling_count = source.rolling(window=window, min_periods=min_periods).count()

    weights_by_date: dict[pd.Timestamp, pd.Series] = {}
    previous_weights: pd.Series | None = None
    cluster_map: dict[str, list[str]] | None = None
    cluster_factors: set[str] = set()
    cluster_eligible_dates = 0
    cluster_rebalance = max(1, int(correlation_rebalance_sessions))
    for date, mean_ic in rolling_ic_df.iterrows():
        count = rolling_count.loc[date].reindex(mean_ic.index).fillna(0)
        std_ic = rolling_std.loc[date].reindex(mean_ic.index)
        ic_ir = mean_ic / std_ic.replace(0, np.nan)
        valid = mean_ic.abs().ge(min_abs_ic) & count.ge(min_periods) & ic_ir.notna()
        if not valid.any():
            continue

        candidates = ic_ir[valid]
        candidate_factors = set(candidates.index)
        should_recluster = (
            cluster_map is None
            or cluster_eligible_dates % cluster_rebalance == 0
            or not candidate_factors.issubset(cluster_factors)
        )
        if should_recluster:
            history = source.loc[source.index <= date, candidates.index].tail(window).dropna(how="all")
            cluster_map = cluster_correlated_factors(history, threshold=correlation_threshold)
            cluster_factors = candidate_factors
        cluster_eligible_dates += 1
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


def _daily_target_factor_corr(factors: pd.DataFrame, target: pd.Series, method: str, min_obs: int) -> pd.Series:
    x = factors.astype(float)
    y = target.astype(float)
    valid = x.notna()
    valid = valid.where(y.notna(), False)
    if not bool(valid.any().any()):
        return pd.Series(np.nan, index=factors.columns, dtype=float)

    if method == "spearman":
        y_frame = pd.DataFrame(np.broadcast_to(y.to_numpy()[:, None], x.shape), index=x.index, columns=x.columns)
        x_values = x.where(valid).rank(axis=0, method="average")
        y_values = y_frame.where(valid).rank(axis=0, method="average")
    else:
        x_values = x.where(valid)
        y_values = pd.DataFrame(np.broadcast_to(y.to_numpy()[:, None], x.shape), index=x.index, columns=x.columns).where(valid)

    count = valid.sum(axis=0)
    sum_x = x_values.sum(axis=0, skipna=True)
    sum_y = y_values.sum(axis=0, skipna=True)
    sum_x2 = x_values.pow(2).sum(axis=0, skipna=True)
    sum_y2 = y_values.pow(2).sum(axis=0, skipna=True)
    sum_xy = x_values.mul(y_values).sum(axis=0, skipna=True)

    cov = sum_xy - (sum_x * sum_y / count.replace(0, np.nan))
    var_x = sum_x2 - sum_x.pow(2) / count.replace(0, np.nan)
    var_y = sum_y2 - sum_y.pow(2) / count.replace(0, np.nan)
    denom = np.sqrt(var_x.clip(lower=0) * var_y.clip(lower=0))
    corr = cov / denom.replace(0, np.nan)
    return corr.where(count.ge(min_obs)).replace([np.inf, -np.inf], np.nan).reindex(factors.columns)


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
        fields = price_df.columns.get_level_values(field_level).astype(str).str.strip().str.lower()
        if "close" not in set(fields):
            raise ValueError("price_df MultiIndex columns must include a close field.")
        close = price_df.loc[:, fields == "close"].copy()
        close.columns = [_normalize_instrument(value) for value in close.columns.get_level_values(1)]
        return _normalize_close_frame(close)
    field_like_columns = {"open", "high", "low", "close", "volume", "vol", "amount", "vwap", "adj_factor"}
    column_names = {str(column).strip().lower() for column in price_df.columns}
    if len(price_df.columns) > 1 and column_names & field_like_columns:
        raise ValueError("Non-MultiIndex price_df must be a close-price panel with instrument columns.")
    close = price_df.copy()
    close.columns = [_normalize_instrument(value) for value in close.columns]
    return _normalize_close_frame(close)


def _normalize_close_frame(close: pd.DataFrame) -> pd.DataFrame:
    result = close.copy()
    raw_dates = pd.DatetimeIndex(pd.to_datetime(result.index, errors="coerce"))
    valid_dates = ~pd.isna(raw_dates)
    if not valid_dates.all():
        result = result.loc[valid_dates].copy()
        raw_dates = raw_dates[valid_dates]
    if result.empty:
        return result

    order = np.argsort(raw_dates.to_numpy(), kind="mergesort")
    if not np.array_equal(order, np.arange(len(raw_dates))):
        result = result.iloc[order].copy()
        raw_dates = raw_dates[order]
    result.index = raw_dates.normalize()
    result.columns = [_normalize_instrument(value) for value in result.columns]
    result = result.loc[:, result.columns != ""]
    if result.columns.has_duplicates:
        result = result.loc[:, ~result.columns.duplicated(keep="last")]
    result = result[~result.index.duplicated(keep="last")].sort_index()
    return result.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _normalize_instrument_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.index = pd.Index([_normalize_instrument(value) for value in result.index], name=frame.index.name)
    result = result[result.index != ""]
    if result.index.has_duplicates:
        result = result[~result.index.duplicated(keep="last")]
    return result


def _normalize_daily_factor_frame(frame: pd.DataFrame, date_level: str | int) -> pd.DataFrame:
    if frame.empty:
        return frame.droplevel(date_level)
    instrument_level = _instrument_level(frame.index, date_level)
    raw_dates = pd.DatetimeIndex(pd.to_datetime(frame.index.get_level_values(date_level), errors="coerce"))
    instruments = [_normalize_instrument(value) for value in frame.index.get_level_values(instrument_level)]
    keep_by_instrument: dict[str, int] = {}
    for position, (raw_date, instrument) in enumerate(zip(raw_dates, instruments)):
        if pd.isna(raw_date) or not instrument:
            continue
        current = keep_by_instrument.get(instrument)
        if current is None or (pd.Timestamp(raw_date), position) >= (pd.Timestamp(raw_dates[current]), current):
            keep_by_instrument[instrument] = position
    if not keep_by_instrument:
        result = frame.iloc[0:0].copy()
        result.index = pd.Index([], name="instrument")
        return result
    keep_positions = sorted(keep_by_instrument.values())
    result = frame.iloc[keep_positions].copy()
    result.index = pd.Index([instruments[position] for position in keep_positions], name="instrument")
    return result


def _instrument_level(index: pd.MultiIndex, date_level: str | int) -> str | int:
    if index.nlevels != 2:
        raise ValueError("factor_df must use two index levels: datetime/instrument.")
    date_position = index.names.index(date_level) if isinstance(date_level, str) else int(date_level)
    return 1 - date_position


def _normalize_instrument_series(series: pd.Series) -> pd.Series:
    result = series.copy()
    result.index = pd.Index([_normalize_instrument(value) for value in result.index], name=series.index.name)
    result = result[result.index != ""]
    if result.index.has_duplicates:
        result = result[~result.index.duplicated(keep="last")]
    return result
