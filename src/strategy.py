from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


FACTOR_GROUP_KEYWORDS = {
    "momentum": ("roc", "mom", "rsi", "bias"),
    "volatility": ("std", "var", "volatility"),
    "volume": ("volume", "vol", "vwap", "amount"),
}
INVERSE_FACTOR_PREFIXES = ("low_", "inverse_", "short_")


def composite_factor(
    factor_df: pd.DataFrame,
    method: str = "momentum",
    factor_weights: pd.Series | dict[str, float] | None = None,
    factor_weights_dynamic: dict[pd.Timestamp, pd.Series] | None = None,
    factor_directions: pd.Series | dict[str, float] | None = None,
    min_obs: int = 5,
) -> pd.Series:
    if factor_df.empty:
        raise ValueError("factor_df is empty.")

    numeric = factor_df.select_dtypes("number")
    if numeric.empty:
        raise ValueError("factor_df has no numeric factor columns.")

    method, direction = _factor_method_parts(method)
    numeric = _select_factor_columns(numeric, method, factor_weights, factor_weights_dynamic)
    if numeric.empty:
        raise ValueError(f"No factor columns matched factor_group='{method}'.")

    clean = _cross_sectional_zscore(numeric, min_obs=min_obs)
    if factor_directions is not None:
        directions = pd.Series(factor_directions, dtype=float)
        common = [col for col in clean.columns if col in directions.index]
        clean = clean.copy()
        clean[common] = clean[common].mul(directions.loc[common], axis=1)
    if method == "ic_weighted":
        if factor_weights_dynamic is not None:
            return (_dynamic_ic_weighted_score(clean, factor_weights_dynamic, factor_weights) * direction).rename("score")
        if factor_weights is None:
            raise ValueError("factor_weights is required when method='ic_weighted'.")
        weights = pd.Series(factor_weights, dtype=float)
        common = [col for col in clean.columns if col in weights.index and weights.loc[col] != 0]
        if not common:
            raise ValueError("No overlapping non-zero factor weights for ic_weighted scoring.")
        selected = clean[common]
        aligned_weights = weights.loc[common]
        score = selected.mul(aligned_weights, axis=1).sum(axis=1, min_count=len(common)) / aligned_weights.abs().sum()
        return (score * direction).rename("score")

    return (_row_mean_with_min_count(clean) * direction).rename("score")


def _select_factor_columns(
    numeric: pd.DataFrame,
    method: str,
    factor_weights: pd.Series | dict[str, float] | None = None,
    factor_weights_dynamic: dict[pd.Timestamp, pd.Series] | None = None,
) -> pd.DataFrame:
    selected_cols = factor_columns_for_method(numeric.columns, method, factor_weights, factor_weights_dynamic)
    return numeric[selected_cols]


def factor_columns_for_method(
    columns: Iterable[object],
    method: str,
    factor_weights: pd.Series | dict[str, float] | None = None,
    factor_weights_dynamic: dict[pd.Timestamp, pd.Series] | None = None,
) -> list[object]:
    method, _ = _factor_method_parts(method)
    column_list = list(columns)
    if method == "all":
        return column_list
    if method == "ic_weighted":
        weighted_cols = _weighted_factor_columns(factor_weights, factor_weights_dynamic)
        if not weighted_cols:
            return column_list
        return [col for col in column_list if str(col) in weighted_cols]

    exact_column = _exact_factor_column(method)
    if exact_column is not None:
        return [col for col in column_list if str(col).lower() == exact_column]

    keywords = FACTOR_GROUP_KEYWORDS.get(method, (method,))
    return [col for col in column_list if any(key in str(col).lower() for key in keywords)]


def _exact_factor_column(method: str) -> str | None:
    for prefix in ("factor:", "column:"):
        if method.startswith(prefix):
            name = method[len(prefix) :].strip().lower()
            return name or None
    return None


def _factor_method_parts(method: str) -> tuple[str, float]:
    method = str(method).strip().lower()
    for prefix in INVERSE_FACTOR_PREFIXES:
        if method.startswith(prefix):
            return method[len(prefix) :], -1.0
    return method, 1.0


def _weighted_factor_columns(
    factor_weights: pd.Series | dict[str, float] | None = None,
    factor_weights_dynamic: dict[pd.Timestamp, pd.Series] | None = None,
) -> set[str]:
    columns: set[str] = set()
    if factor_weights is not None:
        weights = pd.Series(factor_weights, dtype=float)
        columns.update(str(col) for col, value in weights.items() if value != 0)
    if factor_weights_dynamic is not None:
        for weights in factor_weights_dynamic.values():
            series = pd.Series(weights, dtype=float)
            columns.update(str(col) for col, value in series.items() if value != 0)
    return columns


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
    group_map: Mapping[str, object] | pd.Series | None = None,
    max_group_weight: float | None = None,
) -> list[str]:
    scores = _normalize_score_series(score_series)
    ranked = scores.index.astype(str).tolist()
    if top_n <= 0 or not ranked:
        return []
    if previous_holdings is None:
        return _apply_group_cap(ranked, top_n, group_map, max_group_weight)

    previous = _normalize_instruments(previous_holdings)
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
    holdings = _enforce_group_cap(
        holdings,
        ranked,
        previous,
        previous_set,
        rank_map,
        allowed_new,
        top_n,
        group_map,
        max_group_weight,
    )
    return sorted(holdings[:top_n], key=lambda code: rank_map.get(code, float("inf")))


def _apply_group_cap(
    ranked: list[str],
    top_n: int,
    group_map: Mapping[str, object] | pd.Series | None,
    max_group_weight: float | None,
) -> list[str]:
    group_limit = _group_limit(top_n, max_group_weight)
    groups = _normalize_group_map(group_map)
    if group_limit is None or groups is None:
        return ranked[:top_n]
    selected: list[str] = []
    group_counts: dict[str, int] = {}
    for code in ranked:
        if len(selected) >= top_n:
            break
        if not _group_slot_available(code, group_counts, group_limit, groups):
            continue
        selected.append(code)
        _bump_group(code, group_counts, groups)
    if len(selected) < top_n:
        selected.extend([code for code in ranked if code not in selected][: top_n - len(selected)])
    return _dedupe_preserve(selected)[:top_n]


def _enforce_group_cap(
    holdings: list[str],
    ranked: list[str],
    previous: list[str],
    previous_set: set[str],
    rank_map: dict[str, int],
    allowed_new: int,
    top_n: int,
    group_map: Mapping[str, object] | pd.Series | None,
    max_group_weight: float | None,
) -> list[str]:
    group_limit = _group_limit(top_n, max_group_weight)
    groups = _normalize_group_map(group_map)
    if group_limit is None or groups is None or not holdings:
        return holdings
    if _group_cap_satisfied(holdings, group_limit, groups):
        return holdings

    selected: list[str] = []
    group_counts: dict[str, int] = {}
    new_count = 0
    previous_ranked = [code for code in previous if code in rank_map]
    previous_ranked = sorted(_dedupe_preserve(previous_ranked), key=lambda code: rank_map.get(code, float("inf")))
    effective_allowed_new = max(allowed_new, top_n - len(previous_ranked))

    for code in previous_ranked:
        if len(selected) >= top_n:
            break
        if not _group_slot_available(code, group_counts, group_limit, groups):
            continue
        selected.append(code)
        _bump_group(code, group_counts, groups)

    for code in ranked:
        if len(selected) >= top_n:
            break
        if code in selected:
            continue
        is_new = code not in previous_set
        if is_new and new_count >= effective_allowed_new:
            continue
        if not _group_slot_available(code, group_counts, group_limit, groups):
            continue
        selected.append(code)
        _bump_group(code, group_counts, groups)
        if is_new:
            new_count += 1

    if len(selected) < top_n:
        for code in ranked:
            if len(selected) >= top_n:
                break
            if code in selected:
                continue
            is_new = code not in previous_set
            if is_new and new_count >= effective_allowed_new:
                continue
            selected.append(code)
            if is_new:
                new_count += 1
    return _dedupe_preserve(selected)


def _group_limit(top_n: int, max_group_weight: float | None) -> int | None:
    if max_group_weight is None:
        return None
    try:
        weight = float(max_group_weight)
    except (TypeError, ValueError):
        return None
    if top_n <= 0 or not np.isfinite(weight) or weight <= 0 or weight >= 1:
        return None
    return max(1, int(np.floor(top_n * weight + 1e-12)))


def _normalize_score_series(score_series: pd.Series) -> pd.Series:
    scores = score_series.dropna().sort_values(ascending=False)
    if scores.empty:
        return scores

    normalized_index = pd.Index([_normalize_instrument(code) for code in scores.index], name=scores.index.name)
    result = pd.Series(scores.to_numpy(), index=normalized_index, name=scores.name)
    result = result[result.index != ""]
    if result.index.has_duplicates:
        result = result[~result.index.duplicated(keep="first")]
    result.attrs = dict(getattr(score_series, "attrs", {}))
    return result


def _normalize_instrument(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def _normalize_instruments(values: Iterable[object]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        instrument = _normalize_instrument(value)
        if not instrument or instrument in seen:
            continue
        result.append(instrument)
        seen.add(instrument)
    return result


def _normalize_group_map(group_map: Mapping[str, object] | pd.Series | None) -> dict[str, str] | None:
    if group_map is None:
        return None
    items = group_map.items() if isinstance(group_map, Mapping) else pd.Series(group_map).items()
    groups: dict[str, str] = {}
    for code, group in items:
        if pd.isna(code) or pd.isna(group):
            continue
        normalized_code = str(code).strip().upper()
        normalized_group = str(group).strip()
        if not normalized_code or not normalized_group:
            continue
        groups[normalized_code] = normalized_group
    return groups or None


def _group_for(code: str, groups: dict[str, str]) -> str | None:
    return groups.get(str(code).strip().upper())


def _group_slot_available(
    code: str,
    group_counts: dict[str, int],
    group_limit: int,
    groups: dict[str, str],
) -> bool:
    group = _group_for(code, groups)
    if group is None:
        return True
    return group_counts.get(group, 0) < group_limit


def _bump_group(code: str, group_counts: dict[str, int], groups: dict[str, str]) -> None:
    group = _group_for(code, groups)
    if group is None:
        return
    group_counts[group] = group_counts.get(group, 0) + 1


def _group_cap_satisfied(holdings: Iterable[str], group_limit: int, groups: dict[str, str]) -> bool:
    counts: dict[str, int] = {}
    for code in holdings:
        group = _group_for(code, groups)
        if group is None:
            continue
        counts[group] = counts.get(group, 0) + 1
        if counts[group] > group_limit:
            return False
    return True


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
        if not holdings:
            previous = holdings
            continue
        rows.extend({"date": pd.Timestamp(date), "instrument": code, "weight": 1 / len(holdings)} for code in holdings)
        previous = holdings
    if not rows:
        return pd.DataFrame(columns=["date", "instrument", "weight"])
    return pd.DataFrame(rows)


def resample_signals(score_panel: pd.Series, rebalance_freq: str) -> pd.Series:
    if rebalance_freq == "daily":
        return score_panel
    if not isinstance(score_panel.index, pd.MultiIndex):
        raise ValueError("score_panel must use MultiIndex: datetime/instrument.")

    normalized = score_panel.copy()
    normalized.index = pd.MultiIndex.from_arrays(
        [
            pd.to_datetime(score_panel.index.get_level_values(0)).normalize(),
            score_panel.index.get_level_values(1).astype(str),
        ],
        names=score_panel.index.names,
    )
    dates = pd.Index(pd.to_datetime(normalized.index.get_level_values(0).unique())).sort_values()
    date_series = pd.Series(dates, index=dates)
    if rebalance_freq == "weekly":
        keep_dates = set(date_series.resample("W-FRI").last().dropna())
    elif rebalance_freq == "monthly":
        keep_dates = set(date_series.resample("ME").last().dropna())
    else:
        raise ValueError(f"Unsupported rebalance_freq: {rebalance_freq}")
    return normalized[normalized.index.get_level_values(0).isin(keep_dates)].sort_index().rename(score_panel.name)


def _cross_sectional_zscore(df: pd.DataFrame, min_obs: int = 5) -> pd.DataFrame:
    if not isinstance(df.index, pd.MultiIndex):
        if len(df) < min_obs:
            return pd.DataFrame(np.nan, index=df.index, columns=df.columns)
        std = df.std(ddof=0).replace(0, pd.NA)
        return _mask_nonfinite((df - df.mean()) / std)

    date_level = df.index.names[0] or 0
    parts: list[pd.DataFrame] = []
    for _date, daily in df.groupby(level=date_level, sort=False):
        counts = daily.count()
        valid_columns = counts[counts >= min_obs].index
        if len(valid_columns) == 0:
            parts.append(pd.DataFrame(np.nan, index=daily.index, columns=daily.columns, dtype="float32"))
            continue

        numeric = daily.astype("float32")
        stds = numeric[valid_columns].std(ddof=0).replace(0, np.nan)
        scaled = pd.DataFrame(np.nan, index=daily.index, columns=daily.columns, dtype="float32")
        scaled[valid_columns] = (numeric[valid_columns] - numeric[valid_columns].mean()) / stds
        parts.append(_mask_nonfinite(scaled).astype("float32"))

    if not parts:
        return pd.DataFrame(index=df.index, columns=df.columns, dtype="float32")
    return pd.concat(parts).sort_index()


def _mask_nonfinite(df: pd.DataFrame) -> pd.DataFrame:
    values = df.to_numpy(copy=False)
    if np.isfinite(values).all():
        return df
    return df.mask(~np.isfinite(values))
