from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from src.config_loader import resolve_path
from src.factor_ic import calculate_factor_ic, calculate_rolling_ic, make_ic_weights, make_rolling_ic_weights, summarize_ic
from src.strategy import composite_factor, factor_columns_for_method


DYNAMIC_IC_SELECTOR_GROUPS = {"dynamic_ic_selector", "dynamic_ic"}
DEFAULT_DYNAMIC_IC_CANDIDATES = [
    "factor:LOW0",
    "inverse_factor:KUP",
    "inverse_factor:KLEN",
    "inverse_factor:KLOW",
    "inverse_factor:CORR5",
    "factor:ROC20",
    "factor:ROC30",
    "factor:RSV5",
]


def build_strategy_scores(
    factors: pd.DataFrame,
    config: dict,
    price_df: pd.DataFrame | None = None,
    price_file: str | Path | None = None,
) -> pd.Series:
    strategy_cfg = config.get("strategy", {})
    factor_group = str(strategy_cfg.get("factor_group", "momentum")).strip().lower()
    min_obs = int(strategy_cfg.get("min_cross_section_obs", 5))
    ic_cfg = config.get("ic", {})
    price_path = price_file or ic_cfg.get("price_file", "data/prices/ohlcv_adjusted.parquet")
    prices = price_df

    if factor_group in DYNAMIC_IC_SELECTOR_GROUPS:
        prices = prices if prices is not None else _load_ic_price_frame(price_path)
        scores = _build_dynamic_ic_selector_scores(factors, prices, config, min_obs=min_obs)
        return _apply_liquidity_filter(scores, prices, config.get("liquidity_filter", {}))

    if factor_group != "ic_weighted":
        scores = composite_factor(factors, method=factor_group, min_obs=min_obs)
        if _liquidity_filter_enabled(config.get("liquidity_filter", {})):
            prices = prices if prices is not None else _load_ic_price_frame(price_path)
            scores = _apply_liquidity_filter(scores, prices, config.get("liquidity_filter", {}))
        return scores

    prices = prices if prices is not None else _load_ic_price_frame(price_path)
    dynamic_weights = _load_or_compute_dynamic_weights(factors, prices, ic_cfg)
    scores = composite_factor(factors, method=factor_group, factor_weights_dynamic=dynamic_weights, min_obs=min_obs)
    return _apply_liquidity_filter(scores, prices, config.get("liquidity_filter", {}))


def build_latest_strategy_scores(
    factors: pd.DataFrame,
    config: dict,
    signal_date: str | pd.Timestamp | None = None,
    price_df: pd.DataFrame | None = None,
    price_file: str | Path | None = None,
) -> pd.Series:
    strategy_cfg = config.get("strategy", {})
    factor_group = str(strategy_cfg.get("factor_group", "momentum")).strip().lower()
    min_obs = int(strategy_cfg.get("min_cross_section_obs", 5))
    target_date = _resolve_score_date(factors, signal_date)
    latest_factors = _slice_factor_date(factors, target_date)
    if latest_factors.empty:
        raise ValueError(f"No factor rows found for signal date {target_date.date()}.")

    ic_cfg = config.get("ic", {})
    price_path = price_file or ic_cfg.get("price_file", "data/prices/ohlcv_adjusted.parquet")
    prices = price_df

    if factor_group in DYNAMIC_IC_SELECTOR_GROUPS:
        prices = prices if prices is not None else _load_ic_price_frame(price_path)
        weights = _latest_dynamic_ic_selector_weights(factors, prices, config, target_date)
        if weights.empty:
            raise ValueError(f"No usable dynamic IC selector weights found for signal date {target_date.date()}.")
        signed_factors = _signed_candidate_factors(latest_factors, _dynamic_ic_candidates(config))
        scores = composite_factor(signed_factors, method="ic_weighted", factor_weights=weights, min_obs=min_obs)
        return _apply_liquidity_filter(scores, prices, config.get("liquidity_filter", {}))

    if factor_group != "ic_weighted":
        scores = composite_factor(latest_factors, method=factor_group, min_obs=min_obs)
        if _liquidity_filter_enabled(config.get("liquidity_filter", {})):
            prices = prices if prices is not None else _load_ic_price_frame(price_path)
            scores = _apply_liquidity_filter(scores, prices, config.get("liquidity_filter", {}))
        return scores

    prices = prices if prices is not None else _load_ic_price_frame(price_path)
    weights = _latest_ic_weights(factors, prices, ic_cfg, target_date)
    if weights.empty:
        raise ValueError(f"No usable IC weights found for signal date {target_date.date()}.")
    scores = composite_factor(latest_factors, method=factor_group, factor_weights=weights, min_obs=min_obs)
    return _apply_liquidity_filter(scores, prices, config.get("liquidity_filter", {}))


def _load_ic_price_frame(path_value: str | Path) -> pd.DataFrame:
    price_path = resolve_path(path_value)
    if not price_path.exists() and price_path.name in {"ohlcv.parquet", "ohlcv_adjusted.parquet"}:
        fallback_name = "close_adjusted.parquet" if price_path.name == "ohlcv_adjusted.parquet" else "close.parquet"
        fallback_price_file = price_path.with_name(fallback_name)
        if fallback_price_file.exists():
            price_path = fallback_price_file
    if not price_path.exists():
        raise FileNotFoundError(f"Price file not found for rolling IC weights: {price_path}")
    return pd.read_parquet(price_path)


def _build_dynamic_ic_selector_scores(
    factors: pd.DataFrame,
    prices: pd.DataFrame,
    config: dict,
    min_obs: int,
) -> pd.Series:
    signed_factors = _signed_candidate_factors(factors, _dynamic_ic_candidates(config))
    weights = _dynamic_ic_selector_weights(signed_factors, prices, config.get("dynamic_ic_selector", {}))
    fallback = _dynamic_ic_fallback_weights(signed_factors, config.get("dynamic_ic_selector", {}))
    return composite_factor(signed_factors, method="ic_weighted", factor_weights=fallback, factor_weights_dynamic=weights, min_obs=min_obs)


def _latest_dynamic_ic_selector_weights(
    factors: pd.DataFrame,
    prices: pd.DataFrame,
    config: dict,
    target_date: pd.Timestamp,
) -> pd.Series:
    signed_factors = _signed_candidate_factors(factors, _dynamic_ic_candidates(config))
    dynamic_weights = _dynamic_ic_selector_weights(signed_factors, prices, config.get("dynamic_ic_selector", {}))
    eligible_dates = [date for date in dynamic_weights if pd.Timestamp(date).normalize() <= target_date]
    if not eligible_dates:
        return _dynamic_ic_fallback_weights(signed_factors, config.get("dynamic_ic_selector", {}))
    return dynamic_weights[max(eligible_dates)]


def _dynamic_ic_candidates(config: dict) -> list[str]:
    selector_cfg = config.get("dynamic_ic_selector", {})
    candidates = selector_cfg.get("candidates", DEFAULT_DYNAMIC_IC_CANDIDATES)
    return [str(candidate) for candidate in candidates]


def _dynamic_ic_fallback_weights(signed_factors: pd.DataFrame, selector_cfg: dict) -> pd.Series:
    fallback_candidate = selector_cfg.get("fallback_candidate", "factor:LOW0")
    if fallback_candidate in {None, ""}:
        return pd.Series(dtype=float)
    try:
        column, _direction = _candidate_column_and_direction(str(fallback_candidate), signed_factors.columns)
    except ValueError:
        return pd.Series(dtype=float)
    if column not in signed_factors.columns:
        return pd.Series(dtype=float)
    return pd.Series({column: 1.0}, dtype=float)


def _signed_candidate_factors(factors: pd.DataFrame, candidates: list[str]) -> pd.DataFrame:
    numeric = factors.select_dtypes("number")
    if numeric.empty:
        raise ValueError("factor_df has no numeric factor columns.")

    pieces: dict[str, pd.Series] = {}
    for candidate in candidates:
        column, direction = _candidate_column_and_direction(candidate, numeric.columns)
        pieces[column] = numeric[column].astype("float32", copy=False) * direction
    if not pieces:
        raise ValueError("dynamic_ic_selector has no usable candidate factors.")
    return pd.DataFrame(pieces, index=factors.index)


def _candidate_column_and_direction(candidate: str, columns: pd.Index) -> tuple[str, float]:
    method = str(candidate).strip()
    direction = 1.0
    lowered = method.lower()
    for prefix in ("low_", "inverse_", "short_"):
        if lowered.startswith(prefix):
            method = method[len(prefix) :]
            direction = -1.0
            break
    selected = factor_columns_for_method(columns, method)
    if len(selected) != 1:
        raise ValueError(f"dynamic_ic_selector candidate '{candidate}' matched {len(selected)} columns; use factor:<name>.")
    return str(selected[0]), direction


def _dynamic_ic_selector_weights(
    signed_factors: pd.DataFrame,
    prices: pd.DataFrame,
    selector_cfg: dict,
) -> dict[pd.Timestamp, pd.Series]:
    rolling_ic = calculate_rolling_ic(
        signed_factors,
        prices,
        horizon=int(selector_cfg.get("horizon", 20)),
        method=str(selector_cfg.get("method", "spearman")),
        window=int(selector_cfg.get("window", 504)),
        min_periods=int(selector_cfg.get("min_periods", 120)),
        min_obs=int(selector_cfg.get("min_obs", 20)),
    )
    scores = _dynamic_ic_selector_metric(rolling_ic, selector_cfg)
    scores.attrs = {}
    min_score = selector_cfg.get("min_score")
    min_score = float(min_score) if min_score is not None else None

    weights: dict[pd.Timestamp, pd.Series] = {}
    top_k = max(1, int(selector_cfg.get("top_k", 1)))
    for date, row in scores.iterrows():
        candidates = row.dropna()
        if min_score is not None:
            candidates = candidates[candidates >= min_score]
        if candidates.empty:
            continue
        selected = candidates.sort_values(ascending=False).head(top_k)
        selected_weights = selected.clip(lower=0.0)
        if float(selected_weights.sum()) <= 0:
            selected_weights = pd.Series(1.0, index=selected.index, dtype=float)
        weights[pd.Timestamp(date).normalize()] = (selected_weights / selected_weights.sum()).astype(float)
    return weights


def _dynamic_ic_selector_metric(rolling_ic: pd.DataFrame, selector_cfg: dict) -> pd.DataFrame:
    metric = str(selector_cfg.get("metric", "mean")).strip().lower()
    if metric == "mean":
        return rolling_ic
    if metric in {"ir", "ic_ir"}:
        daily_ic = rolling_ic.attrs.get("daily_ic")
        if not isinstance(daily_ic, pd.DataFrame):
            raise ValueError("rolling IC frame is missing daily_ic attrs.")
        horizon = max(1, int(rolling_ic.attrs.get("horizon", selector_cfg.get("horizon", 20))))
        window = int(rolling_ic.attrs.get("window", selector_cfg.get("window", 504)))
        min_periods = int(rolling_ic.attrs.get("min_periods", selector_cfg.get("min_periods", 120)))
        source = daily_ic.shift(horizon)
        rolling_std = source.rolling(window=window, min_periods=min_periods).std(ddof=0)
        return rolling_ic / rolling_std.where(rolling_std != 0)
    raise ValueError(f"Unsupported dynamic_ic_selector metric: {metric}")


def _liquidity_filter_enabled(filter_cfg: dict) -> bool:
    return bool(filter_cfg.get("enabled", False))


def _apply_liquidity_filter(scores: pd.Series, prices: pd.DataFrame, filter_cfg: dict) -> pd.Series:
    if not _liquidity_filter_enabled(filter_cfg):
        return scores
    if scores.empty:
        return scores
    if not isinstance(scores.index, pd.MultiIndex):
        raise ValueError("liquidity filter requires MultiIndex scores: datetime/instrument.")

    amount = _price_field(prices, str(filter_cfg.get("field", "amount")))
    if amount.empty:
        raise ValueError("liquidity filter requires an amount field in the price panel.")
    amount.index = pd.to_datetime(amount.index).normalize()
    adv = amount.rolling(
        window=int(filter_cfg.get("window", 10)),
        min_periods=int(filter_cfg.get("min_periods", 5)),
    ).mean()
    side = str(filter_cfg.get("side", "low")).strip().lower()
    quantile = float(filter_cfg.get("quantile", 0.35))

    if side not in {"low", "high"}:
        raise ValueError(f"Unsupported liquidity filter side: {side}")

    date_level = scores.index.names[0] or 0
    instrument_level = scores.index.names[1] or 1
    score_dates = pd.to_datetime(scores.index.get_level_values(date_level)).normalize()
    score_instruments = scores.index.get_level_values(instrument_level).astype(str)
    lookup_index = pd.MultiIndex.from_arrays([score_dates, score_instruments], names=["datetime", "instrument"])

    adv_stack = adv.stack(future_stack=True).rename("adv")
    adv_stack.index = adv_stack.index.set_names(["datetime", "instrument"])
    aligned_adv = adv_stack.reindex(lookup_index).to_numpy()
    thresholds = adv.quantile(quantile, axis=1).reindex(score_dates).to_numpy()

    if side == "low":
        mask_values = aligned_adv <= thresholds
    else:
        mask_values = aligned_adv >= thresholds
    mask_values &= pd.notna(aligned_adv) & pd.notna(thresholds)
    mask = pd.Series(mask_values, index=scores.index)
    return scores.where(mask).sort_index().rename(scores.name)


def _price_field(prices: pd.DataFrame, field: str) -> pd.DataFrame:
    field = str(field).lower()
    if isinstance(prices.columns, pd.MultiIndex):
        field_names = prices.columns.get_level_values(0).astype(str).str.lower()
        if field not in set(field_names):
            return pd.DataFrame(index=prices.index)
        frame = prices.loc[:, field_names == field].copy()
        frame.columns = frame.columns.get_level_values(1).astype(str)
        return frame
    return pd.DataFrame(index=prices.index)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _load_or_compute_dynamic_weights(factors: pd.DataFrame, prices: pd.DataFrame, ic_cfg: dict) -> dict[pd.Timestamp, pd.Series]:
    cache_file = ic_cfg.get("weights_cache_file")
    expected_meta = _weights_cache_meta(factors, prices, ic_cfg)
    if cache_file and bool(ic_cfg.get("weights_cache_enabled", True)):
        cache_path = resolve_path(cache_file)
        cached = _read_weights_cache(cache_path, expected_meta)
        if cached is not None:
            return cached

    rolling_ic = calculate_rolling_ic(
        factors,
        prices,
        window=int(ic_cfg.get("window", 252)),
        min_periods=int(ic_cfg.get("min_periods", 60)),
    )
    dynamic_weights = make_rolling_ic_weights(
        rolling_ic,
        top_k=int(ic_cfg.get("top_k", 30)),
        min_abs_ic=float(ic_cfg.get("min_abs_ic", 0.02)),
        min_periods=int(ic_cfg.get("min_periods", 60)),
        correlation_threshold=float(ic_cfg.get("corr_threshold", 0.7)),
        weight_smoothing=float(ic_cfg.get("weight_smoothing", 0.0)),
        max_weight_turnover=_optional_float(ic_cfg.get("max_weight_turnover")),
    )
    if cache_file and bool(ic_cfg.get("weights_cache_enabled", True)):
        _write_weights_cache(resolve_path(cache_file), dynamic_weights, expected_meta)
    return dynamic_weights


def _latest_ic_weights(factors: pd.DataFrame, prices: pd.DataFrame, ic_cfg: dict, target_date: pd.Timestamp) -> pd.Series:
    window = int(ic_cfg.get("window", 252))
    min_periods = int(ic_cfg.get("min_periods", 60))
    lookback_sessions = int(ic_cfg.get("latest_weight_lookback_sessions", max(window + min_periods + 5, window + 5)))
    factor_history = _slice_recent_factor_history(factors, target_date, lookback_sessions)
    price_history = _slice_price_history(prices, target_date, lookback_sessions)
    ic_df = calculate_factor_ic(factor_history, price_history, min_obs=int(ic_cfg.get("min_obs", 20)))
    if len(ic_df) < min_periods:
        return pd.Series(dtype=float)
    ic_df = ic_df.tail(window)
    summary = summarize_ic(ic_df)
    top_k = int(ic_cfg.get("top_k", 30))
    min_abs_ic = float(ic_cfg.get("min_abs_ic", 0.02))
    weights = make_ic_weights(summary, top_k=top_k, min_abs_ic=min_abs_ic)
    if _is_empty_weight_vector(weights) and min_abs_ic > 0:
        weights = make_ic_weights(summary, top_k=top_k, min_abs_ic=0.0)
    if _is_empty_weight_vector(weights):
        weights = _mean_ic_weights(summary, top_k=top_k, min_abs_ic=0.0)
    if _is_empty_weight_vector(weights):
        weights = _equal_ic_weights(summary, top_k=top_k)
    return weights


def _is_empty_weight_vector(weights: pd.Series) -> bool:
    return weights.empty or float(weights.abs().sum()) <= 0


def _mean_ic_weights(summary: pd.DataFrame, top_k: int, min_abs_ic: float = 0.0) -> pd.Series:
    if summary.empty or "mean_ic" not in summary.columns:
        return pd.Series(dtype=float)
    scores = summary["mean_ic"].copy()
    if min_abs_ic > 0:
        scores = scores[scores.abs() >= min_abs_ic]
    scores = scores.reindex(scores.abs().sort_values(ascending=False).index).head(top_k)
    return scores.fillna(0)


def _equal_ic_weights(summary: pd.DataFrame, top_k: int) -> pd.Series:
    if summary.empty:
        return pd.Series(dtype=float)
    candidates = summary.copy()
    if "count" in candidates.columns:
        candidates = candidates[pd.to_numeric(candidates["count"], errors="coerce").fillna(0) > 0]
    selected = list(candidates.index[: max(top_k, 1)])
    if not selected:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(selected), index=selected, dtype=float)


def _read_weights_cache(cache_path: Path, expected_meta: dict[str, object]) -> dict[pd.Timestamp, pd.Series] | None:
    meta_path = _weights_cache_meta_path(cache_path)
    if not cache_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta != expected_meta:
            return None
        with cache_path.open("rb") as f:
            weights = pickle.load(f)
    except (OSError, json.JSONDecodeError, pickle.PickleError, EOFError, AttributeError, ValueError):
        return None
    if not isinstance(weights, dict):
        return None
    return {pd.Timestamp(date).normalize(): series for date, series in weights.items() if isinstance(series, pd.Series)}


def _write_weights_cache(cache_path: Path, weights: dict[pd.Timestamp, pd.Series], meta: dict[str, object]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(weights, f, protocol=pickle.HIGHEST_PROTOCOL)
    _weights_cache_meta_path(cache_path).write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def _weights_cache_meta_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.name}.meta.json")


def _resolve_score_date(factors: pd.DataFrame, signal_date: str | pd.Timestamp | None) -> pd.Timestamp:
    if factors.empty or not isinstance(factors.index, pd.MultiIndex):
        raise ValueError("factors must use MultiIndex: datetime/instrument.")
    dates = _factor_dates(factors)
    if str(signal_date).lower() in {"none", "", "latest"} or signal_date is None:
        return pd.Timestamp(dates.max()).normalize()
    target = pd.Timestamp(signal_date).normalize()
    if target not in set(dates):
        raise ValueError(f"Signal date {target.date()} is not present in factor data.")
    return target


def _factor_dates(factors: pd.DataFrame) -> pd.DatetimeIndex:
    date_level = factors.index.names[0] or 0
    return pd.DatetimeIndex(pd.to_datetime(factors.index.get_level_values(date_level)).normalize()).unique().sort_values()


def _slice_factor_date(factors: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    date_level = factors.index.names[0] or 0
    dates = pd.to_datetime(factors.index.get_level_values(date_level)).normalize()
    return factors[dates == target_date]


def _slice_recent_factor_history(factors: pd.DataFrame, target_date: pd.Timestamp, sessions: int) -> pd.DataFrame:
    date_level = factors.index.names[0] or 0
    dates = pd.to_datetime(factors.index.get_level_values(date_level)).normalize()
    eligible = pd.DatetimeIndex(dates[dates <= target_date]).unique().sort_values()
    if eligible.empty:
        return factors.iloc[0:0]
    selected = set(eligible[-max(sessions, 1) :])
    return factors[dates.isin(selected)]


def _slice_price_history(prices: pd.DataFrame, target_date: pd.Timestamp, sessions: int) -> pd.DataFrame:
    if prices.empty:
        return prices
    price_dates = pd.DatetimeIndex(pd.to_datetime(prices.index).normalize())
    eligible = pd.DatetimeIndex(price_dates[price_dates <= target_date]).unique().sort_values()
    if eligible.empty:
        return prices.iloc[0:0]
    selected = set(eligible[-max(sessions, 1) :])
    return prices[price_dates.isin(selected)]


def _weights_cache_meta(factors: pd.DataFrame, prices: pd.DataFrame, ic_cfg: dict) -> dict[str, object]:
    return {
        "params": {
            "window": int(ic_cfg.get("window", 252)),
            "min_periods": int(ic_cfg.get("min_periods", 60)),
            "top_k": int(ic_cfg.get("top_k", 30)),
            "min_abs_ic": float(ic_cfg.get("min_abs_ic", 0.02)),
            "corr_threshold": float(ic_cfg.get("corr_threshold", 0.7)),
            "weight_smoothing": float(ic_cfg.get("weight_smoothing", 0.0)),
            "max_weight_turnover": _optional_float(ic_cfg.get("max_weight_turnover")),
        },
        "factors": _frame_signature(factors),
        "prices": _frame_signature(prices),
    }


def _frame_signature(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {"rows": 0, "columns": [], "start": "", "end": "", "symbols": [], "sample_hash": 0}
    if isinstance(frame.index, pd.MultiIndex):
        dates = pd.to_datetime(frame.index.get_level_values(0)).normalize()
        symbols = sorted(set(frame.index.get_level_values(1).astype(str).str.upper()))
    else:
        dates = pd.to_datetime(frame.index).normalize()
        if isinstance(frame.columns, pd.MultiIndex):
            symbols = sorted(set(frame.columns.get_level_values(-1).astype(str).str.upper()))
        else:
            symbols = sorted(set(frame.columns.astype(str).str.upper()))
    return {
        "rows": int(len(frame)),
        "columns": [str(col) for col in frame.columns],
        "start": str(dates.min().date()),
        "end": str(dates.max().date()),
        "symbols": symbols,
        "sample_hash": _sample_frame_hash(frame),
    }


def _sample_frame_hash(frame: pd.DataFrame, max_samples: int = 512) -> int:
    if frame.empty:
        return 0
    if len(frame) <= max_samples:
        sample = frame
    else:
        positions = sorted({round(index * (len(frame) - 1) / (max_samples - 1)) for index in range(max_samples)})
        sample = frame.iloc[positions]
    stable = sample.copy()
    stable.columns = [str(col) for col in stable.columns]
    hashed = pd.util.hash_pandas_object(stable, index=True)
    return int(hashed.sum() % (2**63 - 1))
