"""模块说明：过滤存在停牌、跌停或价格缺失风险的候选标的。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
import weakref

import numpy as np
import pandas as pd

from src.common import PRICE_FIELD_COLUMNS, looks_like_field_table as _looks_like_field_table, normalize_instrument as _normalize_instrument

_PRICE_FIELD_CACHE: dict[int, tuple[weakref.ReferenceType[pd.DataFrame], set[str], dict[str, pd.DataFrame]]] = {}
_NORMALIZED_PRICE_CACHE: dict[int, tuple[weakref.ReferenceType[pd.DataFrame], pd.DataFrame]] = {}
_NORMALIZED_PRICE_CACHE_MAX_SIZE = 8


def selection_risk_filter_enabled(config: dict[str, Any] | None) -> bool:
    """函数说明：处理 selection_risk_filter_enabled 主要逻辑。"""
    cfg = _selection_risk_config(config or {})
    return bool(cfg.get("enabled", False))


def filter_scores_by_selection_risk(
    scores: pd.Series,
    prices: pd.DataFrame,
    signal_date: str | pd.Timestamp,
    config: dict[str, Any] | None,
) -> pd.Series:
    """函数说明：过滤 filter_scores_by_selection_risk 主要逻辑。"""
    cfg = _selection_risk_config(config or {})
    if not bool(cfg.get("enabled", False)) or scores.empty:
        return scores
    if prices is None or prices.empty:
        raise ValueError("selection_risk_filter requires a non-empty price panel.")

    instruments = [str(code) for code in scores.dropna().index.tolist()]
    eligible = selection_risk_eligible_instruments(prices, signal_date, instruments, config or {})
    mask = pd.Series([_normalize_instrument(code) in eligible for code in scores.index], index=scores.index)
    result = scores.where(mask)
    result.attrs = dict(getattr(scores, "attrs", {}))
    return result.rename(scores.name)


def selection_risk_eligible_instruments(
    prices: pd.DataFrame,
    signal_date: str | pd.Timestamp,
    instruments: Iterable[str],
    config: dict[str, Any] | None,
) -> set[str]:
    """函数说明：处理 selection_risk_eligible_instruments 主要逻辑。"""
    cfg = _selection_risk_config(config or {})
    if not bool(cfg.get("enabled", False)):
        return {_normalize_instrument(code) for code in instruments if _normalize_instrument(code)}

    normalized_prices = _normalize_price_frame(prices)
    signal_ts = pd.Timestamp(signal_date).normalize()

    if not _has_price_field(normalized_prices, "close"):
        raise ValueError("selection_risk_filter requires a close field in the price panel.")
    eligible_dates = pd.DatetimeIndex(normalized_prices.index[normalized_prices.index <= signal_ts]).unique().sort_values()
    if eligible_dates.empty:
        return set()
    lookback = max(1, int(cfg.get("lookback_sessions", 5)))
    lookback_dates = eligible_dates[-lookback:]

    required_fields = [str(value).strip().lower() for value in cfg.get("required_price_fields", ["open", "close"])]
    required_fields = [field for field in required_fields if field]
    max_missing = max(0, int(cfg.get("max_missing_price_sessions", 0)))
    max_limit_down_days = cfg.get("max_limit_down_days", 0)
    max_limit_down_days = None if max_limit_down_days is None else max(0, int(max_limit_down_days))
    require_positive_volume = bool(cfg.get("require_positive_volume", True))

    instrument_index = pd.Index([_normalize_instrument(code) for code in instruments]).drop_duplicates()
    instrument_index = instrument_index[instrument_index != ""]
    if instrument_index.empty:
        return set()

    missing_count = _missing_price_session_counts(
        normalized_prices,
        instrument_index,
        lookback_dates,
        required_fields,
        require_positive_volume=require_positive_volume,
    )
    eligible_mask = missing_count <= max_missing
    if max_limit_down_days is not None:
        limit_down_count = _recent_limit_down_day_counts(normalized_prices, instrument_index, lookback_dates, config or {}, cfg)
        eligible_mask &= limit_down_count <= max_limit_down_days
    return set(eligible_mask[eligible_mask].index.astype(str))


def _selection_risk_config(config: dict[str, Any]) -> dict[str, Any]:
    """函数说明：处理 selection_risk_config 的内部辅助逻辑。"""
    cfg = config.get("selection_risk_filter", {})
    return cfg if isinstance(cfg, dict) else {}


def _missing_price_session_counts(
    prices: pd.DataFrame,
    instruments: pd.Index,
    lookback_dates: pd.DatetimeIndex,
    required_fields: list[str],
    require_positive_volume: bool,
) -> pd.Series:
    """函数说明：处理 missing_price_session_counts 的内部辅助逻辑。"""
    missing_sessions = pd.DataFrame(False, index=lookback_dates, columns=instruments)
    for field in required_fields:
        missing_sessions |= _missing_field_sessions(prices, field, instruments, lookback_dates)
    if require_positive_volume:
        missing_sessions |= _missing_field_sessions(prices, "volume", instruments, lookback_dates)
    return missing_sessions.sum(axis=0).astype("int64")


def _missing_field_sessions(
    prices: pd.DataFrame,
    field: str,
    instruments: pd.Index,
    lookback_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """函数说明：处理 missing_field_sessions 的内部辅助逻辑。"""
    frame = _price_field_slice(prices, field, lookback_dates, instruments)
    if frame.empty:
        return pd.DataFrame(True, index=lookback_dates, columns=instruments)
    values = _numeric_frame(frame)
    return values.isna() | (values <= 0)


def _recent_limit_down_day_counts(
    prices: pd.DataFrame,
    instruments: pd.Index,
    lookback_dates: pd.DatetimeIndex,
    config: dict[str, Any],
    filter_cfg: dict[str, Any],
) -> pd.Series:
    """函数说明：处理 recent_limit_down_day_counts 的内部辅助逻辑。"""
    needed_dates = _lookback_with_previous_dates(prices, lookback_dates)
    close = _price_field_slice(prices, "close", needed_dates, instruments)
    if close.empty:
        return pd.Series(0, index=instruments, dtype="int64")
    low = _price_field_slice(prices, "low", lookback_dates, instruments)
    probe_frame = low if not low.empty else close
    previous_close = _numeric_frame(close.shift(1).reindex(lookback_dates).reindex(columns=instruments))
    probe = _numeric_frame(probe_frame.reindex(lookback_dates).reindex(columns=instruments))
    buffer = max(float(filter_cfg.get("limit_down_buffer", 0.0)), 0.0)
    thresholds = _limit_down_threshold_frame(prices, lookback_dates, instruments, config) - buffer
    thresholds = thresholds.clip(lower=0.0)
    limit_down = probe <= previous_close * (1 - thresholds)
    limit_down &= previous_close.notna() & probe.notna() & (previous_close > 0)
    return limit_down.sum(axis=0).astype("int64")


def _limit_down_threshold_frame(
    prices: pd.DataFrame,
    lookback_dates: pd.DatetimeIndex,
    instruments: pd.Index,
    config: dict[str, Any],
) -> pd.DataFrame:
    """函数说明：处理 limit_down_threshold_frame 的内部辅助逻辑。"""
    thresholds = pd.Series(
        [_base_limit_down_threshold_for_stock(str(stock), config) for stock in instruments],
        index=instruments,
        dtype="float64",
    )
    frame = pd.DataFrame(
        np.tile(thresholds.to_numpy(), (len(lookback_dates), 1)),
        index=lookback_dates,
        columns=instruments,
    )
    is_st = _price_field_slice(prices, "is_st", lookback_dates, instruments)
    if is_st.empty:
        return frame
    st_flags = is_st.fillna(False).astype(bool)
    return frame.where(~st_flags, float(_config_value(config, "st_limit_down_threshold", 0.049)))


def _limit_down_threshold_for_stock(stock: str, prices: pd.DataFrame, date: pd.Timestamp, config: dict[str, Any]) -> float:
    """函数说明：处理 limit_down_threshold_for_stock 的内部辅助逻辑。"""
    if _is_st_on_date(stock, prices, date):
        return float(_config_value(config, "st_limit_down_threshold", 0.049))
    return _base_limit_down_threshold_for_stock(stock, config)


def _base_limit_down_threshold_for_stock(stock: str, config: dict[str, Any]) -> float:
    """函数说明：处理 base_limit_down_threshold_for_stock 的内部辅助逻辑。"""
    lowered = str(stock).lower()
    if lowered.startswith(("688", "689")):
        return float(
            _config_value(
                config,
                "star_limit_down_threshold",
                _config_value(config, "growth_limit_down_threshold", 0.199),
            )
        )
    if lowered.startswith(("300", "301")):
        return float(
            _config_value(
                config,
                "growth_limit_down_threshold",
                _config_value(config, "star_limit_down_threshold", 0.199),
            )
        )
    if lowered.startswith(("8", "4")):
        return float(_config_value(config, "bj_limit_down_threshold", 0.299))
    return float(_config_value(config, "limit_down_threshold", 0.099))


def _is_st_on_date(stock: str, prices: pd.DataFrame, date: pd.Timestamp) -> bool:
    """函数说明：判断 is_st_on_date 是否成立。"""
    is_st = _price_field_slice(prices, "is_st", pd.DatetimeIndex([pd.Timestamp(date).normalize()]), pd.Index([stock]))
    if is_st.empty or stock not in is_st.columns or date not in is_st.index:
        return False
    value = is_st.loc[date, stock]
    if pd.isna(value):
        return False
    return bool(value)


def _config_value(config: dict[str, Any], key: str, default: Any) -> Any:
    """函数说明：处理 config_value 的内部辅助逻辑。"""
    if key in config:
        return config[key]
    backtest = config.get("backtest", {})
    if isinstance(backtest, dict) and key in backtest:
        return backtest[key]
    return default


def _normalize_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """函数说明：规范化 normalize_price_frame 的内部辅助逻辑。"""
    source_prices = prices
    cached = _get_cached_normalized_price_frame(source_prices)
    if cached is not None:
        return cached

    raw_dates = pd.DatetimeIndex(pd.to_datetime(prices.index, errors="coerce"))
    valid_dates = ~pd.isna(raw_dates)
    if not valid_dates.all():
        prices = prices.loc[valid_dates].copy(deep=False)
        raw_dates = raw_dates[valid_dates]

    order = np.argsort(raw_dates.to_numpy(), kind="mergesort")
    if not np.array_equal(order, np.arange(len(raw_dates))):
        prices = prices.iloc[order].copy(deep=False)
        raw_dates = raw_dates[order]
    normalized_index = raw_dates.normalize()
    if isinstance(prices.columns, pd.MultiIndex):
        normalized_columns = pd.MultiIndex.from_arrays(
            [
                [_normalize_price_field(value) for value in prices.columns.get_level_values(0)],
                [_normalize_instrument(value) for value in prices.columns.get_level_values(1)],
            ],
            names=["field", "instrument"],
        )
        columns_need_normalization = (
            not prices.columns.equals(normalized_columns)
            or list(prices.columns.names) != ["field", "instrument"]
        )
        if not prices.index.equals(normalized_index) or columns_need_normalization:
            prices = prices.copy(deep=False)
            prices.index = normalized_index
            prices.columns = normalized_columns
        if prices.index.has_duplicates:
            prices = prices.loc[~prices.index.duplicated(keep="last")]
        non_empty_instruments = prices.columns.get_level_values("instrument") != ""
        if not bool(non_empty_instruments.all()):
            prices = prices.loc[:, non_empty_instruments]
        if prices.columns.has_duplicates:
            prices = prices.loc[:, ~prices.columns.duplicated(keep="last")]
        if not prices.index.is_monotonic_increasing:
            prices = prices.sort_index()
        return _store_normalized_price_frame(source_prices, prices)

    if _looks_like_field_table(prices.columns):
        raise ValueError("Non-MultiIndex price_df must be a close-price panel with instrument columns.")

    result = prices.copy(deep=False)
    result.index = normalized_index
    if result.index.has_duplicates:
        result = result.loc[~result.index.duplicated(keep="last")]
    result.columns = pd.MultiIndex.from_product(
        [["close"], [_normalize_instrument(value) for value in result.columns]],
        names=["field", "instrument"],
    )
    non_empty_instruments = result.columns.get_level_values("instrument") != ""
    if not bool(non_empty_instruments.all()):
        result = result.loc[:, non_empty_instruments]
    if result.columns.has_duplicates:
        result = result.loc[:, ~result.columns.duplicated(keep="last")]
    if not result.index.is_monotonic_increasing:
        result = result.sort_index()
    return _store_normalized_price_frame(source_prices, result)


def _get_cached_normalized_price_frame(prices: pd.DataFrame) -> pd.DataFrame | None:
    """函数说明：读取同一价格面板的规范化缓存。"""
    cached = _NORMALIZED_PRICE_CACHE.get(id(prices))
    if cached is None:
        return None
    cached_source, normalized = cached
    if cached_source() is prices:
        return normalized
    _NORMALIZED_PRICE_CACHE.pop(id(prices), None)
    return None


def _store_normalized_price_frame(source_prices: pd.DataFrame, normalized: pd.DataFrame) -> pd.DataFrame:
    """函数说明：保存价格面板规范化结果，并限制缓存体积。"""
    _prune_normalized_price_cache()
    if len(_NORMALIZED_PRICE_CACHE) >= _NORMALIZED_PRICE_CACHE_MAX_SIZE:
        _NORMALIZED_PRICE_CACHE.pop(next(iter(_NORMALIZED_PRICE_CACHE)), None)
    cache_key = id(source_prices)
    _NORMALIZED_PRICE_CACHE[cache_key] = (
        weakref.ref(source_prices, lambda _ref, key=cache_key: _NORMALIZED_PRICE_CACHE.pop(key, None)),
        normalized,
    )
    return normalized


def _prune_normalized_price_cache() -> None:
    """函数说明：清理已经释放的价格面板缓存项。"""
    dead_keys = [key for key, (source_ref, _) in _NORMALIZED_PRICE_CACHE.items() if source_ref() is None]
    for key in dead_keys:
        _NORMALIZED_PRICE_CACHE.pop(key, None)


def _normalize_price_field(value: object) -> str:
    """函数说明：规范化 normalize_price_field 的内部辅助逻辑。"""
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _has_price_field(prices: pd.DataFrame, field: str) -> bool:
    """函数说明：判断 has_price_field 是否成立。"""
    if not isinstance(prices.columns, pd.MultiIndex):
        return False
    field = _normalize_price_field(field)
    return field in set(_normalize_price_field(value) for value in prices.columns.get_level_values("field"))


def _price_field_slice(
    prices: pd.DataFrame,
    field: str,
    dates: pd.DatetimeIndex,
    instruments: pd.Index,
) -> pd.DataFrame:
    """函数说明：处理 price_field_slice 的内部辅助逻辑。"""
    field = _normalize_price_field(field)
    dates = pd.DatetimeIndex(pd.to_datetime(dates, errors="coerce")).dropna().normalize().unique().sort_values()
    instruments = pd.Index([_normalize_instrument(value) for value in instruments]).drop_duplicates()
    instruments = instruments[instruments != ""]
    if dates.empty or instruments.empty or not _has_price_field(prices, field):
        return pd.DataFrame(index=dates)

    field_frame = _price_field(prices, field)
    if field_frame.empty:
        return pd.DataFrame(index=dates, columns=instruments)
    return field_frame.reindex(dates).reindex(columns=instruments)


def _lookback_with_previous_dates(prices: pd.DataFrame, lookback_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """函数说明：处理 lookback_with_previous_dates 的内部辅助逻辑。"""
    price_dates = pd.DatetimeIndex(prices.index).unique().sort_values()
    lookback_dates = pd.DatetimeIndex(lookback_dates).unique().sort_values()
    if price_dates.empty or lookback_dates.empty:
        return lookback_dates
    positions = price_dates.searchsorted(lookback_dates)
    previous = [price_dates[position - 1] for position in positions if position > 0]
    return pd.DatetimeIndex([*previous, *lookback_dates]).unique().sort_values()


def _price_field(prices: pd.DataFrame, field: str) -> pd.DataFrame:
    """函数说明：处理 price_field 的内部辅助逻辑。"""
    field = _normalize_price_field(field)
    cache_key = id(prices)
    cached = _PRICE_FIELD_CACHE.get(cache_key)
    if cached is None or cached[0]() is not prices:
        field_names = set(_normalize_price_field(value) for value in prices.columns.get_level_values("field"))
        cache: dict[str, pd.DataFrame] = {}
        _PRICE_FIELD_CACHE[cache_key] = (weakref.ref(prices), field_names, cache)
    else:
        _, field_names, cache = cached
    if field in cache:
        return cache[field]
    if field not in field_names:
        frame = pd.DataFrame(index=prices.index)
    else:
        field_values = pd.Index([_normalize_price_field(value) for value in prices.columns.get_level_values("field")])
        frame = prices.loc[:, field_values == field]
        frame.columns = frame.columns.get_level_values("instrument").astype(str)
    cache[field] = frame
    return frame


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """函数说明：处理 numeric_frame 的内部辅助逻辑。"""
    try:
        return frame.astype("float64")
    except (TypeError, ValueError):
        return frame.apply(pd.to_numeric, errors="coerce")
