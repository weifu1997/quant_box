from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.common import normalize_instrument as _normalize_instrument


DEFAULT_DAILY_BASIC_FEATURES = [
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "pe",
    "pe_ttm",
    "pb",
    "ps",
    "ps_ttm",
    "dv_ratio",
    "dv_ttm",
    "total_mv",
    "circ_mv",
]

DEFAULT_PRICE_FEATURES = [
    "low_amount_5",
    "low_amount_20",
    "low_amount_60",
    "amount_log_20",
    "return_20",
    "return_60",
    "volatility_20",
    "illiquidity_20",
]
PRICE_FIELD_NAMES = {"open", "high", "low", "close", "volume", "amount", "vwap", "is_st"}


def append_daily_basic_features(
    factors: pd.DataFrame,
    daily_basic: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = config or {}
    enabled = bool(cfg.get("enabled", False)) and bool(cfg.get("daily_basic", True))
    if factors.empty or daily_basic.empty or not enabled:
        return factors, {"enabled": enabled, "features_added": 0, "dates_matched": 0}
    if not isinstance(factors.index, pd.MultiIndex):
        raise ValueError("factors must use MultiIndex: datetime/instrument.")

    fields = [str(field) for field in cfg.get("daily_basic_fields", DEFAULT_DAILY_BASIC_FEATURES)]
    lag_days = max(0, int(cfg.get("daily_basic_lag_days", 1)))
    min_coverage = float(cfg.get("min_coverage", 0.0))
    basics = _normalize_daily_basic(daily_basic, fields)
    if basics.empty:
        return factors, {"enabled": True, "features_added": 0, "dates_matched": 0}

    factor_dates = pd.to_datetime(factors.index.get_level_values(0)).normalize()
    factor_symbols = pd.Index([_normalize_instrument(value) for value in factors.index.get_level_values(1)])
    lookup_dates = factor_dates - pd.Timedelta(days=lag_days)
    aligned = _align_daily_basic_asof(basics, lookup_dates, factor_symbols, factors.index)
    aligned.index = factors.index
    aligned = _transform_daily_basic_features(aligned)
    if min_coverage > 0:
        coverage = aligned.notna().mean(axis=0)
        aligned = aligned.loc[:, coverage >= min_coverage]
    if aligned.empty:
        return factors, {"enabled": True, "features_added": 0, "dates_matched": 0}

    extended = pd.concat([factors, aligned.add_prefix("DB_")], axis=1)
    dates_matched = int(aligned.notna().any(axis=1).groupby(factor_dates).any().sum())
    return extended, {
        "enabled": True,
        "features_added": int(aligned.shape[1]),
        "dates_matched": dates_matched,
        "lag_days": lag_days,
        "fields": list(aligned.columns),
    }


def append_price_derived_features(
    factors: pd.DataFrame,
    prices: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = config or {}
    enabled = bool(cfg.get("enabled", False)) and bool(cfg.get("price_derived", True))
    if factors.empty or prices.empty or not enabled:
        return factors, {"enabled": enabled, "features_added": 0, "dates_matched": 0}
    if not isinstance(factors.index, pd.MultiIndex):
        raise ValueError("factors must use MultiIndex: datetime/instrument.")

    fields = [str(field).lower() for field in cfg.get("price_features", DEFAULT_PRICE_FEATURES)]
    lag_sessions = max(0, int(cfg.get("price_feature_lag_sessions", 1)))
    min_coverage = float(cfg.get("min_coverage", 0.0))
    feature_frame = _price_feature_frame(prices, fields)
    if feature_frame.empty:
        return factors, {"enabled": True, "features_added": 0, "dates_matched": 0}
    if lag_sessions:
        feature_frame = feature_frame.groupby(level=1, group_keys=False).shift(lag_sessions)

    factor_dates = pd.to_datetime(factors.index.get_level_values(0)).normalize()
    factor_symbols = pd.Index([_normalize_instrument(value) for value in factors.index.get_level_values(1)])
    lookup_index = pd.MultiIndex.from_arrays([factor_dates, factor_symbols], names=["datetime", "instrument"])
    aligned = feature_frame.reindex(lookup_index)
    aligned.index = factors.index
    if min_coverage > 0:
        coverage = aligned.notna().mean(axis=0)
        aligned = aligned.loc[:, coverage >= min_coverage]
    if aligned.empty:
        return factors, {"enabled": True, "features_added": 0, "dates_matched": 0}

    extended = pd.concat([factors, aligned.add_prefix("PX_")], axis=1)
    dates_matched = int(aligned.notna().any(axis=1).groupby(factor_dates).any().sum())
    return extended, {
        "enabled": True,
        "features_added": int(aligned.shape[1]),
        "dates_matched": dates_matched,
        "lag_sessions": lag_sessions,
        "fields": list(aligned.columns),
    }


def _normalize_daily_basic(daily_basic: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    if daily_basic.empty:
        return pd.DataFrame()
    frame = daily_basic.copy()
    if isinstance(frame.index, pd.MultiIndex):
        names = list(frame.index.names)
        date_level = names.index("trade_date") if "trade_date" in names else 0
        symbol_level = names.index("ts_code") if "ts_code" in names else 1
        index = pd.MultiIndex.from_arrays(
            [
                pd.to_datetime(frame.index.get_level_values(date_level)).normalize(),
                [_normalize_instrument(value) for value in frame.index.get_level_values(symbol_level)],
            ],
            names=["datetime", "instrument"],
        )
        frame.index = index
    else:
        if "trade_date" not in frame.columns or "ts_code" not in frame.columns:
            return pd.DataFrame()
        frame = frame.copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
        frame["ts_code"] = [_normalize_instrument(value) for value in frame["ts_code"]]
        frame = frame.dropna(subset=["trade_date", "ts_code"])
        frame = frame[frame["ts_code"] != ""]
        frame = frame.set_index(["trade_date", "ts_code"])
        frame.index = frame.index.set_names(["datetime", "instrument"])

    frame = frame[frame.index.get_level_values("instrument") != ""]
    selected = [field for field in fields if field in frame.columns]
    if not selected:
        return pd.DataFrame()
    numeric = frame[selected].apply(pd.to_numeric, errors="coerce")
    return numeric.sort_index()


def _align_daily_basic_asof(
    basics: pd.DataFrame,
    lookup_dates: pd.DatetimeIndex,
    factor_symbols: pd.Index,
    factor_index: pd.Index,
) -> pd.DataFrame:
    if basics.empty:
        return pd.DataFrame(index=factor_index)

    requests = pd.DataFrame(
        {
            "lookup_date": pd.DatetimeIndex(pd.to_datetime(lookup_dates).normalize()),
            "instrument": [_normalize_instrument(value) for value in factor_symbols],
        },
        index=pd.RangeIndex(len(factor_index)),
    )
    aligned = pd.DataFrame(np.nan, index=requests.index, columns=basics.columns, dtype="float64")
    for instrument, row_index in requests.groupby("instrument", sort=False).groups.items():
        try:
            history = basics.xs(instrument, level=1).sort_index()
        except KeyError:
            continue
        if history.empty:
            continue
        dates = pd.DatetimeIndex(requests.loc[row_index, "lookup_date"])
        positions = history.index.searchsorted(dates, side="right") - 1
        valid = positions >= 0
        if not bool(valid.any()):
            continue
        rows = row_index.to_numpy()[valid]
        aligned.iloc[rows, :] = history.iloc[positions[valid]].to_numpy(dtype="float64", copy=False)
    aligned.index = factor_index
    return aligned


def _transform_daily_basic_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        values = pd.to_numeric(result[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if column.endswith("_mv") or column in {"pe", "pe_ttm", "pb", "ps", "ps_ttm"}:
            values = np.log1p(values.clip(lower=0.0))
        result[column] = values.astype("float32")
    return result


def _price_feature_frame(prices: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    close = _price_field(prices, "close")
    amount = _price_field(prices, "amount")
    if close.empty:
        return pd.DataFrame()
    close = close.replace([np.inf, -np.inf], np.nan)
    amount = amount.replace([np.inf, -np.inf], np.nan) if not amount.empty else pd.DataFrame(index=close.index, columns=close.columns)
    returns = close.pct_change()
    panels: dict[str, pd.DataFrame] = {}
    for field in fields:
        if field.startswith("low_amount_"):
            window = _feature_window(field, "low_amount")
            panels[field.upper()] = -np.log1p(amount.rolling(window, min_periods=max(2, window // 2)).mean().clip(lower=0.0))
        elif field.startswith("amount_log_"):
            window = _feature_window(field, "amount_log")
            panels[field.upper()] = np.log1p(amount.rolling(window, min_periods=max(2, window // 2)).mean().clip(lower=0.0))
        elif field.startswith("return_"):
            window = _feature_window(field, "return")
            panels[field.upper()] = close.divide(close.shift(window)).sub(1.0)
        elif field.startswith("volatility_"):
            window = _feature_window(field, "volatility")
            panels[field.upper()] = returns.rolling(window, min_periods=max(2, window // 2)).std(ddof=0)
        elif field.startswith("illiquidity_"):
            window = _feature_window(field, "illiquidity")
            amt = amount.rolling(window, min_periods=max(2, window // 2)).mean().replace(0.0, np.nan)
            panels[field.upper()] = returns.abs().rolling(window, min_periods=max(2, window // 2)).mean().divide(amt)
    if not panels:
        return pd.DataFrame()

    stacked_parts = []
    for name, panel in panels.items():
        normalized = panel.copy()
        normalized.index = pd.to_datetime(normalized.index).normalize()
        normalized.columns = [_normalize_instrument(value) for value in normalized.columns]
        stacked = normalized.stack(future_stack=True).rename(name)
        stacked_parts.append(stacked)
    result = pd.concat(stacked_parts, axis=1).replace([np.inf, -np.inf], np.nan)
    result.index = result.index.set_names(["datetime", "instrument"])
    return result.astype("float32")


def _price_field(prices: pd.DataFrame, field: str) -> pd.DataFrame:
    field = str(field).strip().lower()
    if prices.empty:
        return pd.DataFrame()
    if isinstance(prices.columns, pd.MultiIndex):
        fields = prices.columns.get_level_values(0).astype(str).str.strip().str.lower()
        if field not in set(fields):
            return pd.DataFrame(index=prices.index)
        frame = prices.loc[:, fields == field].copy()
        frame.columns = [_normalize_instrument(value) for value in frame.columns.get_level_values(-1)]
    elif _looks_like_field_table(prices.columns):
        raise ValueError("Non-MultiIndex price_df must be a close-price panel with instrument columns.")
    elif field == "close":
        frame = prices.copy()
    elif field in {str(column).strip().lower() for column in prices.columns}:
        columns = prices.columns.astype(str).str.strip().str.lower()
        frame = prices.loc[:, columns == field].copy()
        frame.columns = [_normalize_instrument(value) for value in frame.columns]
    else:
        return pd.DataFrame(index=prices.index)
    raw_dates = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="coerce"))
    valid_dates = ~raw_dates.isna()
    frame = frame.loc[valid_dates].copy()
    raw_dates = raw_dates[valid_dates]
    if not frame.empty:
        order = np.argsort(raw_dates.to_numpy(), kind="mergesort")
        frame = frame.iloc[order].copy()
        raw_dates = raw_dates[order]
    frame.index = raw_dates.normalize()
    frame.columns = [_normalize_instrument(value) for value in frame.columns]
    frame = frame.loc[:, frame.columns != ""]
    if frame.columns.has_duplicates:
        frame = frame.loc[:, ~frame.columns.duplicated(keep="last")]
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame.sort_index().apply(pd.to_numeric, errors="coerce")


def _looks_like_field_table(columns: pd.Index) -> bool:
    labels = {str(column).strip().lower() for column in columns}
    return len(labels) > 1 and bool(labels & PRICE_FIELD_NAMES)


def _feature_window(field: str, prefix: str) -> int:
    suffix = field[len(prefix) :].strip("_")
    return max(2, int(suffix))
