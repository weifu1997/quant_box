from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config_loader import resolve_path


REGIME_BULL = "bull"
REGIME_BEAR = "bear"
REGIME_SIDEWAYS = "sideways"
REGIME_STATES = {REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS}
PRICE_FIELD_COLUMNS = {"open", "high", "low", "close", "volume", "vol", "amount", "vwap", "adj_factor", "is_st"}


def market_regime_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("market_regime", {}).get("enabled", False))


def detect_market_regime(price_df: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    cfg = config.get("market_regime", {})
    return _detect_regime(price_df, config, cfg, name="market_regime")


def detect_reporting_regime(price_df: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    base = dict(config.get("market_regime", {}))
    reporting_cfg = dict(config.get("reporting_regime", {}))
    cfg = {**base, **reporting_cfg}
    cfg.setdefault("lag_days", 0)
    return _detect_regime(price_df, config, cfg, name="reporting_regime")


def _detect_regime(price_df: pd.DataFrame, config: dict[str, Any], cfg: dict[str, Any], name: str) -> pd.Series:
    benchmark = _benchmark_close(price_df, config, cfg)
    if benchmark.empty:
        return pd.Series(dtype="object", name=name)

    benchmark = benchmark.sort_index()
    benchmark.index = pd.to_datetime(benchmark.index).normalize()
    benchmark = benchmark[~benchmark.index.duplicated(keep="last")].astype(float)

    ma_window = int(cfg.get("ma_window", 120))
    momentum_window = int(cfg.get("momentum_window", 20))
    vol_window = int(cfg.get("volatility_window", 20))
    min_periods = int(cfg.get("min_periods", min(ma_window, momentum_window, vol_window)))
    min_periods = max(1, min_periods)

    moving_average = benchmark.rolling(ma_window, min_periods=min(min_periods, ma_window)).mean()
    momentum = benchmark.pct_change(momentum_window)
    annual_volatility = benchmark.pct_change().rolling(vol_window, min_periods=min(min_periods, vol_window)).std(ddof=0) * np.sqrt(252)
    high_volatility = _high_volatility_mask(annual_volatility, cfg)
    drawdown_bear = _drawdown_bear_mask(benchmark, cfg, min_periods)

    bull = (
        (benchmark >= moving_average)
        & (momentum >= float(cfg.get("bull_momentum_min", 0.0)))
        & ~high_volatility
        & ~drawdown_bear
    )
    bear = (benchmark < moving_average) & (
        (momentum <= float(cfg.get("bear_momentum_max", 0.0))) | high_volatility
    )
    bear = bear | drawdown_bear

    regime = pd.Series(REGIME_SIDEWAYS, index=benchmark.index, name=name, dtype="object")
    regime.loc[bull.fillna(False)] = REGIME_BULL
    regime.loc[bear.fillna(False)] = REGIME_BEAR

    lag_days = int(cfg.get("lag_days", 1))
    if lag_days > 0:
        regime = regime.shift(lag_days).fillna(REGIME_SIDEWAYS)
    return regime.rename(name)


def regime_for_date(regimes: pd.Series, date: pd.Timestamp | str, default: str = REGIME_SIDEWAYS) -> str:
    if regimes.empty:
        return normalize_regime(default)
    target = pd.Timestamp(date).normalize()
    normalized = _normalize_regime_index(regimes)
    eligible = normalized.loc[normalized.index <= target]
    if eligible.empty:
        return normalize_regime(default)
    return normalize_regime(str(eligible.iloc[-1]), default)


def regimes_for_dates(regimes: pd.Series, dates: pd.Index, default: str = REGIME_SIDEWAYS) -> pd.Series:
    normalized_dates = pd.DatetimeIndex(pd.to_datetime(dates).normalize())
    if regimes.empty:
        return pd.Series(default, index=normalized_dates, dtype="object")
    normalized = _normalize_regime_index(regimes)
    aligned = normalized.reindex(normalized_dates, method="ffill").fillna(default)
    return aligned.map(lambda value: normalize_regime(str(value), default)).rename("market_regime")


def normalize_regime(value: str | None, default: str = REGIME_SIDEWAYS) -> str:
    state = str(value or default).strip().lower()
    return state if state in REGIME_STATES else default


def _normalize_regime_index(regimes: pd.Series) -> pd.Series:
    normalized = regimes.copy()
    raw_dates = pd.DatetimeIndex(pd.to_datetime(normalized.index, errors="coerce"))
    valid_dates = ~raw_dates.isna()
    normalized = normalized.loc[valid_dates].copy()
    raw_dates = raw_dates[valid_dates]
    if not normalized.empty:
        order = np.argsort(raw_dates.to_numpy(), kind="mergesort")
        normalized = normalized.iloc[order].copy()
        raw_dates = raw_dates[order]
    normalized.index = raw_dates.normalize()
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    return normalized.sort_index()


def defensive_exposure_schedule(regimes: pd.Series, config: dict[str, Any], dates: pd.Index | None = None) -> pd.Series:
    cfg = config.get("defensive_timing", {})
    if not bool(cfg.get("enabled", False)):
        target_dates = pd.DatetimeIndex(pd.to_datetime(dates).normalize()) if dates is not None else regimes.index
        return pd.Series(1.0, index=target_dates, name="exposure_scale", dtype=float)

    default = normalize_regime(cfg.get("default_regime"), REGIME_SIDEWAYS)
    target_dates = pd.DatetimeIndex(pd.to_datetime(dates).normalize()) if dates is not None else pd.DatetimeIndex(regimes.index)
    aligned = regimes_for_dates(regimes, target_dates, default=default)
    exposure_by_regime = {
        REGIME_BULL: float(cfg.get("bull_exposure", 1.0)),
        REGIME_SIDEWAYS: float(cfg.get("sideways_exposure", 0.60)),
        REGIME_BEAR: float(cfg.get("bear_exposure", 0.30)),
    }
    schedule = aligned.map(exposure_by_regime).astype(float).clip(lower=0.0)
    return schedule.rename("exposure_scale")


def apply_defensive_timing_to_backtest_config(
    bt_config: dict[str, Any],
    price_df: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    result = dict(bt_config)
    if not market_regime_enabled(config) or not bool(config.get("defensive_timing", {}).get("enabled", False)):
        return result

    regimes = detect_market_regime(price_df, config)
    if regimes.empty:
        return result

    exposure = defensive_exposure_schedule(regimes, config, pd.Index(pd.to_datetime(price_df.index)))
    result["exposure_schedule"] = exposure
    result["exposure_rebalance_threshold"] = float(
        config.get("defensive_timing", {}).get(
            "exposure_rebalance_threshold",
            result.get("exposure_rebalance_threshold", 0.05),
        )
    )
    return result


def defensive_exposure_for_date(
    price_df: pd.DataFrame,
    config: dict[str, Any],
    date: str | pd.Timestamp,
    default: float = 1.0,
) -> float:
    if not market_regime_enabled(config) or not bool(config.get("defensive_timing", {}).get("enabled", False)):
        return float(default)
    regimes = detect_market_regime(price_df, config)
    if regimes.empty:
        return float(default)
    exposure = defensive_exposure_schedule(regimes, config, pd.Index(pd.to_datetime(price_df.index)))
    target = pd.Timestamp(date).normalize()
    eligible = exposure.loc[exposure.index <= target]
    if eligible.empty:
        return float(default)
    return float(eligible.iloc[-1])


def summarize_regime_performance(equity_curve: pd.Series, regimes: pd.Series, config: dict[str, Any] | None = None) -> pd.DataFrame:
    if equity_curve.empty:
        return pd.DataFrame(columns=["regime", "start", "end", "days", "total_return", "annual_return", "max_drawdown"])
    annual_days = int((config or {}).get("annual_trading_days", 252))
    equity = equity_curve.sort_index().astype(float)
    equity.index = pd.to_datetime(equity.index).normalize()
    returns = equity.pct_change(fill_method=None).dropna()
    if returns.empty:
        return pd.DataFrame(columns=["regime", "start", "end", "days", "total_return", "annual_return", "max_drawdown"])
    aligned_regimes = regimes_for_dates(regimes, returns.index)

    rows: list[dict[str, object]] = []
    group_id = (aligned_regimes != aligned_regimes.shift()).cumsum()
    for _group, state_series in aligned_regimes.groupby(group_id):
        segment_returns = pd.to_numeric(returns.reindex(state_series.index), errors="coerce").dropna()
        if segment_returns.empty:
            continue
        total_return = float((1.0 + segment_returns).prod() - 1.0)
        periods = max(len(segment_returns), 1)
        annual_return = float((1 + total_return) ** (annual_days / periods) - 1) if total_return > -1 else -1.0
        wealth = pd.concat(
            [
                pd.Series([1.0], index=[segment_returns.index[0] - pd.Timedelta(nanoseconds=1)]),
                (1.0 + segment_returns).cumprod(),
            ]
        )
        drawdown = wealth / wealth.cummax() - 1
        rows.append(
            {
                "regime": normalize_regime(str(state_series.iloc[0])),
                "start": segment_returns.index.min().date().isoformat(),
                "end": segment_returns.index.max().date().isoformat(),
                "days": int(len(segment_returns)),
                "total_return": total_return,
                "annual_return": annual_return,
                "max_drawdown": float(drawdown.min()),
            }
        )
    return pd.DataFrame(rows)


def aggregate_regime_performance(regime_stats: pd.DataFrame) -> pd.DataFrame:
    if regime_stats.empty:
        return pd.DataFrame(columns=["regime", "segments", "days", "total_return", "weighted_annual_return", "worst_drawdown"])
    rows: list[dict[str, object]] = []
    for regime, frame in regime_stats.groupby("regime", sort=False):
        days = pd.to_numeric(frame["days"], errors="coerce").fillna(0.0)
        total_returns = pd.to_numeric(frame["total_return"], errors="coerce").fillna(0.0)
        annual_returns = pd.to_numeric(frame["annual_return"], errors="coerce").fillna(0.0)
        weighted_annual = float((annual_returns * days).sum() / days.sum()) if float(days.sum()) > 0 else 0.0
        compounded = float((1.0 + total_returns).prod() - 1.0)
        rows.append(
            {
                "regime": regime,
                "segments": int(len(frame)),
                "days": int(days.sum()),
                "total_return": compounded,
                "weighted_annual_return": weighted_annual,
                "worst_drawdown": float(pd.to_numeric(frame["max_drawdown"], errors="coerce").min()),
            }
        )
    return pd.DataFrame(rows)


def _high_volatility_mask(volatility: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    fixed_threshold = cfg.get("high_volatility_threshold")
    if fixed_threshold is not None:
        return volatility >= float(fixed_threshold)
    quantile_window = int(cfg.get("volatility_quantile_window", 252))
    quantile = float(cfg.get("high_volatility_quantile", 0.75))
    threshold = volatility.rolling(quantile_window, min_periods=max(1, int(cfg.get("min_periods", 20)))).quantile(quantile)
    return volatility >= threshold


def _drawdown_bear_mask(benchmark: pd.Series, cfg: dict[str, Any], min_periods: int) -> pd.Series:
    threshold = cfg.get("bear_drawdown_threshold")
    if threshold is None:
        return pd.Series(False, index=benchmark.index)
    threshold_value = abs(float(threshold))
    if threshold_value <= 0:
        return pd.Series(False, index=benchmark.index)
    window = int(cfg.get("drawdown_window", 252) or 0)
    if window > 0:
        peak = benchmark.rolling(window, min_periods=min(max(1, min_periods), window)).max()
    else:
        peak = benchmark.cummax()
    drawdown = benchmark / peak - 1.0
    return (drawdown <= -threshold_value).fillna(False)


def _benchmark_close(price_df: pd.DataFrame, config: dict[str, Any], cfg: dict[str, Any] | None = None) -> pd.Series:
    cfg = cfg or config.get("market_regime", {})
    benchmark_file = cfg.get("benchmark_file")
    if benchmark_file:
        benchmark = _load_benchmark_file(benchmark_file)
        if not benchmark.empty:
            return benchmark

    close = _close_frame(price_df)
    if close.empty:
        return pd.Series(dtype=float)

    symbol = cfg.get("benchmark_symbol")
    if symbol:
        matched = _match_column(close.columns, str(symbol))
        if matched is not None:
            return close[matched].rename("benchmark_close")

    hs300_symbols = _load_hs300_symbols(cfg.get("hs300_constituents_file") or config.get("data", {}).get("hs300_constituents_file"))
    hs300_columns = [column for column in close.columns if _normalize_symbol(str(column)) in hs300_symbols]
    if hs300_columns:
        return _equal_weight_proxy(close[hs300_columns])

    return _equal_weight_proxy(close)


def _load_benchmark_file(path_value: str | Path) -> pd.Series:
    path = resolve_path(path_value)
    if not path.exists():
        return pd.Series(dtype=float)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        frame = pd.read_parquet(path)
    lower_columns = {} if isinstance(frame, pd.Series) else {str(column).strip().lower(): column for column in frame.columns}
    if isinstance(frame, pd.Series):
        series = frame
    elif "close" in lower_columns:
        series = frame[lower_columns["close"]]
    else:
        series = frame.iloc[:, -1]
    if not isinstance(series.index, pd.DatetimeIndex) and lower_columns:
        if "date" in lower_columns:
            series.index = pd.to_datetime(frame[lower_columns["date"]])
        elif "trade_date" in lower_columns:
            series.index = pd.to_datetime(frame[lower_columns["trade_date"]])
    return pd.to_numeric(series, errors="coerce").dropna().rename("benchmark_close")


def _close_frame(price_df: pd.DataFrame) -> pd.DataFrame:
    if price_df.empty:
        return pd.DataFrame()
    if isinstance(price_df.columns, pd.MultiIndex):
        fields = price_df.columns.get_level_values(0).astype(str).str.strip().str.lower()
        if "close" not in set(fields):
            return pd.DataFrame(index=price_df.index)
        close = price_df.loc[:, fields == "close"].copy()
        close.columns = close.columns.get_level_values(-1).astype(str)
        return _normalize_close_frame(close)
    if _looks_like_field_table(price_df.columns):
        raise ValueError("Non-MultiIndex price_df must be a close-price panel with instrument columns.")
    return _normalize_close_frame(price_df.copy())


def _normalize_close_frame(close: pd.DataFrame) -> pd.DataFrame:
    if close.empty:
        return close
    raw_dates = pd.DatetimeIndex(pd.to_datetime(close.index, errors="coerce"))
    valid_dates = ~pd.isna(raw_dates)
    if not valid_dates.all():
        close = close.loc[valid_dates].copy()
        raw_dates = raw_dates[valid_dates]
    if close.empty:
        return close

    order = np.argsort(raw_dates.to_numpy(), kind="mergesort")
    if not np.array_equal(order, np.arange(len(raw_dates))):
        close = close.iloc[order].copy()
        raw_dates = raw_dates[order]
    close.index = raw_dates.normalize()
    close.columns = [_normalize_instrument(value) for value in close.columns]
    close = close.loc[:, close.columns != ""]
    if close.columns.has_duplicates:
        close = close.loc[:, ~close.columns.duplicated(keep="last")]
    if close.index.has_duplicates:
        close = close.loc[~close.index.duplicated(keep="last")]
    return close.sort_index()


def _looks_like_field_table(columns: pd.Index) -> bool:
    labels = {str(column).strip().lower() for column in columns}
    return len(labels) > 1 and bool(labels & PRICE_FIELD_COLUMNS)


def _load_hs300_symbols(path_value: str | Path | None) -> set[str]:
    if not path_value:
        return set()
    path = resolve_path(path_value)
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return set()
    for column in ("ts_code", "con_code", "instrument", "symbol"):
        if column in frame.columns:
            return {_normalize_symbol(value) for value in frame[column].dropna().astype(str)}
    return set()


def _equal_weight_proxy(close: pd.DataFrame) -> pd.Series:
    numeric = close.apply(pd.to_numeric, errors="coerce")
    returns = numeric.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    proxy_returns = returns.mean(axis=1, skipna=True).fillna(0.0)
    proxy = (1.0 + proxy_returns).cumprod()
    proxy.index = pd.to_datetime(proxy.index).normalize()
    return proxy.rename("benchmark_close")


def _match_column(columns: pd.Index, symbol: str) -> str | None:
    target = _normalize_symbol(symbol)
    for column in columns:
        if _normalize_symbol(str(column)) == target:
            return str(column)
    return None


def _normalize_instrument(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def _normalize_symbol(value: str) -> str:
    return str(value).strip().replace("_", ".").lower()
