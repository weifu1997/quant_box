"""模块说明：处理回测仓位比例和择时调仓规则。"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def _exposure_scale(equity_rows: list[tuple[pd.Timestamp, float]], config: dict) -> float:
    """函数说明：处理 exposure_scale 的内部辅助逻辑。"""
    target_vol = config.get("target_vol")
    overlay_scale = _equity_overlay_exposure(equity_rows, config)
    if target_vol is None:
        return overlay_scale
    window = int(config.get("vol_window", 60))
    max_leverage = float(config.get("max_leverage", 1.0))
    if len(equity_rows) <= window:
        return min(overlay_scale, max_leverage)
    equity = pd.Series(dict(equity_rows)).sort_index()
    realized_vol = equity.pct_change().dropna().tail(window).std(ddof=1) * np.sqrt(252)
    if not realized_vol or pd.isna(realized_vol):
        return min(overlay_scale, max_leverage)
    vol_scale = max(0.0, min(float(target_vol) / float(realized_vol), max_leverage))
    return max(0.0, min(overlay_scale * vol_scale, max_leverage))


def _equity_overlay_rebalance_needed(equity_rows: list[tuple[pd.Timestamp, float]], config: dict) -> bool:
    """函数说明：处理 equity_overlay_rebalance_needed 的内部辅助逻辑。"""
    if not _equity_overlay_enabled(config) or len(equity_rows) < 2:
        return False
    cfg = config.get("equity_overlay", {})
    if bool(cfg.get("rebalance_on_signal_only", False)):
        return False
    threshold = float(cfg.get("rebalance_threshold", 0.05))
    current = _equity_overlay_exposure(equity_rows, config)
    previous = _equity_overlay_exposure(equity_rows[:-1], config)
    return abs(current - previous) >= max(threshold, 0.0)


def _equity_overlay_exposure(equity_rows: list[tuple[pd.Timestamp, float]], config: dict) -> float:
    """函数说明：处理 equity_overlay_exposure 的内部辅助逻辑。"""
    if not _equity_overlay_enabled(config) or len(equity_rows) < 2:
        return 1.0
    cfg = config.get("equity_overlay", {})
    equity = pd.Series(dict(equity_rows)).sort_index().astype(float)
    equity = equity[equity > 0]
    if len(equity) < 2:
        return 1.0

    current = float(equity.iloc[-1])
    min_periods = max(1, int(cfg.get("min_periods", 5)))
    ma_window = max(1, int(cfg.get("ma_window", 90)))
    momentum_window = max(1, int(cfg.get("momentum_window", 5)))
    drawdown_window = max(1, int(cfg.get("drawdown_window", 60)))
    drawdown_cut = abs(float(cfg.get("drawdown_cut", 0.20)))

    side_risk = False
    if len(equity) >= min(min_periods, ma_window):
        moving_average = float(equity.tail(ma_window).mean())
        side_risk = side_risk or current < moving_average
    if len(equity) > momentum_window:
        reference = float(equity.iloc[-momentum_window - 1])
        if reference > 0:
            side_risk = side_risk or current / reference - 1.0 < 0.0

    bear_risk = False
    if len(equity) >= min(min_periods, drawdown_window):
        recent_peak = float(equity.tail(drawdown_window).max())
        if recent_peak > 0:
            bear_risk = current / recent_peak - 1.0 <= -drawdown_cut

    if bear_risk:
        exposure = float(cfg.get("bear_exposure", 0.0))
    elif side_risk:
        exposure = float(cfg.get("sideways_exposure", 0.2))
    else:
        exposure = float(cfg.get("bull_exposure", 1.0))
    max_exposure = float(cfg.get("max_exposure", 1.0))
    return max(0.0, min(exposure, max_exposure))


def _equity_overlay_enabled(config: dict) -> bool:
    """函数说明：处理 equity_overlay_enabled 的内部辅助逻辑。"""
    overlay = config.get("equity_overlay", {})
    return isinstance(overlay, dict) and bool(overlay.get("enabled", False))


def _normalize_exposure_schedule(schedule: object, price_dates: pd.Index) -> pd.Series | None:
    """函数说明：规范化 normalize_exposure_schedule 的内部辅助逻辑。"""
    if schedule is None:
        return None
    if isinstance(schedule, pd.Series):
        series = schedule.copy()
    elif isinstance(schedule, Mapping):
        series = pd.Series(schedule, dtype=float)
    else:
        try:
            series = pd.Series(schedule, dtype=float)
        except (TypeError, ValueError):
            return None
    if series.empty:
        return None

    raw_dates = pd.DatetimeIndex(pd.to_datetime(series.index, errors="coerce"))
    valid_dates = ~raw_dates.isna()
    series = pd.to_numeric(series.loc[valid_dates], errors="coerce")
    raw_dates = raw_dates[valid_dates]
    if not series.empty:
        order = np.argsort(raw_dates.to_numpy(), kind="mergesort")
        series = series.iloc[order].copy()
        raw_dates = raw_dates[order]
    series.index = raw_dates.normalize()
    series = series.dropna()
    series = series[~series.index.duplicated(keep="last")]
    series = series.sort_index()
    if series.empty:
        return None

    aligned_dates = pd.DatetimeIndex(pd.to_datetime(price_dates).normalize())
    aligned = series.reindex(aligned_dates, method="ffill").fillna(1.0)
    return aligned.clip(lower=0.0).rename("exposure_scale")


def _scheduled_exposure_scale(exposure_schedule: pd.Series | None, trade_date: pd.Timestamp) -> float:
    """函数说明：处理 scheduled_exposure_scale 的内部辅助逻辑。"""
    if exposure_schedule is None or exposure_schedule.empty:
        return 1.0
    date = pd.Timestamp(trade_date).normalize()
    if date not in exposure_schedule.index:
        eligible = exposure_schedule.loc[exposure_schedule.index <= date]
        return float(eligible.iloc[-1]) if not eligible.empty else 1.0
    return float(exposure_schedule.loc[date])


def _scheduled_exposure_rebalance_needed(
    exposure_schedule: pd.Series | None,
    trade_date: pd.Timestamp,
    previous_date: pd.Timestamp | None,
    config: dict,
) -> bool:
    """函数说明：处理 scheduled_exposure_rebalance_needed 的内部辅助逻辑。"""
    if exposure_schedule is None or previous_date is None:
        return False
    if bool(config.get("exposure_schedule_rebalance_on_signal_only", False)):
        return False
    threshold = float(config.get("exposure_rebalance_threshold", 0.05))
    current = _scheduled_exposure_scale(exposure_schedule, trade_date)
    previous = _scheduled_exposure_scale(exposure_schedule, previous_date)
    return abs(current - previous) >= max(threshold, 0.0)
