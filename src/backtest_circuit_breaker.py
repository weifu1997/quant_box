from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


def _circuit_breaker_target_exposure(config: dict) -> float:
    value = config.get("circuit_breaker_target_exposure")
    if value is None:
        return 0.0
    return max(0.0, min(float(value), 1.0))


def _optional_nonnegative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer or null.") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be greater than or equal to 0.")
    return parsed


def _drawdown_breached(total: float, peak_equity: float, config: dict) -> bool:
    threshold = config.get("circuit_breaker_drawdown")
    if threshold is None or peak_equity <= 0:
        return False
    drawdown = total / peak_equity - 1
    return drawdown <= -abs(float(threshold))


def _annual_drawdown_guard_active(
    total: float,
    year_peak_equity: float,
    trade_year: int,
    active_year: int | None,
    config: dict,
) -> bool:
    cfg = config.get("annual_drawdown_guard", {})
    if not isinstance(cfg, Mapping) or not bool(cfg.get("enabled", False)):
        return False
    if active_year == trade_year:
        return True
    threshold = cfg.get("drawdown")
    if threshold is None or year_peak_equity <= 0:
        return False
    drawdown = total / year_peak_equity - 1
    return drawdown <= -abs(float(threshold))


def _annual_drawdown_guard_released(
    total: float,
    year_peak_equity: float,
    trade_year: int,
    active_year: int | None,
    config: dict,
) -> bool:
    if active_year != trade_year or year_peak_equity <= 0:
        return False
    cfg = config.get("annual_drawdown_guard", {})
    if not isinstance(cfg, Mapping) or not bool(cfg.get("enabled", False)):
        return False
    release = cfg.get("release_drawdown")
    if release is None:
        return False
    drawdown = total / year_peak_equity - 1
    return drawdown >= -abs(float(release))


def _annual_drawdown_guard_target_exposure(config: dict) -> float:
    cfg = config.get("annual_drawdown_guard", {})
    if not isinstance(cfg, Mapping):
        return 0.0
    value = cfg.get("target_exposure", 0.0)
    return max(0.0, min(float(value), 1.0))


def _cooldown_until(price_dates: pd.Index, current_pos: int, cooldown_days: int) -> pd.Timestamp:
    if len(price_dates) == 0:
        return pd.NaT
    target_pos = min(max(current_pos + max(cooldown_days, 0), current_pos), len(price_dates) - 1)
    return pd.Timestamp(price_dates[target_pos])
