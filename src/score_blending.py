from __future__ import annotations

from typing import Any

import pandas as pd

from src.market_regime import REGIME_BEAR, REGIME_BULL, REGIME_SIDEWAYS, regime_for_date


def apply_regime_score_blend(
    scores: pd.Series,
    factors: pd.DataFrame,
    regimes: pd.Series,
    config: dict[str, Any] | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    cfg = config or {}
    if scores.empty or factors.empty or regimes.empty or not bool(cfg.get("enabled", False)):
        return scores, {"enabled": bool(cfg.get("enabled", False)), "dates_blended": 0}
    if not isinstance(scores.index, pd.MultiIndex) or not isinstance(factors.index, pd.MultiIndex):
        raise ValueError("scores and factors must use MultiIndex: datetime/instrument.")

    components = cfg.get("defensive_components") or [
        {"column": "STD20", "direction": -1.0},
        {"column": "BETA20", "direction": -1.0},
        {"column": "ROC20", "direction": 1.0},
    ]
    weights_by_regime = {
        REGIME_BULL: float(cfg.get("bull_defensive_weight", 0.0)),
        REGIME_SIDEWAYS: float(cfg.get("sideways_defensive_weight", 0.5)),
        REGIME_BEAR: float(cfg.get("bear_defensive_weight", 1.0)),
    }

    normalized_factors = _normalize_factor_index(factors)
    parts: list[pd.Series] = []
    dates_blended = 0
    component_hits = 0
    for date, daily_scores in scores.groupby(level=0, sort=True):
        date_key = pd.Timestamp(date).normalize()
        daily = daily_scores.droplevel(0).astype(float)
        state = regime_for_date(regimes, date_key)
        defensive_weight = max(0.0, min(weights_by_regime.get(state, 0.0), 1.0))
        if defensive_weight <= 0:
            parts.append(daily_scores)
            continue
        try:
            daily_factors = normalized_factors.xs(date_key, level=0)
        except KeyError:
            parts.append(daily_scores)
            continue
        defensive = _defensive_score(daily_factors, components)
        if defensive.empty:
            parts.append(daily_scores)
            continue
        ml_rank = _cross_sectional_rank_score(daily)
        blended = (1.0 - defensive_weight) * ml_rank + defensive_weight * defensive.reindex(ml_rank.index)
        blended.index = pd.MultiIndex.from_product([[date_key], blended.index.astype(str)], names=scores.index.names)
        parts.append(blended.rename(scores.name or "score"))
        dates_blended += 1
        component_hits += len(defensive.attrs.get("components", []))

    result = pd.concat(parts).sort_index().rename(scores.name or "score") if parts else scores.copy()
    result.attrs = dict(scores.attrs)
    return result, {
        "enabled": True,
        "dates_blended": dates_blended,
        "average_components": float(component_hits / dates_blended) if dates_blended else 0.0,
        "bear_defensive_weight": weights_by_regime[REGIME_BEAR],
        "sideways_defensive_weight": weights_by_regime[REGIME_SIDEWAYS],
        "bull_defensive_weight": weights_by_regime[REGIME_BULL],
    }


def _normalize_factor_index(factors: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(factors.index.get_level_values(0)).normalize()
    instruments = factors.index.get_level_values(1).astype(str)
    normalized = factors.copy(deep=False)
    normalized.index = pd.MultiIndex.from_arrays([dates, instruments], names=["datetime", "instrument"])
    return normalized.sort_index()


def _defensive_score(factors: pd.DataFrame, components: list[dict[str, object]]) -> pd.Series:
    pieces: list[pd.Series] = []
    used: list[str] = []
    for item in components:
        column = str(item.get("column", ""))
        if column not in factors.columns:
            continue
        direction = float(item.get("direction", 1.0))
        pieces.append(_cross_sectional_rank_score(pd.to_numeric(factors[column], errors="coerce")) * direction)
        used.append(column)
    if not pieces:
        return pd.Series(dtype=float, name="defensive_score")
    score = pd.concat(pieces, axis=1).mean(axis=1, skipna=True).rename("defensive_score")
    score.attrs["components"] = used
    return score


def _cross_sectional_rank_score(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    return clean.rank(pct=True, method="average").sub(0.5).mul(2.0)
