from __future__ import annotations

from typing import Any

import pandas as pd

from src.market_regime import REGIME_BEAR, REGIME_BULL, REGIME_SIDEWAYS, normalize_regime, regime_for_date


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
        daily = _normalize_daily_scores(daily_scores)
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


def apply_regime_score_filter(
    scores: pd.Series,
    factors: pd.DataFrame,
    regimes: pd.Series,
    config: dict[str, Any] | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    cfg = config or {}
    if scores.empty or factors.empty or regimes.empty or not bool(cfg.get("enabled", False)):
        return scores, {"enabled": bool(cfg.get("enabled", False)), "dates_filtered": 0}
    if not isinstance(scores.index, pd.MultiIndex) or not isinstance(factors.index, pd.MultiIndex):
        raise ValueError("scores and factors must use MultiIndex: datetime/instrument.")

    rules = _filter_rules(cfg)
    if not rules:
        return scores, {"enabled": True, "dates_filtered": 0, "rules": 0}

    normalized_factors = _normalize_factor_index(factors)
    parts: list[pd.Series] = []
    dates_filtered = 0
    rows_before = 0
    rows_after = 0
    for date, daily_scores in scores.groupby(level=0, sort=True):
        date_key = pd.Timestamp(date).normalize()
        daily = _normalize_daily_scores(daily_scores)
        rule = _rule_for_regime(rules, regime_for_date(regimes, date_key))
        if rule is None:
            parts.append(daily_scores)
            continue
        try:
            daily_factors = normalized_factors.xs(date_key, level=0)
        except KeyError:
            parts.append(daily_scores)
            continue
        filter_score = _defensive_score(daily_factors, rule["components"])
        if filter_score.empty:
            parts.append(daily_scores)
            continue

        threshold = _filter_threshold(filter_score, rule)
        mask = filter_score.reindex(daily.index) >= threshold
        filtered = daily.where(mask.fillna(False))
        filtered.index = pd.MultiIndex.from_product([[date_key], filtered.index.astype(str)], names=scores.index.names)
        parts.append(filtered.rename(scores.name or "score"))
        dates_filtered += 1
        rows_before += int(daily.notna().sum())
        rows_after += int(filtered.notna().sum())

    result = pd.concat(parts).sort_index().rename(scores.name or "score") if parts else scores.copy()
    result.attrs = dict(scores.attrs)
    return result, {
        "enabled": True,
        "dates_filtered": dates_filtered,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": max(rows_before - rows_after, 0),
        "rules": len(rules),
    }


def _filter_rules(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    default_components = cfg.get("components") or cfg.get("defensive_components") or [
        {"column": "ROC20", "direction": 1.0},
        {"column": "STD20", "direction": -1.0},
    ]
    rules: list[dict[str, Any]] = []
    for item in cfg.get("rules", []):
        regime = normalize_regime(str(item.get("regime", REGIME_BEAR)))
        rules.append(
            {
                "regime": regime,
                "components": item.get("components") or default_components,
                "min_score": float(item.get("min_score", cfg.get("min_score", 0.0))),
                "keep_top_fraction": item.get("keep_top_fraction", cfg.get("keep_top_fraction")),
            }
        )
    if not rules and "bear_min_score" in cfg:
        rules.append(
            {
                "regime": REGIME_BEAR,
                "components": default_components,
                "min_score": float(cfg.get("bear_min_score", 0.0)),
                "keep_top_fraction": cfg.get("bear_keep_top_fraction", cfg.get("keep_top_fraction")),
            }
        )
    return rules


def _rule_for_regime(rules: list[dict[str, Any]], regime: str) -> dict[str, Any] | None:
    normalized = normalize_regime(regime)
    for rule in rules:
        if rule["regime"] == normalized:
            return rule
    return None


def _filter_threshold(filter_score: pd.Series, rule: dict[str, Any]) -> float:
    keep_top_fraction = rule.get("keep_top_fraction")
    min_score = float(rule.get("min_score", 0.0))
    if keep_top_fraction is None:
        return min_score
    fraction = max(0.0, min(float(keep_top_fraction), 1.0))
    if fraction <= 0:
        return float("inf")
    quantile_threshold = float(filter_score.dropna().quantile(max(0.0, 1.0 - fraction)))
    return max(min_score, quantile_threshold)


def _normalize_factor_index(factors: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(factors.index.get_level_values(0)).normalize()
    instruments = [_normalize_instrument(value) for value in factors.index.get_level_values(1)]
    normalized = factors.copy(deep=False)
    normalized.index = pd.MultiIndex.from_arrays([dates, instruments], names=["datetime", "instrument"])
    normalized = normalized[normalized.index.get_level_values("instrument") != ""]
    if normalized.index.has_duplicates:
        normalized = normalized.groupby(level=["datetime", "instrument"], sort=False).last()
    return normalized.sort_index()


def _normalize_daily_scores(daily_scores: pd.Series) -> pd.Series:
    daily = daily_scores.droplevel(0).astype(float).sort_values(ascending=False, kind="mergesort", na_position="last").copy()
    daily.index = [_normalize_instrument(value) for value in daily.index]
    daily = daily[daily.index != ""]
    if daily.index.has_duplicates:
        daily = daily[~daily.index.duplicated(keep="first")]
    return daily


def _normalize_instrument(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


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
