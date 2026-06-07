from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config_loader import resolve_path


def load_industry_map(path: str | Path) -> pd.Series:
    file_path = resolve_path(path)
    if not file_path.exists():
        return pd.Series(dtype=object, name="industry")
    frame = pd.read_csv(file_path)
    if "ts_code" not in frame.columns or "industry" not in frame.columns:
        return pd.Series(dtype=object, name="industry")
    clean = frame.copy()
    clean["ts_code"] = clean["ts_code"].map(_normalize_instrument)
    clean = clean[clean["ts_code"] != ""]
    clean = clean.drop_duplicates("ts_code", keep="last")
    industry = clean.set_index("ts_code")["industry"].fillna("UNKNOWN").astype(str)
    return industry.rename("industry")


def load_daily_basic(path: str | Path) -> pd.DataFrame:
    file_path = resolve_path(path)
    if not file_path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(file_path)
    if frame.empty or "ts_code" not in frame.columns or "trade_date" not in frame.columns:
        return pd.DataFrame()
    frame = frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame["ts_code"] = frame["ts_code"].map(_normalize_instrument)
    frame = frame.dropna(subset=["trade_date", "ts_code"])
    frame = frame[frame["ts_code"] != ""]
    frame = frame.drop_duplicates(["trade_date", "ts_code"], keep="last")
    return frame.set_index(["trade_date", "ts_code"]).sort_index()


def neutralize_score_panel(
    scores: pd.Series,
    industry_map: pd.Series | None = None,
    daily_basic: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    cfg = config or {}
    enabled = bool(cfg.get("enabled", False))
    market_cap_field = str(cfg.get("market_cap_field", "circ_mv"))
    if scores.empty or not enabled:
        return scores, {
            "enabled": enabled,
            "dates_neutralized": 0,
            "industry_dates": 0,
            "market_cap_dates": 0,
            "market_cap_field": market_cap_field,
        }
    if not isinstance(scores.index, pd.MultiIndex):
        raise ValueError("scores must use MultiIndex: datetime/instrument.")

    industry_enabled = bool(cfg.get("industry", True))
    market_cap_enabled = bool(cfg.get("market_cap", True))
    min_obs = max(3, int(cfg.get("min_obs", 20)))
    industry_map = industry_map if industry_map is not None else pd.Series(dtype=object)
    daily_basic = daily_basic if daily_basic is not None else pd.DataFrame()

    parts: list[pd.Series] = []
    dates_neutralized = 0
    industry_dates = 0
    market_cap_dates = 0
    for date, daily_scores in scores.groupby(level=0, sort=True):
        daily = daily_scores.droplevel(0).astype(float).copy()
        neutralized = daily.copy()
        normalized_symbols = pd.Index([_normalize_instrument(value) for value in neutralized.index])
        if industry_enabled and not industry_map.empty:
            normalized_industry = industry_map.copy()
            normalized_industry.index = [_normalize_instrument(value) for value in normalized_industry.index]
            normalized_industry = normalized_industry[normalized_industry.index != ""]
            normalized_industry = normalized_industry[~normalized_industry.index.duplicated(keep="last")]
            groups = pd.Series(normalized_symbols, index=neutralized.index).map(normalized_industry).fillna("UNKNOWN")
            neutralized = neutralized - neutralized.groupby(groups).transform("mean")
            industry_dates += 1

        if market_cap_enabled and not daily_basic.empty and market_cap_field in daily_basic.columns:
            date_key = pd.Timestamp(date).normalize()
            try:
                basics = daily_basic.xs(date_key, level=0)
            except KeyError:
                basics = pd.DataFrame()
            if not basics.empty:
                basics = basics.copy()
                basics.index = [_normalize_instrument(value) for value in basics.index]
                basics = basics[basics.index != ""]
                basics = basics[~basics.index.duplicated(keep="last")]
                cap = pd.to_numeric(basics[market_cap_field].reindex(normalized_symbols), errors="coerce")
                cap.index = neutralized.index
                residual = _market_cap_residual(neutralized, cap, min_obs)
                if residual is not None:
                    neutralized = residual
                    market_cap_dates += 1

        if industry_enabled or market_cap_enabled:
            dates_neutralized += 1
        neutralized.index = pd.MultiIndex.from_product([[pd.Timestamp(date).normalize()], neutralized.index], names=scores.index.names)
        neutralized.attrs = {}
        parts.append(neutralized)

    result = pd.concat(parts).sort_index().rename(scores.name or "score") if parts else scores.copy()
    result.attrs = {}
    return result, {
        "enabled": True,
        "dates_neutralized": dates_neutralized,
        "industry_dates": industry_dates,
        "market_cap_dates": market_cap_dates,
        "market_cap_field": market_cap_field,
    }


def _normalize_instrument(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def _market_cap_residual(scores: pd.Series, market_cap: pd.Series, min_obs: int) -> pd.Series | None:
    cap = np.log1p(pd.to_numeric(market_cap, errors="coerce").astype(float))
    y = scores.astype(float)
    valid = y.notna() & cap.notna() & np.isfinite(y) & np.isfinite(cap)
    if int(valid.sum()) < min_obs:
        return None
    x_values = cap.loc[valid].to_numpy(dtype=float)
    if float(np.nanstd(x_values)) <= 1e-12:
        return None
    X = np.column_stack([np.ones(len(x_values)), x_values])
    beta = np.linalg.lstsq(X, y.loc[valid].to_numpy(dtype=float), rcond=None)[0]
    fitted = X @ beta
    residual = y.copy()
    residual.loc[valid] = y.loc[valid].to_numpy(dtype=float) - fitted
    return residual
