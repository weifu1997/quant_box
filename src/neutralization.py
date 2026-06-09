"""模块说明：对信号分数做行业和市值中性化。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common import normalize_instrument as _normalize_instrument, parse_datetime_values as _parse_datetime_values
from src.config_loader import resolve_path


def load_industry_map(path: str | Path) -> pd.Series:
    """函数说明：加载 load_industry_map 主要逻辑。"""
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
    """函数说明：加载 load_daily_basic 主要逻辑。"""
    file_path = resolve_path(path)
    if not file_path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(file_path)
    if frame.empty or "ts_code" not in frame.columns or "trade_date" not in frame.columns:
        return pd.DataFrame()
    frame = frame.copy()
    raw_trade_dates = _parse_datetime_values(frame["trade_date"])
    frame["_raw_trade_date"] = raw_trade_dates
    frame["trade_date"] = raw_trade_dates.dt.normalize()
    frame["ts_code"] = frame["ts_code"].map(_normalize_instrument)
    frame = frame.dropna(subset=["trade_date", "ts_code"])
    frame = frame[frame["ts_code"] != ""]
    frame["_position"] = range(len(frame))
    frame = frame.sort_values(["trade_date", "ts_code", "_raw_trade_date", "_position"], kind="mergesort")
    frame = frame.drop_duplicates(["trade_date", "ts_code"], keep="last")
    return frame.drop(columns=["_raw_trade_date", "_position"]).set_index(["trade_date", "ts_code"]).sort_index()


def neutralize_score_panel(
    scores: pd.Series,
    industry_map: pd.Series | None = None,
    daily_basic: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """函数说明：处理 neutralize_score_panel 主要逻辑。"""
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
    normalized_scores = _normalize_score_index(scores)
    for date, daily_scores in normalized_scores.groupby(level=0, sort=True):
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


def _normalize_score_index(scores: pd.Series) -> pd.Series:
    """函数说明：规范化 normalize_score_index 的内部辅助逻辑。"""
    raw_dates = _parse_datetime_values(scores.index.get_level_values(0))
    raw_instruments = scores.index.get_level_values(1)
    values = pd.to_numeric(pd.Series(scores.to_numpy()), errors="coerce").to_numpy()
    frame = pd.DataFrame(
        {
            "date": raw_dates.dt.normalize(),
            "raw_date": raw_dates,
            "instrument": [_normalize_instrument(value) for value in raw_instruments],
            "label": list(raw_instruments),
            "score": values,
            "position": range(len(scores)),
        }
    )
    frame = frame[frame["date"].notna() & (frame["instrument"] != "")]
    if frame.empty:
        return pd.Series(dtype=float, name=scores.name)
    frame = frame.sort_values(
        ["date", "instrument", "raw_date", "score", "position"],
        kind="mergesort",
        na_position="first",
    ).drop_duplicates(["date", "instrument"], keep="last")
    index = pd.MultiIndex.from_arrays([frame["date"], frame["label"]], names=scores.index.names)
    result = pd.Series(frame["score"].to_numpy(), index=index, name=scores.name)
    return result.sort_index()


def _market_cap_residual(scores: pd.Series, market_cap: pd.Series, min_obs: int) -> pd.Series | None:
    """函数说明：处理 market_cap_residual 的内部辅助逻辑。"""
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
