"""Factor diagnostic tables for IC stability and quantile return spreads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common import normalize_instrument
from src.config_loader import resolve_path
from src.factor_ic import calculate_factor_ic, make_forward_returns, summarize_ic


def build_factor_diagnostics(
    factor_df: pd.DataFrame,
    price_df: pd.DataFrame,
    *,
    horizon: int = 1,
    method: str = "spearman",
    min_obs: int = 20,
    quantiles: int = 5,
) -> dict[str, pd.DataFrame]:
    """Build IC, yearly IC, and factor quantile-return diagnostic tables."""
    daily_ic = calculate_factor_ic(factor_df, price_df, horizon=horizon, method=method, min_obs=min_obs)
    ic_summary = summarize_ic(daily_ic)
    ic_summary.index.name = "factor"
    return {
        "daily_ic": daily_ic,
        "ic_summary": ic_summary,
        "yearly_ic": build_yearly_ic_summary(daily_ic),
        "group_returns": build_factor_group_returns(
            factor_df,
            price_df,
            horizon=horizon,
            min_obs=min_obs,
            quantiles=quantiles,
        ),
    }


def build_yearly_ic_summary(daily_ic: pd.DataFrame) -> pd.DataFrame:
    """Summarize daily IC by calendar year and factor."""
    if daily_ic.empty:
        return pd.DataFrame(columns=["year", "factor", "mean_ic", "std_ic", "ic_ir", "positive_ratio", "count"])
    rows: list[pd.DataFrame] = []
    for year, group in daily_ic.groupby(pd.DatetimeIndex(daily_ic.index).year):
        summary = summarize_ic(group)
        summary.index.name = "factor"
        frame = summary.reset_index()
        frame.insert(0, "year", int(year))
        rows.append(frame)
    if not rows:
        return pd.DataFrame(columns=["year", "factor", "mean_ic", "std_ic", "ic_ir", "positive_ratio", "count"])
    return pd.concat(rows, ignore_index=True)


def build_factor_group_returns(
    factor_df: pd.DataFrame,
    price_df: pd.DataFrame,
    *,
    horizon: int = 1,
    min_obs: int = 20,
    quantiles: int = 5,
) -> pd.DataFrame:
    """Build per-date factor quantile forward-return spreads."""
    if not isinstance(factor_df.index, pd.MultiIndex):
        raise ValueError("factor_df must use MultiIndex: datetime/instrument.")
    factors = factor_df.select_dtypes("number")
    if factors.empty:
        return _empty_group_returns()
    returns = make_forward_returns(price_df, horizon=horizon)
    date_level = factors.index.names[0] or 0
    factor_dates = pd.DatetimeIndex(pd.to_datetime(factors.index.get_level_values(date_level), errors="coerce").normalize())
    return_dates = set(pd.to_datetime(returns.index.get_level_values(0), errors="coerce").normalize())

    rows: list[dict[str, Any]] = []
    for date_key in pd.DatetimeIndex(factor_dates.dropna().unique()).sort_values():
        if date_key not in return_dates:
            continue
        daily_factors = _daily_factor_frame(factors[factor_dates == date_key], date_level)
        if daily_factors.empty:
            continue
        try:
            daily_returns = _instrument_series(returns.xs(date_key, level=0, drop_level=True))
        except KeyError:
            continue
        aligned = daily_factors.join(daily_returns.rename("forward_return"), how="inner").dropna(subset=["forward_return"])
        if aligned.empty:
            continue
        for factor in factors.columns:
            rows.extend(_factor_quantile_rows(date_key, str(factor), aligned[[factor, "forward_return"]], quantiles, min_obs))
    return pd.DataFrame(
        rows,
        columns=[
            "datetime",
            "factor",
            "quantile",
            "mean_forward_return",
            "median_forward_return",
            "instrument_count",
            "top_minus_bottom",
        ],
    )


def write_factor_diagnostics(tables: dict[str, pd.DataFrame], out_dir: str | Path = "outputs") -> dict[str, str]:
    """Persist factor diagnostic tables under the configured output directory."""
    output_dir = resolve_path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "factor_daily_ic": output_dir / "factor_daily_ic.csv",
        "factor_ic_summary": output_dir / "factor_ic_summary.csv",
        "factor_ic_yearly": output_dir / "factor_ic_yearly.csv",
        "factor_group_returns": output_dir / "factor_group_returns.csv",
    }
    tables["daily_ic"].to_csv(paths["factor_daily_ic"], encoding="utf-8-sig")
    tables["ic_summary"].to_csv(paths["factor_ic_summary"], encoding="utf-8-sig")
    tables["yearly_ic"].to_csv(paths["factor_ic_yearly"], index=False, encoding="utf-8-sig")
    tables["group_returns"].to_csv(paths["factor_group_returns"], index=False, encoding="utf-8-sig")
    return {key: str(path) for key, path in paths.items()}


def _factor_quantile_rows(
    date_key: pd.Timestamp,
    factor: str,
    frame: pd.DataFrame,
    quantiles: int,
    min_obs: int,
) -> list[dict[str, Any]]:
    clean = frame.rename(columns={factor: "factor_value"}).dropna(subset=["factor_value", "forward_return"])
    if len(clean) < min_obs or clean["factor_value"].nunique(dropna=True) < 2:
        return []
    bucket_count = max(2, min(int(quantiles), int(clean["factor_value"].nunique(dropna=True)), len(clean)))
    try:
        buckets = pd.qcut(clean["factor_value"], q=bucket_count, labels=False, duplicates="drop")
    except ValueError:
        return []
    clean = clean.assign(_quantile=pd.to_numeric(buckets, errors="coerce"))
    clean = clean.dropna(subset=["_quantile"])
    if clean.empty:
        return []
    grouped = clean.groupby("_quantile")["forward_return"]
    means = grouped.mean()
    if means.empty:
        return []
    top_minus_bottom = float(means.iloc[-1] - means.iloc[0])
    rows: list[dict[str, Any]] = []
    for quantile, values in grouped:
        rows.append(
            {
                "datetime": pd.Timestamp(date_key).date().isoformat(),
                "factor": factor,
                "quantile": int(quantile) + 1,
                "mean_forward_return": float(values.mean()),
                "median_forward_return": float(values.median()),
                "instrument_count": int(values.count()),
                "top_minus_bottom": top_minus_bottom,
            }
        )
    return rows


def _daily_factor_frame(frame: pd.DataFrame, date_level: str | int) -> pd.DataFrame:
    if frame.empty:
        result = frame.droplevel(date_level)
        result.index = pd.Index([], name="instrument")
        return result
    date_position = frame.index.names.index(date_level) if isinstance(date_level, str) else int(date_level)
    instrument_level = 1 - date_position
    instruments = [normalize_instrument(value) for value in frame.index.get_level_values(instrument_level)]
    result = frame.copy()
    result.index = pd.Index(instruments, name="instrument")
    result = result[result.index != ""]
    if result.index.has_duplicates:
        result = result[~result.index.duplicated(keep="last")]
    return result


def _instrument_series(series: pd.Series) -> pd.Series:
    result = series.copy()
    result.index = pd.Index([normalize_instrument(value) for value in result.index], name="instrument")
    result = result[result.index != ""]
    if result.index.has_duplicates:
        result = result[~result.index.duplicated(keep="last")]
    return result


def _empty_group_returns() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "datetime",
            "factor",
            "quantile",
            "mean_forward_return",
            "median_forward_return",
            "instrument_count",
            "top_minus_bottom",
        ]
    )
