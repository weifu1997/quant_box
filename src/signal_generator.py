from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config_loader import load_config, resolve_path
from src.factor_calculator import load_or_compute_factors
from src.scoring import build_latest_strategy_scores
from src.selection_constraints import load_industry_group_map
from src.selection_risk import filter_scores_by_selection_risk, selection_risk_filter_enabled
from src.strategy import select_stocks
from src.trading_calendar import resolve_target_date_value

logger = logging.getLogger(__name__)


def read_previous_holdings(path: str | Path | None = None) -> list[str]:
    config = load_config()
    holdings_path = resolve_path(path or config["outputs"]["holdings_file"])
    if not holdings_path.exists():
        return []
    df = pd.read_csv(holdings_path)
    col = "instrument" if "instrument" in df.columns else "ticker"
    if col not in df.columns:
        return []
    return _normalize_instruments(df[col].dropna().tolist())


def generate_signal(
    signal_date: str,
    previous_holdings: list[str] | None = None,
    factor_file: str | Path | None = None,
    config: dict | None = None,
    factors: pd.DataFrame | None = None,
    price_df: pd.DataFrame | None = None,
    price_file: str | Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    config = config or load_config()
    data_cfg = config["data"]
    strategy_cfg = config["strategy"]
    use_latest_date = str(signal_date).lower() == "latest"
    factor_end_date = resolve_target_date_value(
        data_cfg["end_date"] if use_latest_date else signal_date,
        config=config,
    )

    if factors is None:
        factors = load_or_compute_factors(
            start_date=data_cfg["start_date"],
            end_date=factor_end_date,
            cache_file=factor_file or config["factors"]["cache_file"],
        )
    score_date = "latest" if use_latest_date else _effective_signal_date(factors, factor_end_date)
    scores = build_latest_strategy_scores(factors, config, signal_date=score_date, price_df=price_df, price_file=price_file)
    latest_date = pd.Timestamp(scores.index.get_level_values(0).max()).normalize()
    if use_latest_date:
        signal_date = latest_date.strftime("%Y-%m-%d")
    else:
        signal_date = str(pd.Timestamp(score_date).date())
    latest_scores = scores.xs(latest_date, level=0, drop_level=True)
    if selection_risk_filter_enabled(config):
        prices = price_df if price_df is not None else _load_price_frame(price_file, config)
        latest_scores = filter_scores_by_selection_risk(latest_scores, prices, latest_date, config)
    latest_scores = _normalize_score_index(latest_scores)
    previous_holdings = _normalize_instruments(previous_holdings if previous_holdings is not None else read_previous_holdings())
    max_industry_weight = strategy_cfg.get("max_industry_weight")
    industry_map = load_industry_group_map(config) if max_industry_weight is not None else None
    holdings = select_stocks(
        latest_scores,
        top_n=int(strategy_cfg.get("top_n", 7)),
        previous_holdings=previous_holdings or None,
        max_turnover=int(strategy_cfg.get("max_turnover", 1)),
        rank_buffer=int(strategy_cfg.get("rank_buffer", 0)),
        group_map=industry_map,
        max_group_weight=max_industry_weight,
    )

    old_set = set(previous_holdings or [])
    new_set = set(holdings)
    rows = []
    for code in holdings:
        rows.append({"date": signal_date, "instrument": code, "action": "HOLD" if code in old_set else "BUY"})
    for code in sorted(old_set - new_set):
        rows.append({"date": signal_date, "instrument": code, "action": "SELL"})
    signal_df = pd.DataFrame(rows, columns=["date", "instrument", "action"])
    signal_df.attrs["signal_date"] = signal_date
    return signal_df, holdings


def _load_price_frame(price_file: str | Path | None, config: dict) -> pd.DataFrame:
    price_path = resolve_path(price_file or config.get("ic", {}).get("price_file", "data/prices/ohlcv_adjusted.parquet"))
    if not price_path.exists():
        raise FileNotFoundError(f"Price file not found for selection risk filter: {price_path}")
    return pd.read_parquet(price_path)


def _effective_signal_date(factors: pd.DataFrame, requested_date: str) -> str:
    requested_ts = pd.Timestamp(requested_date).normalize()
    dates = _factor_dates(factors)
    eligible = dates[dates <= requested_ts]
    if eligible.empty:
        raise ValueError(f"No factor cache date is available on or before requested signal_date {requested_ts.date()}.")
    effective = pd.Timestamp(eligible.max()).normalize()
    if effective != requested_ts:
        logger.warning(
            "Falling back signal date from %s to %s because factor cache has no rows for the requested date.",
            requested_ts.date(),
            effective.date(),
        )
    return str(effective.date())


def _factor_dates(factors: pd.DataFrame) -> pd.DatetimeIndex:
    if factors.empty or not isinstance(factors.index, pd.MultiIndex):
        raise ValueError("factors must use MultiIndex: date/instrument.")
    date_level = factors.index.names[0] or 0
    return pd.DatetimeIndex(pd.to_datetime(factors.index.get_level_values(date_level)).normalize()).unique().sort_values()


def _normalize_score_index(scores: pd.Series) -> pd.Series:
    if scores.empty:
        return scores
    result = scores.sort_values(ascending=False, kind="mergesort", na_position="last").copy()
    result.index = pd.Index([_normalize_instrument(value) for value in result.index], name=result.index.name)
    result = result[result.index != ""]
    if result.index.has_duplicates:
        result = result[~result.index.duplicated(keep="first")]
    result.attrs = dict(getattr(scores, "attrs", {}))
    return result


def _normalize_instrument(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def _normalize_instruments(values: list[str] | pd.Series) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        instrument = _normalize_instrument(value)
        if not instrument or instrument in seen:
            continue
        result.append(instrument)
        seen.add(instrument)
    return result


def save_signal(signal_df: pd.DataFrame, holdings: list[str], signal_date: str, config: dict | None = None) -> tuple[Path, Path]:
    config = config or load_config()
    out_dir = resolve_path(config["outputs"].get("dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    signal_path = out_dir / f"signal_{signal_date}.csv"
    holdings_path = resolve_path(config["outputs"]["holdings_file"])
    signal_df.to_csv(signal_path, index=False, encoding="utf-8-sig")
    pd.DataFrame({"instrument": holdings}).to_csv(holdings_path, index=False, encoding="utf-8-sig")
    return signal_path, holdings_path


def save_candidate_signal(
    signal_df: pd.DataFrame,
    holdings: list[str],
    signal_date: str,
    config: dict | None = None,
) -> tuple[Path, Path]:
    config = config or load_config()
    out_dir = resolve_path(config["outputs"].get("dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    signal_path = out_dir / f"candidate_signal_{signal_date}.csv"
    holdings_path = out_dir / f"candidate_holdings_{signal_date}.csv"
    signal_df.to_csv(signal_path, index=False, encoding="utf-8-sig")
    pd.DataFrame({"instrument": holdings}).to_csv(holdings_path, index=False, encoding="utf-8-sig")
    return signal_path, holdings_path
