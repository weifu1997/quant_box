"""模块说明：从最新策略分数和历史持仓生成调仓信号。"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.common import (
    normalize_datetime_index as _normalize_datetime_index,
    normalize_instrument_index as _normalize_instrument_index,
    normalize_instruments as _normalize_instruments,
)
from src.config_loader import load_config, resolve_path
from src.factor_calculator import load_or_compute_factors
from src.risk_policy import RiskPolicy
from src.scoring import build_latest_strategy_scores
from src.strategy import select_stocks
from src.trading_calendar import resolve_target_date_value

logger = logging.getLogger(__name__)


def read_previous_holdings(path: str | Path | None = None, config: dict | None = None) -> list[str]:
    """函数说明：读取 read_previous_holdings 主要逻辑。"""
    config = config or load_config()
    holdings_path = resolve_path(path or config["outputs"]["holdings_file"])
    if not holdings_path.exists():
        return []
    df = pd.read_csv(holdings_path)
    col = "instrument" if "instrument" in df.columns else "ticker"
    if col not in df.columns:
        return []
    return _normalize_instruments(df[col].dropna().tolist())


def read_signal_previous_holdings(config: dict | None = None) -> tuple[list[str], str]:
    """Read account holdings for signal actions, falling back to latest target holdings."""
    config = config or load_config()
    account_path_value = config.get("account", {}).get("current_holdings_file")
    if account_path_value:
        account_path = resolve_path(account_path_value)
        if account_path.exists():
            account_holdings = _read_holdings_csv(account_path, require_positive_shares=True)
            if account_holdings is not None:
                return account_holdings, "account.current_holdings_file"
    return read_previous_holdings(config=config), "outputs.holdings_file"


def _read_holdings_csv(path: str | Path, require_positive_shares: bool = False) -> list[str] | None:
    holdings_path = resolve_path(path)
    if not holdings_path.exists():
        return []
    df = pd.read_csv(holdings_path)
    col = "instrument" if "instrument" in df.columns else "ticker"
    if col not in df.columns:
        return None
    if require_positive_shares and "shares" in df.columns:
        shares = pd.to_numeric(df["shares"], errors="coerce").fillna(0.0)
        df = df[shares > 0]
    return _normalize_instruments(df[col].dropna().tolist())


def generate_signal(
    signal_date: str,
    previous_holdings: list[str] | None = None,
    factor_file: str | Path | None = None,
    config: dict | None = None,
    factors: pd.DataFrame | None = None,
    scores: pd.Series | None = None,
    price_df: pd.DataFrame | None = None,
    price_file: str | Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """函数说明：生成 generate_signal 主要逻辑。"""
    config = config or load_config()
    data_cfg = config["data"]
    strategy_cfg = config["strategy"]
    risk_policy = RiskPolicy(config)
    use_latest_date = str(signal_date).lower() == "latest"
    factor_end_date = resolve_target_date_value(
        data_cfg["end_date"] if use_latest_date else signal_date,
        config=config,
    )

    if scores is None and factors is None:
        factors = load_or_compute_factors(
            start_date=data_cfg["start_date"],
            end_date=factor_end_date,
            cache_file=factor_file or config["factors"]["cache_file"],
        )
    if scores is None:
        score_date = "latest" if use_latest_date else _effective_signal_date(factors, factor_end_date)
        scores = build_latest_strategy_scores(
            factors,
            config,
            signal_date=score_date,
            price_df=price_df,
            price_file=price_file,
        )
    else:
        score_date = "latest" if use_latest_date else _effective_score_panel_date(scores, factor_end_date)
        if score_date != "latest":
            scores = _slice_score_panel_date(scores, pd.Timestamp(score_date))
    latest_date, latest_scores = _latest_daily_scores(scores)
    if use_latest_date:
        signal_date = latest_date.strftime("%Y-%m-%d")
    else:
        signal_date = str(pd.Timestamp(score_date).date())
    if risk_policy.selection_risk_enabled():
        prices = price_df if price_df is not None else _load_price_frame(price_file, config)
        latest_scores = risk_policy.filter_selection_scores(latest_scores, prices, latest_date)
    latest_scores = _normalize_score_index(latest_scores)
    if previous_holdings is None:
        previous_holdings, _source = read_signal_previous_holdings(config=config)
    previous_holdings = _normalize_instruments(previous_holdings)
    max_industry_weight = risk_policy.max_industry_weight
    industry_map = risk_policy.industry_group_map()
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
    """函数说明：加载 load_price_frame 的内部辅助逻辑。"""
    price_path = resolve_path(price_file or config.get("ic", {}).get("price_file", "data/prices/ohlcv_adjusted.parquet"))
    if not price_path.exists():
        raise FileNotFoundError(f"Price file not found for selection risk filter: {price_path}")
    return pd.read_parquet(price_path)


def _latest_daily_scores(scores: pd.Series) -> tuple[pd.Timestamp, pd.Series]:
    """函数说明：处理 latest_daily_scores 的内部辅助逻辑。"""
    if scores.empty or not isinstance(scores.index, pd.MultiIndex):
        raise ValueError("scores must use MultiIndex: datetime/instrument.")
    raw_dates = _normalize_datetime_index(scores.index.get_level_values(0), normalize=False)
    values = pd.to_numeric(pd.Series(scores.to_numpy()), errors="coerce").to_numpy()
    frame = pd.DataFrame(
        {
            "date": raw_dates.normalize(),
            "raw_date": raw_dates,
            "instrument": _normalize_instrument_index(scores.index.get_level_values(1)),
            "score": values,
            "position": range(len(scores)),
        }
    )
    frame = frame[frame["date"].notna() & (frame["instrument"] != "")]
    if frame.empty:
        raise ValueError("No dated score rows are available.")
    latest_date = pd.Timestamp(frame["date"].max()).normalize()
    latest = frame[frame["date"] == latest_date].copy()
    latest = latest.sort_values(
        ["instrument", "raw_date", "score", "position"],
        kind="mergesort",
        na_position="first",
    )
    latest = latest.drop_duplicates("instrument", keep="last")
    result = pd.Series(latest["score"].to_numpy(), index=pd.Index(latest["instrument"], name=scores.index.names[1]), name=scores.name)
    result.attrs = dict(getattr(scores, "attrs", {}))
    return latest_date, result


def _effective_signal_date(factors: pd.DataFrame, requested_date: str) -> str:
    """函数说明：处理 effective_signal_date 的内部辅助逻辑。"""
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
    """函数说明：处理 factor_dates 的内部辅助逻辑。"""
    if factors.empty or not isinstance(factors.index, pd.MultiIndex):
        raise ValueError("factors must use MultiIndex: date/instrument.")
    date_level = factors.index.names[0] or 0
    return _normalize_datetime_index(factors.index.get_level_values(date_level), dropna=True, unique=True, sort=True)


def _effective_score_panel_date(scores: pd.Series, requested_date: str) -> str:
    requested_ts = pd.Timestamp(requested_date).normalize()
    dates = _score_panel_dates(scores)
    eligible = dates[dates <= requested_ts]
    if eligible.empty:
        raise ValueError(f"No score panel date is available on or before requested signal_date {requested_ts.date()}.")
    effective = pd.Timestamp(eligible.max()).normalize()
    if effective != requested_ts:
        logger.warning(
            "Falling back signal date from %s to %s because score panel has no rows for the requested date.",
            requested_ts.date(),
            effective.date(),
        )
    return str(effective.date())


def _score_panel_dates(scores: pd.Series) -> pd.DatetimeIndex:
    if scores.empty or not isinstance(scores.index, pd.MultiIndex):
        raise ValueError("scores must use MultiIndex: datetime/instrument.")
    date_level = scores.index.names[0] or 0
    return _normalize_datetime_index(scores.index.get_level_values(date_level), dropna=True, unique=True, sort=True)


def _slice_score_panel_date(scores: pd.Series, target_date: pd.Timestamp) -> pd.Series:
    dates = _normalize_datetime_index(scores.index.get_level_values(0), normalize=False)
    mask = dates.normalize() == pd.Timestamp(target_date).normalize()
    sliced = scores.loc[mask].copy()
    sliced.attrs = dict(getattr(scores, "attrs", {}))
    return sliced


def _normalize_score_index(scores: pd.Series) -> pd.Series:
    """函数说明：规范化 normalize_score_index 的内部辅助逻辑。"""
    if scores.empty:
        return scores
    result = scores.sort_values(ascending=False, kind="mergesort", na_position="last").copy()
    result.index = _normalize_instrument_index(result.index, name=result.index.name)
    result = result[result.index != ""]
    if result.index.has_duplicates:
        result = result[~result.index.duplicated(keep="first")]
    result.attrs = dict(getattr(scores, "attrs", {}))
    return result


def save_signal(signal_df: pd.DataFrame, holdings: list[str], signal_date: str, config: dict | None = None) -> tuple[Path, Path]:
    """函数说明：保存 save_signal 主要逻辑。"""
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
    """函数说明：保存 save_candidate_signal 主要逻辑。"""
    config = config or load_config()
    out_dir = resolve_path(config["outputs"].get("dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    signal_path = out_dir / f"candidate_signal_{signal_date}.csv"
    holdings_path = out_dir / f"candidate_holdings_{signal_date}.csv"
    signal_df.to_csv(signal_path, index=False, encoding="utf-8-sig")
    pd.DataFrame({"instrument": holdings}).to_csv(holdings_path, index=False, encoding="utf-8-sig")
    return signal_path, holdings_path
