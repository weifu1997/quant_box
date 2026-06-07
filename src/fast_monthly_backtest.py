from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.strategy import select_stocks


PRICE_FIELD_COLUMNS = {"open", "high", "low", "close", "volume", "vol", "amount", "vwap", "adj_factor", "is_st"}


@dataclass
class FastBacktestResult:
    equity_curve: pd.Series
    weights: pd.DataFrame
    metrics: dict[str, float]


@dataclass
class FastPeriod:
    signal_date: pd.Timestamp
    trade_date: pd.Timestamp
    next_trade_date: pd.Timestamp
    scores: pd.Series
    returns: pd.Series


@dataclass
class FastBacktestData:
    periods: list[FastPeriod]
    price_dates: pd.DatetimeIndex
    initial_date: pd.Timestamp | None


def run_fast_period_backtest(
    score_panel: pd.Series | pd.DataFrame,
    price_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    config: dict[str, Any],
) -> FastBacktestResult:
    """Fast approximate period-to-period backtest for research screening.

    This intentionally omits lot sizing, limit rules, capacity and stale-price handling.
    Use the formal backtest for final validation.
    """

    prepared = prepare_fast_period_data(
        score_panel,
        price_df,
        start_date,
        end_date,
        trade_price_field=str(config.get("trade_price_field", "close")),
    )
    return run_fast_prepared_backtest(prepared, config)


def prepare_fast_period_data(
    score_panel: pd.Series | pd.DataFrame,
    price_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    trade_price_field: str = "close",
) -> FastBacktestData:
    scores = _ensure_score_panel(score_panel)
    trade_prices = _price_frame(price_df, trade_price_field)
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    price_dates = pd.DatetimeIndex(trade_prices.index[(trade_prices.index >= start) & (trade_prices.index <= end)]).unique().sort_values()
    if price_dates.empty:
        return FastBacktestData(periods=[], price_dates=price_dates, initial_date=None)

    signal_dates = pd.DatetimeIndex(pd.to_datetime(scores.index.get_level_values(0)).unique()).sort_values()
    signal_dates = signal_dates[(signal_dates >= start) & (signal_dates <= end)]
    trade_dates = [_next_price_date(price_dates, signal_date) for signal_date in signal_dates]
    schedule = [(signal, trade) for signal, trade in zip(signal_dates, trade_dates) if trade is not None]
    if not schedule:
        return FastBacktestData(periods=[], price_dates=price_dates, initial_date=pd.Timestamp(price_dates[0]))

    periods: list[FastPeriod] = []
    for idx, (signal_date, trade_date) in enumerate(schedule):
        if trade_date >= end:
            break
        next_trade_date = schedule[idx + 1][1] if idx + 1 < len(schedule) else price_dates[-1]
        if next_trade_date <= trade_date:
            continue
        daily_scores = scores.xs(signal_date, level=0, drop_level=True).dropna()
        daily_scores.index = daily_scores.index.astype(str)
        period_returns = _period_returns(trade_prices, trade_date, next_trade_date, trade_prices.columns)
        daily_scores = daily_scores[daily_scores.index.isin(period_returns.index)]
        periods.append(
            FastPeriod(
                signal_date=pd.Timestamp(signal_date),
                trade_date=pd.Timestamp(trade_date),
                next_trade_date=pd.Timestamp(next_trade_date),
                scores=daily_scores,
                returns=period_returns,
            )
        )
    initial_date = periods[0].trade_date if periods else pd.Timestamp(price_dates[0])
    return FastBacktestData(periods=periods, price_dates=price_dates, initial_date=initial_date)


def run_fast_prepared_backtest(prepared: FastBacktestData, config: dict[str, Any]) -> FastBacktestResult:
    if not prepared.periods:
        if prepared.initial_date is None:
            empty = pd.Series(dtype=float, name="equity")
            return FastBacktestResult(empty, pd.DataFrame(), _metrics(empty, config))
        equity = pd.Series(
            [float(config.get("initial_capital", 1_000_000))],
            index=[prepared.initial_date],
            name="equity",
        )
        return FastBacktestResult(equity, pd.DataFrame(), _metrics(equity, config))

    exposure = _exposure_schedule(config.get("exposure_schedule"), prepared.price_dates)
    top_n = int(config.get("top_n", 15))
    max_turnover = int(config.get("max_turnover", top_n))
    rank_buffer = int(config.get("rank_buffer", 0))
    industry_map = config.get("industry_map")
    max_industry_weight = config.get("max_industry_weight")
    max_weight = config.get("max_weight_per_stock")
    max_weight = float(max_weight) if max_weight is not None else None
    drift_threshold = max(float(config.get("rebalance_drift_threshold", 0.0) or 0.0), 0.0)
    score_weighted = bool(config.get("score_weighted", False))
    cost_rate = _round_trip_cost_rate(config)

    capital = float(config.get("initial_capital", 1_000_000))
    equity_rows: list[tuple[pd.Timestamp, float]] = [(pd.Timestamp(prepared.periods[0].trade_date), capital)]
    weight_rows: list[dict[str, object]] = []
    current_weights = pd.Series(dtype=float)
    current_holdings: list[str] = []
    previous_scale: float | None = None
    total_weight_turnover = 0.0

    for period in prepared.periods:
        signal_date = period.signal_date
        trade_date = period.trade_date
        next_trade_date = period.next_trade_date
        daily_scores = period.scores
        holdings = select_stocks(
            daily_scores,
            top_n=top_n,
            previous_holdings=current_holdings,
            max_turnover=max_turnover,
            rank_buffer=rank_buffer,
            group_map=industry_map,
            max_group_weight=max_industry_weight,
        )
        scale = _scale_for_date(exposure, trade_date)
        target_weights = _target_weights(daily_scores, holdings, score_weighted, max_weight) * scale
        if drift_threshold > 0 and previous_scale is not None and abs(scale - previous_scale) <= 1e-12:
            target_weights = _apply_weight_drift_threshold(target_weights, current_weights, drift_threshold)
        all_names = current_weights.index.union(target_weights.index)
        turnover = float((target_weights.reindex(all_names, fill_value=0.0) - current_weights.reindex(all_names, fill_value=0.0)).abs().sum())
        total_weight_turnover += turnover
        capital *= max(0.0, 1.0 - turnover * cost_rate)

        period_returns = period.returns.reindex(target_weights.index).fillna(0.0)
        if not target_weights.empty and target_weights.sum() > 0:
            period_return = float((period_returns.reindex(target_weights.index).fillna(0.0) * target_weights).sum())
            capital *= 1.0 + period_return
        else:
            period_return = 0.0
        equity_rows.append((pd.Timestamp(next_trade_date), capital))
        for stock, weight in target_weights.items():
            weight_rows.append({"date": trade_date, "signal_date": signal_date, "instrument": stock, "weight": float(weight)})
        current_weights = _drift_weights_after_returns(target_weights, period_returns, period_return)
        current_holdings = list(current_weights.index)
        previous_scale = scale

    equity = pd.Series(dict(equity_rows), name="equity").sort_index()
    weights = pd.DataFrame(weight_rows)
    metrics = _metrics(equity, config)
    metrics["total_weight_turnover"] = float(total_weight_turnover)
    metrics["annual_weight_turnover"] = _annual_weight_turnover(total_weight_turnover, equity)
    return FastBacktestResult(equity, weights, metrics)


def _ensure_score_panel(score_panel: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(score_panel, pd.DataFrame):
        if "score" in score_panel.columns:
            score_panel = score_panel["score"]
        elif score_panel.shape[1] == 1:
            score_panel = score_panel.iloc[:, 0]
        else:
            raise ValueError("score_panel DataFrame must have a 'score' column or one column.")
    if not isinstance(score_panel.index, pd.MultiIndex):
        raise ValueError("score_panel must use MultiIndex: datetime/instrument.")
    raw_dates = pd.DatetimeIndex(pd.to_datetime(score_panel.index.get_level_values(0), errors="coerce"))
    values = pd.to_numeric(pd.Series(score_panel.to_numpy()), errors="coerce").to_numpy()
    frame = pd.DataFrame(
        {
            "date": raw_dates.normalize(),
            "raw_date": raw_dates,
            "instrument": [_normalize_instrument(value) for value in score_panel.index.get_level_values(1)],
            "score": values,
            "position": range(len(score_panel)),
        }
    )
    frame = frame[frame["date"].notna() & (frame["instrument"] != "")]
    if frame.empty:
        return pd.Series(dtype=float, name="score")
    frame = frame.sort_values(
        ["date", "instrument", "raw_date", "score", "position"],
        kind="mergesort",
        na_position="first",
    )
    frame = frame.drop_duplicates(["date", "instrument"], keep="last")
    index = pd.MultiIndex.from_arrays([frame["date"], frame["instrument"]], names=["datetime", "instrument"])
    result = pd.Series(frame["score"].to_numpy(), index=index, name="score")
    return result.sort_index()


def _price_frame(price_df: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    if price_df.empty:
        return pd.DataFrame()
    field = str(field or "close").strip().lower()
    if isinstance(price_df.columns, pd.MultiIndex):
        fields = price_df.columns.get_level_values(0).astype(str).str.strip().str.lower()
        selected = price_df.loc[:, fields == field].copy()
        if selected.empty and field != "close":
            selected = price_df.loc[:, fields == "close"].copy()
        selected.columns = [_normalize_instrument(value) for value in selected.columns.get_level_values(-1)]
    else:
        if _looks_like_field_table(price_df.columns):
            raise ValueError("Non-MultiIndex price_df must be a close-price panel with instrument columns.")
        selected = price_df.copy()
        selected.columns = [_normalize_instrument(value) for value in selected.columns]

    raw_dates = pd.DatetimeIndex(pd.to_datetime(selected.index, errors="coerce"))
    valid_dates = ~pd.isna(raw_dates)
    if not valid_dates.all():
        selected = selected.loc[valid_dates].copy()
        raw_dates = raw_dates[valid_dates]
    if selected.empty:
        return selected

    order = np.argsort(raw_dates.to_numpy(), kind="mergesort")
    if not np.array_equal(order, np.arange(len(raw_dates))):
        selected = selected.iloc[order].copy()
        raw_dates = raw_dates[order]
    selected.index = raw_dates.normalize()
    if selected.index.has_duplicates:
        selected = selected.loc[~selected.index.duplicated(keep="last")]
    selected = selected.loc[:, selected.columns != ""]
    if selected.columns.has_duplicates:
        selected = selected.loc[:, ~selected.columns.duplicated(keep="last")]
    selected = selected.sort_index()
    if all(pd.api.types.is_numeric_dtype(dtype) for dtype in selected.dtypes):
        return selected
    return selected.apply(pd.to_numeric, errors="coerce")


def _normalize_instrument(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def _looks_like_field_table(columns: pd.Index) -> bool:
    labels = {str(column).strip().lower() for column in columns}
    return len(labels) > 1 and bool(labels & PRICE_FIELD_COLUMNS)


def _next_price_date(price_dates: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp | None:
    pos = price_dates.searchsorted(pd.Timestamp(date).normalize(), side="right")
    if pos >= len(price_dates):
        return None
    return pd.Timestamp(price_dates[pos])


def _exposure_schedule(value: object, price_dates: pd.DatetimeIndex) -> pd.Series:
    if not isinstance(value, pd.Series):
        return pd.Series(1.0, index=price_dates, dtype=float)
    exposure = value.copy()
    raw_dates = pd.DatetimeIndex(pd.to_datetime(exposure.index, errors="coerce"))
    valid_dates = ~raw_dates.isna()
    exposure = pd.to_numeric(exposure.loc[valid_dates], errors="coerce")
    raw_dates = raw_dates[valid_dates]
    if not exposure.empty:
        order = np.argsort(raw_dates.to_numpy(), kind="mergesort")
        exposure = exposure.iloc[order].copy()
        raw_dates = raw_dates[order]
    exposure.index = raw_dates.normalize()
    exposure = exposure.dropna()
    exposure = exposure[~exposure.index.duplicated(keep="last")].sort_index()
    return exposure.reindex(price_dates).ffill().fillna(1.0).clip(lower=0.0, upper=1.0)


def _scale_for_date(exposure: pd.Series, date: pd.Timestamp) -> float:
    if date in exposure.index:
        return float(exposure.loc[date])
    prior = exposure[exposure.index <= date]
    return float(prior.iloc[-1]) if not prior.empty else 1.0


def _target_weights(scores: pd.Series, holdings: list[str], score_weighted: bool, max_weight: float | None) -> pd.Series:
    if not holdings:
        return pd.Series(dtype=float)
    if score_weighted:
        weights = _score_weights(scores, holdings)
    else:
        weights = pd.Series(1.0 / len(holdings), index=holdings, dtype=float)
    if max_weight is not None:
        weights = weights.clip(upper=max_weight)
    return weights


def _apply_weight_drift_threshold(target_weights: pd.Series, previous_weights: pd.Series, threshold: float) -> pd.Series:
    if target_weights.empty or previous_weights.empty or threshold <= 0:
        return target_weights
    adjusted = target_weights.copy()
    for instrument, target_weight in target_weights.items():
        if instrument not in previous_weights.index:
            continue
        previous_weight = float(previous_weights.loc[instrument])
        if abs(float(target_weight) - previous_weight) <= threshold:
            adjusted.loc[instrument] = previous_weight
    return adjusted


def _drift_weights_after_returns(weights: pd.Series, returns: pd.Series, portfolio_return: float) -> pd.Series:
    if weights.empty:
        return pd.Series(dtype=float)
    denominator = 1.0 + float(portfolio_return)
    if denominator <= 0 or not np.isfinite(denominator):
        return pd.Series(dtype=float)
    aligned_returns = pd.to_numeric(returns.reindex(weights.index), errors="coerce").fillna(0.0)
    drifted = weights.astype(float) * (1.0 + aligned_returns) / denominator
    drifted = drifted.replace([np.inf, -np.inf], np.nan).dropna()
    return drifted[drifted > 1e-12]


def _score_weights(scores: pd.Series, holdings: list[str]) -> pd.Series:
    selected = pd.to_numeric(scores.reindex(holdings), errors="coerce").replace([np.inf, -np.inf], np.nan)
    if selected.notna().sum() == 0:
        return pd.Series(1.0 / len(holdings), index=holdings, dtype=float)
    values = selected.fillna(selected.min()).astype(float)
    shifted = values - min(float(values.min()), 0.0)
    if shifted.sum() <= 0:
        shifted = values - float(values.min())
    if shifted.sum() <= 0:
        return pd.Series(1.0 / len(holdings), index=holdings, dtype=float)
    return shifted / shifted.sum()


def _period_returns(close: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, instruments: pd.Index) -> pd.Series:
    start_prices = pd.to_numeric(close.loc[start].reindex(instruments), errors="coerce").replace([np.inf, -np.inf], np.nan)
    end_prices = pd.to_numeric(close.loc[end].reindex(instruments), errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid_start = start_prices.notna() & (start_prices > 0.0)
    returns = end_prices.divide(start_prices).sub(1.0)
    returns = returns.replace([np.inf, -np.inf], np.nan)
    return returns.loc[valid_start]


def _round_trip_cost_rate(config: dict[str, Any]) -> float:
    commission = float(config.get("commission", 0.0))
    transfer = float(config.get("transfer_fee", 0.0))
    stamp = float(config.get("stamp_tax", 0.0))
    slippage = float(config.get("slippage", 0.0))
    return commission + transfer + slippage + stamp / 2.0


def _metrics(equity_curve: pd.Series, config: dict[str, Any]) -> dict[str, float]:
    if equity_curve.empty:
        return {"total_return": 0.0, "annual_return": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}
    equity = equity_curve.sort_index().astype(float)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if equity.iloc[0] else 0.0
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 252)
    annual_return = float((1.0 + total_return) ** (1.0 / years) - 1.0) if total_return > -1 else -1.0
    drawdown = equity / equity.cummax() - 1.0
    returns = equity.pct_change().dropna()
    periods_per_year = len(returns) / years if years > 0 else 12.0
    sharpe = 0.0
    if len(returns) > 1 and float(returns.std(ddof=1)) > 0:
        sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year))
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": float(drawdown.min()),
        "sharpe": sharpe,
    }


def _annual_weight_turnover(total_weight_turnover: float, equity_curve: pd.Series) -> float:
    if equity_curve.empty or len(equity_curve) <= 1:
        return 0.0
    years = max((equity_curve.index[-1] - equity_curve.index[0]).days / 365.25, 1 / 252)
    return float(total_weight_turnover / years)
