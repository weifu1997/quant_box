from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import weakref

import numpy as np
import pandas as pd

from src.backtest_circuit_breaker import (
    _annual_drawdown_guard_active,
    _annual_drawdown_guard_released,
    _annual_drawdown_guard_target_exposure,
    _circuit_breaker_target_exposure,
    _cooldown_until,
    _drawdown_breached,
    _optional_nonnegative_int,
)
from src.backtest_costs import _commission_cost, _shares_affordable, _transfer_fee_cost
from src.backtest_exposure import (
    _equity_overlay_rebalance_needed,
    _exposure_scale,
    _normalize_exposure_schedule,
    _scheduled_exposure_rebalance_needed,
    _scheduled_exposure_scale,
)
from src.common import PRICE_FIELD_COLUMNS, looks_like_field_table as _looks_like_field_table, normalize_instrument as _normalize_instrument
from src.selection_risk import filter_scores_by_selection_risk
from src.strategy import select_stocks


LOT_SIZE = 100
_PRICE_FIELD_CACHE: dict[int, tuple[weakref.ReferenceType[pd.DataFrame], set[str], dict[str, pd.DataFrame]]] = {}


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    holdings: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float]


def run_backtest(
    score_panel: pd.Series,
    price_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    config: dict,
) -> BacktestResult:
    score_panel = _ensure_score_panel(score_panel)
    prices = _normalize_price_frame(price_df)

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    price_dates = pd.Index(pd.to_datetime(prices.index).sort_values().unique())
    price_dates = pd.Index([date for date in price_dates if start <= date <= end])
    if price_dates.empty:
        empty_equity = pd.Series(dtype=float, name="equity")
        empty_trades = pd.DataFrame()
        return BacktestResult(empty_equity, pd.DataFrame(), empty_trades, calculate_metrics(empty_equity, empty_trades, config))

    signal_dates = sorted(pd.to_datetime(score_panel.index.get_level_values(0).unique()))
    signal_dates = [date for date in signal_dates if start <= date <= end]
    trade_schedule: dict[pd.Timestamp, pd.Timestamp] = {}
    for signal_date in signal_dates:
        trade_date = _next_price_date(price_dates, signal_date, end)
        if trade_date is not None:
            trade_schedule[trade_date] = signal_date
    exposure_schedule = _normalize_exposure_schedule(config.get("exposure_schedule"), price_dates)

    capital = float(config.get("initial_capital", 1_000_000))
    commission = float(config.get("commission", 0.0003))
    stamp_tax = float(config.get("stamp_tax", 0.001))
    transfer_fee = float(config.get("transfer_fee", 0.0))
    min_commission = float(config.get("min_commission_per_order", 0.0))
    slippage = float(config.get("slippage", 0.0))
    trade_price_field = str(config.get("trade_price_field", "close")).lower()
    valuation_price_field = str(config.get("valuation_price_field", "close")).lower()
    top_n = int(config.get("top_n", 7))
    max_turnover = int(config.get("max_turnover", 1))
    rank_buffer = int(config.get("rank_buffer", 0))
    industry_map = config.get("industry_map")
    max_industry_weight = config.get("max_industry_weight")
    max_weight = config.get("max_weight_per_stock")
    max_weight = float(max_weight) if max_weight is not None else None
    rebalance_drift_threshold = float(config.get("rebalance_drift_threshold", 0.0) or 0.0)

    holdings: dict[str, int] = {}
    entry_prices: dict[str, float] = {}
    last_prices: dict[str, float] = {}
    stale_unpriced_days: dict[str, int] = {}
    equity_rows: list[tuple[pd.Timestamp, float]] = []
    holding_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []

    previous_date: pd.Timestamp | None = None
    last_signal_date: pd.Timestamp | None = None
    peak_equity = capital
    year_peak_equity = capital
    annual_guard_year: int | None = None
    circuit_breaker_until: pd.Timestamp | None = None
    circuit_breaker_cooldown_days = _optional_nonnegative_int(config.get("circuit_breaker_cooldown_days"), "circuit_breaker_cooldown_days")
    for date_pos, trade_date in enumerate(price_dates):
        close = _price_row(prices, valuation_price_field, trade_date)
        trade_prices = _price_row(prices, trade_price_field, trade_date)
        close_values = _row_price_dict(close)
        trade_price_values = _row_price_dict(trade_prices)
        last_prices.update(close_values)
        tradability = _tradability(prices, trade_date, previous_date, config)
        capital = _execute_stale_price_exits(
            holdings,
            entry_prices,
            stale_unpriced_days,
            capital,
            close_values,
            last_prices,
            tradability,
            trade_rows,
            trade_date,
            commission,
            stamp_tax,
            transfer_fee,
            min_commission,
            slippage,
            prices,
            config,
        )
        capital = _execute_risk_exits(
            holdings,
            entry_prices,
            capital,
            close_values,
            last_prices,
            tradability,
            trade_rows,
            trade_date,
            commission,
            stamp_tax,
            transfer_fee,
            min_commission,
            slippage,
            prices,
            config,
        )
        total_before_signal = _portfolio_value(capital, holdings, close_values, last_prices)
        trade_year = pd.Timestamp(trade_date).year
        if previous_date is None or pd.Timestamp(previous_date).year != trade_year:
            year_peak_equity = total_before_signal
            annual_guard_year = None
        else:
            year_peak_equity = max(year_peak_equity, total_before_signal)

        if circuit_breaker_until is not None and pd.Timestamp(trade_date) <= circuit_breaker_until:
            risk_off = True
        else:
            if circuit_breaker_until is not None:
                circuit_breaker_until = None
            peak_equity = max(peak_equity, total_before_signal)
            risk_off = bool(holdings) and _drawdown_breached(total_before_signal, peak_equity, config)
            if risk_off and circuit_breaker_cooldown_days is not None:
                circuit_breaker_until = _cooldown_until(price_dates, date_pos, circuit_breaker_cooldown_days)
        if risk_off:
            target_exposure = _circuit_breaker_target_exposure(config)
            if target_exposure <= 0:
                capital = _liquidate_portfolio(
                    holdings,
                    entry_prices,
                    capital,
                    close_values,
                    last_prices,
                    tradability,
                    trade_rows,
                    trade_date,
                    commission,
                    stamp_tax,
                    transfer_fee,
                    min_commission,
                    slippage,
                    prices,
                    config,
                    reason="circuit_breaker",
                )
            else:
                capital = _reduce_portfolio_exposure(
                    holdings,
                    entry_prices,
                    capital,
                    close_values,
                    last_prices,
                    tradability,
                    trade_rows,
                    trade_date,
                    commission,
                    stamp_tax,
                    transfer_fee,
                    min_commission,
                    slippage,
                    prices,
                    config,
                    target_exposure,
                    reason="circuit_breaker",
                )

        if _annual_drawdown_guard_released(total_before_signal, year_peak_equity, trade_year, annual_guard_year, config):
            annual_guard_year = None
        annual_guard_active = _annual_drawdown_guard_active(total_before_signal, year_peak_equity, trade_year, annual_guard_year, config)
        if annual_guard_active:
            annual_guard_year = trade_year
            target_exposure = _annual_drawdown_guard_target_exposure(config)
            if target_exposure <= 0:
                capital = _liquidate_portfolio(
                    holdings,
                    entry_prices,
                    capital,
                    close_values,
                    last_prices,
                    tradability,
                    trade_rows,
                    trade_date,
                    commission,
                    stamp_tax,
                    transfer_fee,
                    min_commission,
                    slippage,
                    prices,
                    config,
                    reason="annual_drawdown_guard",
                )
            else:
                capital = _reduce_portfolio_exposure(
                    holdings,
                    entry_prices,
                    capital,
                    close_values,
                    last_prices,
                    tradability,
                    trade_rows,
                    trade_date,
                    commission,
                    stamp_tax,
                    transfer_fee,
                    min_commission,
                    slippage,
                    prices,
                    config,
                    target_exposure,
                    reason="annual_drawdown_guard",
                )
            risk_off = True

        signal_date = trade_schedule.get(pd.Timestamp(trade_date))
        if signal_date is not None:
            last_signal_date = signal_date
        exposure_rebalance = _scheduled_exposure_rebalance_needed(exposure_schedule, trade_date, previous_date, config)
        exposure_rebalance = exposure_rebalance or _equity_overlay_rebalance_needed(equity_rows, config)
        rebalance_signal_date = signal_date if signal_date is not None else last_signal_date if exposure_rebalance else None
        if rebalance_signal_date is not None and not risk_off:
            daily_scores = score_panel.xs(rebalance_signal_date, level=0, drop_level=True)
            daily_scores.index = daily_scores.index.astype(str)
            daily_scores = daily_scores[daily_scores.index.isin(tradability["priced"])]
            daily_scores = filter_scores_by_selection_risk(daily_scores, prices, rebalance_signal_date, config)

            target_holdings = select_stocks(
                daily_scores,
                top_n=top_n,
                previous_holdings=holdings.keys(),
                max_turnover=max_turnover,
                rank_buffer=rank_buffer,
                group_map=industry_map,
                max_group_weight=max_industry_weight,
            )
            total_before_trade = _portfolio_value(capital, holdings, close_values, last_prices)
            exposure_scale = _exposure_scale(equity_rows, config) * _scheduled_exposure_scale(exposure_schedule, trade_date)
            target_values = _target_values(total_before_trade, target_holdings, daily_scores, max_weight, exposure_scale, config)

            desired_shares = {}
            for stock in target_holdings:
                price = _price_for(stock, trade_price_values, last_prices)
                target_value = target_values.get(stock, 0.0)
                desired_shares[stock] = _round_lot(target_value / price if price > 0 else 0, stock, config)
            drift_threshold = 0.0 if exposure_rebalance else rebalance_drift_threshold
            desired_shares = _apply_rebalance_drift_threshold(
                desired_shares,
                holdings,
                target_holdings,
                target_values,
                trade_price_values,
                last_prices,
                total_before_trade,
                drift_threshold,
            )

            for stock in list(holdings):
                current = holdings.get(stock, 0)
                desired = desired_shares.get(stock, 0)
                sell_shares = current - desired
                if sell_shares <= 0:
                    continue
                if stock not in tradability["sellable"]:
                    trade_rows.append(_blocked_trade(rebalance_signal_date, trade_date, stock, "SELL", sell_shares, "not_sellable"))
                    continue
                base_price = _price_for(stock, trade_price_values, last_prices)
                trade_slippage = _trade_slippage(slippage, prices, trade_date, stock, sell_shares * base_price, config)
                price = base_price * (1 - trade_slippage)
                capacity = _capacity(prices, trade_date, stock, sell_shares * price, config)
                filled_shares, status, reason = _apply_capacity_limit(sell_shares, price, stock, prices, trade_date, config)
                if filled_shares <= 0:
                    trade_rows.append(_blocked_trade(rebalance_signal_date, trade_date, stock, "SELL", sell_shares, "capacity_limited"))
                    continue
                sell_shares = filled_shares
                gross = sell_shares * price
                commission_cost = _commission_cost(gross, commission, min_commission)
                tax_cost = gross * stamp_tax
                transfer_fee_cost = _transfer_fee_cost(gross, transfer_fee)
                proceeds = gross - commission_cost - tax_cost - transfer_fee_cost
                capital += proceeds
                remaining = current - sell_shares
                if remaining > 0:
                    holdings[stock] = remaining
                else:
                    holdings.pop(stock, None)
                    entry_prices.pop(stock, None)
                trade_rows.append(
                    _trade(
                        rebalance_signal_date,
                        trade_date,
                        stock,
                        "SELL",
                        sell_shares,
                        price,
                        proceeds,
                        status=status,
                        reason=reason,
                        commission_cost=commission_cost,
                        tax_cost=tax_cost,
                        transfer_fee_cost=transfer_fee_cost,
                        slippage_cost=sell_shares * base_price * trade_slippage,
                        slippage_rate=trade_slippage,
                        slippage_model=_slippage_model_name(config),
                        capacity=capacity,
                    )
                )

            for stock in target_holdings:
                current = holdings.get(stock, 0)
                desired = desired_shares.get(stock, 0)
                buy_shares = desired - current
                if buy_shares <= 0:
                    continue
                if stock not in tradability["buyable"]:
                    trade_rows.append(_blocked_trade(rebalance_signal_date, trade_date, stock, "BUY", buy_shares, "not_buyable"))
                    continue
                base_price = _price_for(stock, trade_price_values, last_prices)
                trade_slippage = _trade_slippage(slippage, prices, trade_date, stock, buy_shares * base_price, config)
                price = base_price * (1 + trade_slippage)
                capacity = _capacity(prices, trade_date, stock, buy_shares * price, config)
                buy_shares, status, reason = _apply_capacity_limit(buy_shares, price, stock, prices, trade_date, config)
                if buy_shares <= 0:
                    trade_rows.append(_blocked_trade(rebalance_signal_date, trade_date, stock, "BUY", desired - current, "capacity_limited"))
                    continue
                gross = buy_shares * price
                commission_cost = _commission_cost(gross, commission, min_commission)
                transfer_fee_cost = _transfer_fee_cost(gross, transfer_fee)
                cost = gross + commission_cost + transfer_fee_cost
                if cost > capital:
                    buy_shares = _round_lot(_shares_affordable(capital, price, commission, min_commission, transfer_fee), stock, config)
                    gross = buy_shares * price
                    commission_cost = _commission_cost(gross, commission, min_commission) if buy_shares > 0 else 0.0
                    transfer_fee_cost = _transfer_fee_cost(gross, transfer_fee) if buy_shares > 0 else 0.0
                    cost = gross + commission_cost + transfer_fee_cost
                if buy_shares <= 0:
                    continue
                capital -= cost
                old_shares = holdings.get(stock, 0)
                holdings[stock] = holdings.get(stock, 0) + buy_shares
                entry_prices[stock] = _average_entry_price(entry_prices.get(stock), old_shares, price, buy_shares)
                trade_rows.append(
                    _trade(
                        rebalance_signal_date,
                        trade_date,
                        stock,
                        "BUY",
                        buy_shares,
                        price,
                        -cost,
                        status=status,
                        reason=reason,
                        commission_cost=commission_cost,
                        tax_cost=0.0,
                        transfer_fee_cost=transfer_fee_cost,
                        slippage_cost=buy_shares * base_price * trade_slippage,
                        slippage_rate=trade_slippage,
                        slippage_model=_slippage_model_name(config),
                        capacity=capacity,
                    )
                )

        total = _portfolio_value(capital, holdings, close_values, last_prices)
        equity_rows.append((pd.Timestamp(trade_date), total))
        for stock, shares in holdings.items():
            price = _price_for(stock, close_values, last_prices)
            holding_rows.append(
                {
                    "date": trade_date,
                    "instrument": stock,
                    "shares": shares,
                    "price": price,
                    "value": shares * price,
                }
            )
        previous_date = pd.Timestamp(trade_date)

    equity_curve = pd.Series(dict(equity_rows), name="equity").sort_index()
    holdings_df = pd.DataFrame(holding_rows)
    trades_df = pd.DataFrame(trade_rows)
    return BacktestResult(
        equity_curve=equity_curve,
        holdings=holdings_df,
        trades=trades_df,
        metrics=calculate_metrics(equity_curve, trades_df, config),
    )


def calculate_metrics(equity_curve: pd.Series, trades: pd.DataFrame, config: dict) -> dict[str, float]:
    annual_days = int(config.get("annual_trading_days", 252))
    risk_free_rate = float(config.get("risk_free_rate", 0.0))
    top_n = max(int(config.get("top_n", 1)), 1)

    if equity_curve.empty:
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_duration": 0.0,
            "calmar": 0.0,
            "win_rate": 0.0,
            "profit_loss_ratio": 0.0,
            "turnover_count": 0.0,
            "annual_turnover": 0.0,
            "commission_cost": 0.0,
            "tax_cost": 0.0,
            "transfer_fee_cost": 0.0,
            "slippage_cost": 0.0,
            "trade_cost": 0.0,
            "trade_cost_ratio": 0.0,
            "annual_trade_cost_ratio": 0.0,
        }

    returns = equity_curve.pct_change().dropna()
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1 if equity_curve.iloc[0] else 0.0
    periods = max(len(equity_curve) - 1, 1)
    annual_return = (1 + total_return) ** (annual_days / periods) - 1
    excess_daily = returns - risk_free_rate / annual_days
    volatility = returns.std(ddof=1) * np.sqrt(annual_days) if len(returns) > 1 else 0.0
    sharpe = excess_daily.mean() / returns.std(ddof=1) * np.sqrt(annual_days) if len(returns) > 1 and returns.std(ddof=1) else 0.0
    downside = float(np.sqrt(np.mean(np.minimum(excess_daily.to_numpy(), 0.0) ** 2))) if len(excess_daily) else 0.0
    sortino = excess_daily.mean() / downside * np.sqrt(annual_days) if len(excess_daily) > 1 and downside > 0 else 0.0
    drawdown = equity_curve / equity_curve.cummax() - 1
    max_drawdown = float(drawdown.min())
    max_dd_duration = float(_max_drawdown_duration(equity_curve))
    calmar = float(annual_return / abs(max_drawdown)) if max_drawdown < 0 else 0.0
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = float(len(wins) / len(returns)) if len(returns) else 0.0
    profit_loss_ratio = float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else 0.0
    sells = _turnover_sell_count(trades)
    annual_turnover = float(sells / max(periods / annual_days, 1 / annual_days) / top_n * 2)
    commission_cost = _trade_cost_sum(trades, "commission_cost")
    tax_cost = _trade_cost_sum(trades, "tax_cost")
    transfer_fee_cost = _trade_cost_sum(trades, "transfer_fee_cost")
    slippage_cost = _trade_cost_sum(trades, "slippage_cost")
    trade_cost = commission_cost + tax_cost + transfer_fee_cost + slippage_cost
    initial_capital = float(config.get("initial_capital", equity_curve.iloc[0] if len(equity_curve) else 1.0))
    trade_cost_ratio = float(trade_cost / initial_capital) if initial_capital > 0 else 0.0
    years = max(periods / annual_days, 1 / annual_days)
    annual_trade_cost_ratio = float(trade_cost_ratio / years)

    metrics = {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_volatility": float(volatility),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_drawdown,
        "max_drawdown_duration": max_dd_duration,
        "calmar": calmar,
        "win_rate": win_rate,
        "profit_loss_ratio": profit_loss_ratio,
        "turnover_count": float(sells),
        "annual_turnover": annual_turnover,
        "commission_cost": commission_cost,
        "tax_cost": tax_cost,
        "transfer_fee_cost": transfer_fee_cost,
        "slippage_cost": slippage_cost,
        "trade_cost": trade_cost,
        "trade_cost_ratio": trade_cost_ratio,
        "annual_trade_cost_ratio": annual_trade_cost_ratio,
    }
    benchmark = config.get("benchmark_curve")
    if isinstance(benchmark, pd.Series):
        metrics.update(calculate_benchmark_metrics(equity_curve, benchmark, config))
    return metrics


def _trade_cost_sum(trades: pd.DataFrame, column: str) -> float:
    if trades.empty or column not in trades.columns:
        return 0.0
    return float(pd.to_numeric(trades[column], errors="coerce").fillna(0.0).sum())


def _turnover_sell_count(trades: pd.DataFrame) -> int:
    if trades.empty or "side" not in trades.columns:
        return 0
    sell_mask = trades["side"].astype(str).str.strip().str.upper() == "SELL"
    if "status" not in trades.columns:
        return int(sell_mask.sum())
    executable = trades["status"].astype(str).str.strip().str.lower().isin({"filled", "partial", "risk_exit"})
    return int((sell_mask & executable).sum())


def calculate_benchmark_metrics(equity_curve: pd.Series, benchmark_curve: pd.Series, config: dict) -> dict[str, float]:
    annual_days = int(config.get("annual_trading_days", 252))
    aligned = pd.concat(
        [equity_curve.pct_change().rename("portfolio"), benchmark_curve.pct_change().rename("benchmark")],
        axis=1,
    ).dropna()
    if aligned.empty or aligned["benchmark"].var(ddof=1) == 0:
        return {"alpha": 0.0, "beta": 0.0, "information_ratio": 0.0}
    beta = aligned["portfolio"].cov(aligned["benchmark"]) / aligned["benchmark"].var(ddof=1)
    active = aligned["portfolio"] - aligned["benchmark"]
    alpha = (aligned["portfolio"].mean() - beta * aligned["benchmark"].mean()) * annual_days
    ir = active.mean() / active.std(ddof=1) * np.sqrt(annual_days) if active.std(ddof=1) else 0.0
    return {"alpha": float(alpha), "beta": float(beta), "information_ratio": float(ir)}


def _ensure_score_panel(score_panel: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(score_panel, pd.DataFrame):
        if "score" in score_panel.columns:
            score_panel = score_panel["score"]
        elif score_panel.shape[1] == 1:
            score_panel = score_panel.iloc[:, 0]
        else:
            raise ValueError("score_panel DataFrame must have a 'score' column or exactly one column.")
    if not isinstance(score_panel.index, pd.MultiIndex):
        raise ValueError("score_panel must use MultiIndex: date/instrument.")
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
        return pd.Series(dtype=float, name=score_panel.name)
    frame = frame.sort_values(
        ["date", "instrument", "raw_date", "score", "position"],
        kind="mergesort",
        na_position="first",
    )
    frame = frame.drop_duplicates(["date", "instrument"], keep="last")
    normalized_index = pd.MultiIndex.from_arrays([frame["date"], frame["instrument"]], names=["datetime", "instrument"])
    result = pd.Series(frame["score"].to_numpy(), index=normalized_index, name=score_panel.name)
    return result.sort_index()


def _normalize_price_frame(price_df: pd.DataFrame) -> pd.DataFrame:
    prices = price_df
    raw_dates = pd.DatetimeIndex(pd.to_datetime(prices.index, errors="coerce"))
    valid_dates = ~raw_dates.isna()
    prices = prices.loc[valid_dates].copy(deep=False)
    raw_dates = raw_dates[valid_dates]
    if not prices.empty:
        order = np.argsort(raw_dates.to_numpy(), kind="mergesort")
        prices = prices.iloc[order].copy(deep=False)
        raw_dates = raw_dates[order]
    normalized_index = raw_dates.normalize()
    if isinstance(prices.columns, pd.MultiIndex):
        if prices.columns.nlevels != 2:
            raise ValueError("price_df MultiIndex columns must be field/instrument.")
        normalized_columns = pd.MultiIndex.from_arrays(
            [
                [_normalize_price_field(value) for value in prices.columns.get_level_values(0)],
                [_normalize_instrument(value) for value in prices.columns.get_level_values(1)],
            ],
            names=["field", "instrument"],
        )
        columns_need_normalization = (
            not prices.columns.equals(normalized_columns)
            or list(prices.columns.names) != ["field", "instrument"]
        )
        if not prices.index.equals(normalized_index) or columns_need_normalization:
            prices = prices.copy(deep=False)
            prices.index = normalized_index
            prices.columns = normalized_columns
        if prices.index.has_duplicates:
            prices = prices.loc[~prices.index.duplicated(keep="last")]
        prices = prices.loc[:, prices.columns.get_level_values("instrument") != ""]
        if prices.columns.has_duplicates:
            prices = prices.loc[:, ~prices.columns.duplicated(keep="last")]
        if not prices.index.is_monotonic_increasing:
            return prices.sort_index()
        return prices

    if _looks_like_field_table(prices.columns):
        raise ValueError("Non-MultiIndex price_df must be a close-price panel with instrument columns.")

    prices = prices.copy(deep=False)
    prices.index = normalized_index
    if prices.index.has_duplicates:
        prices = prices.loc[~prices.index.duplicated(keep="last")]
    prices.columns = pd.MultiIndex.from_product(
        [["close"], [_normalize_instrument(value) for value in prices.columns]],
        names=["field", "instrument"],
    )
    prices = prices.loc[:, prices.columns.get_level_values("instrument") != ""]
    if prices.columns.has_duplicates:
        prices = prices.loc[:, ~prices.columns.duplicated(keep="last")]
    if not prices.index.is_monotonic_increasing:
        return prices.sort_index()
    return prices


def _normalize_price_field(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _field(prices: pd.DataFrame, field: str) -> pd.DataFrame:
    field = _normalize_price_field(field)
    cache_key = id(prices)
    cached = _PRICE_FIELD_CACHE.get(cache_key)
    if cached is None or cached[0]() is not prices:
        _prune_price_field_cache()
        field_names = set(prices.columns.get_level_values("field"))
        cache: dict[str, pd.DataFrame] = {}
        _PRICE_FIELD_CACHE[cache_key] = (weakref.ref(prices), field_names, cache)
    else:
        _, field_names, cache = cached
    if field in cache:
        return cache[field]

    if field not in field_names:
        frame = pd.DataFrame(index=prices.index)
    else:
        frame = prices.xs(field, level="field", axis=1)
    frame.attrs = {}
    cache[field] = frame
    return frame


def _prune_price_field_cache() -> None:
    dead_keys = [key for key, (prices_ref, _fields, _cache) in _PRICE_FIELD_CACHE.items() if prices_ref() is None]
    for key in dead_keys:
        _PRICE_FIELD_CACHE.pop(key, None)


def _field_on_date(prices: pd.DataFrame, field: str, date: pd.Timestamp) -> pd.Series:
    frame = _field(prices, field)
    if frame.empty or date not in frame.index:
        return pd.Series(dtype=float)
    row = frame.loc[date]
    row.index = row.index.astype(str)
    return row.astype(float)


def _price_row(prices: pd.DataFrame, preferred_field: str, date: pd.Timestamp) -> pd.Series:
    row = _field_on_date(prices, preferred_field, date)
    if not row.empty:
        return row
    return _field_on_date(prices, "close", date)


def _tradability(
    prices: pd.DataFrame,
    trade_date: pd.Timestamp,
    previous_date: pd.Timestamp | None,
    config: dict,
) -> dict[str, set[str]]:
    close = _field_on_date(prices, "close", trade_date)
    priced = set(close.dropna().index.astype(str))
    buyable = set(priced)
    sellable = set(priced)

    volume = _field_on_date(prices, "volume", trade_date)
    if not volume.empty:
        liquid = set(volume[volume > 0].dropna().index.astype(str))
        buyable &= liquid
        sellable &= liquid

    if previous_date is not None:
        prev_close = _field_on_date(prices, "close", previous_date).reindex(close.index)
        valid_prev = prev_close > 0
        high = _field_on_date(prices, "high", trade_date).reindex(close.index)
        low = _field_on_date(prices, "low", trade_date).reindex(close.index)
        up_probe = high.where(high.notna(), close)
        down_probe = low.where(low.notna(), close)
        for stock in close.index[valid_prev]:
            stock = str(stock)
            up_threshold = _limit_threshold_for_stock(stock, prices, trade_date, config, "up")
            down_threshold = _limit_threshold_for_stock(stock, prices, trade_date, config, "down")
            if pd.notna(up_probe.get(stock)) and float(up_probe.get(stock)) >= float(prev_close.get(stock)) * (1 + up_threshold):
                buyable.discard(stock)
            if pd.notna(down_probe.get(stock)) and float(down_probe.get(stock)) <= float(prev_close.get(stock)) * (1 - down_threshold):
                sellable.discard(stock)

    return {"priced": priced, "buyable": buyable, "sellable": sellable}


def _limit_threshold_for_stock(stock: str, prices: pd.DataFrame, trade_date: pd.Timestamp, config: dict, side: str) -> float:
    suffix = "up" if side == "up" else "down"
    if _is_st_on_date(stock, prices, trade_date):
        return float(config.get(f"st_limit_{suffix}_threshold", 0.049))
    lowered = str(stock).lower()
    if lowered.startswith(("688", "689")):
        return float(config.get(f"star_limit_{suffix}_threshold", config.get(f"growth_limit_{suffix}_threshold", 0.199)))
    if lowered.startswith(("300", "301")):
        return float(config.get(f"growth_limit_{suffix}_threshold", config.get(f"star_limit_{suffix}_threshold", 0.199)))
    if lowered.startswith(("8", "4")):
        return float(config.get(f"bj_limit_{suffix}_threshold", 0.299))
    return float(config.get(f"limit_{suffix}_threshold", 0.099))


def _is_st_on_date(stock: str, prices: pd.DataFrame, trade_date: pd.Timestamp) -> bool:
    is_st = _field_on_date(prices, "is_st", trade_date)
    if is_st.empty or stock not in is_st.index or pd.isna(is_st.loc[stock]):
        return False
    return bool(is_st.loc[stock])


def _portfolio_value(
    capital: float,
    holdings: dict[str, int],
    close: pd.Series | Mapping[str, float],
    last_prices: dict[str, float],
) -> float:
    total = capital
    for stock, shares in holdings.items():
        total += shares * _price_for(stock, close, last_prices)
    return float(total)


def _price_for(stock: str, close: pd.Series | Mapping[str, float], last_prices: dict[str, float]) -> float:
    if isinstance(close, Mapping):
        price = close.get(stock)
        if price is not None and pd.notna(price):
            return float(price)
        return float(last_prices.get(stock, 0.0))
    if stock in close.index and pd.notna(close.loc[stock]):
        return float(close.loc[stock])
    return float(last_prices.get(stock, 0.0))


def _row_price_dict(row: pd.Series | Mapping[str, float]) -> Mapping[str, float]:
    if isinstance(row, Mapping):
        return row
    if row.empty:
        return {}
    return row.dropna().to_dict()


def _target_values(
    total: float,
    target_holdings: list[str],
    scores: pd.Series,
    max_weight: float | None,
    exposure_scale: float,
    config: dict,
) -> dict[str, float]:
    if not target_holdings:
        return {}

    weighting = str(config.get("weighting_method", "")).strip().lower()
    score_weighted = bool(config.get("score_weighted", False)) or weighting in {"score", "score_weighted"}
    if score_weighted:
        weights = _score_weights(scores, target_holdings)
    else:
        weights = pd.Series(1.0 / len(target_holdings), index=target_holdings, dtype=float)

    if max_weight is not None:
        weights = weights.clip(upper=max_weight)
    scale = max(float(exposure_scale), 0.0)
    return {stock: float(total) * float(weight) * scale for stock, weight in weights.items()}


def _apply_rebalance_drift_threshold(
    desired_shares: dict[str, int],
    current_holdings: Mapping[str, int],
    target_holdings: list[str],
    target_values: Mapping[str, float],
    trade_prices: pd.Series | Mapping[str, float],
    last_prices: dict[str, float],
    total: float,
    threshold: float,
) -> dict[str, int]:
    if threshold <= 0 or total <= 0 or not desired_shares:
        return desired_shares

    allowed_drift = float(total) * max(float(threshold), 0.0)
    adjusted = dict(desired_shares)
    for stock in target_holdings:
        current_shares = int(current_holdings.get(stock, 0))
        if current_shares <= 0:
            continue
        price = _price_for(stock, trade_prices, last_prices)
        if price <= 0:
            continue
        current_value = current_shares * price
        target_value = float(target_values.get(stock, 0.0))
        if abs(current_value - target_value) <= allowed_drift:
            adjusted[stock] = current_shares
    return adjusted


def _score_weights(scores: pd.Series, target_holdings: list[str]) -> pd.Series:
    selected = pd.to_numeric(scores.reindex(target_holdings), errors="coerce").replace([np.inf, -np.inf], np.nan)
    if selected.notna().sum() == 0:
        return pd.Series(1.0 / len(target_holdings), index=target_holdings, dtype=float)

    values = selected.to_numpy(dtype=float)
    finite = np.isfinite(values)
    min_value = float(np.nanmin(values[finite])) if finite.any() else 0.0
    shifted = values - min_value if min_value <= 0 else values.copy()
    shifted = np.where(finite, shifted, 0.0)
    shifted = np.maximum(shifted, 0.0)
    if float(shifted.sum()) <= 0:
        return pd.Series(1.0 / len(target_holdings), index=target_holdings, dtype=float)
    return pd.Series(shifted / shifted.sum(), index=target_holdings, dtype=float)


def _round_lot(shares: float, stock: str, config: dict) -> int:
    lot_size = _lot_size(stock, config)
    return int(shares / lot_size) * lot_size


def _lot_size(stock: str, config: dict) -> int:
    lot_map = config.get("lot_size_map", {})
    if stock in lot_map:
        return int(lot_map[stock])
    code = str(stock).split(".", 1)[0].upper()
    for prefix, lot_size in lot_map.items():
        if code.startswith(str(prefix).upper()):
            return int(lot_size)
    if code.startswith(("688", "689")):
        return int(config.get("star_market_lot_size", 200))
    return int(config.get("lot_size", LOT_SIZE))


def _trade(
    signal_date: pd.Timestamp,
    trade_date: pd.Timestamp,
    stock: str,
    side: str,
    shares: int,
    price: float,
    cash: float,
    status: str = "filled",
    reason: str | None = None,
    commission_cost: float = 0.0,
    tax_cost: float = 0.0,
    transfer_fee_cost: float = 0.0,
    slippage_cost: float = 0.0,
    slippage_rate: float = 0.0,
    slippage_model: str = "fixed",
    capacity: dict[str, float | bool] | None = None,
) -> dict[str, Any]:
    row = {
        "signal_date": signal_date,
        "date": trade_date,
        "instrument": stock,
        "side": side,
        "shares": shares,
        "price": price,
        "cash": cash,
        "status": status,
        "reason": reason or "",
        "commission_cost": float(commission_cost),
        "tax_cost": float(tax_cost),
        "transfer_fee_cost": float(transfer_fee_cost),
        "slippage_cost": float(slippage_cost),
        "slippage_rate": float(slippage_rate),
        "slippage_model": slippage_model,
    }
    if reason is not None:
        row["reason"] = reason
    if capacity is not None:
        row.update(capacity)
    return row


def _blocked_trade(
    signal_date: pd.Timestamp,
    trade_date: pd.Timestamp,
    stock: str,
    side: str,
    shares: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "signal_date": signal_date,
        "date": trade_date,
        "instrument": stock,
        "side": side,
        "shares": shares,
        "price": np.nan,
        "cash": 0.0,
        "status": "blocked",
        "reason": reason,
        "commission_cost": 0.0,
        "tax_cost": 0.0,
        "transfer_fee_cost": 0.0,
        "slippage_cost": 0.0,
        "slippage_rate": 0.0,
        "slippage_model": "none",
    }


def _max_drawdown_duration(equity_curve: pd.Series) -> int:
    if equity_curve.empty:
        return 0
    running_max = equity_curve.cummax()
    underwater = equity_curve < running_max
    if not bool(underwater.any()):
        return 0
    groups = (~underwater).cumsum()
    return int(underwater.groupby(groups).size().max() - 1)


def _execute_risk_exits(
    holdings: dict[str, int],
    entry_prices: dict[str, float],
    capital: float,
    close: pd.Series | Mapping[str, float],
    last_prices: dict[str, float],
    tradability: dict[str, set[str]],
    trade_rows: list[dict[str, object]],
    trade_date: pd.Timestamp,
    commission: float,
    stamp_tax: float,
    transfer_fee: float,
    min_commission: float,
    slippage: float,
    prices: pd.DataFrame,
    config: dict,
) -> float:
    stop_loss = config.get("stop_loss_pct")
    take_profit = config.get("take_profit_pct")
    if stop_loss is None and take_profit is None:
        return capital

    open_prices = _row_price_dict(_field_on_date(prices, "open", trade_date))
    high_prices = _row_price_dict(_field_on_date(prices, "high", trade_date))
    low_prices = _row_price_dict(_field_on_date(prices, "low", trade_date))
    for stock, shares in list(holdings.items()):
        entry = entry_prices.get(stock)
        if entry is None or entry <= 0 or shares <= 0:
            continue
        reason, execution_price = _risk_exit_decision_from_rows(
            stock,
            entry,
            close,
            open_prices,
            high_prices,
            low_prices,
            last_prices,
            stop_loss,
            take_profit,
            config,
        )
        if reason is None:
            continue
        if stock not in tradability["sellable"]:
            trade_rows.append(_blocked_trade(pd.NaT, trade_date, stock, "SELL", shares, f"{reason}_not_sellable"))
            continue
        capital = _sell_all(
            holdings,
            entry_prices,
            capital,
            close,
            last_prices,
            trade_rows,
            trade_date,
            stock,
            commission,
            stamp_tax,
            transfer_fee,
            min_commission,
            slippage,
            prices,
            config,
            reason,
            execution_price=execution_price,
        )
    return capital


def _execute_stale_price_exits(
    holdings: dict[str, int],
    entry_prices: dict[str, float],
    stale_unpriced_days: dict[str, int],
    capital: float,
    close: pd.Series | Mapping[str, float],
    last_prices: dict[str, float],
    tradability: dict[str, set[str]],
    trade_rows: list[dict[str, object]],
    trade_date: pd.Timestamp,
    commission: float,
    stamp_tax: float,
    transfer_fee: float,
    min_commission: float,
    slippage: float,
    prices: pd.DataFrame,
    config: dict,
) -> float:
    threshold = int(config.get("stale_price_exit_days", 20))
    haircut = float(config.get("stale_price_haircut", 0.5))
    policy = str(config.get("stale_price_exit_policy", "haircut_exit")).strip().lower()
    if threshold <= 0 or not holdings:
        return capital

    close_values = _row_price_dict(close)
    volume_frame = _field(prices, "volume")
    volume_required = not volume_frame.empty
    volume_values = _row_price_dict(_field_on_date(prices, "volume", trade_date)) if volume_required else {}
    for stock, shares in list(holdings.items()):
        has_price = stock in close_values
        has_volume = (not volume_required) or volume_values.get(stock, 0.0) > 0
        if has_price and has_volume:
            stale_unpriced_days[stock] = 0
            continue
        stale_unpriced_days[stock] = stale_unpriced_days.get(stock, 0) + 1
        if stale_unpriced_days[stock] < threshold:
            continue

        if policy not in {"haircut_exit", "discount_exit", "sell"}:
            trade_rows.append(_blocked_trade(pd.NaT, trade_date, stock, "SELL", shares, f"stale_price_{policy}"))
            continue

        base_price = _price_for(stock, close, last_prices) * max(0.0, min(haircut, 1.0))
        capital = _sell_all(
            holdings,
            entry_prices,
            capital,
            close,
            last_prices,
            trade_rows,
            trade_date,
            stock,
            commission,
            stamp_tax,
            transfer_fee,
            min_commission,
            slippage,
            prices,
            config,
            "stale_price_exit",
            execution_price=base_price,
        )
        if stock not in holdings:
            stale_unpriced_days.pop(stock, None)
    return capital


def _liquidate_portfolio(
    holdings: dict[str, int],
    entry_prices: dict[str, float],
    capital: float,
    close: pd.Series | Mapping[str, float],
    last_prices: dict[str, float],
    tradability: dict[str, set[str]],
    trade_rows: list[dict[str, object]],
    trade_date: pd.Timestamp,
    commission: float,
    stamp_tax: float,
    transfer_fee: float,
    min_commission: float,
    slippage: float,
    prices: pd.DataFrame,
    config: dict,
    reason: str,
) -> float:
    for stock, shares in list(holdings.items()):
        if shares <= 0:
            continue
        if stock not in tradability["sellable"]:
            trade_rows.append(_blocked_trade(pd.NaT, trade_date, stock, "SELL", shares, f"{reason}_not_sellable"))
            continue
        capital = _sell_all(
            holdings,
            entry_prices,
            capital,
            close,
            last_prices,
            trade_rows,
            trade_date,
            stock,
            commission,
            stamp_tax,
            transfer_fee,
            min_commission,
            slippage,
            prices,
            config,
            reason,
        )
    return capital


def _reduce_portfolio_exposure(
    holdings: dict[str, int],
    entry_prices: dict[str, float],
    capital: float,
    close: pd.Series | Mapping[str, float],
    last_prices: dict[str, float],
    tradability: dict[str, set[str]],
    trade_rows: list[dict[str, object]],
    trade_date: pd.Timestamp,
    commission: float,
    stamp_tax: float,
    transfer_fee: float,
    min_commission: float,
    slippage: float,
    prices: pd.DataFrame,
    config: dict,
    target_exposure: float,
    reason: str,
) -> float:
    target_exposure = max(0.0, min(float(target_exposure), 1.0))
    if target_exposure <= 0:
        return _liquidate_portfolio(
            holdings,
            entry_prices,
            capital,
            close,
            last_prices,
            tradability,
            trade_rows,
            trade_date,
            commission,
            stamp_tax,
            transfer_fee,
            min_commission,
            slippage,
            prices,
            config,
            reason,
        )
    if target_exposure >= 1 or not holdings:
        return capital

    holding_values: dict[str, tuple[int, float, float]] = {}
    current_invested = 0.0
    for stock, shares in list(holdings.items()):
        if shares <= 0:
            continue
        price = _price_for(stock, close, last_prices)
        value = float(shares) * price
        if price <= 0 or value <= 0:
            continue
        holding_values[stock] = (shares, price, value)
        current_invested += value

    if current_invested <= 0:
        return capital
    total = capital + current_invested
    target_invested = total * target_exposure
    if current_invested <= target_invested:
        return capital

    sell_fraction = min((current_invested - target_invested) / current_invested, 1.0)
    for stock, (shares, _price, _value) in holding_values.items():
        sell_shares = min(_round_lot(shares * sell_fraction, stock, config), shares)
        if sell_shares <= 0:
            continue
        if stock not in tradability["sellable"]:
            trade_rows.append(_blocked_trade(pd.NaT, trade_date, stock, "SELL", sell_shares, f"{reason}_not_sellable"))
            continue
        capital = _sell_all(
            holdings,
            entry_prices,
            capital,
            close,
            last_prices,
            trade_rows,
            trade_date,
            stock,
            commission,
            stamp_tax,
            transfer_fee,
            min_commission,
            slippage,
            prices,
            config,
            reason,
            requested_shares=sell_shares,
        )
    return capital


def _sell_all(
    holdings: dict[str, int],
    entry_prices: dict[str, float],
    capital: float,
    close: pd.Series | Mapping[str, float],
    last_prices: dict[str, float],
    trade_rows: list[dict[str, object]],
    trade_date: pd.Timestamp,
    stock: str,
    commission: float,
    stamp_tax: float,
    transfer_fee: float,
    min_commission: float,
    slippage: float,
    prices: pd.DataFrame,
    config: dict,
    reason: str,
    execution_price: float | None = None,
    requested_shares: int | None = None,
) -> float:
    total_shares = holdings.get(stock, 0)
    shares = total_shares if requested_shares is None else min(max(int(requested_shares), 0), total_shares)
    if shares <= 0:
        return capital
    base_price = execution_price if execution_price is not None else _risk_exit_market_price(stock, close, last_prices, trade_date, prices, config)
    trade_slippage = _trade_slippage(slippage, prices, trade_date, stock, shares * float(base_price), config)
    price = float(base_price) * (1 - trade_slippage)
    filled_shares, status, capacity_reason = _apply_capacity_limit(shares, price, stock, prices, trade_date, config)
    if filled_shares <= 0:
        trade_rows.append(
            _blocked_trade(pd.NaT, trade_date, stock, "SELL", shares, f"{reason}_{capacity_reason or 'capacity_limited'}")
        )
        return capital

    gross = filled_shares * price
    commission_cost = _commission_cost(gross, commission, min_commission)
    tax_cost = gross * stamp_tax
    transfer_fee_cost = _transfer_fee_cost(gross, transfer_fee)
    proceeds = gross - commission_cost - tax_cost - transfer_fee_cost
    capital += proceeds
    remaining = total_shares - filled_shares
    if remaining > 0:
        holdings[stock] = remaining
    else:
        holdings.pop(stock, None)
        entry_prices.pop(stock, None)
    trade_rows.append(
        _trade(
            pd.NaT,
            trade_date,
            stock,
            "SELL",
            filled_shares,
            price,
            proceeds,
            status="risk_exit" if status == "filled" else status,
            reason=reason if capacity_reason is None else f"{reason}_{capacity_reason}",
            commission_cost=commission_cost,
            tax_cost=tax_cost,
            transfer_fee_cost=transfer_fee_cost,
            slippage_cost=filled_shares * float(base_price) * trade_slippage,
            slippage_rate=trade_slippage,
            slippage_model=_slippage_model_name(config),
            capacity=_capacity(prices, trade_date, stock, filled_shares * price, config),
        )
    )
    return capital


def _average_entry_price(old_entry: float | None, old_shares: int, price: float, buy_shares: int) -> float:
    if old_entry is None or old_shares <= 0:
        return float(price)
    return float((old_entry * old_shares + price * buy_shares) / (old_shares + buy_shares))


def _risk_exit_decision_from_rows(
    stock: str,
    entry: float,
    close: pd.Series | Mapping[str, float],
    open_prices: pd.Series | Mapping[str, float],
    high_prices: pd.Series | Mapping[str, float],
    low_prices: pd.Series | Mapping[str, float],
    last_prices: dict[str, float],
    stop_loss: float | None,
    take_profit: float | None,
    config: dict | None = None,
) -> tuple[str | None, float | None]:
    open_price = _price_for(stock, open_prices, last_prices)
    high_price = _price_for(stock, high_prices, last_prices)
    low_price = _price_for(stock, low_prices, last_prices)

    if stop_loss is not None:
        stop_price = entry * (1 - abs(float(stop_loss)))
        if low_price <= stop_price:
            execution = open_price if open_price <= stop_price else _stop_trigger_fill_price(stop_price, low_price, config)
            return "stop_loss", float(execution)

    if take_profit is not None:
        take_price = entry * (1 + abs(float(take_profit)))
        if high_price >= take_price:
            execution = open_price if open_price >= take_price else _take_profit_fill_price(take_price, high_price, config)
            return "take_profit", float(execution)

    close_price = _price_for(stock, close, last_prices)
    if stop_loss is not None and close_price / entry - 1 <= -abs(float(stop_loss)):
        return "stop_loss", float(close_price)
    if take_profit is not None and close_price / entry - 1 >= abs(float(take_profit)):
        return "take_profit", float(close_price)
    return None, None


def _stop_trigger_fill_price(stop_price: float, low_price: float, config: dict | None) -> float:
    cfg = config or {}
    policy = str(cfg.get("stop_fill_policy", "conservative")).strip().lower()
    if policy in {"stop", "stop_price", "ideal"}:
        return float(stop_price)
    buffer = max(float(cfg.get("stop_fill_buffer", 0.005)), 0.0)
    buffered = stop_price * (1 - buffer)
    return float(max(low_price, min(stop_price, buffered)))


def _take_profit_fill_price(take_price: float, high_price: float, config: dict | None) -> float:
    cfg = config or {}
    policy = str(cfg.get("take_profit_fill_policy", cfg.get("stop_fill_policy", "conservative"))).strip().lower()
    if policy in {"target", "take_price", "ideal", "stop_price"}:
        return float(take_price)
    buffer = max(float(cfg.get("take_profit_fill_buffer", 0.005)), 0.0)
    buffered = take_price * (1 - buffer)
    return float(min(high_price, max(0.0, buffered)))


def _risk_exit_market_price(
    stock: str,
    close: pd.Series | Mapping[str, float],
    last_prices: dict[str, float],
    trade_date: pd.Timestamp,
    prices: pd.DataFrame,
    config: dict,
) -> float:
    price_field = str(config.get("risk_exit_price_field", config.get("trade_price_field", "close"))).lower()
    price_row = _price_row(prices, price_field, trade_date)
    if not price_row.empty:
        return _price_for(stock, price_row, last_prices)
    return _price_for(stock, close, last_prices)


def _trade_slippage(base_slippage: float, prices: pd.DataFrame, trade_date: pd.Timestamp, stock: str, notional: float, config: dict) -> float:
    base = max(float(base_slippage), 0.0)
    if not bool(config.get("dynamic_slippage_enabled", False)):
        return base
    window = int(config.get("capacity_window", 20))
    amount_unit = float(config.get("amount_unit", 1000.0))
    adv = _prior_adv(prices, trade_date, stock, window, amount_unit)
    if adv <= 0:
        return base
    ratio = max(float(abs(notional) / adv), 0.0)
    threshold = max(float(config.get("dynamic_slippage_threshold", 0.02)), 0.0)
    multiplier = max(float(config.get("dynamic_slippage_multiplier", 2.0)), 0.0)
    cap = max(float(config.get("max_slippage", 0.03)), base)
    extra = max(ratio - threshold, 0.0) * multiplier
    return float(min(base + extra, cap))


def _slippage_model_name(config: dict) -> str:
    return "dynamic_adv" if bool(config.get("dynamic_slippage_enabled", False)) else "fixed"


def _capacity(prices: pd.DataFrame, trade_date: pd.Timestamp, stock: str, notional: float, config: dict) -> dict[str, float | bool]:
    window = int(config.get("capacity_window", 20))
    amount_unit = float(config.get("amount_unit", 1000.0))
    warn_threshold = float(config.get("capacity_warning_threshold", 0.05))
    adv = _prior_adv(prices, trade_date, stock, window, amount_unit)
    ratio = float(abs(notional) / adv) if adv > 0 else 0.0
    return {"capacity_ratio": ratio, "capacity_warning": bool(ratio > warn_threshold)}


def _apply_capacity_limit(
    requested_shares: int,
    price: float,
    stock: str,
    prices: pd.DataFrame,
    trade_date: pd.Timestamp,
    config: dict,
) -> tuple[int, str, str | None]:
    participation = config.get("max_participation_rate")
    if participation is None:
        return requested_shares, "filled", None
    participation = float(participation)
    if participation <= 0 or requested_shares <= 0 or price <= 0:
        return requested_shares, "filled", None

    amount_unit = float(config.get("amount_unit", 1000.0))
    window = int(config.get("capacity_window", 20))
    max_notional = _prior_adv(prices, trade_date, stock, window, amount_unit) * participation
    if not np.isfinite(max_notional) or max_notional <= 0:
        return 0, "blocked", "capacity_limited"

    max_shares = _round_lot(max_notional / price, stock, config)
    filled = min(requested_shares, max_shares)
    if filled < requested_shares:
        return filled, "partial" if filled > 0 else "blocked", "capacity_limited"
    return requested_shares, "filled", None


def _prior_adv(prices: pd.DataFrame, trade_date: pd.Timestamp, stock: str, window: int, amount_unit: float) -> float:
    amount = _field(prices, "amount")
    if amount.empty or stock not in amount.columns:
        return 0.0
    history = amount.loc[amount.index < trade_date, stock].dropna().tail(max(window, 1))
    if history.empty:
        return 0.0
    return float(history.mean() * amount_unit)


def _next_price_date(price_dates: pd.Index, signal_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Timestamp | None:
    pos = price_dates.searchsorted(signal_date, side="right")
    if pos >= len(price_dates):
        return None
    trade_date = pd.Timestamp(price_dates[pos])
    if trade_date > end_date:
        return None
    return trade_date
