from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.strategy import select_stocks


LOT_SIZE = 100


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

    capital = float(config.get("initial_capital", 1_000_000))
    commission = float(config.get("commission", 0.0003))
    stamp_tax = float(config.get("stamp_tax", 0.001))
    slippage = float(config.get("slippage", 0.0))
    top_n = int(config.get("top_n", 7))
    max_turnover = int(config.get("max_turnover", 1))
    rank_buffer = int(config.get("rank_buffer", 0))
    max_weight = config.get("max_weight_per_stock")
    max_weight = float(max_weight) if max_weight is not None else None

    holdings: dict[str, int] = {}
    last_prices: dict[str, float] = {}
    equity_rows: list[tuple[pd.Timestamp, float]] = []
    holding_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []

    previous_date: pd.Timestamp | None = None
    for trade_date in price_dates:
        close = _field_on_date(prices, "close", trade_date)
        last_prices.update({str(code): float(price) for code, price in close.dropna().items()})
        tradability = _tradability(prices, trade_date, previous_date, config)

        signal_date = trade_schedule.get(pd.Timestamp(trade_date))
        if signal_date is not None:
            daily_scores = score_panel.xs(signal_date, level=0, drop_level=True)
            daily_scores.index = daily_scores.index.astype(str)
            daily_scores = daily_scores[daily_scores.index.isin(tradability["priced"])]

            target_holdings = select_stocks(
                daily_scores,
                top_n=top_n,
                previous_holdings=holdings.keys(),
                max_turnover=max_turnover,
                rank_buffer=rank_buffer,
            )
            total_before_trade = _portfolio_value(capital, holdings, close, last_prices)
            target_value = _target_value(total_before_trade, target_holdings, max_weight)

            desired_shares = {}
            for stock in target_holdings:
                price = _price_for(stock, close, last_prices)
                desired_shares[stock] = _round_lot(target_value / price if price > 0 else 0, stock, config)

            for stock in list(holdings):
                current = holdings.get(stock, 0)
                desired = desired_shares.get(stock, 0)
                sell_shares = current - desired
                if sell_shares <= 0:
                    continue
                if stock not in tradability["sellable"]:
                    trade_rows.append(_blocked_trade(signal_date, trade_date, stock, "SELL", sell_shares, "not_sellable"))
                    continue
                price = _price_for(stock, close, last_prices) * (1 - slippage)
                proceeds = sell_shares * price * (1 - commission - stamp_tax)
                capital += proceeds
                remaining = current - sell_shares
                if remaining > 0:
                    holdings[stock] = remaining
                else:
                    holdings.pop(stock, None)
                trade_rows.append(_trade(signal_date, trade_date, stock, "SELL", sell_shares, price, proceeds))

            for stock in target_holdings:
                current = holdings.get(stock, 0)
                desired = desired_shares.get(stock, 0)
                buy_shares = desired - current
                if buy_shares <= 0:
                    continue
                if stock not in tradability["buyable"]:
                    trade_rows.append(_blocked_trade(signal_date, trade_date, stock, "BUY", buy_shares, "not_buyable"))
                    continue
                price = _price_for(stock, close, last_prices) * (1 + slippage)
                cost = buy_shares * price * (1 + commission)
                if cost > capital:
                    buy_shares = _round_lot(capital / (price * (1 + commission)), stock, config)
                    cost = buy_shares * price * (1 + commission)
                if buy_shares <= 0:
                    continue
                capital -= cost
                holdings[stock] = holdings.get(stock, 0) + buy_shares
                trade_rows.append(_trade(signal_date, trade_date, stock, "BUY", buy_shares, price, -cost))

        total = _portfolio_value(capital, holdings, close, last_prices)
        equity_rows.append((pd.Timestamp(trade_date), total))
        for stock, shares in holdings.items():
            price = _price_for(stock, close, last_prices)
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
        }

    returns = equity_curve.pct_change().dropna()
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1 if equity_curve.iloc[0] else 0.0
    periods = max(len(equity_curve) - 1, 1)
    annual_return = (1 + total_return) ** (annual_days / periods) - 1
    excess_daily = returns - risk_free_rate / annual_days
    volatility = returns.std(ddof=1) * np.sqrt(annual_days) if len(returns) > 1 else 0.0
    sharpe = excess_daily.mean() / returns.std(ddof=1) * np.sqrt(annual_days) if len(returns) > 1 and returns.std(ddof=1) else 0.0
    downside = excess_daily[excess_daily < 0].std(ddof=1)
    sortino = excess_daily.mean() / downside * np.sqrt(annual_days) if len(excess_daily) > 1 and downside else 0.0
    drawdown = equity_curve / equity_curve.cummax() - 1
    max_drawdown = float(drawdown.min())
    max_dd_duration = float(_max_drawdown_duration(equity_curve))
    calmar = float(annual_return / abs(max_drawdown)) if max_drawdown < 0 else 0.0
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = float(len(wins) / len(returns)) if len(returns) else 0.0
    profit_loss_ratio = float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else 0.0
    sells = int((trades.get("side") == "SELL").sum()) if not trades.empty else 0
    annual_turnover = float(sells / max(periods / annual_days, 1 / annual_days) / top_n * 2)

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
    }
    benchmark = config.get("benchmark_curve")
    if isinstance(benchmark, pd.Series):
        metrics.update(calculate_benchmark_metrics(equity_curve, benchmark, config))
    return metrics


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
    return score_panel.sort_index()


def _normalize_price_frame(price_df: pd.DataFrame) -> pd.DataFrame:
    prices = price_df.copy()
    prices.index = pd.to_datetime(prices.index)
    if isinstance(prices.columns, pd.MultiIndex):
        if prices.columns.nlevels != 2:
            raise ValueError("price_df MultiIndex columns must be field/instrument.")
        prices.columns = pd.MultiIndex.from_arrays(
            [
                prices.columns.get_level_values(0).astype(str).str.lower(),
                prices.columns.get_level_values(1).astype(str),
            ],
            names=["field", "instrument"],
        )
        return prices.sort_index()

    prices.columns = pd.MultiIndex.from_product([["close"], prices.columns.astype(str)], names=["field", "instrument"])
    return prices.sort_index()


def _field(prices: pd.DataFrame, field: str) -> pd.DataFrame:
    if field not in prices.columns.get_level_values("field"):
        return pd.DataFrame(index=prices.index)
    return prices.xs(field, level="field", axis=1)


def _field_on_date(prices: pd.DataFrame, field: str, date: pd.Timestamp) -> pd.Series:
    frame = _field(prices, field)
    if frame.empty or date not in frame.index:
        return pd.Series(dtype=float)
    row = frame.loc[date]
    row.index = row.index.astype(str)
    return row.astype(float)


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
        up_threshold = float(config.get("limit_up_threshold", 0.099))
        down_threshold = float(config.get("limit_down_threshold", 0.099))
        limit_up = close[valid_prev] >= prev_close[valid_prev] * (1 + up_threshold)
        limit_down = close[valid_prev] <= prev_close[valid_prev] * (1 - down_threshold)
        buyable -= set(limit_up[limit_up].index.astype(str))
        sellable -= set(limit_down[limit_down].index.astype(str))

    return {"priced": priced, "buyable": buyable, "sellable": sellable}


def _portfolio_value(capital: float, holdings: dict[str, int], close: pd.Series, last_prices: dict[str, float]) -> float:
    total = capital
    for stock, shares in holdings.items():
        total += shares * _price_for(stock, close, last_prices)
    return float(total)


def _price_for(stock: str, close: pd.Series, last_prices: dict[str, float]) -> float:
    if stock in close.index and pd.notna(close.loc[stock]):
        return float(close.loc[stock])
    return float(last_prices.get(stock, 0.0))


def _target_value(total: float, target_holdings: list[str], max_weight: float | None) -> float:
    if not target_holdings:
        return 0.0
    equal_weight = 1 / len(target_holdings)
    weight = min(equal_weight, max_weight) if max_weight is not None else equal_weight
    return total * weight


def _round_lot(shares: float, stock: str, config: dict) -> int:
    lot_size = _lot_size(stock, config)
    return int(shares / lot_size) * lot_size


def _lot_size(stock: str, config: dict) -> int:
    lot_map = config.get("lot_size_map", {})
    if stock in lot_map:
        return int(lot_map[stock])
    if str(stock).lower().startswith(("688", "689")):
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
) -> dict[str, Any]:
    return {
        "signal_date": signal_date,
        "date": trade_date,
        "instrument": stock,
        "side": side,
        "shares": shares,
        "price": price,
        "cash": cash,
        "status": "filled",
    }


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
    }


def _max_drawdown_duration(equity_curve: pd.Series) -> int:
    running_max = equity_curve.cummax()
    underwater = equity_curve < running_max
    longest = current = 0
    for is_underwater in underwater:
        current = current + 1 if is_underwater else 0
        longest = max(longest, current)
    return longest


def _next_price_date(price_dates: pd.Index, signal_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Timestamp | None:
    pos = price_dates.searchsorted(signal_date, side="right")
    if pos >= len(price_dates):
        return None
    trade_date = pd.Timestamp(price_dates[pos])
    if trade_date > end_date:
        return None
    return trade_date
