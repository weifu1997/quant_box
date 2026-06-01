from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.strategy import select_stocks


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
    prices = price_df.copy()
    prices.index = pd.to_datetime(prices.index)
    prices.columns = prices.columns.astype(str)

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    dates = sorted(pd.to_datetime(score_panel.index.get_level_values(0).unique()))
    signal_dates = [date for date in dates if start <= date <= end]
    price_dates = pd.Index(sorted(prices.index))

    capital = float(config.get("initial_capital", 1_000_000))
    commission = float(config.get("commission", 0.0003))
    stamp_tax = float(config.get("stamp_tax", 0.001))
    top_n = int(config.get("top_n", 7))
    max_turnover = int(config.get("max_turnover", 1))
    rank_buffer = int(config.get("rank_buffer", 0))

    holdings: dict[str, int] = {}
    equity_rows: list[tuple[pd.Timestamp, float]] = []
    holding_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []

    for signal_date in signal_dates:
        trade_date = _next_price_date(price_dates, signal_date, end)
        if trade_date is None:
            break

        daily_scores = score_panel.xs(signal_date, level=0, drop_level=True)
        tradable = _available_prices(prices.loc[trade_date])
        daily_scores = daily_scores[daily_scores.index.astype(str).isin(tradable.index)]

        target_holdings = select_stocks(
            daily_scores,
            top_n=top_n,
            previous_holdings=holdings.keys(),
            max_turnover=max_turnover,
            rank_buffer=rank_buffer,
        )
        target_holdings = [stock for stock in target_holdings if stock in tradable.index]

        total_before_trade = capital
        for stock, shares in holdings.items():
            if stock in tradable.index:
                total_before_trade += shares * float(tradable.loc[stock])
        target_value = total_before_trade / len(target_holdings) if target_holdings else 0

        desired_shares = {}
        for stock in target_holdings:
            price = float(tradable.loc[stock])
            desired_shares[stock] = int(target_value / price / 100) * 100 if price > 0 else 0

        for stock in list(holdings):
            if stock not in tradable.index:
                continue
            current = holdings.get(stock, 0)
            desired = desired_shares.get(stock, 0)
            sell_shares = current - desired
            if sell_shares <= 0:
                continue
            price = float(tradable.loc[stock])
            proceeds = sell_shares * price * (1 - commission - stamp_tax)
            capital += proceeds
            remaining = current - sell_shares
            if remaining > 0:
                holdings[stock] = remaining
            else:
                holdings.pop(stock, None)
            trade_rows.append(
                {
                    "signal_date": signal_date,
                    "date": trade_date,
                    "instrument": stock,
                    "side": "SELL",
                    "shares": sell_shares,
                    "price": price,
                    "cash": proceeds,
                }
            )

        for stock in target_holdings:
            current = holdings.get(stock, 0)
            desired = desired_shares.get(stock, 0)
            buy_shares = desired - current
            if buy_shares <= 0:
                continue
            price = float(tradable.loc[stock])
            cost = buy_shares * price * (1 + commission)
            if cost > capital:
                buy_shares = int(capital / (price * (1 + commission)) / 100) * 100
                cost = buy_shares * price * (1 + commission)
            if buy_shares <= 0:
                continue
            capital -= cost
            holdings[stock] = holdings.get(stock, 0) + buy_shares
            trade_rows.append(
                {
                    "signal_date": signal_date,
                    "date": trade_date,
                    "instrument": stock,
                    "side": "BUY",
                    "shares": buy_shares,
                    "price": price,
                    "cash": -cost,
                }
            )

        total = capital
        for stock, shares in holdings.items():
            if stock in tradable.index:
                value = shares * float(tradable.loc[stock])
                total += value
                holding_rows.append(
                    {
                        "signal_date": signal_date,
                        "date": trade_date,
                        "instrument": stock,
                        "shares": shares,
                        "price": float(tradable.loc[stock]),
                        "value": value,
                    }
                )
        equity_rows.append((trade_date, total))

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
    if equity_curve.empty:
        return {"total_return": 0.0, "annual_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "turnover_count": 0.0}
    returns = equity_curve.pct_change().dropna()
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1 if equity_curve.iloc[0] else 0.0
    elapsed_days = max((equity_curve.index[-1] - equity_curve.index[0]).days, 1)
    elapsed_years = elapsed_days / 365.25
    annual_return = (1 + total_return) ** (1 / elapsed_years) - 1
    periods_per_year = len(returns) / elapsed_years if elapsed_years > 0 else 0
    sharpe = (
        returns.mean() / returns.std(ddof=0) * np.sqrt(periods_per_year)
        if len(returns) > 1 and returns.std(ddof=0) and periods_per_year
        else 0.0
    )
    drawdown = equity_curve / equity_curve.cummax() - 1
    sells = int((trades.get("side") == "SELL").sum()) if not trades.empty else 0
    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "turnover_count": float(sells),
    }


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


def _available_prices(row: pd.Series) -> pd.Series:
    return row.dropna().astype(float)


def _next_price_date(price_dates: pd.Index, signal_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Timestamp | None:
    pos = price_dates.searchsorted(signal_date, side="right")
    if pos >= len(price_dates):
        return None
    trade_date = pd.Timestamp(price_dates[pos])
    if trade_date > end_date:
        return None
    return trade_date
