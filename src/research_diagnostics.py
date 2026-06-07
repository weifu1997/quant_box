from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config_loader import resolve_path
from src.market_regime import detect_reporting_regime


def build_research_diagnostics(
    equity_curve: pd.Series | pd.DataFrame,
    holdings: pd.DataFrame,
    trades: pd.DataFrame,
    price_df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Build research diagnostics for a completed backtest.

    The outputs intentionally favor point-in-time evidence over optimistic
    summaries: benchmark-relative returns, explicit cost drag, holdings-based
    return attribution, and current exposure concentration.
    """
    equity = _equity_series(equity_curve)
    if equity.empty:
        return {"enabled": False, "issues": ["empty_equity_curve"]}, {}

    annual_days = int(config.get("backtest", {}).get("annual_trading_days", 252))
    benchmark = _benchmark_curve(price_df, config, equity.index)
    benchmark_summary = _benchmark_comparison(equity, benchmark, annual_days)
    drawdown_summary, drawdown_tables = _drawdown_diagnostics(equity, benchmark, trades)
    regime_summary, regime_tables = _regime_return_diagnostics(equity, benchmark, price_df, config, annual_days)
    regime_trade_summary, regime_trade_tables = _regime_trade_diagnostics(trades, price_df, config)
    cost_summary = _cost_attribution(trades, equity)
    turnover_summary, turnover_tables = _turnover_attribution(trades, holdings, equity, config, annual_days)
    attribution_summary, attribution_tables = _holding_return_attribution(holdings, price_df, config, drawdown_summary)
    exposure_summary, exposure_tables = _exposure_diagnostics(holdings, config)

    diagnostics = {
        "enabled": True,
        "benchmark": benchmark_summary,
        "drawdown": drawdown_summary,
        "regime_returns": regime_summary,
        "regime_trades": regime_trade_summary,
        "cost_attribution": cost_summary,
        "turnover_attribution": turnover_summary,
        "holding_attribution": attribution_summary,
        "exposure": exposure_summary,
        "issues": [
            *benchmark_summary.get("issues", []),
            *regime_summary.get("issues", []),
            *regime_trade_summary.get("issues", []),
            *attribution_summary.get("issues", []),
            *exposure_summary.get("issues", []),
        ],
    }
    tables: dict[str, pd.DataFrame] = {}
    if benchmark is not None and not benchmark.empty:
        aligned = pd.concat(
            [
                equity.rename("strategy_equity"),
                benchmark.reindex(equity.index).rename("benchmark_equity"),
            ],
            axis=1,
        )
        tables["benchmark_curve"] = aligned.reset_index().rename(columns={"index": "date"})
    tables.update(drawdown_tables)
    tables.update(regime_tables)
    tables.update(regime_trade_tables)
    tables.update(turnover_tables)
    tables.update(attribution_tables)
    tables.update(exposure_tables)
    return diagnostics, tables


def write_research_diagnostics(
    diagnostics: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    out_dir: str | Path,
    prefix: str = "auto_research",
) -> dict[str, str]:
    output_dir = resolve_path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    json_path = output_dir / f"{prefix}_diagnostics.json"
    json_path.write_text(json.dumps(_json_safe(diagnostics), indent=2, ensure_ascii=False), encoding="utf-8")
    paths["research_diagnostics"] = str(json_path)
    for name, table in tables.items():
        if table.empty:
            continue
        path = output_dir / f"{prefix}_{name}.csv"
        table.to_csv(path, index=False, encoding="utf-8-sig")
        paths[f"research_{name}"] = str(path)
    return paths


def _equity_series(equity_curve: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(equity_curve, pd.DataFrame):
        if equity_curve.empty:
            return pd.Series(dtype=float, name="equity")
        if "equity" in equity_curve.columns:
            series = equity_curve["equity"]
        else:
            series = equity_curve.iloc[:, 0]
    else:
        series = equity_curve
    result = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    result.index = pd.to_datetime(result.index).normalize()
    result = result[~result.index.duplicated(keep="last")].sort_index()
    return result.rename("equity")


def _benchmark_curve(price_df: pd.DataFrame, config: dict[str, Any], target_index: pd.Index) -> pd.Series | None:
    research_cfg = config.get("research", {})
    benchmark_cfg = research_cfg.get("benchmark", {})
    benchmark_file = benchmark_cfg.get("file") or config.get("market_regime", {}).get("benchmark_file")
    if benchmark_file:
        path = resolve_path(benchmark_file)
        if path.exists():
            return _read_benchmark_series(path, target_index)

    close = _price_field(price_df, "close")
    if close.empty:
        return None

    symbol = benchmark_cfg.get("symbol") or config.get("market_regime", {}).get("benchmark_symbol")
    if symbol:
        normalized_symbol = _normalize_instrument(symbol)
        if normalized_symbol in close.columns:
            return _normalize_curve(close[normalized_symbol], target_index)

    method = str(benchmark_cfg.get("method", "equal_weight_universe")).lower()
    if method in {"hs300_equal_weight", "csi300_equal_weight"}:
        hs300_file = resolve_path(config.get("data", {}).get("hs300_constituents_file", "data/raw/hs300_constituents.csv"))
        if hs300_file.exists():
            frame = pd.read_csv(hs300_file)
            code_col = "con_code" if "con_code" in frame.columns else "ts_code" if "ts_code" in frame.columns else None
            if code_col is not None:
                symbols = list(dict.fromkeys(_normalize_instrument(value) for value in frame[code_col].dropna().astype(str)))
                selected = [symbol for symbol in symbols if symbol in close.columns]
                if selected:
                    close = close[selected]

    returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    benchmark_returns = returns.mean(axis=1, skipna=True)
    if benchmark_returns.dropna().empty and close.dropna(how="all").empty:
        return None
    benchmark_returns = benchmark_returns.fillna(0.0)
    curve = (1.0 + benchmark_returns).cumprod()
    return _normalize_curve(curve, target_index)


def _read_benchmark_series(path: Path, target_index: pd.Index) -> pd.Series:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        date_col = next((col for col in ["date", "trade_date", "datetime"] if col in frame.columns), frame.columns[0])
        value_col = next((col for col in ["close", "benchmark", "equity", "value"] if col in frame.columns), frame.columns[-1])
        series = pd.Series(pd.to_numeric(frame[value_col], errors="coerce").to_numpy(), index=pd.to_datetime(frame[date_col], errors="coerce"))
    else:
        loaded = pd.read_parquet(path)
        if isinstance(loaded, pd.Series):
            series = loaded
        else:
            value_col = "close" if "close" in loaded.columns else loaded.columns[0]
            series = loaded[value_col]
    return _normalize_curve(series, target_index)


def _normalize_curve(series: pd.Series, target_index: pd.Index) -> pd.Series:
    result = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    result.index = pd.to_datetime(result.index).normalize()
    result = result[~result.index.duplicated(keep="last")].sort_index()
    target_dates = pd.DatetimeIndex(pd.to_datetime(target_index).normalize())
    result = result.reindex(target_dates).ffill().dropna()
    if result.empty:
        return result.rename("benchmark")
    first = float(result.iloc[0])
    if first != 0:
        result = result / first
    return result.rename("benchmark")


def _benchmark_comparison(equity: pd.Series, benchmark: pd.Series | None, annual_days: int) -> dict[str, Any]:
    strategy_returns = equity.pct_change(fill_method=None).dropna()
    summary: dict[str, Any] = {
        "strategy_total_return": _total_return(equity),
        "strategy_annual_return": _annual_return(equity, annual_days),
        "strategy_max_drawdown": _max_drawdown(equity),
        "strategy_sharpe": _sharpe(strategy_returns, annual_days),
        "issues": [],
    }
    if benchmark is None or benchmark.empty:
        summary["issues"].append("benchmark_unavailable")
        return summary

    aligned = pd.concat([equity.rename("strategy"), benchmark.rename("benchmark")], axis=1).dropna()
    if len(aligned) < 3:
        summary["issues"].append("benchmark_overlap_too_short")
        return summary
    strategy = aligned["strategy"]
    bench = aligned["benchmark"]
    strategy_ret = strategy.pct_change(fill_method=None).dropna()
    bench_ret = bench.pct_change(fill_method=None).dropna()
    active = strategy_ret.sub(bench_ret, fill_value=0.0).dropna()
    beta = _beta(strategy_ret, bench_ret)
    summary.update(
        {
            "benchmark_total_return": _total_return(bench),
            "benchmark_annual_return": _annual_return(bench, annual_days),
            "benchmark_max_drawdown": _max_drawdown(bench),
            "active_total_return": _total_return(strategy) - _total_return(bench),
            "active_annual_return": _annual_return(strategy, annual_days) - _annual_return(bench, annual_days),
            "tracking_error": float(active.std(ddof=0) * np.sqrt(annual_days)) if len(active) else None,
            "information_ratio": _information_ratio(active, annual_days),
            "beta": beta,
            "correlation": float(strategy_ret.corr(bench_ret)) if len(strategy_ret) and len(bench_ret) else None,
        }
    )
    return summary


def _drawdown_diagnostics(
    equity: pd.Series,
    benchmark: pd.Series | None,
    trades: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    if equity.empty:
        return {"enabled": False, "issues": ["empty_equity_curve"]}, {}
    episodes = _drawdown_episodes(equity, benchmark, trades)
    if episodes.empty:
        return {
            "enabled": True,
            "max_drawdown": 0.0,
            "max_drawdown_peak_date": None,
            "max_drawdown_trough_date": None,
            "max_drawdown_recovery_date": None,
            "top_drawdowns": [],
            "issues": [],
        }, {"drawdown_periods": episodes}

    worst = episodes.sort_values("max_drawdown").iloc[0]
    summary = {
        "enabled": True,
        "max_drawdown": float(worst["max_drawdown"]),
        "max_drawdown_peak_date": _date_text(worst["peak_date"]),
        "max_drawdown_start_date": _date_text(worst["start_date"]),
        "max_drawdown_trough_date": _date_text(worst["trough_date"]),
        "max_drawdown_recovery_date": _date_text(worst["recovery_date"]),
        "max_drawdown_days_to_trough": int(worst["days_to_trough"]),
        "max_drawdown_days_to_recovery": _optional_int(worst["days_to_recovery"]),
        "max_drawdown_benchmark_return_peak_to_trough": _optional_float(worst["benchmark_return_peak_to_trough"]),
        "trades_peak_to_trough": int(worst["trades_peak_to_trough"]),
        "risk_exit_trades_peak_to_trough": int(worst["risk_exit_trades_peak_to_trough"]),
        "blocked_trades_peak_to_trough": int(worst["blocked_trades_peak_to_trough"]),
        "top_drawdowns": episodes.sort_values("max_drawdown").head(5).to_dict(orient="records"),
        "issues": [],
    }
    tables = {"drawdown_periods": episodes}
    tables.update(_max_drawdown_trade_tables(trades, worst["peak_date"], worst["trough_date"]))
    return summary, tables


def _max_drawdown_trade_tables(trades: pd.DataFrame, peak_date: object, trough_date: object) -> dict[str, pd.DataFrame]:
    if trades.empty or "date" not in trades.columns or pd.isna(peak_date) or pd.isna(trough_date):
        return {}
    start = pd.Timestamp(peak_date).normalize()
    end = pd.Timestamp(trough_date).normalize()
    frame = trades.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame[(frame["date"] >= start) & (frame["date"] <= end)].copy()
    if frame.empty:
        return {}

    frame["instrument"] = _column_or_default(frame, "instrument", "").map(_normalize_instrument)
    frame["side"] = _normalize_trade_side(_column_or_default(frame, "side", ""))
    frame["status"] = _normalize_trade_status(_column_or_default(frame, "status", ""))
    frame["reason"] = _normalize_trade_reason(_column_or_default(frame, "reason", "")).replace("", "rebalance")
    frame["shares"] = pd.to_numeric(_column_or_default(frame, "shares", 0.0), errors="coerce").fillna(0.0)
    frame["price"] = pd.to_numeric(_column_or_default(frame, "price", 0.0), errors="coerce").fillna(0.0)
    frame["cash"] = pd.to_numeric(_column_or_default(frame, "cash", 0.0), errors="coerce").fillna(0.0)
    frame["notional"] = frame["cash"].abs()
    fallback_notional = (frame["shares"] * frame["price"]).abs()
    frame.loc[frame["notional"] <= 0, "notional"] = fallback_notional.loc[frame["notional"] <= 0]
    cost_columns = ["commission_cost", "tax_cost", "transfer_fee_cost", "slippage_cost"]
    for column in cost_columns:
        frame[column] = pd.to_numeric(_column_or_default(frame, column, 0.0), errors="coerce").fillna(0.0)
    frame["trade_cost"] = frame[cost_columns].sum(axis=1)

    by_status_reason = (
        frame.groupby(["side", "status", "reason"], as_index=False)
        .agg(trade_count=("date", "size"), notional=("notional", "sum"), trade_cost=("trade_cost", "sum"))
        .sort_values(["trade_cost", "notional"], ascending=[False, False])
    )
    by_instrument = (
        frame.groupby(["instrument", "side", "status"], as_index=False)
        .agg(trade_count=("date", "size"), notional=("notional", "sum"), trade_cost=("trade_cost", "sum"))
        .sort_values(["trade_cost", "notional"], ascending=[False, False])
    )
    return {
        "max_drawdown_trades": frame,
        "max_drawdown_trade_costs_by_status_reason": by_status_reason,
        "max_drawdown_trade_costs_by_instrument": by_instrument,
    }


def _drawdown_episodes(equity: pd.Series, benchmark: pd.Series | None, trades: pd.DataFrame) -> pd.DataFrame:
    series = equity.sort_index().astype(float)
    running_peak = series.cummax()
    drawdown = series / running_peak - 1.0
    peak_dates = pd.Series(index=series.index, dtype="datetime64[ns]")
    last_peak = series.index[0]
    for date, value in series.items():
        if value >= running_peak.loc[date]:
            last_peak = date
        peak_dates.loc[date] = last_peak

    rows: list[dict[str, Any]] = []
    in_drawdown = False
    segment_start: pd.Timestamp | None = None
    drawdown_dates = list(series.index)
    for date in drawdown_dates:
        is_down = bool(drawdown.loc[date] < 0)
        if is_down and not in_drawdown:
            segment_start = pd.Timestamp(date)
            in_drawdown = True
        recovered = in_drawdown and not is_down
        if recovered and segment_start is not None:
            rows.append(_drawdown_episode_row(series, drawdown, peak_dates, benchmark, trades, segment_start, pd.Timestamp(date)))
            in_drawdown = False
            segment_start = None
    if in_drawdown and segment_start is not None:
        rows.append(_drawdown_episode_row(series, drawdown, peak_dates, benchmark, trades, segment_start, None))
    return pd.DataFrame(rows)


def _drawdown_episode_row(
    equity: pd.Series,
    drawdown: pd.Series,
    peak_dates: pd.Series,
    benchmark: pd.Series | None,
    trades: pd.DataFrame,
    start_date: pd.Timestamp,
    recovery_date: pd.Timestamp | None,
) -> dict[str, Any]:
    end_date = recovery_date if recovery_date is not None else pd.Timestamp(equity.index[-1])
    period_drawdown = drawdown.loc[(drawdown.index >= start_date) & (drawdown.index <= end_date)]
    trough_date = pd.Timestamp(period_drawdown.idxmin())
    peak_date = pd.Timestamp(peak_dates.loc[start_date])
    peak_equity = float(equity.loc[peak_date])
    trough_equity = float(equity.loc[trough_date])
    trade_counts = _trade_counts_between(trades, peak_date, trough_date)
    return {
        "peak_date": peak_date,
        "start_date": start_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
        "max_drawdown": float(period_drawdown.min()),
        "peak_equity": peak_equity,
        "trough_equity": trough_equity,
        "recovery_equity": float(equity.loc[recovery_date]) if recovery_date is not None else np.nan,
        "days_to_trough": int((trough_date - peak_date).days),
        "days_to_recovery": int((recovery_date - peak_date).days) if recovery_date is not None else np.nan,
        "benchmark_return_peak_to_trough": _benchmark_return_between(benchmark, peak_date, trough_date),
        **trade_counts,
    }


def _trade_counts_between(trades: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> dict[str, int]:
    if trades.empty or "date" not in trades.columns:
        return {
            "trades_peak_to_trough": 0,
            "buy_trades_peak_to_trough": 0,
            "sell_trades_peak_to_trough": 0,
            "risk_exit_trades_peak_to_trough": 0,
            "blocked_trades_peak_to_trough": 0,
        }
    frame = trades.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame[(frame["date"] >= pd.Timestamp(start_date).normalize()) & (frame["date"] <= pd.Timestamp(end_date).normalize())]
    side = _normalize_trade_side(frame.get("side", pd.Series(dtype=str)))
    status = _normalize_trade_status(frame.get("status", pd.Series(dtype=str)))
    return {
        "trades_peak_to_trough": int(len(frame)),
        "buy_trades_peak_to_trough": int((side == "BUY").sum()),
        "sell_trades_peak_to_trough": int((side == "SELL").sum()),
        "risk_exit_trades_peak_to_trough": int((status == "risk_exit").sum()),
        "blocked_trades_peak_to_trough": int((status == "blocked").sum()),
    }


def _benchmark_return_between(benchmark: pd.Series | None, start_date: pd.Timestamp, end_date: pd.Timestamp) -> float | None:
    if benchmark is None or benchmark.empty:
        return None
    series = benchmark.sort_index().astype(float)
    window = series[(series.index >= pd.Timestamp(start_date).normalize()) & (series.index <= pd.Timestamp(end_date).normalize())]
    if len(window) < 2 or float(window.iloc[0]) == 0:
        return None
    return float(window.iloc[-1] / window.iloc[0] - 1.0)


def _date_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(pd.Timestamp(value).date())


def _optional_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _regime_return_diagnostics(
    equity: pd.Series,
    benchmark: pd.Series | None,
    price_df: pd.DataFrame,
    config: dict[str, Any],
    annual_days: int,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    if equity.empty or price_df.empty:
        return {"enabled": False, "issues": ["regime_returns_unavailable"]}, {}
    regimes = detect_reporting_regime(price_df, config)
    if regimes.empty:
        return {"enabled": False, "issues": ["regime_labels_unavailable"]}, {}

    strategy_returns = equity.sort_index().pct_change(fill_method=None).dropna().rename("strategy_return")
    if strategy_returns.empty:
        return {"enabled": False, "issues": ["strategy_returns_unavailable"]}, {}
    frame = pd.DataFrame({"strategy_return": strategy_returns})
    frame["regime"] = regimes.reindex(frame.index, method="ffill")
    frame = frame.dropna(subset=["regime"])
    if frame.empty:
        return {"enabled": False, "issues": ["regime_return_overlap_empty"]}, {}

    if benchmark is not None and not benchmark.empty:
        benchmark_returns = benchmark.sort_index().pct_change(fill_method=None).rename("benchmark_return")
        frame["benchmark_return"] = benchmark_returns.reindex(frame.index)
    else:
        frame["benchmark_return"] = np.nan
    frame["active_return"] = frame["strategy_return"] - frame["benchmark_return"]

    rows: list[dict[str, Any]] = []
    for regime, group in frame.groupby("regime", sort=False):
        strategy = pd.to_numeric(group["strategy_return"], errors="coerce").dropna()
        benchmark_group = pd.to_numeric(group["benchmark_return"], errors="coerce").dropna()
        active = pd.to_numeric(group["active_return"], errors="coerce").dropna()
        row = {
            "regime": str(regime),
            "days": int(len(group)),
            "strategy_total_return": _compound_returns(strategy),
            "strategy_annual_return": _annualized_return_from_returns(strategy, annual_days),
            "strategy_max_drawdown": _max_drawdown_from_returns(strategy),
            "strategy_hit_rate": float((strategy > 0).mean()) if len(strategy) else None,
            "benchmark_total_return": _compound_returns(benchmark_group),
            "benchmark_annual_return": _annualized_return_from_returns(benchmark_group, annual_days),
            "active_total_return": _compound_returns(active),
            "active_mean_daily_return": float(active.mean()) if len(active) else None,
        }
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return {"enabled": False, "issues": ["regime_return_table_empty"]}, {}

    worst_active = table.dropna(subset=["active_mean_daily_return"]).sort_values("active_mean_daily_return").head(1)
    worst_drawdown = table.dropna(subset=["strategy_max_drawdown"]).sort_values("strategy_max_drawdown").head(1)
    summary = {
        "enabled": True,
        "regime_count": int(table["regime"].nunique()),
        "worst_active_regime": str(worst_active.iloc[0]["regime"]) if not worst_active.empty else None,
        "worst_drawdown_regime": str(worst_drawdown.iloc[0]["regime"]) if not worst_drawdown.empty else None,
        "records": table.to_dict(orient="records"),
        "issues": [],
    }
    daily = frame.reset_index().rename(columns={"index": "date"})
    return summary, {"regime_returns": table, "regime_daily_returns": daily}


def _compound_returns(returns: pd.Series) -> float | None:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return None
    return float((1.0 + clean).prod() - 1.0)


def _annualized_return_from_returns(returns: pd.Series, annual_days: int) -> float | None:
    total = _compound_returns(returns)
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if total is None or clean.empty:
        return None
    return float((1.0 + total) ** (annual_days / max(len(clean), 1)) - 1.0) if total > -1 else -1.0


def _max_drawdown_from_returns(returns: pd.Series) -> float | None:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return None
    wealth = np.concatenate([[1.0], (1.0 + clean).cumprod().to_numpy(dtype=float)])
    running_peak = np.maximum.accumulate(wealth)
    return float((wealth / running_peak - 1.0).min())


def _regime_trade_diagnostics(
    trades: pd.DataFrame,
    price_df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    if trades.empty or "date" not in trades.columns or price_df.empty:
        return {"enabled": False, "issues": ["regime_trade_costs_unavailable"]}, {}
    regimes = detect_reporting_regime(price_df, config)
    if regimes.empty:
        return {"enabled": False, "issues": ["regime_trade_labels_unavailable"]}, {}

    frame = trades.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["date"])
    if frame.empty:
        return {"enabled": False, "issues": ["regime_trade_dates_unavailable"]}, {}
    frame["regime"] = regimes.reindex(pd.DatetimeIndex(frame["date"]), method="ffill").to_numpy()
    frame = frame.dropna(subset=["regime"])
    if frame.empty:
        return {"enabled": False, "issues": ["regime_trade_overlap_empty"]}, {}

    frame["side"] = _normalize_trade_side(_column_or_default(frame, "side", ""))
    frame["status"] = _normalize_trade_status(_column_or_default(frame, "status", ""))
    frame["reason"] = _normalize_trade_reason(_column_or_default(frame, "reason", "")).replace("", "rebalance")
    frame["cash"] = pd.to_numeric(_column_or_default(frame, "cash", 0.0), errors="coerce").fillna(0.0)
    frame["shares"] = pd.to_numeric(_column_or_default(frame, "shares", 0.0), errors="coerce").fillna(0.0)
    frame["price"] = pd.to_numeric(_column_or_default(frame, "price", 0.0), errors="coerce").fillna(0.0)
    frame["notional"] = frame["cash"].abs()
    fallback_notional = (frame["shares"] * frame["price"]).abs()
    frame.loc[frame["notional"] <= 0, "notional"] = fallback_notional.loc[frame["notional"] <= 0]
    cost_columns = ["commission_cost", "tax_cost", "transfer_fee_cost", "slippage_cost"]
    for column in cost_columns:
        frame[column] = pd.to_numeric(_column_or_default(frame, column, 0.0), errors="coerce").fillna(0.0)
    frame["trade_cost"] = frame[cost_columns].sum(axis=1)

    by_regime = (
        frame.groupby("regime", as_index=False)
        .agg(
            trade_count=("side", "size"),
            buy_count=("side", lambda value: int((value == "BUY").sum())),
            sell_count=("side", lambda value: int((value == "SELL").sum())),
            risk_exit_count=("status", lambda value: int((value == "risk_exit").sum())),
            blocked_count=("status", lambda value: int((value == "blocked").sum())),
            notional=("notional", "sum"),
            trade_cost=("trade_cost", "sum"),
            commission_cost=("commission_cost", "sum"),
            tax_cost=("tax_cost", "sum"),
            transfer_fee_cost=("transfer_fee_cost", "sum"),
            slippage_cost=("slippage_cost", "sum"),
        )
        .sort_values("trade_cost", ascending=False)
    )
    by_reason = (
        frame.groupby(["regime", "status", "reason"], as_index=False)
        .agg(trade_count=("side", "size"), notional=("notional", "sum"), trade_cost=("trade_cost", "sum"))
        .sort_values(["regime", "trade_count"], ascending=[True, False])
    )
    if not by_regime.empty:
        by_regime["cost_per_trade"] = by_regime["trade_cost"] / by_regime["trade_count"].replace(0, np.nan)
        by_regime["cost_per_notional"] = by_regime["trade_cost"] / by_regime["notional"].replace(0, np.nan)

    worst_cost = by_regime.sort_values("trade_cost", ascending=False).head(1)
    summary = {
        "enabled": True,
        "regime_count": int(by_regime["regime"].nunique()) if not by_regime.empty else 0,
        "highest_cost_regime": str(worst_cost.iloc[0]["regime"]) if not worst_cost.empty else None,
        "records": by_regime.to_dict(orient="records"),
        "issues": [],
    }
    return summary, {"regime_trade_costs": by_regime, "regime_trade_costs_by_reason": by_reason}


def _cost_attribution(trades: pd.DataFrame, equity: pd.Series) -> dict[str, Any]:
    initial = float(equity.iloc[0]) if not equity.empty else 0.0
    summary: dict[str, Any] = {"trade_count": int(len(trades)), "initial_equity": initial}
    cost_columns = ["commission_cost", "tax_cost", "transfer_fee_cost", "slippage_cost"]
    total = 0.0
    for column in cost_columns:
        value = float(pd.to_numeric(trades.get(column, pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
        summary[column] = value
        total += value
    summary["total_trade_cost"] = total
    summary["cost_drag_on_initial_equity"] = float(total / initial) if initial > 0 else None
    if not trades.empty and "status" in trades.columns:
        summary["trade_status_counts"] = _normalize_trade_status(trades["status"]).value_counts().to_dict()
    if not trades.empty and "capacity_warning" in trades.columns:
        summary["capacity_warning_count"] = int(pd.Series(trades["capacity_warning"]).astype(bool).sum())
    return summary


def _turnover_attribution(
    trades: pd.DataFrame,
    holdings: pd.DataFrame,
    equity: pd.Series,
    config: dict[str, Any],
    annual_days: int,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    if trades.empty or "side" not in trades.columns:
        return {"enabled": False, "issues": ["trades_unavailable"]}, {}

    frame = trades.copy()
    frame["side"] = _normalize_trade_side(frame["side"])
    frame["status"] = _normalize_trade_status(_column_or_default(frame, "status", ""))
    frame["reason"] = _normalize_trade_reason(_column_or_default(frame, "reason", ""))
    frame["instrument"] = _column_or_default(frame, "instrument", "").map(_normalize_instrument)
    frame["date"] = pd.to_datetime(_column_or_default(frame, "date", pd.NaT), errors="coerce").dt.normalize()
    frame["shares"] = pd.to_numeric(_column_or_default(frame, "shares", 0.0), errors="coerce").fillna(0.0)
    frame["price"] = pd.to_numeric(_column_or_default(frame, "price", 0.0), errors="coerce").fillna(0.0)
    frame["cash"] = pd.to_numeric(_column_or_default(frame, "cash", 0.0), errors="coerce").fillna(0.0)
    frame["notional"] = frame["cash"].abs()
    fallback_notional = (frame["shares"] * frame["price"]).abs()
    frame.loc[frame["notional"] <= 0, "notional"] = fallback_notional.loc[frame["notional"] <= 0]
    cost_columns = ["commission_cost", "tax_cost", "transfer_fee_cost", "slippage_cost"]
    for column in cost_columns:
        frame[column] = pd.to_numeric(_column_or_default(frame, column, 0.0), errors="coerce").fillna(0.0)
    frame["trade_cost"] = frame[cost_columns].sum(axis=1)
    positions_after_trade = _positions_after_trade(holdings)
    frame["turnover_category"] = _turnover_categories(frame, positions_after_trade)

    sells = frame[frame["side"] == "SELL"].copy()
    executable_statuses = {"filled", "partial", "risk_exit"}
    executable = sells["status"].isin(executable_statuses)
    risk_exit = sells["status"].eq("risk_exit")
    normal_rebalance = executable & sells["status"].isin({"filled", "partial"}) & sells["reason"].eq("")
    blocked = sells["status"].eq("blocked")
    normal_rebalance_sells = sells[normal_rebalance].copy()
    rebalance_trim_count: int | None = None
    rebalance_exit_count: int | None = None
    if positions_after_trade:
        rebalance_trim_count = int((normal_rebalance_sells["turnover_category"] == "rebalance_trim").sum())
        rebalance_exit_count = int((normal_rebalance_sells["turnover_category"] == "rebalance_exit").sum())
    top_n = int(config.get("strategy", {}).get("top_n", config.get("top_n", 1)) or 1)
    periods = max(len(equity) - 1, 0)
    years = max(periods / max(annual_days, 1), 1 / max(annual_days, 1))
    initial = float(equity.iloc[0]) if not equity.empty else 0.0

    by_status_reason = (
        sells.assign(reason=sells["reason"].replace("", "rebalance"))
        .groupby(["status", "reason"], as_index=False)
        .agg(trade_count=("side", "size"), shares=("shares", "sum"), notional=("notional", "sum"), trade_cost=("trade_cost", "sum"))
        .sort_values(["trade_count", "notional"], ascending=[False, False])
    )
    by_category = (
        frame.groupby("turnover_category", as_index=False)
        .agg(trade_count=("side", "size"), notional=("notional", "sum"), trade_cost=("trade_cost", "sum"))
        .sort_values(["trade_count", "notional"], ascending=[False, False])
    )
    if initial > 0 and not by_category.empty:
        by_category["trade_cost_drag_on_initial_equity"] = by_category["trade_cost"] / initial
    dated_sells = sells[sells["date"].notna()].copy()
    by_year_status = pd.DataFrame()
    if not dated_sells.empty:
        dated_sells["year"] = dated_sells["date"].dt.year.astype(int)
        by_year_status = (
            dated_sells.groupby(["year", "status"], as_index=False)
            .agg(sell_count=("side", "size"), notional=("notional", "sum"))
            .sort_values(["year", "status"])
        )

    executable_sell_count = int(executable.sum())
    category_metrics = _category_metrics(by_category, initial)
    total_trade_cost = float(frame["trade_cost"].sum())
    turnover_without_trim = None
    if rebalance_trim_count is not None:
        turnover_without_trim = float(max(executable_sell_count - rebalance_trim_count, 0) / years / max(top_n, 1) * 2)
    summary = {
        "enabled": True,
        "sell_count": int(len(sells)),
        "executable_sell_count": executable_sell_count,
        "blocked_sell_count": int(blocked.sum()),
        "normal_rebalance_sell_count": int(normal_rebalance.sum()),
        "rebalance_trim_sell_count": rebalance_trim_count,
        "rebalance_exit_sell_count": rebalance_exit_count,
        "rebalance_trim_share_of_normal_sells": (
            float(rebalance_trim_count / normal_rebalance.sum()) if rebalance_trim_count is not None and normal_rebalance.sum() else None
        ),
        "risk_exit_sell_count": int(risk_exit.sum()),
        "risk_exit_share_of_executable_sells": float(risk_exit.sum() / executable_sell_count) if executable_sell_count else 0.0,
        "annual_turnover_estimate": float(executable_sell_count / years / max(top_n, 1) * 2),
        "annual_turnover_without_rebalance_trims_estimate": turnover_without_trim,
        **category_metrics,
        "top_status_reasons": by_status_reason.head(10).to_dict(orient="records"),
        "issues": [],
    }
    trim_cost = summary.get("rebalance_trim_trade_cost")
    summary["rebalance_trim_cost_share_of_total_trade_cost"] = (
        float(trim_cost / total_trade_cost) if trim_cost is not None and total_trade_cost > 0 else None
    )
    return summary, {
        "turnover_by_category": by_category,
        "turnover_by_status_reason": by_status_reason,
        "turnover_by_year_status": by_year_status,
    }


def _turnover_categories(frame: pd.DataFrame, positions_after_trade: set[tuple[pd.Timestamp, str]]) -> pd.Series:
    categories = pd.Series("other", index=frame.index, dtype=object)
    buy = frame["side"] == "BUY"
    sell = frame["side"] == "SELL"
    categories.loc[buy] = "buy"
    categories.loc[sell & frame["status"].eq("blocked")] = "blocked_sell"
    categories.loc[sell & frame["status"].eq("risk_exit")] = "risk_exit"
    normal_rebalance = sell & frame["status"].isin({"filled", "partial"}) & frame["reason"].eq("")
    if positions_after_trade:
        still_held = pd.Series(
            [(date, instrument) in positions_after_trade for date, instrument in zip(frame["date"], frame["instrument"], strict=False)],
            index=frame.index,
        )
        categories.loc[normal_rebalance & still_held] = "rebalance_trim"
        categories.loc[normal_rebalance & ~still_held] = "rebalance_exit"
    else:
        categories.loc[normal_rebalance] = "rebalance"
    categories.loc[sell & frame["status"].isin({"filled", "partial"}) & frame["reason"].ne("")] = "other_sell"
    return categories


def _category_metrics(by_category: pd.DataFrame, initial_equity: float) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if by_category.empty:
        return metrics
    indexed = by_category.set_index("turnover_category")
    for category in ["rebalance_trim", "rebalance_exit", "risk_exit", "blocked_sell", "buy"]:
        if category not in indexed.index:
            continue
        row = indexed.loc[category]
        metrics[f"{category}_trade_count"] = int(row["trade_count"])
        metrics[f"{category}_notional"] = float(row["notional"])
        metrics[f"{category}_trade_cost"] = float(row["trade_cost"])
        metrics[f"{category}_trade_cost_drag_on_initial_equity"] = (
            float(row["trade_cost"] / initial_equity) if initial_equity > 0 else None
        )
    return metrics


def _positions_after_trade(holdings: pd.DataFrame) -> set[tuple[pd.Timestamp, str]]:
    frame = _normalize_holdings(holdings)
    if frame.empty:
        return set()
    return set(zip(frame["date"], frame["instrument"], strict=False))


def _column_or_default(frame: pd.DataFrame, column: str, default: Any) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series([default] * len(frame), index=frame.index)


def _holding_return_attribution(
    holdings: pd.DataFrame,
    price_df: pd.DataFrame,
    config: dict[str, Any],
    drawdown_summary: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    if holdings.empty:
        return {"enabled": False, "issues": ["empty_holdings"]}, {}
    close = _price_field(price_df, "close")
    if close.empty:
        return {"enabled": False, "issues": ["close_prices_unavailable"]}, {}

    frame = _normalize_holdings(holdings)
    if frame.empty:
        return {"enabled": False, "issues": ["holdings_missing_required_columns"]}, {}
    returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    price_dates = pd.DatetimeIndex(close.index)
    rows: list[dict[str, Any]] = []
    for date, daily in frame.groupby("date", sort=True):
        next_dates = price_dates[price_dates > pd.Timestamp(date)]
        if next_dates.empty:
            continue
        next_date = next_dates[0]
        total_value = float(daily["value"].sum())
        if total_value <= 0 or next_date not in returns.index:
            continue
        daily_ret = returns.loc[next_date]
        for _, row in daily.iterrows():
            instrument = row["instrument"]
            if instrument not in daily_ret.index or pd.isna(daily_ret.loc[instrument]):
                continue
            weight = float(row["value"] / total_value)
            stock_return = float(daily_ret.loc[instrument])
            rows.append(
                {
                    "date": date,
                    "next_date": next_date,
                    "instrument": instrument,
                    "weight": weight,
                    "stock_return": stock_return,
                    "gross_contribution": weight * stock_return,
                }
            )
    contribution = pd.DataFrame(rows)
    if contribution.empty:
        return {"enabled": False, "issues": ["holding_contributions_unavailable"]}, {}

    industry = _load_industry_map(config)
    if not industry.empty:
        contribution["industry"] = contribution["instrument"].map(industry).fillna("UNKNOWN")
    else:
        contribution["industry"] = "UNKNOWN"
    contribution = _attach_regime_to_contributions(contribution, price_df, config)

    by_instrument = (
        contribution.groupby("instrument", as_index=False)["gross_contribution"]
        .sum()
        .sort_values("gross_contribution", ascending=False)
    )
    by_industry = (
        contribution.groupby("industry", as_index=False)["gross_contribution"]
        .sum()
        .sort_values("gross_contribution", ascending=False)
    )
    by_regime_instrument = _group_contribution_by_regime(contribution, "instrument")
    by_regime_industry = _group_contribution_by_regime(contribution, "industry")
    drawdown_contribution_summary, drawdown_contribution_tables = _max_drawdown_contribution_attribution(
        contribution,
        drawdown_summary,
    )
    summary = {
        "enabled": True,
        "gross_close_to_close_contribution": float(contribution["gross_contribution"].sum()),
        "top_positive_instruments": _top_records(by_instrument, "gross_contribution", ascending=False),
        "top_negative_instruments": _top_records(by_instrument, "gross_contribution", ascending=True),
        "top_positive_industries": _top_records(by_industry, "gross_contribution", ascending=False),
        "top_negative_industries": _top_records(by_industry, "gross_contribution", ascending=True),
        "top_negative_regime_industries": _top_records_by_regime(by_regime_industry, "gross_contribution", ascending=True),
        "top_positive_regime_industries": _top_records_by_regime(by_regime_industry, "gross_contribution", ascending=False),
        **drawdown_contribution_summary,
        "issues": [],
    }
    tables = {
        "holding_contributions": contribution,
        "instrument_attribution": by_instrument,
        "industry_attribution": by_industry,
    }
    if not by_regime_instrument.empty:
        tables["regime_instrument_attribution"] = by_regime_instrument
    if not by_regime_industry.empty:
        tables["regime_industry_attribution"] = by_regime_industry
    tables.update(drawdown_contribution_tables)
    return summary, tables


def _max_drawdown_contribution_attribution(
    contribution: pd.DataFrame,
    drawdown_summary: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    start, end = _drawdown_contribution_window(drawdown_summary)
    if start is None or end is None:
        return {"max_drawdown_contribution_enabled": False}, {}
    frame = contribution.copy()
    next_dates = pd.to_datetime(frame["next_date"], errors="coerce").dt.normalize()
    window = frame[(next_dates > start) & (next_dates <= end)].copy()
    if window.empty:
        return {
            "max_drawdown_contribution_enabled": False,
            "max_drawdown_contribution_start_date": _date_text(start),
            "max_drawdown_contribution_end_date": _date_text(end),
        }, {}

    by_instrument = (
        window.groupby("instrument", as_index=False)["gross_contribution"]
        .sum()
        .sort_values("gross_contribution", ascending=False)
    )
    by_industry = (
        window.groupby("industry", as_index=False)["gross_contribution"]
        .sum()
        .sort_values("gross_contribution", ascending=False)
    )
    summary = {
        "max_drawdown_contribution_enabled": True,
        "max_drawdown_contribution_start_date": _date_text(start),
        "max_drawdown_contribution_end_date": _date_text(end),
        "max_drawdown_gross_close_to_close_contribution": float(window["gross_contribution"].sum()),
        "max_drawdown_top_negative_instruments": _top_records(by_instrument, "gross_contribution", ascending=True),
        "max_drawdown_top_positive_instruments": _top_records(by_instrument, "gross_contribution", ascending=False),
        "max_drawdown_top_negative_industries": _top_records(by_industry, "gross_contribution", ascending=True),
        "max_drawdown_top_positive_industries": _top_records(by_industry, "gross_contribution", ascending=False),
    }
    return summary, {
        "max_drawdown_holding_contributions": window,
        "max_drawdown_instrument_attribution": by_instrument,
        "max_drawdown_industry_attribution": by_industry,
    }


def _drawdown_contribution_window(drawdown_summary: dict[str, Any] | None) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if not drawdown_summary or not bool(drawdown_summary.get("enabled", False)):
        return None, None
    peak = drawdown_summary.get("max_drawdown_peak_date")
    trough = drawdown_summary.get("max_drawdown_trough_date")
    if not peak or not trough:
        return None, None
    return pd.Timestamp(peak).normalize(), pd.Timestamp(trough).normalize()


def _attach_regime_to_contributions(contribution: pd.DataFrame, price_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if contribution.empty or "next_date" not in contribution.columns or price_df.empty:
        return contribution
    regimes = detect_reporting_regime(price_df, config)
    if regimes.empty:
        return contribution
    frame = contribution.copy()
    next_dates = pd.DatetimeIndex(pd.to_datetime(frame["next_date"], errors="coerce").dt.normalize())
    aligned = regimes.reindex(next_dates, method="ffill")
    frame["regime"] = aligned.to_numpy()
    return frame


def _group_contribution_by_regime(contribution: pd.DataFrame, column: str) -> pd.DataFrame:
    if contribution.empty or "regime" not in contribution.columns or column not in contribution.columns:
        return pd.DataFrame()
    frame = contribution.dropna(subset=["regime"])
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby(["regime", column], as_index=False)["gross_contribution"]
        .sum()
        .sort_values(["regime", "gross_contribution"], ascending=[True, False])
    )


def _top_records_by_regime(frame: pd.DataFrame, column: str, ascending: bool, limit: int = 3) -> list[dict[str, Any]]:
    if frame.empty or "regime" not in frame.columns or column not in frame.columns:
        return []
    rows: list[dict[str, Any]] = []
    for regime, group in frame.groupby("regime", sort=False):
        selected = group.sort_values(column, ascending=ascending).head(limit).copy()
        selected["regime"] = regime
        rows.extend(selected.to_dict(orient="records"))
    return rows


def _exposure_diagnostics(holdings: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    frame = _normalize_holdings(holdings)
    if frame.empty:
        return {"enabled": False, "issues": ["empty_holdings"]}, {}

    frame["weight"] = frame["value"] / frame.groupby("date")["value"].transform("sum").replace(0.0, np.nan)
    industry = _load_industry_map(config)
    frame["industry"] = frame["instrument"].map(industry).fillna("UNKNOWN") if not industry.empty else "UNKNOWN"
    industry_exposure = (
        frame.groupby(["date", "industry"], as_index=False)["weight"]
        .sum()
        .sort_values(["date", "weight"], ascending=[True, False])
    )

    latest_date = frame["date"].max()
    latest_industry = industry_exposure[industry_exposure["date"] == latest_date].sort_values("weight", ascending=False)
    summary: dict[str, Any] = {
        "enabled": True,
        "latest_date": str(pd.Timestamp(latest_date).date()),
        "latest_max_industry_weight": float(latest_industry["weight"].max()) if not latest_industry.empty else None,
        "latest_top_industries": latest_industry.head(10).to_dict(orient="records"),
        "latest_position_count": int(frame[frame["date"] == latest_date]["instrument"].nunique()),
        "latest_top_position_weight": float(frame[frame["date"] == latest_date]["weight"].max()),
        "issues": [],
    }

    cap_exposure = _market_cap_exposure(frame[frame["date"] == latest_date], latest_date, config)
    tables = {"industry_exposure": industry_exposure}
    if not cap_exposure.empty:
        tables["market_cap_exposure"] = cap_exposure
        summary["market_cap_buckets"] = cap_exposure.to_dict(orient="records")
        cap_summary = _market_cap_exposure_summary(cap_exposure, latest_date, config)
        summary.update({key: value for key, value in cap_summary.items() if key != "issues"})
        summary["issues"].extend(cap_summary.get("issues", []))
    else:
        summary["issues"].append("market_cap_exposure_unavailable")
    return summary, tables


def _market_cap_exposure(latest_holdings: pd.DataFrame, latest_date: pd.Timestamp, config: dict[str, Any]) -> pd.DataFrame:
    research_cfg = config.get("research", {}).get("exposure", {})
    daily_basic_path = resolve_path(
        research_cfg.get("daily_basic_file")
        or config.get("data", {}).get("daily_basic_file", "data/factors/daily_basic.parquet")
    )
    market_cap_field = str(research_cfg.get("market_cap_field", "circ_mv"))
    if not daily_basic_path.exists():
        return pd.DataFrame()
    daily_basic = pd.read_parquet(daily_basic_path)
    if daily_basic.empty or "trade_date" not in daily_basic.columns or "ts_code" not in daily_basic.columns or market_cap_field not in daily_basic.columns:
        return pd.DataFrame()
    frame = daily_basic.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame["ts_code"] = frame["ts_code"].map(_normalize_instrument)
    frame[market_cap_field] = pd.to_numeric(frame[market_cap_field], errors="coerce")
    frame = frame[
        (frame["ts_code"] != "")
        & (frame["trade_date"] <= pd.Timestamp(latest_date).normalize())
        & frame[market_cap_field].notna()
    ]
    if frame.empty:
        return pd.DataFrame()
    last_basic_date = frame["trade_date"].max()
    cross_section = frame[frame["trade_date"] == last_basic_date].drop_duplicates("ts_code", keep="last").copy()
    if cross_section.empty:
        return pd.DataFrame()
    cross_section["market_cap_rank_pct"] = cross_section[market_cap_field].rank(pct=True)
    cross_section["bucket"] = "mid"
    cross_section.loc[cross_section["market_cap_rank_pct"] <= 1 / 3, "bucket"] = "small"
    cross_section.loc[cross_section["market_cap_rank_pct"] >= 2 / 3, "bucket"] = "large"
    bucket_map = pd.Series(cross_section["bucket"].to_numpy(), index=cross_section["ts_code"]).to_dict()
    cap_map = pd.Series(cross_section[market_cap_field].to_numpy(), index=cross_section["ts_code"]).to_dict()

    latest = latest_holdings.groupby("instrument", as_index=False)["weight"].sum()
    latest["market_cap"] = latest["instrument"].map(cap_map)
    latest["bucket"] = latest["instrument"].map(bucket_map).fillna("unknown")
    latest["known_position"] = latest["market_cap"].notna()
    result = (
        latest.groupby("bucket", as_index=False)
        .agg(
            weight=("weight", "sum"),
            position_count=("instrument", "nunique"),
            known_position_count=("known_position", "sum"),
            market_cap_median=("market_cap", "median"),
            market_cap_min=("market_cap", "min"),
            market_cap_max=("market_cap", "max"),
        )
        .sort_values("weight", ascending=False)
    )
    result["known_position_count"] = result["known_position_count"].astype(int)
    result["unknown_position_count"] = result["position_count"] - result["known_position_count"]
    result["asof_date"] = str(pd.Timestamp(last_basic_date).date())
    result["market_cap_field"] = market_cap_field
    return result


def _market_cap_exposure_summary(cap_exposure: pd.DataFrame, latest_date: pd.Timestamp, config: dict[str, Any]) -> dict[str, Any]:
    if cap_exposure.empty:
        return {}
    issues: list[str] = []
    asof_date = pd.to_datetime(cap_exposure.get("asof_date", pd.Series(dtype=object)).dropna().head(1), errors="coerce")
    normalized_latest = pd.Timestamp(latest_date).normalize()
    normalized_asof = pd.Timestamp(asof_date.iloc[0]).normalize() if len(asof_date) and not pd.isna(asof_date.iloc[0]) else None
    known_positions = int(_numeric_column_sum(cap_exposure, "known_position_count"))
    total_positions = int(_numeric_column_sum(cap_exposure, "position_count"))
    unknown_positions = max(total_positions - known_positions, 0)
    unknown_weight = float(cap_exposure.loc[cap_exposure["bucket"].eq("unknown"), "weight"].sum()) if "bucket" in cap_exposure.columns else 0.0
    matched_weight = float(max(1.0 - unknown_weight, 0.0))

    research_cfg = config.get("research", {}).get("exposure", {})
    min_matched_weight = float(research_cfg.get("market_cap_min_matched_weight", 0.8))
    if matched_weight < min_matched_weight:
        issues.append(f"market_cap_matched_weight_below_threshold:{matched_weight:.4f}<{min_matched_weight:.4f}")
    max_staleness_days = research_cfg.get("market_cap_max_staleness_days", 5)
    staleness_days: int | None = None
    if normalized_asof is not None:
        staleness_days = int((normalized_latest - normalized_asof).days)
        if max_staleness_days is not None and staleness_days > int(max_staleness_days):
            issues.append(f"market_cap_asof_stale:{staleness_days}>{int(max_staleness_days)}")

    return {
        "market_cap_asof_date": str(normalized_asof.date()) if normalized_asof is not None else None,
        "market_cap_staleness_days": staleness_days,
        "market_cap_matched_position_count": known_positions,
        "market_cap_unknown_position_count": unknown_positions,
        "market_cap_matched_weight": matched_weight,
        "market_cap_unknown_weight": unknown_weight,
        "issues": issues,
    }


def _numeric_column_sum(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())


def _normalize_holdings(holdings: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "instrument", "value"}
    if holdings.empty or not required.issubset(set(holdings.columns)):
        return pd.DataFrame()
    frame = holdings[["date", "instrument", "value"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["instrument"] = frame["instrument"].map(_normalize_instrument)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["date", "instrument", "value"])
    frame = frame[(frame["instrument"] != "") & (frame["value"] > 0)]
    return frame.sort_values(["date", "instrument"]).reset_index(drop=True)


def _price_field(price_df: pd.DataFrame, field: str) -> pd.DataFrame:
    if price_df.empty:
        return pd.DataFrame()
    field = str(field).strip().lower()
    if isinstance(price_df.columns, pd.MultiIndex):
        fields = price_df.columns.get_level_values(0).astype(str).str.strip().str.lower()
        if field not in set(fields):
            return pd.DataFrame(index=price_df.index)
        frame = price_df.loc[:, fields == field].copy()
        frame.columns = [_normalize_instrument(value) for value in frame.columns.get_level_values(-1)]
    elif field == "close":
        frame = price_df.copy()
    else:
        return pd.DataFrame(index=price_df.index)
    frame.index = pd.to_datetime(frame.index).normalize()
    frame.columns = [_normalize_instrument(value) for value in frame.columns]
    frame = frame.loc[:, frame.columns != ""]
    frame = frame.loc[:, ~frame.columns.duplicated(keep="last")]
    return frame.sort_index().apply(pd.to_numeric, errors="coerce")


def _load_industry_map(config: dict[str, Any]) -> pd.Series:
    research_cfg = config.get("research", {}).get("exposure", {})
    path = resolve_path(research_cfg.get("industry_file") or config.get("data", {}).get("constituents_file", "data/raw/mainboard_a_stocks.csv"))
    if not path.exists():
        return pd.Series(dtype=object)
    frame = pd.read_csv(path)
    if "industry" not in frame.columns:
        return pd.Series(dtype=object)
    code_col = next((col for col in ["ts_code", "con_code", "instrument", "code"] if col in frame.columns), None)
    if code_col is None:
        return pd.Series(dtype=object)
    result = frame.dropna(subset=[code_col]).drop_duplicates(code_col, keep="last")
    return pd.Series(
        result["industry"].fillna("UNKNOWN").astype(str).to_numpy(),
        index=result[code_col].map(_normalize_instrument),
        name="industry",
    )


def _total_return(curve: pd.Series) -> float | None:
    if curve.empty:
        return None
    first = float(curve.iloc[0])
    if first == 0:
        return None
    return float(curve.iloc[-1] / first - 1.0)


def _annual_return(curve: pd.Series, annual_days: int) -> float | None:
    total = _total_return(curve)
    if total is None or len(curve) <= 1:
        return None
    return float((1.0 + total) ** (annual_days / max(len(curve) - 1, 1)) - 1.0)


def _max_drawdown(curve: pd.Series) -> float | None:
    if curve.empty:
        return None
    drawdown = curve / curve.cummax() - 1.0
    return float(drawdown.min())


def _sharpe(returns: pd.Series, annual_days: int) -> float | None:
    returns = returns.dropna()
    if returns.empty:
        return None
    std = float(returns.std(ddof=0))
    if std <= 0:
        return None
    return float(returns.mean() / std * np.sqrt(annual_days))


def _information_ratio(active_returns: pd.Series, annual_days: int) -> float | None:
    active_returns = active_returns.dropna()
    if active_returns.empty:
        return None
    std = float(active_returns.std(ddof=0))
    if std <= 0:
        return None
    return float(active_returns.mean() / std * np.sqrt(annual_days))


def _beta(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float | None:
    aligned = pd.concat([strategy_returns.rename("strategy"), benchmark_returns.rename("benchmark")], axis=1).dropna()
    if len(aligned) < 3:
        return None
    benchmark_var = float(aligned["benchmark"].var(ddof=0))
    if benchmark_var <= 0:
        return None
    return float(aligned["strategy"].cov(aligned["benchmark"], ddof=0) / benchmark_var)


def _top_records(frame: pd.DataFrame, column: str, ascending: bool) -> list[dict[str, Any]]:
    if frame.empty or column not in frame.columns:
        return []
    return frame.sort_values(column, ascending=ascending).head(5).to_dict(orient="records")


def _normalize_instrument(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def _normalize_trade_side(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip().str.upper()


def _normalize_trade_status(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip().str.lower()


def _normalize_trade_reason(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value).date())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value
