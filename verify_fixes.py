from __future__ import annotations

import pandas as pd

from src.backtest import run_backtest


def build_demo_data() -> tuple[pd.Series, pd.DataFrame]:
    dates = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
        ]
    )
    instruments = ["600001.SH", "000001.SZ"]
    score_index = pd.MultiIndex.from_product([dates[:-1], instruments], names=["datetime", "instrument"])
    scores = pd.Series(
        [
            10,
            1,
            10,
            1,
            9,
            2,
            8,
            3,
            7,
            4,
        ],
        index=score_index,
        name="score",
    )

    prices = pd.concat(
        {
            "open": pd.DataFrame(
                {
                    "600001.SH": [10.0, 20.0, 14.0, 16.0, 17.0, 18.0],
                    "000001.SZ": [10.0, 10.0, 10.2, 10.4, 10.5, 10.6],
                },
                index=dates,
            ),
            "high": pd.DataFrame(
                {
                    "600001.SH": [10.2, 20.5, 15.0, 17.0, 18.0, 18.5],
                    "000001.SZ": [10.1, 10.2, 10.3, 10.5, 10.6, 10.7],
                },
                index=dates,
            ),
            "low": pd.DataFrame(
                {
                    "600001.SH": [9.8, 9.5, 13.8, 15.8, 16.8, 17.8],
                    "000001.SZ": [9.9, 9.9, 10.1, 10.3, 10.4, 10.5],
                },
                index=dates,
            ),
            "close": pd.DataFrame(
                {
                    "600001.SH": [10.0, 10.0, 14.0, 16.0, 17.0, 18.0],
                    "000001.SZ": [10.0, 10.1, 10.2, 10.3, 10.4, 10.5],
                },
                index=dates,
            ),
            "volume": pd.DataFrame(
                {
                    "600001.SH": [100000, 100000, 100000, 100000, 100000, 100000],
                    "000001.SZ": [100000, 100000, 100000, 100000, 100000, 100000],
                },
                index=dates,
            ),
            "amount": pd.DataFrame(
                {
                    "600001.SH": [1_000_000, 40_000, 50_000, 60_000, 60_000, 60_000],
                    "000001.SZ": [1_000_000, 1_000_000, 1_000_000, 1_000_000, 1_000_000, 1_000_000],
                },
                index=dates,
            ),
        },
        axis=1,
    )
    return scores, prices


def summarize(name: str, result) -> dict[str, float | int | str]:
    trades = result.trades
    partial_count = int((trades.get("status") == "partial").sum()) if not trades.empty else 0
    blocked_count = int((trades.get("status") == "blocked").sum()) if not trades.empty else 0
    return {
        "case": name,
        "total_return": result.metrics["total_return"],
        "sharpe": result.metrics["sharpe"],
        "max_drawdown": result.metrics["max_drawdown"],
        "win_rate": result.metrics["win_rate"],
        "commission_tax": float(trades.get("commission_cost", pd.Series(dtype=float)).sum())
        + float(trades.get("tax_cost", pd.Series(dtype=float)).sum()),
        "slippage_cost": float(trades.get("slippage_cost", pd.Series(dtype=float)).sum()),
        "not_full_count": partial_count + blocked_count,
    }


def main() -> None:
    scores, prices = build_demo_data()
    common = {
        "initial_capital": 100000.0,
        "commission": 0.0003,
        "stamp_tax": 0.001,
        "annual_trading_days": 252,
        "top_n": 1,
        "max_turnover": 1,
        "rank_buffer": 0,
        "limit_up_threshold": 2.0,
        "limit_down_threshold": 2.0,
    }
    legacy = run_backtest(
        scores,
        prices,
        "2024-01-02",
        "2024-01-09",
        {
            **common,
            "trade_price_field": "close",
            "valuation_price_field": "close",
            "slippage": 0.0,
            "max_participation_rate": None,
        },
    )
    fixed = run_backtest(
        scores,
        prices,
        "2024-01-02",
        "2024-01-09",
        {
            **common,
            "trade_price_field": "open",
            "valuation_price_field": "close",
            "slippage": 0.001,
            "max_participation_rate": 0.2,
            "amount_unit": 1.0,
            "capacity_warning_threshold": 0.05,
            "capacity_window": 2,
        },
    )

    rows = [summarize("legacy_close_fill", legacy), summarize("fixed_open_capacity", fixed)]
    report = pd.DataFrame(rows).set_index("case")
    print(report.to_string(float_format=lambda value: f"{value:.6f}"))

    assert legacy.metrics["total_return"] > fixed.metrics["total_return"], "fixed model should remove inflated close-fill return"
    assert rows[1]["slippage_cost"] > rows[0]["slippage_cost"], "fixed model should account for slippage"
    assert rows[1]["not_full_count"] > rows[0]["not_full_count"], "fixed model should expose capacity-limited fills"

    print("\nAssertions passed: fixed backtest is more conservative and closer to manual execution reality.")


if __name__ == "__main__":
    main()
