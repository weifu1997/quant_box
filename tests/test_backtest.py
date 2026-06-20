"""模块说明：覆盖 test_backtest 相关行为的测试用例。"""

from __future__ import annotations

import gc
import unittest
import weakref

import numpy as np
import pandas as pd

from src.backtest import (
    _NORMALIZED_PRICE_CACHE,
    _PRICE_FIELD_CACHE,
    _PRICE_FIELD_CACHE_MAX_SIZE,
    _field,
    _lot_size,
    _max_drawdown_duration,
    _normalize_price_frame,
    calculate_metrics,
    run_backtest,
)
from tests.fixtures.real_data import require_real_market_data


class BacktestTests(unittest.TestCase):
    """类说明：组织 BacktestTests 测试用例。"""
    def test_calculate_metrics_turnover_ignores_blocked_sells(self) -> None:
        """函数说明：验证 test_calculate_metrics_turnover_ignores_blocked_sells 覆盖的行为场景。"""
        equity = pd.Series([100000.0, 100100.0, 100200.0], index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
        trades = pd.DataFrame(
            [
                {"side": " sell ", "status": " blocked "},
                {"side": " sell ", "status": " filled "},
                {"side": "SELL", "status": " PARTIAL "},
                {"side": "SELL", "status": " RISK_EXIT "},
                {"side": "BUY", "status": "filled"},
            ]
        )

        metrics = calculate_metrics(equity, trades, {"annual_trading_days": 252, "top_n": 1})

        self.assertEqual(metrics["turnover_count"], 3.0)

    def test_run_backtest_accepts_string_score_dates(self) -> None:
        """函数说明：验证 test_run_backtest_accepts_string_score_dates 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        stock = "000001.SZ"
        scores = pd.Series(
            [1.0],
            index=pd.MultiIndex.from_tuples([("2024-01-02", stock)], names=["datetime", "instrument"]),
            name="score",
        )
        prices = pd.concat(
            {
                "close": pd.DataFrame({stock: [10.0, 10.0]}, index=dates),
                "volume": pd.DataFrame({stock: [1000.0, 1000.0]}, index=dates),
            },
            axis=1,
        )

        result = run_backtest(scores, prices, "2024-01-02", "2024-01-03", {"initial_capital": 100000, "top_n": 1})

        self.assertTrue((result.trades["side"] == "BUY").any())

    def test_run_backtest_normalizes_intraday_price_timestamps_to_trade_dates(self) -> None:
        """函数说明：验证 test_run_backtest_normalizes_intraday_price_timestamps_to_trade_dates 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02 15:00", "2024-01-03 15:00"])
        stock = "000001.SZ"
        scores = pd.Series(
            [1.0],
            index=pd.MultiIndex.from_tuples([("2024-01-02", stock)], names=["datetime", "instrument"]),
            name="score",
        )
        prices = pd.concat(
            {
                "close": pd.DataFrame({stock: [10.0, 10.0]}, index=dates),
                "volume": pd.DataFrame({stock: [1000.0, 1000.0]}, index=dates),
            },
            axis=1,
        )

        result = run_backtest(scores, prices, "2024-01-02", "2024-01-03", {"initial_capital": 100000, "top_n": 1})

        buys = result.trades[result.trades["side"] == "BUY"]
        self.assertFalse(buys.empty)
        self.assertEqual(pd.Timestamp(buys.iloc[0]["date"]), pd.Timestamp("2024-01-03"))
        self.assertEqual(result.equity_curve.index.tolist(), [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")])

    def test_run_backtest_uses_latest_intraday_price_per_trade_date(self) -> None:
        """函数说明：验证 test_run_backtest_uses_latest_intraday_price_per_trade_date 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-01 15:00", "2024-01-02 15:00", "2024-01-02 09:30", "2024-01-03 15:00"])
        stock = "000001.SZ"
        scores = pd.Series(
            [1.0],
            index=pd.MultiIndex.from_tuples([("2024-01-01", stock)], names=["datetime", "instrument"]),
            name="score",
        )
        prices = pd.concat(
            {
                "close": pd.DataFrame({stock: [10.0, 10.0, 30.0, 20.0]}, index=dates),
                "volume": pd.DataFrame({stock: [1000.0, 1000.0, 1000.0, 1000.0]}, index=dates),
            },
            axis=1,
        )

        result = run_backtest(
            scores,
            prices,
            "2024-01-01",
            "2024-01-03",
            {"initial_capital": 100000, "top_n": 1, "commission": 0.0, "stamp_tax": 0.0, "slippage": 0.0},
        )

        buy = result.trades[result.trades["side"] == "BUY"].iloc[0]
        self.assertEqual(pd.Timestamp(buy["date"]), pd.Timestamp("2024-01-02"))
        self.assertAlmostEqual(float(buy["price"]), 10.0)

    def test_run_backtest_applies_selection_schedule_by_signal_date(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        stocks = ["000001.SZ", "000002.SZ"]
        scores = pd.Series(
            [2.0, 1.0, 2.0, 3.0],
            index=pd.MultiIndex.from_tuples(
                [
                    (pd.Timestamp("2024-01-02"), stocks[0]),
                    (pd.Timestamp("2024-01-02"), stocks[1]),
                    (pd.Timestamp("2024-01-03"), stocks[0]),
                    (pd.Timestamp("2024-01-03"), stocks[1]),
                ],
                names=["datetime", "instrument"],
            ),
            name="score",
        )
        prices = pd.concat(
            {
                "close": pd.DataFrame({stock: [10.0, 10.0, 10.0] for stock in stocks}, index=dates),
                "volume": pd.DataFrame({stock: [1000.0, 1000.0, 1000.0] for stock in stocks}, index=dates),
            },
            axis=1,
        )

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-04",
            {
                "initial_capital": 100000,
                "top_n": 1,
                "max_turnover": 1,
                "rank_buffer": 0,
                "commission": 0.0,
                "stamp_tax": 0.0,
                "slippage": 0.0,
                "selection_schedule": {
                    "2024-01-03": {"top_n": 2, "max_turnover": 2, "rank_buffer": 0},
                },
            },
        )

        latest = result.holdings[result.holdings["date"] == pd.Timestamp("2024-01-04")]
        self.assertEqual(set(latest["instrument"]), set(stocks))

    def test_normalized_price_frame_cache_reuses_same_source_panel(self) -> None:
        """函数说明：验证回测会复用同一价格面板的规范化结果。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        stock = "000001.SZ"
        prices = pd.concat(
            {
                "close": pd.DataFrame({stock: [10.0, 10.1]}, index=dates),
                "volume": pd.DataFrame({stock: [1000.0, 1000.0]}, index=dates),
            },
            axis=1,
        )
        _NORMALIZED_PRICE_CACHE.clear()

        first = _normalize_price_frame(prices)
        second = _normalize_price_frame(prices)

        self.assertIs(first, second)
        self.assertEqual(len(_NORMALIZED_PRICE_CACHE), 1)

    def test_run_backtest_matches_score_and_price_instruments_case_insensitively(self) -> None:
        """函数说明：验证 test_run_backtest_matches_score_and_price_instruments_case_insensitively 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        scores = pd.Series(
            [1.0],
            index=pd.MultiIndex.from_tuples([(dates[0], "000001.SZ")], names=["datetime", "instrument"]),
            name="score",
        )
        prices = pd.concat(
            {
                " close ": pd.DataFrame({" 000001.sz ": [10.0, 10.0]}, index=dates),
                " volume ": pd.DataFrame({" 000001.sz ": [1000.0, 1000.0]}, index=dates),
            },
            axis=1,
        )

        result = run_backtest(scores, prices, "2024-01-02", "2024-01-03", {"initial_capital": 100000, "top_n": 1})

        buys = result.trades[result.trades["side"] == "BUY"]
        self.assertFalse(buys.empty)
        self.assertEqual(buys.iloc[0]["instrument"], "000001.SZ")

    def test_run_backtest_keeps_highest_score_when_normalized_codes_duplicate(self) -> None:
        """函数说明：验证 test_run_backtest_keeps_highest_score_when_normalized_codes_duplicate 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        scores = pd.Series(
            [10.0, 1.0, 5.0],
            index=pd.MultiIndex.from_tuples(
                [(dates[0], " a "), (dates[0], "A"), (dates[0], "B")],
                names=["datetime", "instrument"],
            ),
            name="score",
        )
        prices = pd.concat(
            {
                "close": pd.DataFrame({"A": [10.0, 10.0], "B": [10.0, 10.0]}, index=dates),
                "volume": pd.DataFrame({"A": [1000.0, 1000.0], "B": [1000.0, 1000.0]}, index=dates),
            },
            axis=1,
        )

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-03",
            {"initial_capital": 100000, "top_n": 1, "max_turnover": 1, "commission": 0.0, "stamp_tax": 0.0},
        )

        buys = result.trades[result.trades["side"] == "BUY"]
        self.assertEqual(buys["instrument"].tolist(), ["A"])

    def test_run_backtest_uses_latest_intraday_score_per_trade_date(self) -> None:
        """函数说明：验证 test_run_backtest_uses_latest_intraday_score_per_trade_date 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        scores = pd.Series(
            [100.0, 1.0, 1.0, 50.0],
            index=pd.MultiIndex.from_tuples(
                [
                    (pd.Timestamp("2024-01-02 09:30"), "A"),
                    (pd.Timestamp("2024-01-02 09:30"), "B"),
                    (pd.Timestamp("2024-01-02 15:00"), "A"),
                    (pd.Timestamp("2024-01-02 15:00"), "B"),
                ],
                names=["datetime", "instrument"],
            ),
            name="score",
        )
        prices = pd.concat(
            {
                "close": pd.DataFrame({"A": [10.0, 10.0], "B": [10.0, 10.0]}, index=dates),
                "volume": pd.DataFrame({"A": [1000.0, 1000.0], "B": [1000.0, 1000.0]}, index=dates),
            },
            axis=1,
        )

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-03",
            {"initial_capital": 100000, "top_n": 1, "max_turnover": 1, "commission": 0.0, "stamp_tax": 0.0},
        )

        buys = result.trades[result.trades["side"] == "BUY"]
        self.assertEqual(buys["instrument"].tolist(), ["B"])

    def test_run_backtest_rejects_flat_ohlcv_price_frame(self) -> None:
        """函数说明：验证 test_run_backtest_rejects_flat_ohlcv_price_frame 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        scores = pd.Series(
            [1.0],
            index=pd.MultiIndex.from_tuples([(dates[0], "000001.SZ")], names=["datetime", "instrument"]),
            name="score",
        )
        prices = pd.DataFrame(
            {
                "open": [10.0, 10.5],
                "close": [10.2, 10.8],
                "volume": [1000.0, 1200.0],
            },
            index=dates,
        )

        with self.assertRaisesRegex(ValueError, "close-price panel"):
            run_backtest(scores, prices, "2024-01-02", "2024-01-03", {"initial_capital": 100000, "top_n": 1})

    def test_calculate_metrics_sortino_uses_downside_deviation_over_all_returns(self) -> None:
        """函数说明：验证 test_calculate_metrics_sortino_uses_downside_deviation_over_all_returns 覆盖的行为场景。"""
        equity = pd.Series(
            [100.0, 110.0, 104.5, 104.5],
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
        )
        returns = equity.pct_change().dropna()
        downside = np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2))
        expected = returns.mean() / downside * np.sqrt(252)

        metrics = calculate_metrics(equity, pd.DataFrame(), {"annual_trading_days": 252, "top_n": 1})

        self.assertAlmostEqual(metrics["sortino"], float(expected))

    def test_max_drawdown_duration_matches_longest_underwater_run(self) -> None:
        """函数说明：验证 test_max_drawdown_duration_matches_longest_underwater_run 覆盖的行为场景。"""
        equity = pd.Series([100.0, 90.0, 95.0, 101.0, 99.0, 98.0, 102.0])

        self.assertEqual(_max_drawdown_duration(equity), 2)

    def test_lot_size_keeps_non_star_boards_at_default_lot(self) -> None:
        """函数说明：验证 test_lot_size_keeps_non_star_boards_at_default_lot 覆盖的行为场景。"""
        config = {"lot_size": 100, "star_market_lot_size": 200}

        self.assertEqual(_lot_size("688001.SH", config), 200)
        self.assertEqual(_lot_size("300001.SZ", config), 100)
        self.assertEqual(_lot_size("830001.BJ", config), 100)

    def test_price_field_cache_prunes_dead_price_frames(self) -> None:
        """函数说明：验证 test_price_field_cache_prunes_dead_price_frames 覆盖的行为场景。"""
        dead_key = -12345
        dead_frame = pd.DataFrame({"A": [1.0]})
        _PRICE_FIELD_CACHE[dead_key] = (weakref.ref(dead_frame), set(), {})
        del dead_frame
        gc.collect()
        prices = pd.concat(
            {"close": pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2024-01-02")])},
            axis=1,
        )
        prices.columns = pd.MultiIndex.from_tuples(prices.columns, names=["field", "instrument"])

        try:
            _field(prices, "close")

            self.assertNotIn(dead_key, _PRICE_FIELD_CACHE)
        finally:
            _PRICE_FIELD_CACHE.pop(dead_key, None)

    def test_price_field_cache_is_size_limited(self) -> None:
        """函数说明：验证价格字段缓存会限制源面板数量。"""
        _PRICE_FIELD_CACHE.clear()
        frames = []
        try:
            for idx in range(_PRICE_FIELD_CACHE_MAX_SIZE + 2):
                prices = pd.concat(
                    {"close": pd.DataFrame({f"A{idx}": [1.0]}, index=[pd.Timestamp("2024-01-02")])},
                    axis=1,
                )
                prices.columns = pd.MultiIndex.from_tuples(prices.columns, names=["field", "instrument"])
                frames.append(prices)
                _field(prices, "close")

            self.assertLessEqual(len(_PRICE_FIELD_CACHE), _PRICE_FIELD_CACHE_MAX_SIZE)
        finally:
            _PRICE_FIELD_CACHE.clear()

    def test_run_backtest_produces_equity_curve(self) -> None:
        """函数说明：验证 test_run_backtest_produces_equity_curve 覆盖的行为场景。"""
        market = require_real_market_data(start="2024-01-02", end="2024-01-12")
        scores = market.factors["LOW0"].rename("score")

        result = run_backtest(
            scores,
            market.prices,
            str(market.start.date()),
            str(market.end.date()),
            {
                "initial_capital": 100000,
                "commission": 0.0003,
                "stamp_tax": 0.001,
                "top_n": 2,
                "max_turnover": 2,
                "annual_trading_days": 252,
            },
        )

        self.assertGreater(len(result.equity_curve), 3)
        self.assertIn("total_return", result.metrics)
        self.assertIn("trade_cost", result.metrics)
        self.assertAlmostEqual(float(result.equity_curve.iloc[0]), 100000.0)
        self.assertGreater(result.metrics["total_return"], -1.0)
        self.assertLess(result.metrics["total_return"], 100.0)
        self.assertGreater(result.metrics["trade_cost"], 0.0)
        self.assertGreater(result.metrics["annual_trade_cost_ratio"], 0.0)
        self.assertFalse(result.trades.empty)
        self.assertTrue(set(result.trades["instrument"]).issubset(set(market.instruments)))

    def test_limit_up_blocks_buy(self) -> None:
        """函数说明：验证 test_limit_up_blocks_buy 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        index = pd.MultiIndex.from_product([dates, ["A"]], names=["datetime", "instrument"])
        scores = pd.Series([10, 10], index=index, name="score")
        prices = pd.concat(
            {
                "close": pd.DataFrame({"A": [10.0, 10.5]}, index=dates),
                "high": pd.DataFrame({"A": [10.0, 11.0]}, index=dates),
                "volume": pd.DataFrame({"A": [1000.0, 1000.0]}, index=dates),
            },
            axis=1,
        )

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-03",
            {
                "initial_capital": 100000,
                "top_n": 1,
                "max_turnover": 1,
                "limit_up_threshold": 0.099,
            },
        )

        self.assertTrue((result.trades["status"] == "blocked").any())
        self.assertEqual(result.equity_curve.iloc[-1], 100000)

    def test_growth_board_uses_twenty_percent_limit_threshold(self) -> None:
        """函数说明：验证 test_growth_board_uses_twenty_percent_limit_threshold 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        stock = "300001.SZ"
        index = pd.MultiIndex.from_product([dates, [stock]], names=["datetime", "instrument"])
        scores = pd.Series([10, 10], index=index, name="score")
        prices = pd.concat(
            {
                "close": pd.DataFrame({stock: [10.0, 11.5]}, index=dates),
                "high": pd.DataFrame({stock: [10.0, 11.5]}, index=dates),
                "volume": pd.DataFrame({stock: [1000.0, 1000.0]}, index=dates),
            },
            axis=1,
        )

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-03",
            {"initial_capital": 100000, "top_n": 1, "max_turnover": 1, "limit_up_threshold": 0.099},
        )

        self.assertTrue((result.trades["side"] == "BUY").any())
        self.assertFalse((result.trades["reason"] == "not_buyable").any())

    def test_growth_board_uses_growth_limit_threshold_when_star_threshold_differs(self) -> None:
        """函数说明：验证 test_growth_board_uses_growth_limit_threshold_when_star_threshold_differs 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        stock = "300001.SZ"
        index = pd.MultiIndex.from_product([dates, [stock]], names=["datetime", "instrument"])
        scores = pd.Series([10, 10], index=index, name="score")
        prices = pd.concat(
            {
                "close": pd.DataFrame({stock: [10.0, 11.2]}, index=dates),
                "high": pd.DataFrame({stock: [10.0, 11.2]}, index=dates),
                "volume": pd.DataFrame({stock: [1000.0, 1000.0]}, index=dates),
            },
            axis=1,
        )

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-03",
            {
                "initial_capital": 100000,
                "top_n": 1,
                "max_turnover": 1,
                "star_limit_up_threshold": 0.099,
                "growth_limit_up_threshold": 0.199,
            },
        )

        self.assertTrue((result.trades["side"] == "BUY").any())
        self.assertFalse((result.trades["reason"] == "not_buyable").any())

    def test_st_uses_five_percent_limit_threshold(self) -> None:
        """函数说明：验证 test_st_uses_five_percent_limit_threshold 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        stock = "600000.SH"
        index = pd.MultiIndex.from_product([dates, [stock]], names=["datetime", "instrument"])
        scores = pd.Series([10, 10], index=index, name="score")
        prices = pd.concat(
            {
                "close": pd.DataFrame({stock: [10.0, 10.6]}, index=dates),
                "high": pd.DataFrame({stock: [10.0, 10.6]}, index=dates),
                "volume": pd.DataFrame({stock: [1000.0, 1000.0]}, index=dates),
                "is_st": pd.DataFrame({stock: [0.0, 1.0]}, index=dates),
            },
            axis=1,
        )

        result = run_backtest(scores, prices, "2024-01-02", "2024-01-03", {"initial_capital": 100000, "top_n": 1})

        blocked = result.trades[result.trades["status"] == "blocked"]
        self.assertFalse(blocked.empty)
        self.assertEqual(blocked.iloc[0]["reason"], "not_buyable")

    def test_capacity_limit_uses_prior_amount_not_trade_day_amount(self) -> None:
        """函数说明：验证 test_capacity_limit_uses_prior_amount_not_trade_day_amount 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        index = pd.MultiIndex.from_product([[dates[0]], ["A"]], names=["datetime", "instrument"])
        scores = pd.Series([10], index=index, name="score")
        prices = pd.concat(
            {
                "open": pd.DataFrame({"A": [10.0, 10.0]}, index=dates),
                "close": pd.DataFrame({"A": [10.0, 10.0]}, index=dates),
                "volume": pd.DataFrame({"A": [1000.0, 1_000_000.0]}, index=dates),
                "amount": pd.DataFrame({"A": [1.0, 1_000_000.0]}, index=dates),
            },
            axis=1,
        )

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-03",
            {
                "initial_capital": 100000,
                "top_n": 1,
                "max_turnover": 1,
                "trade_price_field": "open",
                "max_participation_rate": 0.1,
                "amount_unit": 1000.0,
                "capacity_window": 20,
            },
        )

        blocked = result.trades[result.trades["status"] == "blocked"]
        self.assertFalse(blocked.empty)
        self.assertEqual(blocked.iloc[0]["reason"], "capacity_limited")

    def test_stop_loss_forces_exit(self) -> None:
        """函数说明：验证 test_stop_loss_forces_exit 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        index = pd.MultiIndex.from_product([[dates[0]], ["A"]], names=["datetime", "instrument"])
        scores = pd.Series([10], index=index, name="score")
        prices = pd.concat(
            {
                "close": pd.DataFrame({"A": [10.0, 10.0, 9.4]}, index=dates),
                "volume": pd.DataFrame({"A": [1000.0, 1000.0, 1000.0]}, index=dates),
                "amount": pd.DataFrame({"A": [1000.0, 1000.0, 1000.0]}, index=dates),
            },
            axis=1,
        )

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-04",
            {"initial_capital": 100000, "top_n": 1, "max_turnover": 1, "stop_loss_pct": 0.05},
        )

        risk_trades = result.trades[result.trades["status"] == "risk_exit"]
        self.assertFalse(risk_trades.empty)
        self.assertEqual(risk_trades.iloc[0]["reason"], "stop_loss")

    def test_rebalance_after_risk_exit_refills_from_last_signal_without_rebuying_exit(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        index = pd.MultiIndex.from_product([[dates[0]], ["A", "B"]], names=["datetime", "instrument"])
        scores = pd.Series([10.0, 9.0], index=index, name="score")
        prices = pd.concat(
            {
                "open": pd.DataFrame({"A": [10.0, 10.0, 12.0], "B": [10.0, 10.0, 10.0]}, index=dates),
                "high": pd.DataFrame({"A": [10.0, 10.2, 12.2], "B": [10.0, 10.2, 10.2]}, index=dates),
                "low": pd.DataFrame({"A": [10.0, 9.8, 11.8], "B": [10.0, 9.8, 9.8]}, index=dates),
                "close": pd.DataFrame({"A": [10.0, 10.0, 12.0], "B": [10.0, 10.0, 10.0]}, index=dates),
                "volume": pd.DataFrame({"A": [1000.0, 1000.0, 1000.0], "B": [1000.0, 1000.0, 1000.0]}, index=dates),
                "amount": pd.DataFrame({"A": [1000.0, 1000.0, 1000.0], "B": [1000.0, 1000.0, 1000.0]}, index=dates),
            },
            axis=1,
        )

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-04",
            {
                "initial_capital": 100000,
                "top_n": 1,
                "max_turnover": 1,
                "trade_price_field": "open",
                "take_profit_pct": 0.10,
                "rebalance_after_risk_exit": True,
                "commission": 0.0,
                "stamp_tax": 0.0,
                "slippage": 0.0,
            },
        )

        same_day = result.trades[result.trades["date"] == dates[-1]]
        self.assertEqual(same_day[same_day["status"] == "risk_exit"].iloc[0]["instrument"], "A")
        buys = same_day[same_day["side"] == "BUY"]
        self.assertEqual(buys.iloc[0]["instrument"], "B")
        self.assertNotIn("A", buys["instrument"].tolist())

    def test_risk_exit_min_positions_blocks_exit_that_would_reduce_position_count(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        index = pd.MultiIndex.from_product([[dates[0]], ["A", "B"]], names=["datetime", "instrument"])
        scores = pd.Series([10.0, 9.0], index=index, name="score")
        prices = pd.concat(
            {
                "open": pd.DataFrame({"A": [10.0, 10.0, 14.0], "B": [10.0, 10.0, 10.0]}, index=dates),
                "high": pd.DataFrame({"A": [10.0, 10.0, 14.0], "B": [10.0, 10.0, 10.0]}, index=dates),
                "low": pd.DataFrame({"A": [10.0, 10.0, 14.0], "B": [10.0, 10.0, 10.0]}, index=dates),
                "close": pd.DataFrame({"A": [10.0, 10.0, 14.0], "B": [10.0, 10.0, 10.0]}, index=dates),
                "volume": pd.DataFrame({"A": [1000.0, 1000.0, 1000.0], "B": [1000.0, 1000.0, 1000.0]}, index=dates),
                "amount": pd.DataFrame({"A": [1000.0, 1000.0, 1000.0], "B": [1000.0, 1000.0, 1000.0]}, index=dates),
            },
            axis=1,
        )

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-04",
            {
                "initial_capital": 100000,
                "top_n": 2,
                "max_turnover": 2,
                "take_profit_pct": 0.10,
                "risk_exit_min_positions": 2,
                "commission": 0.0,
                "stamp_tax": 0.0,
                "slippage": 0.0,
            },
        )

        blocked = result.trades[result.trades["reason"] == "take_profit_min_positions"]
        self.assertEqual(len(blocked), 1)
        self.assertEqual(blocked.iloc[0]["status"], "blocked")
        latest = result.holdings[result.holdings["date"].astype(str) == "2024-01-04"]
        self.assertEqual(set(latest["instrument"]), {"A", "B"})

    def test_stop_loss_exit_respects_capacity_limit_and_keeps_remaining_shares(self) -> None:
        """函数说明：验证 test_stop_loss_exit_respects_capacity_limit_and_keeps_remaining_shares 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        index = pd.MultiIndex.from_product([[dates[0]], ["A"]], names=["datetime", "instrument"])
        scores = pd.Series([10], index=index, name="score")
        prices = pd.concat(
            {
                "open": pd.DataFrame({"A": [10.0, 10.0, 9.6]}, index=dates),
                "high": pd.DataFrame({"A": [10.0, 10.2, 9.7]}, index=dates),
                "low": pd.DataFrame({"A": [10.0, 9.9, 9.4]}, index=dates),
                "close": pd.DataFrame({"A": [10.0, 10.0, 9.4]}, index=dates),
                "volume": pd.DataFrame({"A": [1000.0, 1000.0, 1000.0]}, index=dates),
                "amount": pd.DataFrame({"A": [1000.0, 100.0, 1000.0]}, index=dates),
            },
            axis=1,
        )

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-04",
            {
                "initial_capital": 100000,
                "top_n": 1,
                "max_turnover": 1,
                "trade_price_field": "open",
                "stop_loss_pct": 0.05,
                "max_participation_rate": 0.1,
                "amount_unit": 1000.0,
                "capacity_window": 1,
                "commission": 0.0,
                "stamp_tax": 0.0,
                "slippage": 0.0,
            },
        )

        sell = result.trades[(result.trades["side"] == "SELL") & (result.trades["reason"] == "stop_loss_capacity_limited")].iloc[0]
        self.assertEqual(sell["status"], "partial")
        self.assertEqual(int(sell["shares"]), 1000)
        final_holding = result.holdings[result.holdings["date"] == dates[-1]].iloc[0]
        self.assertEqual(int(final_holding["shares"]), 9000)

    def test_stop_loss_uses_intraday_trigger_not_close_fill(self) -> None:
        """函数说明：验证 test_stop_loss_uses_intraday_trigger_not_close_fill 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        index = pd.MultiIndex.from_product([[dates[0]], ["A"]], names=["datetime", "instrument"])
        scores = pd.Series([10], index=index, name="score")
        prices = pd.concat(
            {
                "open": pd.DataFrame({"A": [10.0, 10.0, 10.4]}, index=dates),
                "high": pd.DataFrame({"A": [10.0, 10.2, 10.5]}, index=dates),
                "low": pd.DataFrame({"A": [10.0, 9.8, 9.2]}, index=dates),
                "close": pd.DataFrame({"A": [10.0, 10.1, 10.3]}, index=dates),
                "volume": pd.DataFrame({"A": [1000.0, 1000.0, 1000.0]}, index=dates),
                "amount": pd.DataFrame({"A": [1000.0, 1000.0, 1000.0]}, index=dates),
            },
            axis=1,
        )

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-04",
            {
                "initial_capital": 100000,
                "top_n": 1,
                "max_turnover": 1,
                "trade_price_field": "open",
                "stop_loss_pct": 0.05,
                "limit_down_threshold": 0.2,
                "slippage": 0.0,
            },
        )

        risk_trade = result.trades[result.trades["status"] == "risk_exit"].iloc[0]
        self.assertEqual(risk_trade["reason"], "stop_loss")
        self.assertAlmostEqual(float(risk_trade["price"]), 9.4525)
        self.assertLess(float(risk_trade["price"]), 9.5)

    def test_gap_down_stop_loss_uses_open_price(self) -> None:
        """函数说明：验证 test_gap_down_stop_loss_uses_open_price 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        index = pd.MultiIndex.from_product([[dates[0]], ["A"]], names=["datetime", "instrument"])
        scores = pd.Series([10], index=index, name="score")
        prices = pd.concat(
            {
                "open": pd.DataFrame({"A": [10.0, 10.0, 9.0]}, index=dates),
                "high": pd.DataFrame({"A": [10.0, 10.2, 9.4]}, index=dates),
                "low": pd.DataFrame({"A": [10.0, 9.8, 8.8]}, index=dates),
                "close": pd.DataFrame({"A": [10.0, 10.1, 9.2]}, index=dates),
                "volume": pd.DataFrame({"A": [1000.0, 1000.0, 1000.0]}, index=dates),
                "amount": pd.DataFrame({"A": [1000.0, 1000.0, 1000.0]}, index=dates),
            },
            axis=1,
        )

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-04",
            {
                "initial_capital": 20000,
                "top_n": 1,
                "max_turnover": 1,
                "trade_price_field": "open",
                "stop_loss_pct": 0.05,
                "limit_down_threshold": 0.2,
                "slippage": 0.0,
            },
        )

        risk_trade = result.trades[result.trades["status"] == "risk_exit"].iloc[0]
        self.assertAlmostEqual(float(risk_trade["price"]), 9.0)

    def test_capacity_warning_is_recorded(self) -> None:
        """函数说明：验证 test_capacity_warning_is_recorded 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        index = pd.MultiIndex.from_product([[dates[0]], ["A"]], names=["datetime", "instrument"])
        scores = pd.Series([10], index=index, name="score")
        prices = pd.concat(
            {
                "close": pd.DataFrame({"A": [10.0, 10.0]}, index=dates),
                "volume": pd.DataFrame({"A": [1000.0, 1000.0]}, index=dates),
                "amount": pd.DataFrame({"A": [1.0, 1.0]}, index=dates),
            },
            axis=1,
        )

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-03",
            {
                "initial_capital": 100000,
                "top_n": 1,
                "max_turnover": 1,
                "capacity_warning_threshold": 0.05,
                "amount_unit": 1000.0,
            },
        )

        filled = result.trades[result.trades["status"] == "filled"]
        self.assertTrue(bool(filled.iloc[0]["capacity_warning"]))

    def test_dynamic_slippage_increases_with_adv_participation(self) -> None:
        """函数说明：验证 test_dynamic_slippage_increases_with_adv_participation 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        index = pd.MultiIndex.from_product([[dates[0]], ["A"]], names=["datetime", "instrument"])
        scores = pd.Series([10], index=index, name="score")
        prices = pd.concat(
            {
                "open": pd.DataFrame({"A": [10.0, 10.0]}, index=dates),
                "close": pd.DataFrame({"A": [10.0, 10.0]}, index=dates),
                "volume": pd.DataFrame({"A": [1000.0, 1000.0]}, index=dates),
                "amount": pd.DataFrame({"A": [1.0, 1.0]}, index=dates),
            },
            axis=1,
        )

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-03",
            {
                "initial_capital": 100000,
                "top_n": 1,
                "max_turnover": 1,
                "trade_price_field": "open",
                "slippage": 0.001,
                "dynamic_slippage_enabled": True,
                "dynamic_slippage_threshold": 0.02,
                "dynamic_slippage_multiplier": 0.1,
                "max_slippage": 0.05,
                "amount_unit": 1000.0,
            },
        )

        trade = result.trades[result.trades["side"] == "BUY"].iloc[0]
        self.assertGreater(float(trade["slippage_rate"]), 0.001)
        self.assertEqual(trade["slippage_model"], "dynamic_adv")

    def test_min_commission_and_transfer_fee_are_recorded(self) -> None:
        """函数说明：验证 test_min_commission_and_transfer_fee_are_recorded 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        index = pd.MultiIndex.from_product([[dates[0]], ["A"]], names=["datetime", "instrument"])
        scores = pd.Series([10], index=index, name="score")
        prices = pd.DataFrame({"A": [10.0, 10.0]}, index=dates)

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-03",
            {
                "initial_capital": 20000,
                "top_n": 1,
                "max_turnover": 1,
                "commission": 0.0001,
                "min_commission_per_order": 5.0,
                "transfer_fee": 0.00001,
            },
        )

        trade = result.trades.iloc[0]
        self.assertEqual(float(trade["commission_cost"]), 5.0)
        self.assertGreater(float(trade["transfer_fee_cost"]), 0.0)
        self.assertGreater(result.metrics["transfer_fee_cost"], 0.0)

    def test_stale_price_exit_applies_haircut(self) -> None:
        """函数说明：验证 test_stale_price_exit_applies_haircut 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
        index = pd.MultiIndex.from_product([[dates[0]], ["A"]], names=["datetime", "instrument"])
        scores = pd.Series([10], index=index, name="score")
        prices = pd.concat(
            {
                "close": pd.DataFrame({"A": [10.0, 10.0, None, None]}, index=dates),
                "volume": pd.DataFrame({"A": [1000.0, 1000.0, 0.0, 0.0]}, index=dates),
                "amount": pd.DataFrame({"A": [1000.0, 1000.0, 0.0, 0.0]}, index=dates),
            },
            axis=1,
        )

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-05",
            {
                "initial_capital": 100000,
                "top_n": 1,
                "max_turnover": 1,
                "stale_price_exit_days": 2,
                "stale_price_exit_policy": "haircut_exit",
                "stale_price_haircut": 0.5,
                "slippage": 0.0,
            },
        )

        stale_trade = result.trades[result.trades["reason"] == "stale_price_exit"].iloc[0]
        self.assertEqual(stale_trade["status"], "risk_exit")
        self.assertAlmostEqual(float(stale_trade["price"]), 5.0)

    def test_stale_price_exit_haircuts_by_default(self) -> None:
        """函数说明：验证 test_stale_price_exit_haircuts_by_default 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
        index = pd.MultiIndex.from_product([[dates[0]], ["A"]], names=["datetime", "instrument"])
        scores = pd.Series([10], index=index, name="score")
        prices = pd.concat(
            {
                "close": pd.DataFrame({"A": [10.0, 10.0, None, None]}, index=dates),
                "volume": pd.DataFrame({"A": [1000.0, 1000.0, 0.0, 0.0]}, index=dates),
                "amount": pd.DataFrame({"A": [1000.0, 1000.0, 0.0, 0.0]}, index=dates),
            },
            axis=1,
        )

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-05",
            {
                "initial_capital": 100000,
                "top_n": 1,
                "max_turnover": 1,
                "stale_price_exit_days": 2,
                "slippage": 0.0,
            },
        )

        stale_trade = result.trades[result.trades["reason"] == "stale_price_exit"].iloc[0]
        self.assertEqual(stale_trade["status"], "risk_exit")
        self.assertAlmostEqual(float(stale_trade["price"]), 5.0)

    def test_stale_price_exit_does_not_require_missing_volume_field(self) -> None:
        """函数说明：验证 test_stale_price_exit_does_not_require_missing_volume_field 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
        index = pd.MultiIndex.from_product([[dates[0]], ["A"]], names=["datetime", "instrument"])
        scores = pd.Series([10], index=index, name="score")
        prices = pd.DataFrame({"A": [10.0, 10.0, 10.0, 10.0]}, index=dates)

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-05",
            {
                "initial_capital": 100000,
                "top_n": 1,
                "max_turnover": 1,
                "stale_price_exit_days": 2,
                "slippage": 0.0,
            },
        )

        self.assertFalse((result.trades["reason"] == "stale_price_exit").any())
        self.assertEqual(result.holdings[result.holdings["date"] == dates[-1]]["instrument"].tolist(), ["A"])

    def test_circuit_breaker_cooldown_allows_reentry(self) -> None:
        """函数说明：验证 test_circuit_breaker_cooldown_allows_reentry 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"])
        index = pd.MultiIndex.from_product([dates[:-1], ["A"]], names=["datetime", "instrument"])
        scores = pd.Series(10, index=index, name="score")
        prices = pd.concat(
            {
                "close": pd.DataFrame({"A": [10.0, 10.0, 9.0, 9.0, 9.0, 10.0]}, index=dates),
                "volume": pd.DataFrame({"A": [1000.0] * len(dates)}, index=dates),
                "amount": pd.DataFrame({"A": [1000.0] * len(dates)}, index=dates),
            },
            axis=1,
        )

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-09",
            {
                "initial_capital": 100000,
                "top_n": 1,
                "max_turnover": 1,
                "circuit_breaker_drawdown": 0.05,
                "circuit_breaker_cooldown_days": 1,
                "commission": 0.0,
                "stamp_tax": 0.0,
                "slippage": 0.0,
            },
        )

        circuit_sell = result.trades[result.trades["reason"] == "circuit_breaker"].iloc[0]
        later_buys = result.trades[
            (result.trades["side"] == "BUY")
            & (pd.to_datetime(result.trades["date"]) > pd.Timestamp(circuit_sell["date"]))
        ]
        self.assertFalse(later_buys.empty)

    def test_circuit_breaker_cooldown_keeps_historical_peak_for_second_drawdown(self) -> None:
        """函数说明：验证 test_circuit_breaker_cooldown_keeps_historical_peak_for_second_drawdown 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"])
        index = pd.MultiIndex.from_product([[dates[0], dates[4]], ["A"]], names=["datetime", "instrument"])
        scores = pd.Series(10, index=index, name="score")
        prices = pd.concat(
            {
                "close": pd.DataFrame({"A": [10.0, 10.0, 12.0, 11.3, 12.0, 12.0, 11.55]}, index=dates),
                "open": pd.DataFrame({"A": [10.0, 10.0, 12.0, 11.3, 12.0, 12.0, 11.55]}, index=dates),
                "volume": pd.DataFrame({"A": [100000.0] * len(dates)}, index=dates),
                "amount": pd.DataFrame({"A": [1000000.0] * len(dates)}, index=dates),
            },
            axis=1,
        )

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-10",
            {
                "initial_capital": 100000,
                "top_n": 1,
                "max_turnover": 1,
                "circuit_breaker_drawdown": 0.05,
                "circuit_breaker_cooldown_days": 1,
                "circuit_breaker_target_exposure": 0.5,
                "commission": 0.0,
                "stamp_tax": 0.0,
                "slippage": 0.0,
            },
        )

        circuit_sells = result.trades[
            (result.trades["side"] == "SELL") & (result.trades["reason"] == "circuit_breaker")
        ]
        self.assertEqual(
            pd.to_datetime(circuit_sells["date"]).dt.strftime("%Y-%m-%d").tolist(),
            ["2024-01-05", "2024-01-08", "2024-01-10"],
        )

    def test_circuit_breaker_keeps_historical_peak_after_cooldown(self) -> None:
        """函数说明：验证 test_circuit_breaker_keeps_historical_peak_after_cooldown 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"])
        signal_dates = pd.to_datetime(["2024-01-02", "2024-01-05"])
        index = pd.MultiIndex.from_product([signal_dates, ["A"]], names=["datetime", "instrument"])
        scores = pd.Series(10, index=index, name="score")
        prices = pd.concat(
            {
                "close": pd.DataFrame({"A": [10.0, 10.0, 9.0, 9.0, 9.5, 9.2]}, index=dates),
                "volume": pd.DataFrame({"A": [1000.0] * len(dates)}, index=dates),
                "amount": pd.DataFrame({"A": [1000.0] * len(dates)}, index=dates),
            },
            axis=1,
        )

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-09",
            {
                "initial_capital": 100000,
                "top_n": 1,
                "max_turnover": 1,
                "circuit_breaker_drawdown": 0.08,
                "circuit_breaker_cooldown_days": 1,
                "commission": 0.0,
                "stamp_tax": 0.0,
                "slippage": 0.0,
                "limit_down_threshold": 1.0,
            },
        )

        circuit_sells = result.trades[(result.trades["side"] == "SELL") & (result.trades["reason"] == "circuit_breaker")]
        self.assertEqual(pd.to_datetime(circuit_sells["date"]).tolist(), [pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-09")])

    def test_circuit_breaker_target_exposure_reduces_instead_of_liquidates(self) -> None:
        """函数说明：验证 test_circuit_breaker_target_exposure_reduces_instead_of_liquidates 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        index = pd.MultiIndex.from_product([[dates[0]], ["A"]], names=["datetime", "instrument"])
        scores = pd.Series(10, index=index, name="score")
        prices = pd.concat(
            {
                "close": pd.DataFrame({"A": [10.0, 10.0, 9.0]}, index=dates),
                "open": pd.DataFrame({"A": [10.0, 10.0, 9.0]}, index=dates),
                "volume": pd.DataFrame({"A": [1000.0] * len(dates)}, index=dates),
                "amount": pd.DataFrame({"A": [1000.0] * len(dates)}, index=dates),
            },
            axis=1,
        )

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-04",
            {
                "initial_capital": 100000,
                "top_n": 1,
                "max_turnover": 1,
                "circuit_breaker_drawdown": 0.05,
                "circuit_breaker_cooldown_days": 1,
                "circuit_breaker_target_exposure": 0.30,
                "commission": 0.0,
                "stamp_tax": 0.0,
                "slippage": 0.0,
                "limit_down_threshold": 1.0,
            },
        )

        circuit_sell = result.trades[result.trades["reason"] == "circuit_breaker"].iloc[0]
        self.assertEqual(int(circuit_sell["shares"]), 7000)
        final_holding = result.holdings[result.holdings["date"] == dates[-1]].iloc[0]
        self.assertEqual(int(final_holding["shares"]), 3000)

    def test_circuit_breaker_allows_null_cooldown_days(self) -> None:
        """函数说明：验证 test_circuit_breaker_allows_null_cooldown_days 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        index = pd.MultiIndex.from_product([[dates[0]], ["A"]], names=["datetime", "instrument"])
        scores = pd.Series(10, index=index, name="score")
        prices = pd.concat(
            {
                "close": pd.DataFrame({"A": [10.0, 10.0, 9.0]}, index=dates),
                "open": pd.DataFrame({"A": [10.0, 10.0, 9.0]}, index=dates),
                "volume": pd.DataFrame({"A": [1000.0] * len(dates)}, index=dates),
                "amount": pd.DataFrame({"A": [1000.0] * len(dates)}, index=dates),
            },
            axis=1,
        )

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-04",
            {
                "initial_capital": 100000,
                "top_n": 1,
                "max_turnover": 1,
                "circuit_breaker_drawdown": 0.05,
                "circuit_breaker_cooldown_days": None,
                "commission": 0.0,
                "stamp_tax": 0.0,
                "slippage": 0.0,
            },
        )

        self.assertFalse(result.trades.empty)

    def test_circuit_breaker_rejects_invalid_cooldown_days(self) -> None:
        """函数说明：验证 test_circuit_breaker_rejects_invalid_cooldown_days 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        scores = pd.Series(
            [10],
            index=pd.MultiIndex.from_tuples([(dates[0], "A")], names=["datetime", "instrument"]),
            name="score",
        )
        prices = pd.concat(
            {
                "close": pd.DataFrame({"A": [10.0, 10.0]}, index=dates),
                "volume": pd.DataFrame({"A": [1000.0, 1000.0]}, index=dates),
                "amount": pd.DataFrame({"A": [1000.0, 1000.0]}, index=dates),
            },
            axis=1,
        )

        with self.assertRaisesRegex(ValueError, "circuit_breaker_cooldown_days"):
            run_backtest(
                scores,
                prices,
                "2024-01-02",
                "2024-01-03",
                {
                    "initial_capital": 100000,
                    "top_n": 1,
                    "circuit_breaker_cooldown_days": "bad",
                },
            )

    def test_annual_drawdown_guard_resets_next_year(self) -> None:
        """函数说明：验证 test_annual_drawdown_guard_resets_next_year 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2025-01-02", "2025-01-03"])
        index = pd.MultiIndex.from_product([[dates[0], dates[3]], ["A"]], names=["datetime", "instrument"])
        scores = pd.Series([10, 10], index=index, name="score")
        prices = pd.concat(
            {
                "close": pd.DataFrame({"A": [10.0, 10.0, 8.0, 8.0, 8.0]}, index=dates),
                "open": pd.DataFrame({"A": [10.0, 10.0, 8.0, 8.0, 8.0]}, index=dates),
                "volume": pd.DataFrame({"A": [1000.0] * len(dates)}, index=dates),
                "amount": pd.DataFrame({"A": [1000.0] * len(dates)}, index=dates),
            },
            axis=1,
        )

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2025-01-03",
            {
                "initial_capital": 100000,
                "top_n": 1,
                "max_turnover": 1,
                "annual_drawdown_guard": {"enabled": True, "drawdown": 0.10, "target_exposure": 0.0},
                "commission": 0.0,
                "stamp_tax": 0.0,
                "slippage": 0.0,
                "limit_down_threshold": 1.0,
            },
        )

        guard_sell = result.trades[result.trades["reason"] == "annual_drawdown_guard"].iloc[0]
        self.assertEqual(pd.Timestamp(guard_sell["date"]), dates[2])
        later_buys = result.trades[
            (result.trades["side"] == "BUY")
            & (pd.to_datetime(result.trades["date"]) > pd.Timestamp(guard_sell["date"]))
        ]
        self.assertFalse(later_buys.empty)

    def test_annual_drawdown_guard_release_allows_same_year_reentry(self) -> None:
        """函数说明：验证 test_annual_drawdown_guard_release_allows_same_year_reentry 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])
        index = pd.MultiIndex.from_product([[dates[0], dates[3]], ["A"]], names=["datetime", "instrument"])
        scores = pd.Series([10, 10], index=index, name="score")
        prices = pd.concat(
            {
                "close": pd.DataFrame({"A": [10.0, 10.0, 8.0, 9.5, 9.5]}, index=dates),
                "open": pd.DataFrame({"A": [10.0, 10.0, 8.0, 9.5, 9.5]}, index=dates),
                "volume": pd.DataFrame({"A": [1000.0] * len(dates)}, index=dates),
                "amount": pd.DataFrame({"A": [1000.0] * len(dates)}, index=dates),
            },
            axis=1,
        )

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-08",
            {
                "initial_capital": 100000,
                "top_n": 1,
                "max_turnover": 1,
                "annual_drawdown_guard": {
                    "enabled": True,
                    "drawdown": 0.18,
                    "release_drawdown": 0.15,
                    "target_exposure": 0.50,
                },
                "commission": 0.0,
                "stamp_tax": 0.0,
                "slippage": 0.0,
                "limit_down_threshold": 1.0,
            },
        )

        guard_sell = result.trades[result.trades["reason"] == "annual_drawdown_guard"].iloc[0]
        later_buys = result.trades[
            (result.trades["side"] == "BUY")
            & (pd.to_datetime(result.trades["date"]) > pd.Timestamp(guard_sell["date"]))
        ]
        self.assertFalse(later_buys.empty)

    def test_exposure_schedule_change_rebalances_with_latest_signal(self) -> None:
        """函数说明：验证 test_exposure_schedule_change_rebalances_with_latest_signal 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        index = pd.MultiIndex.from_product([[dates[0]], ["A"]], names=["datetime", "instrument"])
        scores = pd.Series([10], index=index, name="score")
        prices = pd.concat(
            {
                "close": pd.DataFrame({"A": [10.0, 10.0, 10.0]}, index=dates),
                "volume": pd.DataFrame({"A": [1000.0, 1000.0, 1000.0]}, index=dates),
                "amount": pd.DataFrame({"A": [1000.0, 1000.0, 1000.0]}, index=dates),
            },
            axis=1,
        )
        exposure = pd.Series([1.0, 1.0, 0.5], index=dates)

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-04",
            {
                "initial_capital": 100000,
                "top_n": 1,
                "max_turnover": 1,
                "commission": 0.0,
                "stamp_tax": 0.0,
                "slippage": 0.0,
                "exposure_schedule": exposure,
                "exposure_rebalance_threshold": 0.1,
                "rebalance_drift_threshold": 1.0,
            },
        )

        sells = result.trades[result.trades["side"] == "SELL"]
        self.assertFalse(sells.empty)
        self.assertEqual(pd.Timestamp(sells.iloc[0]["date"]), dates[-1])

    def test_exposure_schedule_uses_latest_intraday_value_per_trade_date(self) -> None:
        """函数说明：验证 test_exposure_schedule_uses_latest_intraday_value_per_trade_date 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        index = pd.MultiIndex.from_product([[dates[0]], ["A"]], names=["datetime", "instrument"])
        scores = pd.Series([10], index=index, name="score")
        prices = pd.concat(
            {
                "close": pd.DataFrame({"A": [10.0, 10.0, 10.0]}, index=dates),
                "volume": pd.DataFrame({"A": [1000.0, 1000.0, 1000.0]}, index=dates),
                "amount": pd.DataFrame({"A": [1000.0, 1000.0, 1000.0]}, index=dates),
            },
            axis=1,
        )
        exposure = pd.Series(
            [0.5, 1.0],
            index=pd.to_datetime(["2024-01-03 15:00", "2024-01-03 09:30"]),
        )

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-04",
            {
                "initial_capital": 100000,
                "top_n": 1,
                "max_turnover": 1,
                "commission": 0.0,
                "stamp_tax": 0.0,
                "slippage": 0.0,
                "exposure_schedule": exposure,
            },
        )

        buy = result.trades[result.trades["side"] == "BUY"].iloc[0]
        self.assertEqual(int(buy["shares"]), 5000)

    def test_exposure_schedule_signal_only_does_not_rebalance_between_signals(self) -> None:
        """函数说明：验证 test_exposure_schedule_signal_only_does_not_rebalance_between_signals 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        index = pd.MultiIndex.from_product([[dates[0]], ["A"]], names=["datetime", "instrument"])
        scores = pd.Series([10], index=index, name="score")
        prices = pd.concat(
            {
                "close": pd.DataFrame({"A": [10.0, 10.0, 10.0]}, index=dates),
                "volume": pd.DataFrame({"A": [1000.0, 1000.0, 1000.0]}, index=dates),
                "amount": pd.DataFrame({"A": [1000.0, 1000.0, 1000.0]}, index=dates),
            },
            axis=1,
        )
        exposure = pd.Series([1.0, 1.0, 0.5], index=dates)

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-04",
            {
                "initial_capital": 100000,
                "top_n": 1,
                "max_turnover": 1,
                "commission": 0.0,
                "stamp_tax": 0.0,
                "slippage": 0.0,
                "exposure_schedule": exposure,
                "exposure_rebalance_threshold": 0.1,
                "exposure_schedule_rebalance_on_signal_only": True,
                "rebalance_drift_threshold": 1.0,
            },
        )

        self.assertTrue(result.trades[result.trades["side"] == "SELL"].empty)

    def test_rebalance_drift_threshold_skips_small_weight_trades(self) -> None:
        """函数说明：验证 test_rebalance_drift_threshold_skips_small_weight_trades 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        index = pd.MultiIndex.from_product([dates[:2], ["A", "B"]], names=["datetime", "instrument"])
        scores = pd.Series([2.0, 1.0, 2.0, 1.0], index=index, name="score")
        prices = pd.DataFrame({"A": [10.0, 10.0, 11.0], "B": [10.0, 10.0, 10.0]}, index=dates)

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-04",
            {
                "initial_capital": 100000,
                "top_n": 2,
                "max_turnover": 2,
                "commission": 0.0,
                "stamp_tax": 0.0,
                "slippage": 0.0,
                "rebalance_drift_threshold": 0.03,
            },
        )

        self.assertEqual(result.trades["side"].tolist(), ["BUY", "BUY"])

    def test_equity_overlay_rebalances_from_last_signal_after_drawdown(self) -> None:
        """函数说明：验证 test_equity_overlay_rebalances_from_last_signal_after_drawdown 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
        index = pd.MultiIndex.from_product([[dates[0]], ["A"]], names=["datetime", "instrument"])
        scores = pd.Series([1.0], index=index, name="score")
        prices = pd.DataFrame({"A": [10.0, 10.0, 8.0, 8.0]}, index=dates)

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-05",
            {
                "initial_capital": 100000,
                "top_n": 1,
                "max_turnover": 1,
                "commission": 0.0,
                "stamp_tax": 0.0,
                "slippage": 0.0,
                "equity_overlay": {
                    "enabled": True,
                    "ma_window": 2,
                    "momentum_window": 1,
                    "drawdown_window": 2,
                    "drawdown_cut": 0.05,
                    "sideways_exposure": 0.5,
                    "bear_exposure": 0.5,
                    "rebalance_threshold": 0.0,
                },
            },
        )

        sells = result.trades[result.trades["side"] == "SELL"]
        self.assertFalse(sells.empty)
        self.assertEqual(pd.Timestamp(sells.iloc[0]["signal_date"]), dates[0])
        self.assertEqual(pd.Timestamp(sells.iloc[0]["date"]), dates[-1])

    def test_equity_overlay_signal_only_does_not_rebalance_between_signals(self) -> None:
        """函数说明：验证 test_equity_overlay_signal_only_does_not_rebalance_between_signals 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
        index = pd.MultiIndex.from_product([[dates[0]], ["A"]], names=["datetime", "instrument"])
        scores = pd.Series([1.0], index=index, name="score")
        prices = pd.DataFrame({"A": [10.0, 10.0, 8.0, 8.0]}, index=dates)

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-05",
            {
                "initial_capital": 100000,
                "top_n": 1,
                "max_turnover": 1,
                "commission": 0.0,
                "stamp_tax": 0.0,
                "slippage": 0.0,
                "equity_overlay": {
                    "enabled": True,
                    "ma_window": 2,
                    "momentum_window": 1,
                    "drawdown_window": 2,
                    "drawdown_cut": 0.05,
                    "sideways_exposure": 0.5,
                    "bear_exposure": 0.5,
                    "rebalance_threshold": 0.0,
                    "rebalance_on_signal_only": True,
                },
            },
        )

        self.assertTrue(result.trades[result.trades["side"] == "SELL"].empty)

    def test_rebalance_drift_threshold_does_not_keep_dropped_holding(self) -> None:
        """函数说明：验证 test_rebalance_drift_threshold_does_not_keep_dropped_holding 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        index = pd.MultiIndex.from_product([dates[:2], ["A", "B", "C"]], names=["datetime", "instrument"])
        scores = pd.Series([3.0, 2.0, 1.0, 1.0, 3.0, 2.0], index=index, name="score")
        prices = pd.DataFrame({"A": [10.0, 10.0, 10.0], "B": [10.0, 10.0, 10.0], "C": [10.0, 10.0, 10.0]}, index=dates)

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-04",
            {
                "initial_capital": 100000,
                "top_n": 2,
                "max_turnover": 2,
                "commission": 0.0,
                "stamp_tax": 0.0,
                "slippage": 0.0,
                "rebalance_drift_threshold": 1.0,
            },
        )

        sells = result.trades[result.trades["side"] == "SELL"]
        buys = result.trades[result.trades["side"] == "BUY"]
        self.assertIn("A", sells["instrument"].tolist())
        self.assertIn("C", buys["instrument"].tolist())

    def test_score_weighted_backtest_allocates_more_to_higher_scores(self) -> None:
        """函数说明：验证 test_score_weighted_backtest_allocates_more_to_higher_scores 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        index = pd.MultiIndex.from_product([[dates[0]], ["A", "B"]], names=["datetime", "instrument"])
        scores = pd.Series([10.0, 1.0], index=index, name="score")
        prices = pd.DataFrame({"A": [10.0, 10.0], "B": [10.0, 10.0]}, index=dates)

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-03",
            {
                "initial_capital": 100000,
                "top_n": 2,
                "max_turnover": 2,
                "commission": 0.0,
                "stamp_tax": 0.0,
                "slippage": 0.0,
                "score_weighted": True,
            },
        )

        buys = result.trades[result.trades["side"] == "BUY"].set_index("instrument")
        self.assertGreater(int(buys.loc["A", "shares"]), int(buys.loc["B", "shares"]))

    def test_backtest_applies_max_industry_weight_to_selection(self) -> None:
        """函数说明：验证 test_backtest_applies_max_industry_weight_to_selection 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        index = pd.MultiIndex.from_product([[dates[0]], ["A", "B", "C", "D"]], names=["datetime", "instrument"])
        scores = pd.Series([10.0, 9.0, 8.0, 7.0], index=index, name="score")
        prices = pd.DataFrame({code: [10.0, 10.0] for code in ["A", "B", "C", "D"]}, index=dates)

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-03",
            {
                "initial_capital": 100000,
                "top_n": 3,
                "max_turnover": 3,
                "commission": 0.0,
                "stamp_tax": 0.0,
                "slippage": 0.0,
                "industry_map": pd.Series({"A": "bank", "B": "bank", "C": "tech", "D": "health"}),
                "max_industry_weight": 0.5,
            },
        )

        bought = result.trades[result.trades["side"] == "BUY"]["instrument"].tolist()
        self.assertEqual(bought, ["A", "C", "D"])

    def test_backtest_applies_selection_risk_filter_on_signal_date(self) -> None:
        """函数说明：验证 test_backtest_applies_selection_risk_filter_on_signal_date 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        index = pd.MultiIndex.from_product([[dates[1]], ["A", "B"]], names=["datetime", "instrument"])
        scores = pd.Series([2.0, 1.0], index=index, name="score")
        prices = pd.concat(
            {
                "open": pd.DataFrame({"A": [10.0, 9.0, 9.2], "B": [10.0, 10.1, 10.2]}, index=dates),
                "close": pd.DataFrame({"A": [10.0, 9.0, 9.2], "B": [10.0, 10.1, 10.2]}, index=dates),
                "low": pd.DataFrame({"A": [10.0, 9.0, 9.1], "B": [10.0, 10.0, 10.1]}, index=dates),
                "volume": pd.DataFrame({"A": [1000.0, 1000.0, 1000.0], "B": [1000.0, 1000.0, 1000.0]}, index=dates),
            },
            axis=1,
        )

        result = run_backtest(
            scores,
            prices,
            "2024-01-02",
            "2024-01-04",
            {
                "initial_capital": 100000,
                "top_n": 1,
                "max_turnover": 1,
                "commission": 0.0,
                "stamp_tax": 0.0,
                "slippage": 0.0,
                "limit_down_threshold": 0.099,
                "selection_risk_filter": {
                    "enabled": True,
                    "lookback_sessions": 2,
                    "required_price_fields": ["open", "close"],
                    "max_missing_price_sessions": 0,
                    "max_limit_down_days": 0,
                    "require_positive_volume": True,
                },
            },
        )

        bought = result.trades[result.trades["side"] == "BUY"]["instrument"].tolist()
        self.assertEqual(bought, ["B"])


if __name__ == "__main__":
    unittest.main()
