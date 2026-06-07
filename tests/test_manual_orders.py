from __future__ import annotations

from pathlib import Path
import unittest
from tempfile import TemporaryDirectory

import pandas as pd

from src.manual_orders import (
    AccountState,
    apply_fill_feedback,
    generate_fill_feedback_template,
    generate_manual_orders,
    generate_order_confirmation_template,
    load_current_holdings,
    save_execution_templates,
    validate_fill_feedback,
)


class ManualOrdersTests(unittest.TestCase):
    def test_generate_manual_orders_calculates_target_and_order_shares(self) -> None:
        signal = pd.DataFrame(
            [
                {"date": "2024-01-03", "instrument": "000001.SZ", "action": "HOLD"},
                {"date": "2024-01-03", "instrument": "600519.SH", "action": "BUY"},
                {"date": "2024-01-03", "instrument": "000002.SZ", "action": "SELL"},
            ]
        )
        prices = _prices(
            ["2024-01-03", "2024-01-04"],
            {
                "000001.SZ": [10.0, 20.0],
                "600519.SH": [100.0, 200.0],
                "000002.SZ": [20.0, 40.0],
            },
        )
        current = pd.DataFrame({"instrument": ["000001.SZ", "000002.SZ"], "shares": [3000, 500]})
        account = AccountState(
            total_asset=100000,
            cash=10000,
            max_position_pct=None,
            lot_size=100,
            star_market_lot_size=200,
            source_file="",
            holdings_file="",
            holdings_loaded=True,
        )

        orders = generate_manual_orders(
            signal,
            ["000001.SZ", "600519.SH"],
            prices,
            signal_date="2024-01-03",
            intended_trade_date="2024-01-04",
            account=account,
            current_holdings=current,
            config={"strategy": {}},
        )

        by_code = orders.set_index("instrument")
        self.assertEqual(float(by_code.loc["000001.SZ", "target_shares"]), 2700.0)
        self.assertEqual(float(by_code.loc["000001.SZ", "order_shares"]), -300.0)
        self.assertEqual(float(by_code.loc["600519.SH", "target_shares"]), 200.0)
        self.assertEqual(float(by_code.loc["600519.SH", "order_shares"]), 200.0)
        self.assertEqual(float(by_code.loc["000002.SZ", "target_shares"]), 0.0)
        self.assertEqual(float(by_code.loc["000002.SZ", "order_shares"]), -500.0)
        self.assertEqual(by_code.loc["000001.SZ", "reference_price_date"], "2024-01-04")
        self.assertTrue(bool(by_code.loc["600519.SH", "is_order_actionable"]))
        self.assertEqual(by_code.loc["600519.SH", "reference_price_source"], "intended_trade_date_close")
        self.assertIn("suggested_limit_price", orders.columns)

    def test_generate_manual_orders_marks_limit_flags_with_board_and_st_thresholds(self) -> None:
        dates = pd.DatetimeIndex(["2024-01-03", "2024-01-04"])
        star = "688001.SH"
        growth = "300001.SZ"
        st_stock = "600000.SH"
        signal = pd.DataFrame(
            [
                {"date": "2024-01-03", "instrument": star, "action": "BUY"},
                {"date": "2024-01-03", "instrument": growth, "action": "BUY"},
                {"date": "2024-01-03", "instrument": st_stock, "action": "BUY"},
            ]
        )
        prices = _price_panel(
            dates,
            {
                "close": {star: [10.0, 11.2], growth: [10.0, 11.2], st_stock: [10.0, 10.6]},
                "high": {star: [10.0, 11.2], growth: [10.0, 11.2], st_stock: [10.0, 10.6]},
                "is_st": {star: [0.0, 0.0], growth: [0.0, 0.0], st_stock: [0.0, 1.0]},
            },
        )
        account = AccountState(
            total_asset=100000,
            cash=100000,
            max_position_pct=None,
            lot_size=100,
            star_market_lot_size=200,
            source_file="",
            holdings_file="",
            holdings_loaded=True,
        )

        orders = generate_manual_orders(
            signal,
            [star, growth, st_stock],
            prices,
            signal_date="2024-01-03",
            intended_trade_date="2024-01-04",
            account=account,
            current_holdings=pd.DataFrame(columns=["instrument", "shares"]),
            config={
                "strategy": {},
                "backtest": {
                    "star_limit_up_threshold": 0.099,
                    "growth_limit_up_threshold": 0.199,
                    "limit_up_threshold": 0.099,
                    "st_limit_up_threshold": 0.049,
                },
            },
        )

        by_code = orders.set_index("instrument")
        self.assertTrue(bool(by_code.loc[star, "is_limit_up"]))
        self.assertFalse(bool(by_code.loc[growth, "is_limit_up"]))
        self.assertTrue(bool(by_code.loc[st_stock, "is_st"]))
        self.assertTrue(bool(by_code.loc[st_stock, "is_limit_up"]))

    def test_generate_manual_orders_marks_blocked_candidate(self) -> None:
        signal = pd.DataFrame([{"date": "2024-01-03", "instrument": "000001.SZ", "action": "BUY"}])
        prices = _prices("2024-01-03", {"000001.SZ": 10.0})
        account = AccountState(
            total_asset=100000,
            cash=10000,
            max_position_pct=None,
            lot_size=100,
            star_market_lot_size=200,
            source_file="",
            holdings_file="",
            holdings_loaded=False,
        )

        orders = generate_manual_orders(
            signal,
            ["000001.SZ"],
            prices,
            signal_date="2024-01-03",
            intended_trade_date="2024-01-04",
            account=account,
            current_holdings=pd.DataFrame(columns=["instrument", "shares"]),
            config={"strategy": {}},
            is_executable=False,
            block_reasons=["data:stale"],
        )

        self.assertFalse(bool(orders.iloc[0]["is_executable"]))
        self.assertIn("blocked:data:stale", orders.iloc[0]["note"])
        self.assertIn("current_shares_missing", orders.iloc[0]["note"])
        self.assertIn("reference_price_from_signal_date", orders.iloc[0]["note"])

    def test_generate_manual_orders_applies_defensive_exposure_to_target_weight(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=6)
        signal_date = str(dates[-1].date())
        signal = pd.DataFrame([{"date": signal_date, "instrument": "000001.SZ", "action": "BUY"}])
        prices = _prices(
            [str(date.date()) for date in dates],
            {"000001.SZ": [10.0, 9.8, 9.5, 9.2, 9.0, 8.8]},
        )
        account = AccountState(
            total_asset=100000,
            cash=10000,
            max_position_pct=None,
            lot_size=100,
            star_market_lot_size=200,
            source_file="",
            holdings_file="",
            holdings_loaded=True,
        )
        config = {
            "strategy": {},
            "market_regime": {
                "enabled": True,
                "ma_window": 2,
                "momentum_window": 1,
                "volatility_window": 2,
                "min_periods": 1,
                "high_volatility_threshold": 10.0,
                "lag_days": 0,
            },
            "defensive_timing": {"enabled": True, "bear_exposure": 0.4, "sideways_exposure": 0.8, "bull_exposure": 1.0},
        }

        orders = generate_manual_orders(
            signal,
            ["000001.SZ"],
            prices,
            signal_date=signal_date,
            intended_trade_date=None,
            account=account,
            current_holdings=pd.DataFrame({"instrument": ["000001.SZ"], "shares": [0]}),
            config=config,
        )

        self.assertEqual(float(orders.iloc[0]["target_weight"]), 0.4)
        self.assertEqual(float(orders.iloc[0]["target_value"]), 40000.0)

    def test_generate_manual_orders_falls_back_to_signal_date_when_intended_price_is_missing(self) -> None:
        signal = pd.DataFrame([{"date": "2024-01-03", "instrument": "000001.SZ", "action": "BUY"}])
        prices = _prices("2024-01-03", {"000001.SZ": 10.0})
        account = AccountState(
            total_asset=100000,
            cash=10000,
            max_position_pct=None,
            lot_size=100,
            star_market_lot_size=200,
            source_file="",
            holdings_file="",
            holdings_loaded=True,
        )

        orders = generate_manual_orders(
            signal,
            ["000001.SZ"],
            prices,
            signal_date="2024-01-03",
            intended_trade_date="2024-01-04",
            account=account,
            current_holdings=pd.DataFrame({"instrument": ["000001.SZ"], "shares": [0]}),
            config={"strategy": {}},
        )

        self.assertEqual(orders.iloc[0]["reference_price_date"], "2024-01-03")
        self.assertEqual(float(orders.iloc[0]["reference_price"]), 10.0)
        self.assertEqual(float(orders.iloc[0]["indicative_target_shares"]), 10000.0)
        self.assertTrue(pd.isna(orders.iloc[0]["target_shares"]))
        self.assertTrue(pd.isna(orders.iloc[0]["order_shares"]))
        self.assertFalse(bool(orders.iloc[0]["is_order_actionable"]))
        self.assertIn("reference_price_from_signal_date", orders.iloc[0]["note"])

    def test_generate_manual_orders_redistributes_leftover_cash_after_lot_rounding(self) -> None:
        signal = pd.DataFrame(
            [
                {"date": "2024-01-03", "instrument": "000001.SZ", "action": "BUY"},
                {"date": "2024-01-03", "instrument": "000002.SZ", "action": "BUY"},
            ]
        )
        prices = _prices("2024-01-04", {"000001.SZ": 10.0, "000002.SZ": 10.0})
        account = AccountState(
            total_asset=103000,
            cash=103000,
            max_position_pct=None,
            lot_size=100,
            star_market_lot_size=200,
            source_file="",
            holdings_file="",
            holdings_loaded=True,
        )

        orders = generate_manual_orders(
            signal,
            ["000001.SZ", "000002.SZ"],
            prices,
            signal_date="2024-01-03",
            intended_trade_date="2024-01-04",
            account=account,
            current_holdings=pd.DataFrame(columns=["instrument", "shares"]),
            config={"strategy": {}, "manual_orders": {"cash_redistribution_overweight_tolerance": 0.10}},
        )

        self.assertEqual(float(orders["target_shares"].sum()), 10300.0)

    def test_generate_manual_orders_normalizes_targets_and_price_columns(self) -> None:
        signal = pd.DataFrame([{"date": "2024-01-03", "instrument": "000001.sz", "action": "BUY"}])
        prices = _prices("2024-01-04", {" 000001.sz ": 20.0})
        account = AccountState(
            total_asset=100000,
            cash=10000,
            max_position_pct=None,
            lot_size=100,
            star_market_lot_size=200,
            source_file="",
            holdings_file="",
            holdings_loaded=True,
        )

        orders = generate_manual_orders(
            signal,
            ["000001.sz", " 000001.SZ "],
            prices,
            signal_date="2024-01-03",
            intended_trade_date="2024-01-04",
            account=account,
            current_holdings=pd.DataFrame({"instrument": ["000001.sz"], "shares": [0]}),
            config={"strategy": {}},
        )

        self.assertEqual(orders["instrument"].tolist(), ["000001.SZ"])
        self.assertEqual(float(orders.iloc[0]["target_weight"]), 1.0)
        self.assertEqual(float(orders.iloc[0]["reference_price"]), 20.0)
        self.assertEqual(float(orders.iloc[0]["target_shares"]), 5000.0)

    def test_generate_manual_orders_does_not_treat_plain_close_panel_as_st_flags(self) -> None:
        signal = pd.DataFrame([{"date": "2024-01-03", "instrument": "600000.SH", "action": "BUY"}])
        prices = pd.DataFrame(
            {"600000.SH": [10.0, 10.6]},
            index=pd.to_datetime(["2024-01-03", "2024-01-04"]),
        )
        account = AccountState(
            total_asset=100000,
            cash=100000,
            max_position_pct=None,
            lot_size=100,
            star_market_lot_size=200,
            source_file="",
            holdings_file="",
            holdings_loaded=True,
        )

        orders = generate_manual_orders(
            signal,
            ["600000.SH"],
            prices,
            signal_date="2024-01-03",
            intended_trade_date="2024-01-04",
            account=account,
            current_holdings=pd.DataFrame(columns=["instrument", "shares"]),
            config={
                "strategy": {},
                "backtest": {
                    "limit_up_threshold": 0.099,
                    "st_limit_up_threshold": 0.049,
                },
            },
        )

        self.assertFalse(bool(orders.iloc[0]["is_st"]))
        self.assertFalse(bool(orders.iloc[0]["is_limit_up"]))

    def test_generate_manual_orders_rejects_flat_ohlcv_price_frame(self) -> None:
        signal = pd.DataFrame([{"date": "2024-01-03", "instrument": "000001.SZ", "action": "BUY"}])
        prices = pd.DataFrame(
            {
                "open": [10.0, 10.5],
                "close": [10.2, 10.8],
                "volume": [1000.0, 1200.0],
            },
            index=pd.to_datetime(["2024-01-03", "2024-01-04"]),
        )
        account = AccountState(
            total_asset=100000,
            cash=100000,
            max_position_pct=None,
            lot_size=100,
            star_market_lot_size=200,
            source_file="",
            holdings_file="",
            holdings_loaded=True,
        )

        with self.assertRaisesRegex(ValueError, "close-price panel"):
            generate_manual_orders(
                signal,
                ["000001.SZ"],
                prices,
                signal_date="2024-01-03",
                intended_trade_date="2024-01-04",
                account=account,
                current_holdings=pd.DataFrame(columns=["instrument", "shares"]),
                config={"strategy": {}},
            )

    def test_load_current_holdings_normalizes_and_deduplicates_instruments(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "current_holdings.csv"
            pd.DataFrame(
                {
                    "instrument": [" 000001.sz ", None, "000001.SZ", "600519.sh"],
                    "shares": [100, 200, 300, 400],
                }
            ).to_csv(path, index=False)

            holdings = load_current_holdings({"account": {"current_holdings_file": str(path)}})

        by_code = holdings.set_index("instrument")
        self.assertEqual(sorted(by_code.index.tolist()), ["000001.SZ", "600519.SH"])
        self.assertEqual(float(by_code.loc["000001.SZ", "shares"]), 300.0)
        self.assertEqual(float(by_code.loc["600519.SH", "shares"]), 400.0)

    def test_execution_templates_mark_actionable_orders_for_confirmation_and_fill_feedback(self) -> None:
        orders = pd.DataFrame(
            [
                {
                    "signal_date": "2024-01-03",
                    "intended_trade_date": "2024-01-04",
                    "instrument": "000001.SZ",
                    "action": "BUY",
                    "is_order_actionable": True,
                    "reference_price": 10.0,
                    "suggested_limit_price": 10.02,
                    "target_shares": 1000,
                    "order_shares": 500,
                    "capacity_ratio": 0.01,
                },
                {
                    "signal_date": "2024-01-03",
                    "intended_trade_date": "2024-01-04",
                    "instrument": "600519.SH",
                    "action": "HOLD",
                    "is_order_actionable": False,
                    "reference_price": 100.0,
                    "suggested_limit_price": 100.0,
                    "target_shares": 0,
                    "order_shares": 0,
                    "capacity_ratio": None,
                },
            ]
        )

        confirmation = generate_order_confirmation_template(orders, "2024-01-03", "2024-01-04")
        feedback = generate_fill_feedback_template(orders, "2024-01-03", "2024-01-04")

        self.assertEqual(confirmation.loc[0, "confirmation_status"], "PENDING")
        self.assertEqual(float(confirmation.loc[0, "confirmed_order_shares"]), 500.0)
        self.assertEqual(confirmation.loc[1, "confirmation_status"], "NO_ORDER")
        self.assertEqual(feedback.loc[0, "fill_status"], "PENDING")
        self.assertEqual(feedback.loc[0, "side"], "BUY")
        self.assertEqual(feedback.loc[1, "fill_status"], "SKIPPED")

    def test_save_execution_templates_uses_configured_outputs_dir_for_defaults(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            confirmation = pd.DataFrame({"instrument": ["000001.SZ"]})
            feedback = pd.DataFrame({"instrument": ["000001.SZ"]})

            files = save_execution_templates(
                confirmation,
                feedback,
                "2024-01-03",
                {"outputs": {"dir": str(root)}},
                executable=False,
            )

            self.assertTrue((root / "order_confirmations" / "order_confirmation_candidate_2024-01-03.csv").exists())
            self.assertTrue((root / "fill_feedback" / "fill_feedback_candidate_2024-01-03.csv").exists())
            self.assertIn("order_confirmation", files)
            self.assertIn("fill_feedback", files)

    def test_apply_fill_feedback_updates_holdings_with_buy_sell_and_partial_fills(self) -> None:
        current = pd.DataFrame({"instrument": ["000001.SZ", "600519.SH"], "shares": [1000, 300]})
        fills = pd.DataFrame(
            [
                {
                    "instrument": "000001.SZ",
                    "side": "BUY",
                    "planned_order_shares": 200,
                    "executed_shares": 200,
                    "fill_status": " filled ",
                },
                {
                    "instrument": "600519.SH",
                    "side": "SELL",
                    "planned_order_shares": -100,
                    "executed_shares": 100,
                    "fill_status": " partial ",
                },
                {
                    "instrument": "000002.SZ",
                    "side": "BUY",
                    "planned_order_shares": 500,
                    "executed_shares": 500,
                    "fill_status": "CANCELLED",
                },
            ]
        )

        updated = apply_fill_feedback(current, fills).set_index("instrument")

        self.assertEqual(float(updated.loc["000001.SZ", "shares"]), 1200.0)
        self.assertEqual(float(updated.loc["600519.SH", "shares"]), 200.0)
        self.assertNotIn("000002.SZ", updated.index)

    def test_validate_fill_feedback_blocks_unfinished_or_invalid_manual_entries(self) -> None:
        current = pd.DataFrame({"instrument": ["000001.SZ"], "shares": [100]})
        fills = pd.DataFrame(
            [
                {"instrument": "000001.SZ", "side": "SELL", "planned_order_shares": -200, "executed_shares": 200, "fill_status": "FILLED"},
                {"instrument": "600519.SH", "side": "BUY", "planned_order_shares": 100, "executed_shares": pd.NA, "fill_status": "PARTIAL"},
                {"instrument": "000002.SZ", "side": "BUY", "planned_order_shares": 100, "executed_shares": 100, "fill_status": "PENDING"},
            ]
        )

        issues = validate_fill_feedback(current, fills)

        self.assertIn("fill_would_make_negative_position:000001.SZ", issues)
        self.assertIn("executed_shares_missing:600519.SH", issues)
        self.assertIn("pending_fill_status:000002.SZ", issues)
        with self.assertRaises(ValueError):
            apply_fill_feedback(current, fills)

    def test_validate_fill_feedback_requires_status_side_and_planned_shares(self) -> None:
        current = pd.DataFrame({"instrument": ["000001.SZ"], "shares": [100]})
        fills = pd.DataFrame({"instrument": ["000001.SZ"], "executed_shares": [100]})

        issues = validate_fill_feedback(current, fills)

        self.assertEqual(
            issues,
            ["fill_feedback_missing_columns:side,planned_order_shares,fill_status"],
        )
        with self.assertRaises(ValueError):
            apply_fill_feedback(current, fills)


def _prices(date: str | list[str], close_values: dict[str, float | list[float]]) -> pd.DataFrame:
    dates = [date] if isinstance(date, str) else date
    instruments = list(close_values)
    columns = pd.MultiIndex.from_product([["close"], instruments], names=["field", "instrument"])
    rows = []
    for idx, _date in enumerate(dates):
        row = []
        for instrument in instruments:
            values = close_values[instrument]
            row.append(values[idx] if isinstance(values, list) else values)
        rows.append(row)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates), columns=columns)


def _price_panel(dates: pd.DatetimeIndex, fields: dict[str, dict[str, list[float]]]) -> pd.DataFrame:
    return pd.concat({field: pd.DataFrame(values, index=dates) for field, values in fields.items()}, axis=1)


if __name__ == "__main__":
    unittest.main()
