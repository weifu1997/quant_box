from __future__ import annotations

from pathlib import Path
import unittest
from tempfile import TemporaryDirectory

import pandas as pd

from src.manual_orders import AccountState, generate_manual_orders, load_current_holdings


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
        self.assertEqual(float(by_code.loc["000001.SZ", "target_shares"]), 2500.0)
        self.assertEqual(float(by_code.loc["000001.SZ", "order_shares"]), -500.0)
        self.assertEqual(float(by_code.loc["600519.SH", "target_shares"]), 200.0)
        self.assertEqual(float(by_code.loc["600519.SH", "order_shares"]), 200.0)
        self.assertEqual(float(by_code.loc["000002.SZ", "target_shares"]), 0.0)
        self.assertEqual(float(by_code.loc["000002.SZ", "order_shares"]), -500.0)
        self.assertEqual(by_code.loc["000001.SZ", "reference_price_date"], "2024-01-04")

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
        self.assertIn("reference_price_from_signal_date", orders.iloc[0]["note"])

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


if __name__ == "__main__":
    unittest.main()
