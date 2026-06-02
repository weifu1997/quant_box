from __future__ import annotations

import unittest

import pandas as pd

from src.manual_orders import AccountState, generate_manual_orders


class ManualOrdersTests(unittest.TestCase):
    def test_generate_manual_orders_calculates_target_and_order_shares(self) -> None:
        signal = pd.DataFrame(
            [
                {"date": "2024-01-03", "instrument": "000001.SZ", "action": "HOLD"},
                {"date": "2024-01-03", "instrument": "600519.SH", "action": "BUY"},
                {"date": "2024-01-03", "instrument": "000002.SZ", "action": "SELL"},
            ]
        )
        prices = _prices("2024-01-03", {"000001.SZ": 10.0, "600519.SH": 100.0, "000002.SZ": 20.0})
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
        self.assertEqual(float(by_code.loc["000001.SZ", "target_shares"]), 5000.0)
        self.assertEqual(float(by_code.loc["000001.SZ", "order_shares"]), 2000.0)
        self.assertEqual(float(by_code.loc["600519.SH", "target_shares"]), 500.0)
        self.assertEqual(float(by_code.loc["600519.SH", "order_shares"]), 500.0)
        self.assertEqual(float(by_code.loc["000002.SZ", "target_shares"]), 0.0)
        self.assertEqual(float(by_code.loc["000002.SZ", "order_shares"]), -500.0)

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


def _prices(date: str, close_values: dict[str, float]) -> pd.DataFrame:
    columns = pd.MultiIndex.from_product([["close"], list(close_values)], names=["field", "instrument"])
    return pd.DataFrame([list(close_values.values())], index=pd.DatetimeIndex([date]), columns=columns)


if __name__ == "__main__":
    unittest.main()
