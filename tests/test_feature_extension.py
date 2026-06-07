from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.feature_extension import _price_field, append_daily_basic_features, append_price_derived_features


class FeatureExtensionTests(unittest.TestCase):
    def test_append_daily_basic_features_uses_lagged_available_date(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        index = pd.MultiIndex.from_product([dates, ["000001.SZ"]], names=["datetime", "instrument"])
        factors = pd.DataFrame({"F1": [1.0, 2.0]}, index=index)
        daily_basic = pd.DataFrame(
            {
                "trade_date": ["2024-01-02", "2024-01-03"],
                "ts_code": ["000001.SZ", "000001.SZ"],
                "circ_mv": [100.0, 999.0],
                "turnover_rate": [1.5, 2.5],
            }
        )

        extended, summary = append_daily_basic_features(
            factors,
            daily_basic,
            {
                "enabled": True,
                "daily_basic_lag_days": 1,
                "daily_basic_fields": ["circ_mv", "turnover_rate"],
            },
        )

        self.assertEqual(summary["features_added"], 2)
        self.assertTrue(pd.isna(extended.loc[(dates[0], "000001.SZ"), "DB_circ_mv"]))
        self.assertAlmostEqual(float(extended.loc[(dates[1], "000001.SZ"), "DB_turnover_rate"]), 1.5)

    def test_append_daily_basic_features_uses_most_recent_available_lagged_date(self) -> None:
        dates = pd.to_datetime(["2024-01-08", "2024-01-10"])
        index = pd.MultiIndex.from_product([dates, ["000001.SZ"]], names=["datetime", "instrument"])
        factors = pd.DataFrame({"F1": [1.0, 2.0]}, index=index)
        daily_basic = pd.DataFrame(
            {
                "trade_date": ["2024-01-05", "2024-01-08", "2024-01-10"],
                "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
                "turnover_rate": [1.5, 2.5, 9.9],
            }
        )

        extended, summary = append_daily_basic_features(
            factors,
            daily_basic,
            {
                "enabled": True,
                "daily_basic_lag_days": 1,
                "daily_basic_fields": ["turnover_rate"],
            },
        )

        self.assertEqual(summary["features_added"], 1)
        self.assertAlmostEqual(float(extended.loc[(dates[0], "000001.SZ"), "DB_turnover_rate"]), 1.5)
        self.assertAlmostEqual(float(extended.loc[(dates[1], "000001.SZ"), "DB_turnover_rate"]), 2.5)

    def test_append_daily_basic_features_respects_daily_basic_toggle(self) -> None:
        date = pd.Timestamp("2024-01-03")
        index = pd.MultiIndex.from_product([[date], ["000001.SZ"]], names=["datetime", "instrument"])
        factors = pd.DataFrame({"F1": [1.0]}, index=index)
        daily_basic = pd.DataFrame(
            {
                "trade_date": ["2024-01-02"],
                "ts_code": ["000001.SZ"],
                "circ_mv": [100.0],
            }
        )

        extended, summary = append_daily_basic_features(
            factors,
            daily_basic,
            {"enabled": True, "daily_basic": False, "daily_basic_fields": ["circ_mv"]},
        )

        self.assertEqual(extended.columns.to_list(), ["F1"])
        self.assertFalse(summary["enabled"])
        self.assertEqual(summary["features_added"], 0)

    def test_append_daily_basic_features_normalizes_symbol_inputs(self) -> None:
        date = pd.Timestamp("2024-01-03")
        index = pd.MultiIndex.from_product([[date], [" 000001.sz "]], names=["datetime", "instrument"])
        factors = pd.DataFrame({"F1": [1.0]}, index=index)
        daily_basic = pd.DataFrame(
            {
                "trade_date": ["2024-01-03"],
                "ts_code": ["000001.SZ"],
                "turnover_rate": [1.5],
            }
        )

        extended, summary = append_daily_basic_features(
            factors,
            daily_basic,
            {
                "enabled": True,
                "daily_basic_lag_days": 0,
                "daily_basic_fields": ["turnover_rate"],
            },
        )

        self.assertEqual(summary["features_added"], 1)
        self.assertAlmostEqual(float(extended.loc[(date, " 000001.sz "), "DB_turnover_rate"]), 1.5)

    def test_append_price_derived_features_uses_lagged_price_panel(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        index = pd.MultiIndex.from_product([dates, ["000001.SZ"]], names=["datetime", "instrument"])
        factors = pd.DataFrame({"F1": [1.0, 2.0, 3.0]}, index=index)
        prices = pd.DataFrame(
            {
                ("close", "000001.SZ"): [10.0, 11.0, 12.0],
                ("amount", "000001.SZ"): [100.0, 200.0, 400.0],
            },
            index=dates,
        )
        prices.columns = pd.MultiIndex.from_tuples(prices.columns, names=["field", "instrument"])

        extended, summary = append_price_derived_features(
            factors,
            prices,
            {
                "enabled": True,
                "price_derived": True,
                "price_feature_lag_sessions": 1,
                "price_features": ["low_amount_2"],
            },
        )

        self.assertEqual(summary["features_added"], 1)
        self.assertTrue(pd.isna(extended.loc[(dates[0], "000001.SZ"), "PX_LOW_AMOUNT_2"]))
        expected = -np.log1p(150.0)
        self.assertAlmostEqual(float(extended.loc[(dates[2], "000001.SZ"), "PX_LOW_AMOUNT_2"]), expected, places=6)

    def test_price_field_keeps_latest_intraday_price_per_session(self) -> None:
        prices = pd.DataFrame(
            {("close", "000001.sz"): [10.0, 30.0, 20.0]},
            index=pd.to_datetime(["2024-01-02 15:00", "2024-01-02 09:30", "2024-01-03 15:00"]),
        )
        prices.columns = pd.MultiIndex.from_tuples(prices.columns, names=["field", "instrument"])

        close = _price_field(prices, "close")

        self.assertEqual(close.index.to_list(), list(pd.to_datetime(["2024-01-02", "2024-01-03"])))
        self.assertAlmostEqual(float(close.loc[pd.Timestamp("2024-01-02"), "000001.SZ"]), 10.0)

    def test_append_price_derived_features_accepts_plain_close_panel(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        index = pd.MultiIndex.from_product([dates, ["000001.SZ"]], names=["datetime", "instrument"])
        factors = pd.DataFrame({"F1": [1.0, 2.0, 3.0]}, index=index)
        prices = pd.DataFrame({"000001.sz": [10.0, 11.0, 12.0]}, index=dates)

        extended, summary = append_price_derived_features(
            factors,
            prices,
            {
                "enabled": True,
                "price_derived": True,
                "price_feature_lag_sessions": 0,
                "price_features": ["return_2"],
            },
        )

        self.assertEqual(summary["features_added"], 1)
        self.assertEqual(summary["dates_matched"], 1)
        self.assertTrue(pd.isna(extended.loc[(dates[1], "000001.SZ"), "PX_RETURN_2"]))
        self.assertAlmostEqual(float(extended.loc[(dates[2], "000001.SZ"), "PX_RETURN_2"]), 0.2, places=6)

    def test_append_price_derived_features_rejects_flat_ohlcv_price_frame(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        index = pd.MultiIndex.from_product([dates, ["000001.SZ"]], names=["datetime", "instrument"])
        factors = pd.DataFrame({"F1": [1.0, 2.0, 3.0]}, index=index)
        prices = pd.DataFrame(
            {
                "open": [10.0, 10.5, 11.0],
                "close": [10.0, 11.0, 12.0],
                "amount": [100.0, 200.0, 300.0],
            },
            index=dates,
        )

        with self.assertRaisesRegex(ValueError, "close-price panel"):
            append_price_derived_features(
                factors,
                prices,
                {
                    "enabled": True,
                    "price_derived": True,
                    "price_feature_lag_sessions": 0,
                    "price_features": ["return_2"],
                },
            )

    def test_append_price_derived_features_normalizes_factor_symbols(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        index = pd.MultiIndex.from_product([dates, [" 000001.sz "]], names=["datetime", "instrument"])
        factors = pd.DataFrame({"F1": [1.0, 2.0, 3.0]}, index=index)
        prices = pd.DataFrame({"000001.SZ": [10.0, 11.0, 12.0]}, index=dates)

        extended, summary = append_price_derived_features(
            factors,
            prices,
            {
                "enabled": True,
                "price_derived": True,
                "price_feature_lag_sessions": 0,
                "price_features": ["return_2"],
            },
        )

        self.assertEqual(summary["features_added"], 1)
        self.assertAlmostEqual(float(extended.loc[(dates[2], " 000001.sz "), "PX_RETURN_2"]), 0.2, places=6)


if __name__ == "__main__":
    unittest.main()
