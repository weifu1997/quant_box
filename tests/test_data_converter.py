"""模块说明：覆盖 test_data_converter 相关行为的测试用例。"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from src.data_converter import convert_to_qlib_format
from src.data_converter import _apply_adjustment


class DataConverterTests(unittest.TestCase):
    """类说明：组织 DataConverterTests 测试用例。"""
    def test_convert_to_qlib_format_ignores_metadata_csv_files(self) -> None:
        """函数说明：验证 test_convert_to_qlib_format_ignores_metadata_csv_files 覆盖的行为场景。"""
        config = {
            "data": {
                "raw_dir": "unused",
                "constituents_file": "data/raw/mainboard_a_stocks.csv",
                "st_calendar_file": "data/raw/st_calendar.csv",
            },
            "qlib": {"provider_uri": "unused", "instruments": "mainboard_a"},
        }

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            qlib_dir = root / "qlib"
            raw_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "2024-01-02",
                        "open": 10.0,
                        "high": 11.0,
                        "low": 9.0,
                        "close": 10.5,
                        "vol": 1000.0,
                        "amount": 10000.0,
                        "adj_factor": 1.0,
                    }
                ]
            ).to_csv(raw_dir / "000001.SZ.csv", index=False)
            pd.DataFrame([{"ts_code": "000982.SZ", "reason": "empty_or_failed_fetch"}]).to_csv(
                raw_dir / "failed_fetches.csv",
                index=False,
            )
            pd.DataFrame([{"ts_code": "000001.SZ", "name": "*ST TEST", "st_start_date": "2024-01-02"}]).to_csv(
                raw_dir / "st_calendar.csv",
                index=False,
            )
            pd.DataFrame([{"index_code": "000300.SH", "con_code": "000001.SZ", "trade_date": "2024-01-02"}]).to_csv(
                raw_dir / "index_constituents.csv",
                index=False,
            )
            pd.DataFrame([{"trade_date": "2024-01-02", "instrument": "000001.SZ", "sources": "hs300"}]).to_csv(
                raw_dir / "historical_universe.csv",
                index=False,
            )

            def fake_resolve_path(value: str | Path) -> Path:
                """函数说明：处理 fake_resolve_path 主要逻辑。"""
                if str(value) == "data/prices":
                    return root / "prices"
                path = Path(value)
                return path if path.is_absolute() else root / path

            with patch("src.data_converter.load_config", return_value=config), patch(
                "src.data_converter.resolve_path",
                side_effect=fake_resolve_path,
            ):
                result = convert_to_qlib_format(raw_dir=raw_dir, qlib_dir=qlib_dir)

            self.assertEqual(result["instruments"], 1)
            self.assertTrue(Path(result["ohlcv_price_file"]).exists())
            self.assertTrue(Path(result["adjusted_ohlcv_price_file"]).exists())

    def test_convert_to_qlib_format_writes_raw_and_adjusted_price_panels(self) -> None:
        """函数说明：验证 test_convert_to_qlib_format_writes_raw_and_adjusted_price_panels 覆盖的行为场景。"""
        config = {
            "data": {"raw_dir": "unused", "constituents_file": "data/raw/mainboard_a_stocks.csv"},
            "qlib": {"provider_uri": "unused", "instruments": "mainboard_a"},
        }

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            qlib_dir = root / "qlib"
            raw_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "2024-01-02",
                        "open": 10.0,
                        "high": 10.0,
                        "low": 10.0,
                        "close": 10.0,
                        "vol": 2000.0,
                        "amount": 2000.0,
                        "adj_factor": 1.0,
                    },
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "2024-01-03",
                        "open": 20.0,
                        "high": 20.0,
                        "low": 20.0,
                        "close": 20.0,
                        "vol": 1000.0,
                        "amount": 2000.0,
                        "adj_factor": 2.0,
                    },
                ]
            ).to_csv(raw_dir / "000001.SZ.csv", index=False)

            def fake_resolve_path(value: str | Path) -> Path:
                """函数说明：处理 fake_resolve_path 主要逻辑。"""
                if str(value) == "data/prices":
                    return root / "prices"
                path = Path(value)
                return path if path.is_absolute() else root / path

            with patch("src.data_converter.load_config", return_value=config), patch(
                "src.data_converter.resolve_path",
                side_effect=fake_resolve_path,
            ):
                result = convert_to_qlib_format(raw_dir=raw_dir, qlib_dir=qlib_dir)

            raw_panel = pd.read_parquet(result["ohlcv_price_file"])
            adjusted_panel = pd.read_parquet(result["adjusted_ohlcv_price_file"])
            qlib_features = pd.read_parquet(qlib_dir / "features" / "000001.sz" / "day.parquet")

            first_date = pd.Timestamp("2024-01-02")
            self.assertAlmostEqual(float(raw_panel.loc[first_date, ("close", "000001.sz")]), 10.0)
            self.assertAlmostEqual(float(adjusted_panel.loc[first_date, ("close", "000001.sz")]), 5.0)
            self.assertAlmostEqual(float(adjusted_panel.loc[first_date, ("volume", "000001.sz")]), 4000.0)
            self.assertAlmostEqual(float(qlib_features.loc[0, "close"]), 5.0)

    def test_convert_to_qlib_format_deduplicates_raw_daily_rows(self) -> None:
        """函数说明：验证 test_convert_to_qlib_format_deduplicates_raw_daily_rows 覆盖的行为场景。"""
        config = {
            "data": {"raw_dir": "unused", "constituents_file": "data/raw/mainboard_a_stocks.csv"},
            "qlib": {"provider_uri": "unused", "instruments": "mainboard_a"},
        }

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            qlib_dir = root / "qlib"
            raw_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "2024-01-02",
                        "open": 10.0,
                        "high": 10.0,
                        "low": 10.0,
                        "close": 10.0,
                        "vol": 1000.0,
                        "amount": 10000.0,
                        "adj_factor": 1.0,
                    },
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "2024-01-02",
                        "open": 11.0,
                        "high": 11.0,
                        "low": 11.0,
                        "close": 11.0,
                        "vol": 1100.0,
                        "amount": 12100.0,
                        "adj_factor": 1.0,
                    },
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "2024-01-03",
                        "open": 20.0,
                        "high": 20.0,
                        "low": 20.0,
                        "close": 20.0,
                        "vol": 1000.0,
                        "amount": 20000.0,
                        "adj_factor": 2.0,
                    },
                ]
            ).to_csv(raw_dir / "000001.SZ.csv", index=False)

            def fake_resolve_path(value: str | Path) -> Path:
                """函数说明：处理 fake_resolve_path 主要逻辑。"""
                if str(value) == "data/prices":
                    return root / "prices"
                path = Path(value)
                return path if path.is_absolute() else root / path

            with patch("src.data_converter.load_config", return_value=config), patch(
                "src.data_converter.resolve_path",
                side_effect=fake_resolve_path,
            ):
                result = convert_to_qlib_format(raw_dir=raw_dir, qlib_dir=qlib_dir)

            raw_panel = pd.read_parquet(result["ohlcv_price_file"])
            adjusted_features = pd.read_parquet(qlib_dir / "features" / "000001.sz" / "day.parquet")
            first_date = pd.Timestamp("2024-01-02")

            self.assertTrue(raw_panel.index.is_unique)
            self.assertAlmostEqual(float(raw_panel.loc[first_date, ("close", "000001.sz")]), 11.0)
            self.assertEqual(adjusted_features["date"].tolist(), [first_date, pd.Timestamp("2024-01-03")])
            self.assertAlmostEqual(float(adjusted_features.loc[0, "close"]), 5.5)

    def test_convert_to_qlib_format_accepts_compact_calendar_dates(self) -> None:
        """函数说明：验证 test_convert_to_qlib_format_accepts_compact_calendar_dates 覆盖的行为场景。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            qlib_dir = root / "qlib"
            calendar_file = root / "calendar.csv"
            raw_dir.mkdir()
            calendar_file.write_text("20240102\n20240103\n", encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "2024-01-02",
                        "open": 10.0,
                        "high": 10.0,
                        "low": 10.0,
                        "close": 10.0,
                        "vol": 1000.0,
                        "amount": 10000.0,
                        "adj_factor": 1.0,
                    },
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "2024-01-03",
                        "open": 11.0,
                        "high": 11.0,
                        "low": 11.0,
                        "close": 11.0,
                        "vol": 1000.0,
                        "amount": 11000.0,
                        "adj_factor": 1.0,
                    },
                ]
            ).to_csv(raw_dir / "000001.SZ.csv", index=False)
            config = {
                "data": {
                    "raw_dir": "unused",
                    "calendar_file": str(calendar_file),
                    "constituents_file": "data/raw/mainboard_a_stocks.csv",
                },
                "qlib": {"provider_uri": "unused", "instruments": "mainboard_a"},
            }

            def fake_resolve_path(value: str | Path) -> Path:
                """函数说明：处理 fake_resolve_path 主要逻辑。"""
                if str(value) == "data/prices":
                    return root / "prices"
                path = Path(value)
                return path if path.is_absolute() else root / path

            with patch("src.data_converter.load_config", return_value=config), patch(
                "src.data_converter.resolve_path",
                side_effect=fake_resolve_path,
            ):
                result = convert_to_qlib_format(raw_dir=raw_dir, qlib_dir=qlib_dir)

            self.assertEqual(result["calendar_days"], 2)
            self.assertEqual((qlib_dir / "calendars" / "day.txt").read_text(encoding="utf-8").splitlines()[0], "2024-01-02")

    def test_convert_to_qlib_format_filters_closed_calendar_rows(self) -> None:
        """函数说明：验证 test_convert_to_qlib_format_filters_closed_calendar_rows 覆盖的行为场景。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            qlib_dir = root / "qlib"
            calendar_file = root / "calendar.csv"
            raw_dir.mkdir()
            calendar_file.write_text("cal_date,is_open\n20240102, TRUE \n20240103,0\n", encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "2024-01-02",
                        "open": 10.0,
                        "high": 10.0,
                        "low": 10.0,
                        "close": 10.0,
                        "vol": 1000.0,
                        "amount": 10000.0,
                        "adj_factor": 1.0,
                    },
                ]
            ).to_csv(raw_dir / "000001.SZ.csv", index=False)
            config = {
                "data": {
                    "raw_dir": "unused",
                    "calendar_file": str(calendar_file),
                    "constituents_file": "data/raw/mainboard_a_stocks.csv",
                },
                "qlib": {"provider_uri": "unused", "instruments": "mainboard_a"},
            }

            def fake_resolve_path(value: str | Path) -> Path:
                """函数说明：处理 fake_resolve_path 主要逻辑。"""
                if str(value) == "data/prices":
                    return root / "prices"
                path = Path(value)
                return path if path.is_absolute() else root / path

            with patch("src.data_converter.load_config", return_value=config), patch(
                "src.data_converter.resolve_path",
                side_effect=fake_resolve_path,
            ):
                result = convert_to_qlib_format(raw_dir=raw_dir, qlib_dir=qlib_dir)

            self.assertEqual(result["calendar_days"], 1)
            self.assertEqual((qlib_dir / "calendars" / "day.txt").read_text(encoding="utf-8").splitlines(), ["2024-01-02"])

    def test_convert_to_qlib_format_rejects_dates_missing_from_calendar(self) -> None:
        """函数说明：验证 test_convert_to_qlib_format_rejects_dates_missing_from_calendar 覆盖的行为场景。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            qlib_dir = root / "qlib"
            calendar_file = root / "calendar.csv"
            raw_dir.mkdir()
            calendar_file.write_text("20240102\n", encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "2024-01-03",
                        "open": 10.0,
                        "high": 10.0,
                        "low": 10.0,
                        "close": 10.0,
                        "vol": 1000.0,
                        "amount": 10000.0,
                        "adj_factor": 1.0,
                    },
                ]
            ).to_csv(raw_dir / "000001.SZ.csv", index=False)
            config = {
                "data": {
                    "raw_dir": "unused",
                    "calendar_file": str(calendar_file),
                    "constituents_file": "data/raw/mainboard_a_stocks.csv",
                },
                "qlib": {"provider_uri": "unused", "instruments": "mainboard_a"},
            }

            def fake_resolve_path(value: str | Path) -> Path:
                """函数说明：处理 fake_resolve_path 主要逻辑。"""
                if str(value) == "data/prices":
                    return root / "prices"
                path = Path(value)
                return path if path.is_absolute() else root / path

            with patch("src.data_converter.load_config", return_value=config), patch(
                "src.data_converter.resolve_path",
                side_effect=fake_resolve_path,
            ):
                with self.assertRaisesRegex(ValueError, "missing from the configured trading calendar"):
                    convert_to_qlib_format(raw_dir=raw_dir, qlib_dir=qlib_dir)

    def test_convert_to_qlib_format_sanitizes_nonpositive_market_values(self) -> None:
        """函数说明：验证 test_convert_to_qlib_format_sanitizes_nonpositive_market_values 覆盖的行为场景。"""
        config = {
            "data": {"raw_dir": "unused", "constituents_file": "data/raw/mainboard_a_stocks.csv"},
            "qlib": {"provider_uri": "unused", "instruments": "mainboard_a"},
        }

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            qlib_dir = root / "qlib"
            raw_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "2024-01-02",
                        "open": 10.0,
                        "high": 10.0,
                        "low": 10.0,
                        "close": 10.0,
                        "vol": 0.0,
                        "amount": 0.0,
                        "adj_factor": 1.0,
                    }
                ]
            ).to_csv(raw_dir / "000001.SZ.csv", index=False)

            def fake_resolve_path(value: str | Path) -> Path:
                """函数说明：处理 fake_resolve_path 主要逻辑。"""
                if str(value) == "data/prices":
                    return root / "prices"
                path = Path(value)
                return path if path.is_absolute() else root / path

            with patch("src.data_converter.load_config", return_value=config), patch(
                "src.data_converter.resolve_path",
                side_effect=fake_resolve_path,
            ):
                result = convert_to_qlib_format(raw_dir=raw_dir, qlib_dir=qlib_dir)

            raw_panel = pd.read_parquet(result["ohlcv_price_file"])
            qlib_features = pd.read_parquet(qlib_dir / "features" / "000001.sz" / "day.parquet")
            date = pd.Timestamp("2024-01-02")

            self.assertTrue(pd.isna(raw_panel.loc[date, ("volume", "000001.sz")]))
            self.assertTrue(pd.isna(raw_panel.loc[date, ("amount", "000001.sz")]))
            self.assertTrue(pd.isna(qlib_features.loc[0, "volume"]))
            self.assertTrue(pd.isna(qlib_features.loc[0, "amount"]))
            self.assertTrue(pd.isna(qlib_features.loc[0, "vwap"]))

    def test_convert_to_qlib_format_removes_stale_price_files_when_no_stock_remains(self) -> None:
        """函数说明：验证 test_convert_to_qlib_format_removes_stale_price_files_when_no_stock_remains 覆盖的行为场景。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            qlib_dir = root / "qlib"
            prices_dir = root / "prices"
            raw_dir.mkdir()
            prices_dir.mkdir()
            close_path = prices_dir / "close.parquet"
            ohlcv_path = prices_dir / "ohlcv.parquet"
            adjusted_close_path = prices_dir / "close_adjusted.parquet"
            adjusted_ohlcv_path = prices_dir / "ohlcv_adjusted.parquet"
            close_path.write_text("stale", encoding="utf-8")
            ohlcv_path.write_text("stale", encoding="utf-8")
            adjusted_close_path.write_text("stale", encoding="utf-8")
            adjusted_ohlcv_path.write_text("stale", encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "2024-01-02",
                        "open": 10.0,
                        "high": 11.0,
                        "low": 9.0,
                        "close": 10.5,
                        "vol": 1000.0,
                        "amount": 10000.0,
                        "adj_factor": 1.0,
                    }
                ]
            ).to_csv(raw_dir / "000001.SZ.csv", index=False)
            tradable_file = root / "tradable.csv"
            pd.DataFrame([{"ts_code": "600000.SH"}]).to_csv(tradable_file, index=False)
            config = {
                "data": {
                    "raw_dir": str(raw_dir),
                    "constituents_file": str(root / "mainboard_a_stocks.csv"),
                    "tradable_file": str(tradable_file),
                },
                "qlib": {"provider_uri": str(qlib_dir), "instruments": "mainboard_a"},
            }

            def fake_resolve_path(value: str | Path) -> Path:
                """函数说明：处理 fake_resolve_path 主要逻辑。"""
                if str(value) == "data/prices":
                    return prices_dir
                path = Path(value)
                return path if path.is_absolute() else root / path

            with patch("src.data_converter.load_config", return_value=config), patch(
                "src.data_converter.resolve_path",
                side_effect=fake_resolve_path,
            ):
                with self.assertRaisesRegex(ValueError, "No valid stock data remained"):
                    convert_to_qlib_format(raw_dir=raw_dir, qlib_dir=qlib_dir)

            self.assertFalse(close_path.exists())
            self.assertFalse(ohlcv_path.exists())
            self.assertFalse(adjusted_close_path.exists())
            self.assertFalse(adjusted_ohlcv_path.exists())

    def test_convert_to_qlib_format_normalizes_tradable_symbols(self) -> None:
        """函数说明：验证 test_convert_to_qlib_format_normalizes_tradable_symbols 覆盖的行为场景。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            qlib_dir = root / "qlib"
            tradable_file = root / "tradable.csv"
            raw_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "2024-01-02",
                        "open": 10.0,
                        "high": 10.0,
                        "low": 10.0,
                        "close": 10.0,
                        "vol": 1000.0,
                        "amount": 10000.0,
                        "adj_factor": 1.0,
                    }
                ]
            ).to_csv(raw_dir / "000001.SZ.csv", index=False)
            pd.DataFrame([{"ts_code": " 000001.sz "}]).to_csv(tradable_file, index=False)
            config = {
                "data": {
                    "raw_dir": str(raw_dir),
                    "constituents_file": str(root / "mainboard_a_stocks.csv"),
                    "tradable_file": str(tradable_file),
                },
                "qlib": {"provider_uri": str(qlib_dir), "instruments": "mainboard_a"},
            }

            def fake_resolve_path(value: str | Path) -> Path:
                """函数说明：处理 fake_resolve_path 主要逻辑。"""
                if str(value) == "data/prices":
                    return root / "prices"
                path = Path(value)
                return path if path.is_absolute() else root / path

            with patch("src.data_converter.load_config", return_value=config), patch(
                "src.data_converter.resolve_path",
                side_effect=fake_resolve_path,
            ):
                result = convert_to_qlib_format(raw_dir=raw_dir, qlib_dir=qlib_dir)

            self.assertEqual(result["instruments"], 1)
            self.assertTrue(Path(result["close_price_file"]).exists())

    def test_adjustment_scales_volume_opposite_to_prices(self) -> None:
        """函数说明：验证 test_adjustment_scales_volume_opposite_to_prices 覆盖的行为场景。"""
        feature_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "open": [10.0, 20.0],
                "high": [10.0, 20.0],
                "low": [10.0, 20.0],
                "close": [10.0, 20.0],
                "volume": [2000.0, 1000.0],
                "amount": [2000.0, 2000.0],
                "vwap": [10.0, 20.0],
                "adj_factor": [1.0, 2.0],
            }
        )

        adjusted = _apply_adjustment(feature_df)

        self.assertAlmostEqual(float(adjusted.loc[0, "close"]), 5.0)
        self.assertAlmostEqual(float(adjusted.loc[0, "volume"]), 4000.0)
        self.assertAlmostEqual(float(adjusted.loc[0, "amount"]), 2000.0)


if __name__ == "__main__":
    unittest.main()
