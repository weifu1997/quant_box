from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from src.data_converter import convert_to_qlib_format
from src.data_converter import _apply_adjustment


class DataConverterTests(unittest.TestCase):
    def test_convert_to_qlib_format_ignores_failed_fetches_manifest(self) -> None:
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

            def fake_resolve_path(value: str | Path) -> Path:
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

    def test_convert_to_qlib_format_removes_stale_price_files_when_no_stock_remains(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            qlib_dir = root / "qlib"
            prices_dir = root / "prices"
            raw_dir.mkdir()
            prices_dir.mkdir()
            close_path = prices_dir / "close.parquet"
            ohlcv_path = prices_dir / "ohlcv.parquet"
            close_path.write_text("stale", encoding="utf-8")
            ohlcv_path.write_text("stale", encoding="utf-8")
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

    def test_adjustment_scales_volume_opposite_to_prices(self) -> None:
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
