from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.data_health import build_data_health_report


class DataHealthTests(unittest.TestCase):
    def test_build_data_health_report_accepts_fresh_complete_data(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            universe_file = raw_dir / "mainboard_a_stocks.csv"
            pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "600519.SH"],
                    "name": ["A", "B"],
                    "list_status": ["L", "L"],
                    "list_date": ["20200101", "20200101"],
                }
            ).to_csv(universe_file, index=False)
            _raw(raw_dir / "000001.SZ.csv", "000001.SZ", "2024-01-03")
            _raw(raw_dir / "600519.SH.csv", "600519.SH", "2024-01-03")

            config = _config(raw_dir, universe_file)
            prices = _prices("2024-01-03", ["000001.SZ", "600519.SH"])
            factors = _factors("2024-01-03", ["000001.SZ", "600519.SH"])

            report = build_data_health_report(config, price_df=prices, factor_df=factors)

            self.assertTrue(report.is_healthy)
            self.assertEqual(report.raw_target_coverage, 1.0)
            self.assertEqual(report.price_target_coverage, 1.0)
            self.assertEqual(report.factor_target_coverage, 1.0)

    def test_build_data_health_report_blocks_stale_price_and_factor_data(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            universe_file = raw_dir / "mainboard_a_stocks.csv"
            pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "600519.SH"],
                    "name": ["A", "B"],
                    "list_status": ["L", "L"],
                    "list_date": ["20200101", "20200101"],
                }
            ).to_csv(universe_file, index=False)
            _raw(raw_dir / "000001.SZ.csv", "000001.SZ", "2024-01-03")
            _raw(raw_dir / "600519.SH.csv", "600519.SH", "2024-01-02")

            config = _config(raw_dir, universe_file)
            prices = _prices("2024-01-02", ["000001.SZ", "600519.SH"])
            factors = _factors("2024-01-02", ["000001.SZ", "600519.SH"])

            report = build_data_health_report(config, price_df=prices, factor_df=factors)

            self.assertFalse(report.is_healthy)
            self.assertIn("raw_latest_before_end:2024-01-02<2024-01-03", report.issues)
            self.assertIn("price_latest_before_end:2024-01-02<2024-01-03", report.issues)
            self.assertIn("factor_latest_before_end:2024-01-02<2024-01-03", report.issues)


def _config(raw_dir: Path, universe_file: Path) -> dict:
    return {
        "data": {
            "end_date": "2024-01-03",
            "raw_dir": str(raw_dir),
            "constituents_file": str(universe_file),
            "universe": "mainboard_a",
            "exclude_st": False,
        },
        "quality": {
            "min_raw_coverage": 1.0,
            "min_price_coverage": 1.0,
            "min_factor_coverage": 1.0,
            "require_latest_end_date": True,
        },
        "ic": {"price_file": "unused.parquet"},
        "factors": {"cache_file": "unused.parquet"},
    }


def _raw(path: Path, code: str, trade_date: str) -> None:
    pd.DataFrame({"ts_code": [code], "trade_date": [trade_date], "close": [10.0]}).to_csv(path, index=False)


def _prices(date: str, instruments: list[str]) -> pd.DataFrame:
    columns = pd.MultiIndex.from_product([["close"], instruments], names=["field", "instrument"])
    return pd.DataFrame([[10.0 for _ in instruments]], index=pd.DatetimeIndex([date]), columns=columns)


def _factors(date: str, instruments: list[str]) -> pd.DataFrame:
    index = pd.MultiIndex.from_product([[pd.Timestamp(date)], instruments], names=["datetime", "instrument"])
    return pd.DataFrame({"ROC5": [1.0 for _ in instruments]}, index=index)


if __name__ == "__main__":
    unittest.main()
