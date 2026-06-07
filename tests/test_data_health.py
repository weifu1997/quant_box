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
            self.assertEqual(report.raw_latest_target_coverage, 1.0)
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
            self.assertIn("raw_latest_coverage_below_threshold:0.5000<1.0000", report.issues)
            self.assertIn("price_latest_before_end:2024-01-02<2024-01-03", report.issues)
            self.assertIn("factor_latest_before_end:2024-01-02<2024-01-03", report.issues)

    def test_build_data_health_report_checks_latest_target_price_and_factor_coverage(self) -> None:
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
            prices = _price_panel(
                ["2024-01-02", "2024-01-03"],
                {
                    "000001.SZ": [10.0, None],
                    "600519.SH": [20.0, None],
                    "000002.SZ": [None, 30.0],
                },
            )
            factors = _factor_rows(
                [
                    ("2024-01-02", "000001.SZ"),
                    ("2024-01-02", "600519.SH"),
                    ("2024-01-03", "000002.SZ"),
                ]
            )

            report = build_data_health_report(config, price_df=prices, factor_df=factors)

            self.assertFalse(report.is_healthy)
            self.assertEqual(report.price_latest_target_symbols, 0)
            self.assertEqual(report.factor_latest_target_symbols, 0)
            self.assertIn("price_latest_coverage_below_threshold:0.0000<1.0000", report.issues)
            self.assertIn("factor_latest_coverage_below_threshold:0.0000<1.0000", report.issues)

    def test_build_data_health_report_allows_sparse_stale_raw_symbols_above_threshold(self) -> None:
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
            config["quality"]["min_raw_coverage"] = 0.5
            prices = _prices("2024-01-03", ["000001.SZ", "600519.SH"])
            factors = _factors("2024-01-03", ["000001.SZ", "600519.SH"])

            report = build_data_health_report(config, price_df=prices, factor_df=factors)

            self.assertTrue(report.is_healthy)
            self.assertEqual(report.raw_latest_target_symbols, 1)
            self.assertEqual(report.raw_latest_target_coverage, 0.5)

    def test_build_data_health_report_allows_sparse_latest_price_and_factor_above_threshold(self) -> None:
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
            config["quality"]["min_price_coverage"] = 0.5
            config["quality"]["min_factor_coverage"] = 0.5
            prices = _price_panel(
                ["2024-01-02", "2024-01-03"],
                {
                    "000001.SZ": [10.0, 10.1],
                    "600519.SH": [20.0, None],
                },
            )
            factors = _factor_rows(
                [
                    ("2024-01-03", "000001.SZ"),
                    ("2024-01-02", "600519.SH"),
                ]
            )

            report = build_data_health_report(config, price_df=prices, factor_df=factors)

            self.assertTrue(report.is_healthy)
            self.assertEqual(report.price_latest_target_symbols, 1)
            self.assertEqual(report.factor_latest_target_symbols, 1)
            self.assertEqual(report.price_latest_target_coverage, 0.5)
            self.assertEqual(report.factor_latest_target_coverage, 0.5)

    def test_build_data_health_report_uses_latest_intraday_price_for_latest_coverage(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            universe_file = raw_dir / "mainboard_a_stocks.csv"
            pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "name": ["A"],
                    "list_status": ["L"],
                    "list_date": ["20200101"],
                }
            ).to_csv(universe_file, index=False)
            _raw(raw_dir / "000001.SZ.csv", "000001.SZ", "2024-01-03")

            config = _config(raw_dir, universe_file)
            prices = _price_panel(
                ["2024-01-03 15:00", "2024-01-03 09:30"],
                {"000001.SZ": [10.0, None]},
            )
            factors = _factors("2024-01-03", ["000001.SZ"])

            report = build_data_health_report(config, price_df=prices, factor_df=factors)

            self.assertTrue(report.is_healthy)
            self.assertEqual(report.price_latest_target_symbols, 1)
            self.assertEqual(report.price_latest_target_coverage, 1.0)

    def test_build_data_health_report_normalizes_symbol_whitespace_and_case(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            universe_file = raw_dir / "mainboard_a_stocks.csv"
            pd.DataFrame(
                {
                    "ts_code": [" 000001.sz "],
                    "name": ["A"],
                    "list_status": ["L"],
                    "list_date": ["20200101"],
                }
            ).to_csv(universe_file, index=False)
            _raw(raw_dir / "000001.SZ.csv", "000001.SZ", "2024-01-03")

            config = _config(raw_dir, universe_file)
            prices = _prices("2024-01-03", [" 000001.sz "])
            factors = _factors("2024-01-03", [" 000001.sz "])

            report = build_data_health_report(config, price_df=prices, factor_df=factors)

            self.assertTrue(report.is_healthy)
            self.assertEqual(report.price_target_symbols, 1)
            self.assertEqual(report.factor_target_symbols, 1)

    def test_build_data_health_report_rejects_flat_ohlcv_price_frame(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            universe_file = raw_dir / "mainboard_a_stocks.csv"
            pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "name": ["A"],
                    "list_status": ["L"],
                    "list_date": ["20200101"],
                }
            ).to_csv(universe_file, index=False)
            _raw(raw_dir / "000001.SZ.csv", "000001.SZ", "2024-01-03")

            config = _config(raw_dir, universe_file)
            prices = pd.DataFrame(
                {
                    "open": [10.0],
                    "close": [10.2],
                    "volume": [1000.0],
                },
                index=pd.DatetimeIndex(["2024-01-03"]),
            )
            factors = _factors("2024-01-03", ["000001.SZ"])

            with self.assertRaisesRegex(ValueError, "close-price panel"):
                build_data_health_report(config, price_df=prices, factor_df=factors)


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


def _price_panel(dates: list[str], values: dict[str, list[float | None]]) -> pd.DataFrame:
    columns = pd.MultiIndex.from_product([["close"], list(values)], names=["field", "instrument"])
    rows = list(zip(*values.values()))
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates), columns=columns)


def _factor_rows(rows: list[tuple[str, str]]) -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(date), instrument) for date, instrument in rows],
        names=["datetime", "instrument"],
    )
    return pd.DataFrame({"ROC5": [1.0 for _ in rows]}, index=index)


if __name__ == "__main__":
    unittest.main()
