"""模块说明：覆盖 test_universe_coverage 相关行为的测试用例。"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from src.universe_coverage import summarize_universe_coverage


class UniverseCoverageTests(unittest.TestCase):
    """类说明：组织 UniverseCoverageTests 测试用例。"""
    def test_summarize_universe_coverage_reports_local_and_price_panel_counts(self) -> None:
        """函数说明：验证 test_summarize_universe_coverage_reports_local_and_price_panel_counts 覆盖的行为场景。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            (raw_dir / "000001.SZ.csv").write_text("", encoding="utf-8")
            (raw_dir / "600000.SH.csv").write_text("", encoding="utf-8")
            (raw_dir / "failed_fetches.csv").write_text("", encoding="utf-8")
            universe_file = root / "mainboard_a_stocks.csv"
            pd.DataFrame(
                [
                    {"ts_code": "000001.SZ", "name": "PINGAN", "list_date": "19910403", "list_status": "L"},
                    {"ts_code": "600000.SH", "name": "SPDB", "list_date": "19991110", "list_status": "L"},
                    {"ts_code": "000002.SZ", "name": "VANKE", "list_date": "19910129", "list_status": "L"},
                ]
            ).to_csv(universe_file, index=False)
            prices = pd.DataFrame({"000001.SZ": [10.0], "600000.SH": [11.0]})
            config = {
                "data": {
                    "raw_dir": str(raw_dir),
                    "constituents_file": str(universe_file),
                    "universe": "mainboard_a",
                    "end_date": "2024-01-01",
                    "exclude_st": True,
                }
            }

            with patch("src.universe_coverage.resolve_path", side_effect=lambda value: Path(value)):
                coverage = summarize_universe_coverage(config, price_df=prices)

        self.assertEqual(coverage["target_symbols"], 3)
        self.assertEqual(coverage["raw_stock_files"], 2)
        self.assertEqual(coverage["price_panel_symbols"], 2)
        self.assertAlmostEqual(float(coverage["price_target_coverage"]), 2 / 3)

    def test_summarize_universe_coverage_normalizes_symbol_whitespace_and_case(self) -> None:
        """函数说明：验证 test_summarize_universe_coverage_normalizes_symbol_whitespace_and_case 覆盖的行为场景。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            (raw_dir / "000001.SZ.csv").write_text("", encoding="utf-8")
            universe_file = root / "mainboard_a_stocks.csv"
            pd.DataFrame(
                [
                    {"ts_code": " 000001.sz ", "name": "PINGAN", "list_date": "19910403", "list_status": "L"},
                ]
            ).to_csv(universe_file, index=False)
            prices = pd.DataFrame({" 000001.sz ": [10.0]})
            config = {
                "data": {
                    "raw_dir": str(raw_dir),
                    "constituents_file": str(universe_file),
                    "universe": "mainboard_a",
                    "end_date": "2024-01-01",
                    "exclude_st": True,
                }
            }

            with patch("src.universe_coverage.resolve_path", side_effect=lambda value: Path(value)):
                coverage = summarize_universe_coverage(config, price_df=prices)

        self.assertEqual(coverage["target_symbols"], 1)
        self.assertEqual(coverage["raw_target_symbols"], 1)
        self.assertEqual(coverage["price_target_symbols"], 1)
        self.assertEqual(coverage["price_target_coverage"], 1.0)

    def test_summarize_universe_coverage_rejects_flat_ohlcv_price_frame(self) -> None:
        """函数说明：验证 test_summarize_universe_coverage_rejects_flat_ohlcv_price_frame 覆盖的行为场景。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            universe_file = root / "mainboard_a_stocks.csv"
            pd.DataFrame(
                [
                    {"ts_code": "000001.SZ", "name": "PINGAN", "list_date": "19910403", "list_status": "L"},
                ]
            ).to_csv(universe_file, index=False)
            prices = pd.DataFrame(
                {
                    "open": [10.0],
                    "close": [10.2],
                    "volume": [1000.0],
                }
            )
            config = {
                "data": {
                    "raw_dir": str(raw_dir),
                    "constituents_file": str(universe_file),
                    "universe": "mainboard_a",
                    "end_date": "2024-01-01",
                    "exclude_st": True,
                }
            }

            with patch("src.universe_coverage.resolve_path", side_effect=lambda value: Path(value)):
                with self.assertRaisesRegex(ValueError, "close-price panel"):
                    summarize_universe_coverage(config, price_df=prices)


if __name__ == "__main__":
    unittest.main()
