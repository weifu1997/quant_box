"""模块说明：覆盖 common 规范化工具。"""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from src.common import (
    is_adj_factor_stock_csv,
    is_stock_csv,
    normalize_datetime_index,
    normalize_instrument_index,
    normalize_instruments,
    normalize_multiindex_date_instrument,
)


class CommonNormalizationTests(unittest.TestCase):
    """类说明：组织公共规范化工具测试。"""

    def test_normalize_instruments_drops_blanks_and_deduplicates(self) -> None:
        values = [" 000001.sz ", "000001.SZ", None, "", "600000.sh"]

        result = normalize_instruments(values)

        self.assertEqual(result, ["000001.SZ", "600000.SH"])

    def test_normalize_instrument_index_preserves_length(self) -> None:
        index = normalize_instrument_index([" a ", None, "b"], name="instrument")

        self.assertEqual(index.tolist(), ["A", "", "B"])
        self.assertEqual(index.name, "instrument")

    def test_normalize_datetime_index_can_drop_unique_and_sort(self) -> None:
        dates = normalize_datetime_index(
            ["2024-01-03 15:00", "bad", "2024-01-02 09:30", "2024-01-03 09:30"],
            dropna=True,
            unique=True,
            sort=True,
        )

        self.assertEqual(dates.tolist(), [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")])

    def test_normalize_multiindex_date_instrument_drops_invalid_rows(self) -> None:
        index = pd.MultiIndex.from_tuples(
            [
                ("2024-01-02 15:00", " a "),
                ("bad", "B"),
                ("2024-01-03", ""),
            ],
            names=["datetime", "instrument"],
        )

        normalized = normalize_multiindex_date_instrument(index)

        self.assertEqual(normalized.tolist(), [(pd.Timestamp("2024-01-02"), "A")])

    def test_stock_csv_recognizes_supported_a_share_exchanges(self) -> None:
        self.assertTrue(is_stock_csv(Path("000001.SZ.csv")))
        self.assertTrue(is_stock_csv(Path("600000.SH.csv")))
        self.assertTrue(is_stock_csv(Path("830001.BJ.csv")))
        self.assertFalse(is_stock_csv(Path("index_constituents.csv")))
        self.assertFalse(is_stock_csv(Path("000001.HK.csv")))

    def test_adj_factor_stock_csv_includes_bj_and_excludes_index_benchmarks(self) -> None:
        self.assertTrue(is_adj_factor_stock_csv(Path("830001.BJ.csv")))
        self.assertTrue(is_adj_factor_stock_csv(Path("600000.SH.csv")))
        self.assertFalse(is_adj_factor_stock_csv(Path("000300.SH.csv")))
        self.assertFalse(is_adj_factor_stock_csv(Path("000905.SH.csv")))


if __name__ == "__main__":
    unittest.main()
