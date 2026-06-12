"""模块说明：覆盖 common 规范化工具。"""

from __future__ import annotations

import unittest

import pandas as pd

from src.common import normalize_datetime_index, normalize_instrument_index, normalize_instruments, normalize_multiindex_date_instrument


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


if __name__ == "__main__":
    unittest.main()
