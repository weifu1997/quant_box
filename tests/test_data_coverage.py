"""模块说明：覆盖 test_data_coverage 相关行为的测试用例。"""

from __future__ import annotations

import unittest

import pandas as pd

from src.data_coverage import (
    build_price_data_gaps,
    build_skipped_months,
    build_yearly_equity_coverage,
    price_coverage_summary,
)


class DataCoverageTests(unittest.TestCase):
    """类说明：组织 DataCoverageTests 测试用例。"""
    def test_price_data_gaps_report_missing_symbols_by_date(self) -> None:
        """函数说明：验证 test_price_data_gaps_report_missing_symbols_by_date 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        close = pd.DataFrame({"A": [10.0, None], "B": [20.0, 21.0]}, index=dates)
        prices = pd.concat({"close": close}, axis=1)

        gaps = build_price_data_gaps(prices, "2024-01-02", "2024-01-03")
        summary = price_coverage_summary(prices, "2024-01-02", "2024-01-03")

        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps.iloc[0]["date"], "2024-01-03")
        self.assertEqual(gaps.iloc[0]["missing_instruments"], "A")
        self.assertEqual(summary["gap_dates"], 1)
        self.assertAlmostEqual(float(summary["min_coverage"]), 0.5)

    def test_price_coverage_uses_latest_intraday_snapshot_per_date(self) -> None:
        """函数说明：验证 test_price_coverage_uses_latest_intraday_snapshot_per_date 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02 15:00", "2024-01-02 09:30", "2024-01-03 15:00"])
        close = pd.DataFrame({"A": [10.0, None, 11.0], "B": [20.0, 21.0, 22.0]}, index=dates)
        prices = pd.concat({"close": close}, axis=1)

        gaps = build_price_data_gaps(prices, "2024-01-02", "2024-01-03")
        summary = price_coverage_summary(prices, "2024-01-02", "2024-01-03")

        self.assertTrue(gaps.empty)
        self.assertEqual(summary["price_dates"], 2)
        self.assertEqual(summary["gap_dates"], 0)
        self.assertAlmostEqual(float(summary["min_coverage"]), 1.0)

    def test_price_coverage_rejects_flat_ohlcv_price_frame(self) -> None:
        """函数说明：验证 test_price_coverage_rejects_flat_ohlcv_price_frame 覆盖的行为场景。"""
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        prices = pd.DataFrame(
            {
                "open": [10.0, 10.1],
                "close": [10.0, 10.2],
                "volume": [1000.0, 1200.0],
            },
            index=dates,
        )

        with self.assertRaisesRegex(ValueError, "close-price panel"):
            price_coverage_summary(prices, "2024-01-02", "2024-01-03")

    def test_skipped_months_filters_non_empty_reasons(self) -> None:
        """函数说明：验证 test_skipped_months_filters_non_empty_reasons 覆盖的行为场景。"""
        diagnostics = pd.DataFrame(
            [
                {"signal_date": "2024-01-31", "skip_reason": ""},
                {"signal_date": "2024-02-29", "skip_reason": "insufficient_train_rows"},
            ]
        )

        skipped = build_skipped_months(diagnostics)

        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped.iloc[0]["signal_date"], "2024-02-29")

    def test_yearly_equity_coverage_flags_missing_years(self) -> None:
        """函数说明：验证 test_yearly_equity_coverage_flags_missing_years 覆盖的行为场景。"""
        equity = pd.Series(
            [100.0, 101.0],
            index=pd.to_datetime(["2024-01-02", "2026-01-02"]),
        )

        coverage = build_yearly_equity_coverage(equity, "2024-01-01", "2026-12-31")

        self.assertEqual(coverage["year"].to_list(), [2024, 2025, 2026])
        missing = coverage[coverage["year"] == 2025].iloc[0]
        self.assertFalse(bool(missing["has_equity"]))


if __name__ == "__main__":
    unittest.main()
