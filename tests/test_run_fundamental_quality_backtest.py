from __future__ import annotations

import unittest

import pandas as pd

from scripts.run_fundamental_quality_backtest import (
    combine_quality_and_price_scores,
    fundamental_quality_scores,
    month_end_signal_dates,
)


class RunFundamentalQualityBacktestTests(unittest.TestCase):
    def test_month_end_signal_dates_use_last_trading_day_in_each_month(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-31", "2024-02-01", "2024-02-29", "2024-03-01"])

        result = month_end_signal_dates(dates, start_date="2024-01-01", end_date="2024-02-29")

        self.assertEqual([value.strftime("%Y-%m-%d") for value in result], ["2024-01-31", "2024-02-29"])

    def test_fundamental_quality_scores_rank_quality_dividend_cash_and_low_debt(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "review_status": "PASS",
                    "total_score": 7,
                    "roe": 0.20,
                    "dividend_yield_ttm": 0.04,
                    "fcf_yield": 0.08,
                    "debt_to_assets": 0.30,
                },
                {
                    "ts_code": "000002.SZ",
                    "review_status": "WATCH",
                    "total_score": 4,
                    "roe": 0.09,
                    "dividend_yield_ttm": 0.02,
                    "fcf_yield": 0.01,
                    "debt_to_assets": 0.55,
                },
                {
                    "ts_code": "000003.SZ",
                    "review_status": "REJECT",
                    "total_score": 3,
                    "roe": 0.30,
                    "dividend_yield_ttm": 0.06,
                    "fcf_yield": 0.10,
                    "debt_to_assets": 0.20,
                },
            ]
        )

        scores = fundamental_quality_scores(frame, min_total_score=4, statuses={"PASS", "WATCH"})

        self.assertEqual(scores.index.tolist(), ["000001.SZ", "000002.SZ"])
        self.assertGreater(scores.loc["000001.SZ"], scores.loc["000002.SZ"])

    def test_fundamental_quality_scores_can_keep_all_statuses_with_score_floor(self) -> None:
        frame = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "review_status": "REJECT", "total_score": 5, "roe": 0.2},
                {"ts_code": "000002.SZ", "review_status": "REJECT", "total_score": 2, "roe": 0.3},
            ]
        )

        scores = fundamental_quality_scores(frame, min_total_score=4, statuses=None)

        self.assertEqual(scores.index.tolist(), ["000001.SZ"])

    def test_combine_quality_and_price_scores_filters_price_universe(self) -> None:
        index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-31"), "000001.SZ"),
                (pd.Timestamp("2024-01-31"), "000002.SZ"),
                (pd.Timestamp("2024-01-31"), "000003.SZ"),
            ],
            names=["date", "instrument"],
        )
        price_scores = pd.Series([1.0, 3.0, 2.0], index=index)
        quality_scores = pd.Series([10.0, 5.0], index=index[:2])

        combined = combine_quality_and_price_scores(
            price_scores=price_scores,
            quality_scores=quality_scores,
            mode="filter_price",
        )

        self.assertEqual(combined.droplevel(0).index.tolist(), ["000001.SZ", "000002.SZ"])
        self.assertEqual(combined.loc[(pd.Timestamp("2024-01-31"), "000002.SZ")], 3.0)

    def test_combine_quality_and_price_scores_blends_quality_with_price(self) -> None:
        index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-31"), "000001.SZ"),
                (pd.Timestamp("2024-01-31"), "000002.SZ"),
            ],
            names=["date", "instrument"],
        )
        price_scores = pd.Series([1.0, 1.1], index=index)
        quality_scores = pd.Series([100.0, 1.0], index=index)

        combined = combine_quality_and_price_scores(
            price_scores=price_scores,
            quality_scores=quality_scores,
            mode="blend",
            quality_weight=2.0,
        )

        self.assertGreater(
            combined.loc[(pd.Timestamp("2024-01-31"), "000001.SZ")],
            combined.loc[(pd.Timestamp("2024-01-31"), "000002.SZ")],
        )


if __name__ == "__main__":
    unittest.main()
