"""Tests for rolling candidate-equity selector helpers."""

from __future__ import annotations

import unittest

import pandas as pd

from scripts.run_candidate_equity_selector import (
    rebalance_schedule,
    run_equity_selector,
    trailing_candidate_scores,
)


class RunCandidateEquitySelectorTests(unittest.TestCase):
    def test_trailing_scores_exclude_as_of_date_return(self) -> None:
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        equities = pd.DataFrame(
            {
                "past_winner": [1.0, 1.1, 1.2, 1.3, 1.3],
                "same_day_jump": [1.0, 1.0, 1.0, 1.0, 3.0],
            },
            index=dates,
        )

        scores = trailing_candidate_scores(
            equities,
            pd.Timestamp("2024-01-05"),
            lookback_days=4,
            min_periods=3,
        )

        self.assertGreater(float(scores["past_winner"]), float(scores["same_day_jump"]))

    def test_rebalance_schedule_uses_last_trading_day_of_month(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-31", "2024-02-01", "2024-02-29"])

        schedule = rebalance_schedule(pd.DatetimeIndex(dates), "monthly")

        self.assertEqual(schedule, [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")])

    def test_run_equity_selector_switches_to_trailing_winner(self) -> None:
        dates = pd.date_range("2024-01-01", periods=8, freq="D")
        equities = pd.DataFrame(
            {
                "steady": [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07],
                "flat": [1.0] * 8,
            },
            index=dates,
        )

        run = run_equity_selector(
            equities,
            lookback_days=4,
            top_k=1,
            min_periods=3,
            rebalance_freq="weekly",
        )

        selected = "|".join(run.selections["selected_candidates"].astype(str).tolist())
        self.assertIn("steady", selected)
        self.assertGreater(float(run.equity.iloc[-1]), 1.0)


if __name__ == "__main__":
    unittest.main()
