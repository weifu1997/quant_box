from __future__ import annotations

import unittest

import pandas as pd

from scripts.run_goal_bottleneck_scan import build_bottleneck_summary


class RunGoalBottleneckScanTests(unittest.TestCase):
    def test_build_bottleneck_summary_counts_yearly_passes(self) -> None:
        yearly = pd.DataFrame(
            [
                {"year": 2022, "annual_return": 0.25, "max_drawdown": -0.10, "candidate_file": "a_years.csv"},
                {"year": 2022, "annual_return": 0.30, "max_drawdown": -0.25, "candidate_file": "b_years.csv"},
                {"year": 2023, "annual_return": 0.10, "max_drawdown": -0.08, "candidate_file": "a_years.csv"},
                {"year": 2023, "annual_return": 0.18, "max_drawdown": -0.30, "candidate_file": "b_years.csv"},
            ]
        )

        summary = build_bottleneck_summary(yearly, return_target=0.20, drawdown_limit=-0.20)
        rows = summary.set_index("year")

        self.assertEqual(rows.loc[2022, "return_pass_count"], 2)
        self.assertEqual(rows.loc[2022, "drawdown_pass_count"], 1)
        self.assertEqual(rows.loc[2022, "both_pass_count"], 1)
        self.assertEqual(rows.loc[2022, "best_annual_return_file"], "b_years.csv")
        self.assertAlmostEqual(rows.loc[2023, "return_target_gap"], 0.02)
        self.assertEqual(rows.loc[2023, "both_pass_count"], 0)


if __name__ == "__main__":
    unittest.main()
