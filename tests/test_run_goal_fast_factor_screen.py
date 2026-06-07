from __future__ import annotations

import unittest

from scripts.run_goal_fast_factor_screen import _screen_quality_fields


class RunGoalFastFactorScreenTests(unittest.TestCase):
    def test_screen_quality_fields_use_configured_thresholds(self) -> None:
        config = {
            "quality": {
                "min_backtest_annual_return": 0.25,
                "min_yearly_annual_return": 0.10,
                "max_backtest_drawdown_limit": -0.15,
                "max_yearly_drawdown_limit": -0.20,
            }
        }

        fields = _screen_quality_fields(
            {
                "annual_return": 0.24,
                "max_drawdown": -0.16,
                "min_year_annual_return": 0.11,
                "worst_year_drawdown": -0.18,
            },
            config,
        )
        passing = _screen_quality_fields(
            {
                "annual_return": 0.26,
                "max_drawdown": -0.14,
                "min_year_annual_return": 0.11,
                "worst_year_drawdown": -0.18,
            },
            config,
        )

        self.assertFalse(fields["meets_full_target"])
        self.assertAlmostEqual(fields["target_gap"], 0.02)
        self.assertTrue(passing["meets_full_target"])
        self.assertEqual(passing["target_gap"], 0.0)

    def test_screen_quality_fields_reject_weak_yearly_metrics(self) -> None:
        config = {
            "quality": {
                "min_backtest_annual_return": 0.20,
                "min_yearly_annual_return": 0.10,
                "max_backtest_drawdown_limit": -0.20,
                "max_yearly_drawdown_limit": -0.20,
            }
        }

        fields = _screen_quality_fields(
            {
                "annual_return": 0.30,
                "max_drawdown": -0.10,
                "min_year_annual_return": -0.05,
                "worst_year_drawdown": -0.25,
            },
            config,
        )

        self.assertFalse(fields["meets_full_target"])
        self.assertFalse(fields["yearly_annual_return_pass"])
        self.assertFalse(fields["yearly_drawdown_pass"])
        self.assertAlmostEqual(fields["target_gap"], 0.20)


if __name__ == "__main__":
    unittest.main()
