"""模块说明：覆盖 test_run_goal_fast_factor_screen 相关行为的测试用例。"""

from __future__ import annotations

import unittest

from scripts.run_goal_fast_factor_screen import _screen_quality_fields


class RunGoalFastFactorScreenTests(unittest.TestCase):
    """类说明：组织 RunGoalFastFactorScreenTests 测试用例。"""
    def test_screen_quality_fields_use_configured_thresholds(self) -> None:
        """函数说明：验证 test_screen_quality_fields_use_configured_thresholds 覆盖的行为场景。"""
        config = {
            "quality": {
                "min_backtest_annual_return": 0.25,
                "max_backtest_drawdown_limit": -0.15,
            }
        }

        fields = _screen_quality_fields({"annual_return": 0.24, "max_drawdown": -0.16}, config)
        passing = _screen_quality_fields({"annual_return": 0.26, "max_drawdown": -0.14}, config)

        self.assertFalse(fields["meets_full_target"])
        self.assertAlmostEqual(fields["target_gap"], 0.02)
        self.assertTrue(passing["meets_full_target"])
        self.assertEqual(passing["target_gap"], 0.0)


if __name__ == "__main__":
    unittest.main()
