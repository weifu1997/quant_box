"""模块说明：覆盖 test_run_ml_experiments 相关行为的测试用例。"""

from __future__ import annotations

import importlib
import unittest


class RunMlExperimentsTests(unittest.TestCase):
    """类说明：组织 RunMlExperimentsTests 测试用例。"""
    def test_module_imports_with_fast_backtest_helper(self) -> None:
        """函数说明：验证 test_module_imports_with_fast_backtest_helper 覆盖的行为场景。"""
        module = importlib.import_module("scripts.run_ml_experiments")

        self.assertTrue(callable(module.main))


if __name__ == "__main__":
    unittest.main()
