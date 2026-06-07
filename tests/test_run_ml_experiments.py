from __future__ import annotations

import importlib
import unittest


class RunMlExperimentsTests(unittest.TestCase):
    def test_module_imports_with_fast_backtest_helper(self) -> None:
        module = importlib.import_module("scripts.run_ml_experiments")

        self.assertTrue(callable(module.main))


if __name__ == "__main__":
    unittest.main()
