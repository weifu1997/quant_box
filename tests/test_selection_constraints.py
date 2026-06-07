from __future__ import annotations

import unittest

from src.selection_constraints import apply_selection_constraints_to_backtest_config


class SelectionConstraintsTests(unittest.TestCase):
    def test_apply_selection_constraints_carries_selection_risk_filter(self) -> None:
        result = apply_selection_constraints_to_backtest_config(
            {"top_n": 1},
            {"selection_risk_filter": {"enabled": True, "lookback_sessions": 3}},
        )

        self.assertEqual(result["selection_risk_filter"]["lookback_sessions"], 3)


if __name__ == "__main__":
    unittest.main()
