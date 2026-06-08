from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.selection_constraints import apply_selection_constraints_to_backtest_config, load_industry_group_map


class SelectionConstraintsTests(unittest.TestCase):
    def test_apply_selection_constraints_carries_selection_risk_filter(self) -> None:
        result = apply_selection_constraints_to_backtest_config(
            {"top_n": 1},
            {"selection_risk_filter": {"enabled": True, "lookback_sessions": 3}},
        )

        self.assertEqual(result["selection_risk_filter"]["lookback_sessions"], 3)

    def test_apply_selection_constraints_loads_industry_map_when_weight_cap_enabled(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "industry.csv"
            pd.DataFrame(
                [
                    {"ts_code": "000001.SZ", "industry": "bank"},
                    {"ts_code": "000002.SZ", "industry": "tech"},
                ]
            ).to_csv(path, index=False)

            result = apply_selection_constraints_to_backtest_config(
                {"top_n": 2, "max_industry_weight": 0.5},
                {"industry_file": str(path)},
            )

            self.assertEqual(result["industry_map"].loc["000001.SZ"], "bank")
            self.assertEqual(result["industry_map"].loc["000002.SZ"], "tech")

    def test_apply_selection_constraints_does_not_load_industry_map_when_disabled(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "industry.csv"
            pd.DataFrame([{"ts_code": "000001.SZ", "industry": "bank"}]).to_csv(path, index=False)

            result = apply_selection_constraints_to_backtest_config(
                {"top_n": 2, "max_industry_weight": None},
                {"industry_file": str(path)},
            )

            self.assertNotIn("industry_map", result)

    def test_load_industry_group_map_uses_last_duplicate_code(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "industry.csv"
            pd.DataFrame(
                [
                    {"ts_code": "000001.SZ", "industry": "old"},
                    {"ts_code": "000001.SZ", "industry": "new"},
                    {"ts_code": "000002.SZ", "industry": None},
                ]
            ).to_csv(path, index=False)

            industry_map = load_industry_group_map({"industry_file": str(path)})

            self.assertEqual(industry_map.loc["000001.SZ"], "new")
            self.assertEqual(industry_map.loc["000002.SZ"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
