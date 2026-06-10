from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from scripts.run_goal_audit import audit_yearly_goal, load_equity, write_audit_outputs


class RunGoalAuditTests(unittest.TestCase):
    def test_audit_yearly_goal_identifies_failed_years_and_buffers(self) -> None:
        yearly = pd.DataFrame(
            [
                {"year": 2022, "annual_return": 0.24, "max_drawdown": -0.18},
                {"year": 2023, "annual_return": 0.08, "max_drawdown": -0.25},
            ]
        )

        audited, summary = audit_yearly_goal(yearly, return_target=0.20, drawdown_limit=-0.20)

        self.assertEqual(summary["year_count"], 2)
        self.assertEqual(summary["year_return_pass_count"], 1)
        self.assertEqual(summary["year_drawdown_pass_count"], 1)
        self.assertFalse(summary["is_goal_met"])
        self.assertEqual(summary["years_below_return_target"], [2023])
        self.assertEqual(summary["years_breaching_drawdown_limit"], [2023])
        self.assertAlmostEqual(float(audited.loc[audited["year"] == 2023, "annual_return_gap"].iloc[0]), -0.12)
        self.assertAlmostEqual(float(audited.loc[audited["year"] == 2023, "drawdown_buffer"].iloc[0]), -0.05)

    def test_load_equity_reads_backtest_csv_shape(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "backtest_equity.csv"
            path.write_text(",equity\n2024-01-02,100.0\n2024-01-03,101.0\n", encoding="utf-8")

            equity = load_equity(path)

            self.assertEqual(equity.name, "equity")
            self.assertEqual(list(equity.index.strftime("%Y-%m-%d")), ["2024-01-02", "2024-01-03"])
            self.assertEqual(equity.iloc[-1], 101.0)

    def test_write_audit_outputs_persists_json_markdown_and_years(self) -> None:
        with TemporaryDirectory() as tmp:
            yearly, summary = audit_yearly_goal(
                pd.DataFrame([{"year": 2024, "annual_return": 0.30, "max_drawdown": -0.10}]),
                return_target=0.20,
                drawdown_limit=-0.20,
            )

            paths = write_audit_outputs(
                output_prefix=Path(tmp) / "audit",
                yearly=yearly,
                summary=summary,
                metrics={"annual_return": 0.30, "max_drawdown": -0.10},
            )

            self.assertTrue(Path(paths["years"]).exists())
            self.assertTrue(Path(paths["json"]).exists())
            self.assertTrue(Path(paths["markdown"]).exists())
            self.assertIn("Status: PASS", Path(paths["markdown"]).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
