"""模块说明：覆盖 test_reporting 相关行为的测试用例。"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.reporting import archive_run, signal_action_summary, write_daily_signal_report


class ReportingTests(unittest.TestCase):
    """类说明：组织 ReportingTests 测试用例。"""
    def test_write_daily_signal_report_renders_quality_and_repair_context(self) -> None:
        """函数说明：验证 test_write_daily_signal_report_renders_quality_and_repair_context 覆盖的行为场景。"""
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "outputs"
            report = {
                "signal_date": "2024-01-03",
                "intended_trade_date": "2024-01-04",
                "is_executable": False,
                "block_reasons": ["data_unhealthy"],
                "data_health": {
                    "is_healthy": False,
                    "target_symbols": 10,
                    "raw_target_coverage": 0.8,
                    "issues": ["raw_missing"],
                },
                "data_governance": {
                    "repair_actions": [
                        {
                            "component": "daily_basic",
                            "reason": "coverage_gap",
                            "commands": ["python scripts/run_update_point_in_time_data.py"],
                        }
                    ],
                },
                "signal_summary": {"BUY": 1, "HOLD": 2, "SELL": 3},
                "files": {"manual_orders": "outputs/manual_orders.csv"},
            }

            path = write_daily_signal_report(report, out_dir)

            text = path.read_text(encoding="utf-8")
            self.assertEqual(path, out_dir / "daily_signal_report.md")
            self.assertIn("# Daily Signal Report", text)
            self.assertIn("- Signal date: 2024-01-03", text)
            self.assertIn("- Block reasons: data_unhealthy", text)
            self.assertIn("- Raw coverage: 80.00%", text)
            self.assertIn("Repair action: daily_basic (coverage_gap)", text)
            self.assertIn("- SELL: 3", text)

    def test_archive_run_copies_existing_files_under_signal_date(self) -> None:
        """函数说明：验证 test_archive_run_copies_existing_files_under_signal_date 覆盖的行为场景。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "auto_signal_report.json"
            source.write_text('{"ok": true}', encoding="utf-8")

            target = archive_run([source, root / "missing.json"], root / "history", "2024-01-03")

            self.assertEqual(target, root / "history" / "2024-01-03")
            self.assertEqual((target / source.name).read_text(encoding="utf-8"), '{"ok": true}')
            self.assertFalse((target / "missing.json").exists())

    def test_signal_action_summary_counts_known_actions_case_insensitively(self) -> None:
        """函数说明：验证 test_signal_action_summary_counts_known_actions_case_insensitively 覆盖的行为场景。"""
        frame = pd.DataFrame({"action": ["buy", "BUY", "hold", "sell", "ignore"]})

        self.assertEqual(signal_action_summary(frame), {"BUY": 2, "HOLD": 1, "SELL": 1})


if __name__ == "__main__":
    unittest.main()
