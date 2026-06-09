"""模块说明：覆盖 test_run_apply_fills 相关行为的测试用例。"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd


class RunApplyFillsTests(unittest.TestCase):
    """类说明：组织 RunApplyFillsTests 测试用例。"""
    def test_apply_fills_updates_current_holdings_and_writes_audit(self) -> None:
        """函数说明：验证 test_apply_fills_updates_current_holdings_and_writes_audit 覆盖的行为场景。"""
        module = importlib.import_module("scripts.run_apply_fills")
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            holdings_file = root / "current_holdings.csv"
            fill_file = root / "fill_feedback_2024-01-03.csv"
            pd.DataFrame({"instrument": ["000001.SZ"], "shares": [100]}).to_csv(holdings_file, index=False)
            pd.DataFrame(
                [
                    {
                        "signal_date": "2024-01-03",
                        "instrument": "000001.SZ",
                        "side": "BUY",
                        "planned_order_shares": 200,
                        "executed_shares": 200,
                        "fill_status": " filled ",
                    },
                    {
                        "signal_date": "2024-01-03",
                        "instrument": "600519.SH",
                        "side": "BUY",
                        "planned_order_shares": 100,
                        "executed_shares": 100,
                        "fill_status": "PARTIAL",
                    },
                ]
            ).to_csv(fill_file, index=False)
            config = {
                "account": {"current_holdings_file": str(holdings_file)},
                "outputs": {"dir": str(root / "outputs")},
            }

            with patch.object(sys, "argv", ["run_apply_fills.py", str(fill_file)]), patch.object(
                module, "load_config", return_value=config
            ):
                module.main()

            updated = pd.read_csv(holdings_file).set_index("instrument")
            self.assertEqual(float(updated.loc["000001.SZ", "shares"]), 300.0)
            self.assertEqual(float(updated.loc["600519.SH", "shares"]), 100.0)
            audit = json.loads((root / "outputs" / "fill_apply_audit_2024-01-03.json").read_text(encoding="utf-8"))
            self.assertFalse(audit["dry_run"])
            self.assertEqual(audit["applied_fill_rows"], 2)

    def test_apply_fills_dry_run_keeps_current_holdings_unchanged(self) -> None:
        """函数说明：验证 test_apply_fills_dry_run_keeps_current_holdings_unchanged 覆盖的行为场景。"""
        module = importlib.import_module("scripts.run_apply_fills")
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            holdings_file = root / "current_holdings.csv"
            fill_file = root / "fill_feedback_2024-01-03.csv"
            pd.DataFrame({"instrument": ["000001.SZ"], "shares": [100]}).to_csv(holdings_file, index=False)
            pd.DataFrame(
                [
                    {
                        "signal_date": "2024-01-03",
                        "instrument": "000001.SZ",
                        "side": "BUY",
                        "planned_order_shares": 200,
                        "executed_shares": 200,
                        "fill_status": "FILLED",
                    }
                ]
            ).to_csv(fill_file, index=False)
            config = {
                "account": {"current_holdings_file": str(holdings_file)},
                "outputs": {"dir": str(root / "outputs")},
            }

            with patch.object(sys, "argv", ["run_apply_fills.py", str(fill_file), "--dry-run"]), patch.object(
                module, "load_config", return_value=config
            ):
                module.main()

            unchanged = pd.read_csv(holdings_file)
            self.assertEqual(float(unchanged.loc[0, "shares"]), 100.0)
            audit = json.loads((root / "outputs" / "fill_apply_audit_2024-01-03.json").read_text(encoding="utf-8"))
            self.assertTrue(audit["dry_run"])
            self.assertEqual(audit["updated_positions"], 1)


if __name__ == "__main__":
    unittest.main()
