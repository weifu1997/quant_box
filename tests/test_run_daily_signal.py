"""模块说明：覆盖 test_run_daily_signal 相关行为的测试用例。"""

from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


class RunDailySignalTests(unittest.TestCase):
    """类说明：组织 RunDailySignalTests 测试用例。"""
    def test_default_run_writes_candidate_without_overwriting_latest_holdings(self) -> None:
        """函数说明：验证 test_default_run_writes_candidate_without_overwriting_latest_holdings 覆盖的行为场景。"""
        module = importlib.import_module("scripts.run_daily_signal")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            latest = root / "latest_holdings.csv"
            latest.write_text("instrument\nOLD.SZ\n", encoding="utf-8")
            config = {"outputs": {"dir": str(root), "holdings_file": str(latest)}}
            signal = pd.DataFrame([{"date": "2024-01-03", "instrument": "000001.SZ", "action": "BUY"}])

            with patch.object(sys, "argv", ["run_daily_signal.py", "--date", "latest"]), patch(
                "scripts.run_daily_signal.read_previous_holdings",
                return_value=["OLD.SZ"],
            ), patch(
                "scripts.run_daily_signal.generate_signal",
                return_value=(signal, ["000001.SZ"]),
            ), patch("src.signal_generator.load_config", return_value=config):
                module.main()

            self.assertTrue((root / "candidate_signal_2024-01-03.csv").exists())
            self.assertTrue((root / "candidate_holdings_2024-01-03.csv").exists())
            self.assertEqual(latest.read_text(encoding="utf-8"), "instrument\nOLD.SZ\n")

    def test_empty_latest_signal_uses_signal_date_metadata_for_candidate_outputs(self) -> None:
        """函数说明：验证 test_empty_latest_signal_uses_signal_date_metadata_for_candidate_outputs 覆盖的行为场景。"""
        module = importlib.import_module("scripts.run_daily_signal")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            latest = root / "latest_holdings.csv"
            latest.write_text("instrument\n", encoding="utf-8")
            config = {"outputs": {"dir": str(root), "holdings_file": str(latest)}}
            signal = pd.DataFrame(columns=["date", "instrument", "action"])
            signal.attrs["signal_date"] = "2024-01-03"

            with patch.object(sys, "argv", ["run_daily_signal.py", "--date", "latest"]), patch(
                "scripts.run_daily_signal.read_previous_holdings",
                return_value=[],
            ), patch(
                "scripts.run_daily_signal.generate_signal",
                return_value=(signal, []),
            ), patch("src.signal_generator.load_config", return_value=config):
                module.main()

            self.assertTrue((root / "candidate_signal_2024-01-03.csv").exists())
            self.assertTrue((root / "candidate_holdings_2024-01-03.csv").exists())
            self.assertFalse((root / "candidate_signal_latest.csv").exists())

    def test_official_run_overwrites_latest_holdings(self) -> None:
        """函数说明：验证 test_official_run_overwrites_latest_holdings 覆盖的行为场景。"""
        module = importlib.import_module("scripts.run_daily_signal")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            latest = root / "latest_holdings.csv"
            latest.write_text("instrument\nOLD.SZ\n", encoding="utf-8")
            config = {"outputs": {"dir": str(root), "holdings_file": str(latest)}}
            signal = pd.DataFrame([{"date": "2024-01-03", "instrument": "000001.SZ", "action": "BUY"}])

            with patch.object(sys, "argv", ["run_daily_signal.py", "--date", "latest", "--official"]), patch(
                "scripts.run_daily_signal.read_previous_holdings",
                return_value=["OLD.SZ"],
            ), patch(
                "scripts.run_daily_signal.generate_signal",
                return_value=(signal, ["000001.SZ"]),
            ), patch("src.signal_generator.load_config", return_value=config):
                module.main()

            self.assertTrue((root / "signal_2024-01-03.csv").exists())
            latest_frame = pd.read_csv(latest)
            self.assertEqual(latest_frame["instrument"].tolist(), ["000001.SZ"])


if __name__ == "__main__":
    unittest.main()
