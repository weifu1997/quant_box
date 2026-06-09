"""模块说明：覆盖 test_run_auto_signal_supervised 相关行为的测试用例。"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


class RunAutoSignalSupervisedTests(unittest.TestCase):
    """类说明：组织 RunAutoSignalSupervisedTests 测试用例。"""
    def test_normalize_run_args_strips_optional_separator(self) -> None:
        """函数说明：验证 test_normalize_run_args_strips_optional_separator 覆盖的行为场景。"""
        module = importlib.import_module("scripts.run_auto_signal_supervised")

        self.assertEqual(module._normalize_run_args(["--", "--skip-update"]), ["--skip-update"])
        self.assertEqual(module._normalize_run_args(["--skip-update"]), ["--skip-update"])

    def test_start_background_run_writes_job_and_log_with_passed_args(self) -> None:
        """函数说明：验证 test_start_background_run_writes_job_and_log_with_passed_args 覆盖的行为场景。"""
        module = importlib.import_module("scripts.run_auto_signal_supervised")

        class FakeProcess:
            """类说明：提供 FakeProcess 测试替身实现。"""
            pid = 12345

        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "outputs"
            calls = []

            def fake_popen(command, **kwargs):
                """函数说明：处理 fake_popen 主要逻辑。"""
                calls.append((command, kwargs))
                return FakeProcess()

            with patch.object(module, "load_config", return_value={"outputs": {"dir": str(out_dir)}}), patch.object(
                module.subprocess,
                "Popen",
                side_effect=fake_popen,
            ):
                job = module.start_background_run(["--skip-update", "--skip-convert"], python_executable="python-test")

            self.assertEqual(job["pid"], 12345)
            self.assertEqual(job["run_args"], ["--skip-update", "--skip-convert"])
            self.assertEqual(calls[0][0][0], "python-test")
            self.assertEqual(calls[0][0][-2:], ["--skip-update", "--skip-convert"])
            self.assertEqual(calls[0][1]["cwd"], module.ROOT)
            self.assertTrue((out_dir / "auto_signal_job.json").exists())
            self.assertTrue(Path(job["log_file"]).exists())
            saved = json.loads((out_dir / "auto_signal_job.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["pid"], 12345)
            self.assertEqual(saved["status_file"], str(out_dir / "auto_run_status.json"))

    def test_background_status_reads_job_and_latest_stage(self) -> None:
        """函数说明：验证 test_background_status_reads_job_and_latest_stage 覆盖的行为场景。"""
        module = importlib.import_module("scripts.run_auto_signal_supervised")

        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "outputs"
            out_dir.mkdir()
            (out_dir / "auto_signal_job.json").write_text(
                json.dumps({"pid": 12345, "log_file": str(out_dir / "job.log")}),
                encoding="utf-8",
            )
            (out_dir / "auto_run_status.json").write_text(
                json.dumps({"status": "running", "stages": [{"name": "optimize_params", "state": "running"}]}),
                encoding="utf-8",
            )

            with patch.object(module, "load_config", return_value={"outputs": {"dir": str(out_dir)}}), patch.object(
                module,
                "_process_running",
                return_value=True,
            ):
                status = module.background_status()

        self.assertTrue(status["process_running"])
        self.assertEqual(status["latest_stage"], {"name": "optimize_params", "state": "running"})


if __name__ == "__main__":
    unittest.main()
