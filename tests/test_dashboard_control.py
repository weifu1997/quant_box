"""Tests for controlled dashboard run actions."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.dashboard_api import create_dashboard_app
from src.dashboard_control import (
    DashboardJobConflictError,
    DashboardJobNotFoundError,
    DashboardJobStartError,
    DashboardJobStopError,
    build_dashboard_job_command,
    list_dashboard_jobs,
    stop_dashboard_job,
)


class DashboardControlTests(unittest.TestCase):
    def test_repair_command_uses_daily_basic_repair_action(self) -> None:
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            (out_dir / "data_governance_report.json").write_text(
                json.dumps({"repair_actions": []}),
                encoding="utf-8",
            )
            (out_dir / "auto_signal_report.json").write_text(
                json.dumps(
                    {
                        "data_governance": {
                            "repair_actions": [
                                {
                                    "component": "daily_basic",
                                    "start_date": "2015-01-05",
                                    "end_date": "2026-06-22",
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )

            command, label = build_dashboard_job_command(
                "repair_point_in_time",
                out_dir=out_dir,
                python_executable="python-test",
            )

            self.assertEqual(command[0], "python-test")
            self.assertEqual(command[1], str(Path("scripts/run_update_point_in_time_data.py").resolve()))
            self.assertEqual(command[command.index("--start-date") + 1], "2015-01-05")
            self.assertEqual(command[command.index("--end-date") + 1], "2026-06-22")
            self.assertIn("--skip-index-constituents", command)
            self.assertIn("--skip-st-calendar", command)
            self.assertIn("daily_basic", label)

    def test_run_auto_signal_candidate_mode_keeps_candidate_only_flag(self) -> None:
        command, label = build_dashboard_job_command(
            "run_auto_signal",
            {"mode": "candidate"},
            python_executable="python-test",
        )

        self.assertEqual(command[0], "python-test")
        self.assertIn("--no-archive", command)
        self.assertIn("--candidate-only", command)
        self.assertIn("候选输出", label)

    def test_run_auto_signal_normal_mode_omits_candidate_only_flag(self) -> None:
        command, label = build_dashboard_job_command(
            "run_auto_signal",
            {"mode": "normal"},
            python_executable="python-test",
        )

        self.assertIn("--no-archive", command)
        self.assertNotIn("--candidate-only", command)
        self.assertIn("正常门槛输出", label)

    def test_run_auto_signal_rejects_invalid_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "mode must be candidate or normal"):
            build_dashboard_job_command("run_auto_signal", {"mode": "force"})

    def test_unknown_dashboard_action_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported dashboard action"):
            build_dashboard_job_command("delete_everything")

    def test_list_dashboard_jobs_skips_malformed_job_json(self) -> None:
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            job_dir = out_dir / "dashboard_jobs"
            job_dir.mkdir()
            (job_dir / "bad.json").write_text("{not-json", encoding="utf-8")
            with patch("src.dashboard_control._output_dir", return_value=out_dir):
                self.assertEqual(list_dashboard_jobs(), [])

    def test_dashboard_jobs_api_marks_running_job_active(self) -> None:
        job = {
            "id": "job-1",
            "label": "重跑自动信号（候选输出）",
            "status": "running",
            "message": "任务已启动。",
            "command": ["python", "scripts/run_auto_signal.py", "--candidate-only"],
            "log_tail": [],
        }
        with patch("src.dashboard_api.list_dashboard_jobs", return_value=[job]):
            response = TestClient(create_dashboard_app()).get("/api/dashboard/jobs")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["jobs"], [job])
        self.assertEqual(body["active_job"], job)

    def test_dashboard_jobs_api_marks_stopping_job_active(self) -> None:
        job = {
            "id": "job-1",
            "label": "重跑自动信号（正常门槛输出）",
            "status": "stopping",
            "message": "正在停止任务，请稍候。",
            "command": ["python", "scripts/run_auto_signal.py"],
            "log_tail": [],
        }
        with patch("src.dashboard_api.list_dashboard_jobs", return_value=[job]):
            response = TestClient(create_dashboard_app()).get("/api/dashboard/jobs")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["active_job"], job)

    def test_stop_dashboard_job_cancels_running_pid(self) -> None:
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            job_dir = out_dir / "dashboard_jobs"
            job_dir.mkdir()
            (job_dir / "job-1.json").write_text(
                json.dumps(
                    {
                        "id": "job-1",
                        "label": "重跑自动信号（正常门槛输出）",
                        "status": "running",
                        "message": "任务已启动。",
                        "command": ["python", "scripts/run_auto_signal.py"],
                        "started_at": "2026-06-24T01:07:39",
                        "completed_at": None,
                        "return_code": None,
                        "log_path": str(out_dir / "job.log"),
                        "pid": 123,
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch("src.dashboard_control._output_dir", return_value=out_dir),
                patch("src.dashboard_control._pid_is_running", return_value=True),
                patch("src.dashboard_control._pid_matches_job", return_value=True),
                patch("src.dashboard_control._terminate_job_process", return_value=(True, -15, "任务已停止。")),
            ):
                job = stop_dashboard_job("job-1")

            self.assertEqual(job["status"], "cancelled")
            self.assertEqual(job["message"], "任务已停止。")
            self.assertEqual(job["return_code"], -15)

    def test_stop_dashboard_job_rejects_completed_job(self) -> None:
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            job_dir = out_dir / "dashboard_jobs"
            job_dir.mkdir()
            (job_dir / "job-1.json").write_text(
                json.dumps(
                    {
                        "id": "job-1",
                        "label": "已完成任务",
                        "status": "succeeded",
                        "message": "任务完成。",
                        "command": ["python", "scripts/run_auto_signal.py"],
                        "started_at": "2026-06-24T01:07:39",
                        "completed_at": "2026-06-24T01:08:39",
                        "return_code": 0,
                        "log_path": str(out_dir / "job.log"),
                    }
                ),
                encoding="utf-8",
            )

            with patch("src.dashboard_control._output_dir", return_value=out_dir):
                with self.assertRaisesRegex(DashboardJobStopError, "not running"):
                    stop_dashboard_job("job-1")

    def test_stop_dashboard_job_reports_missing_job(self) -> None:
        with TemporaryDirectory() as tmp:
            with patch("src.dashboard_control._output_dir", return_value=Path(tmp)):
                with self.assertRaisesRegex(DashboardJobNotFoundError, "not found"):
                    stop_dashboard_job("missing")

    def test_dashboard_jobs_api_stops_job(self) -> None:
        job = {
            "id": "job-1",
            "label": "重跑自动信号（正常门槛输出）",
            "status": "cancelled",
            "message": "任务已停止。",
            "command": ["python", "scripts/run_auto_signal.py"],
            "log_tail": [],
        }
        with patch("src.dashboard_api.stop_dashboard_job", return_value=job):
            response = TestClient(create_dashboard_app()).post("/api/dashboard/jobs/job-1/stop")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["job"], job)

    def test_dashboard_jobs_api_reports_missing_stop_job(self) -> None:
        with patch("src.dashboard_api.stop_dashboard_job", side_effect=DashboardJobNotFoundError("Dashboard job not found")):
            response = TestClient(create_dashboard_app()).post("/api/dashboard/jobs/missing/stop")

        self.assertEqual(response.status_code, 404)

    def test_dashboard_jobs_api_rejects_stop_for_finished_job(self) -> None:
        with patch("src.dashboard_api.stop_dashboard_job", side_effect=DashboardJobStopError("Dashboard job is not running")):
            response = TestClient(create_dashboard_app()).post("/api/dashboard/jobs/job-1/stop")

        self.assertEqual(response.status_code, 409)

    def test_dashboard_jobs_api_rejects_invalid_action(self) -> None:
        with patch("src.dashboard_api.start_dashboard_job", side_effect=ValueError("Unsupported dashboard action")):
            response = TestClient(create_dashboard_app()).post("/api/dashboard/jobs", json={"action": "nope"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Unsupported dashboard action")

    def test_dashboard_jobs_api_rejects_concurrent_job(self) -> None:
        with patch("src.dashboard_api.start_dashboard_job", side_effect=DashboardJobConflictError("Dashboard job already running")):
            response = TestClient(create_dashboard_app()).post(
                "/api/dashboard/jobs",
                json={"action": "run_auto_signal", "mode": "candidate"},
            )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"], "Dashboard job already running")

    def test_dashboard_jobs_api_reports_start_failure(self) -> None:
        with patch("src.dashboard_api.start_dashboard_job", side_effect=DashboardJobStartError("Dashboard job failed to start")):
            response = TestClient(create_dashboard_app()).post(
                "/api/dashboard/jobs",
                json={"action": "repair_point_in_time"},
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "Dashboard job failed to start")


if __name__ == "__main__":
    unittest.main()
