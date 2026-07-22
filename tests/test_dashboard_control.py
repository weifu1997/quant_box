"""Tests for controlled dashboard run actions."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.dashboard_api import create_dashboard_app
from src.dashboard_control import (
    DashboardJobConflictError,
    DashboardJobNotFoundError,
    DashboardJobStartError,
    DashboardJobStopError,
    _finalize_job,
    _pid_matches_job,
    _write_job,
    build_dashboard_job_command,
    list_dashboard_jobs,
    list_dashboard_workflows,
    start_dashboard_job,
    stop_dashboard_job,
)


class DashboardControlTests(unittest.TestCase):
    def test_start_dashboard_job_serializes_the_single_job_reservation(self) -> None:
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            start_barrier = threading.Barrier(2)
            release_process = threading.Event()

            class FakeProcess:
                pid = 101
                returncode = 0

                def wait(self) -> int:
                    release_process.wait(timeout=2)
                    return 0

                def poll(self) -> int | None:
                    return 0 if release_process.is_set() else None

            def start_one() -> dict:
                start_barrier.wait()
                return start_dashboard_job("check_tushare_config")

            with (
                patch("src.dashboard_control._output_dir", return_value=out_dir),
                patch("src.dashboard_control.subprocess.Popen", return_value=FakeProcess()) as popen,
                patch.dict("src.dashboard_control._RUNNING_PROCESSES", {}, clear=True),
            ):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [executor.submit(start_one) for _index in range(2)]
                    jobs = []
                    conflicts = []
                    for future in futures:
                        try:
                            jobs.append(future.result(timeout=2))
                        except DashboardJobConflictError as exc:
                            conflicts.append(exc)
                self.assertEqual(len(jobs), 1)
                self.assertEqual(len(conflicts), 1)
                popen.assert_called_once()
                release_process.set()
                status_path = next((out_dir / "dashboard_jobs").glob("*.json"))
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    status = json.loads(status_path.read_text(encoding="utf-8"))
                    if status.get("status") == "succeeded":
                        break
                    time.sleep(0.01)
                self.assertEqual(status.get("status"), "succeeded")

    def test_finalize_job_uses_latest_persisted_stop_state(self) -> None:
        with TemporaryDirectory() as tmp:
            job_dir = Path(tmp)
            stale = {"id": "job-1", "status": "running", "message": "任务已启动。"}
            _write_job(job_dir, {**stale, "status": "stopping", "message": "正在停止任务。"})

            finalized = _finalize_job(job_dir, stale, -15)

            self.assertEqual(finalized["status"], "cancelled")
            self.assertEqual(finalized["message"], "任务已停止。")
            persisted = json.loads((job_dir / "job-1.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["status"], "cancelled")
            self.assertEqual(list(job_dir.glob(".job-1-*.json")), [])

    def test_write_job_replaces_status_file_atomically(self) -> None:
        with TemporaryDirectory() as tmp:
            job_dir = Path(tmp)
            with patch("src.dashboard_control.os.replace", wraps=os.replace) as replace:
                _write_job(job_dir, {"id": "job-1", "status": "running"})

            replace.assert_called_once()
            self.assertEqual(json.loads((job_dir / "job-1.json").read_text(encoding="utf-8"))["status"], "running")

    def test_pid_match_requires_command_line_evidence(self) -> None:
        with patch("src.dashboard_control._pid_command_line", return_value=""):
            self.assertFalse(_pid_matches_job(123, ["python", "scripts/run_auto_signal.py"]))

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

    def test_historical_universe_web_action_tolerates_partial_index_windows(self) -> None:
        command, label = build_dashboard_job_command(
            "build_historical_universe",
            python_executable="python-test",
        )

        self.assertEqual(command[1], str(Path("scripts/run_build_universe.py").resolve()))
        self.assertIn("--skip-index-errors", command)
        self.assertNotIn("--skip-fetch", command)
        self.assertIn("历史股票池", label)

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

    def test_workflow_catalog_does_not_expose_command_arguments(self) -> None:
        workflows = list_dashboard_workflows()

        self.assertTrue(any(item["action"] == "update_market_data" for item in workflows))
        self.assertTrue(any(item["action"] == "run_backtest" for item in workflows))
        self.assertTrue(all("script" not in item and "args" not in item for item in workflows))
        self.assertTrue(all("flag" not in parameter for item in workflows for parameter in item.get("parameters", [])))

    def test_core_workflow_commands_use_current_python_and_repo_scripts(self) -> None:
        cases = {
            "update_market_data": "run_update_data.py",
            "convert_data": "run_convert_data.py",
            "calculate_factors": "run_calc_factors.py",
            "optimize_parameters": "run_optimize.py",
            "run_backtest": "run_backtest.py",
            "generate_candidate_signal": "run_daily_signal.py",
        }
        for action, script in cases.items():
            with self.subTest(action=action):
                command, label = build_dashboard_job_command(action, python_executable="python-test")
                self.assertEqual(command[0], "python-test")
                self.assertEqual(Path(command[1]).name, script)
                self.assertTrue(label)

    def test_generate_candidate_signal_never_adds_official_flag(self) -> None:
        command, _label = build_dashboard_job_command("generate_candidate_signal", python_executable="python-test")

        self.assertNotIn("--official", command)
        self.assertIn("--date", command)

    def test_workflow_parameters_build_typed_command_arguments(self) -> None:
        command, _label = build_dashboard_job_command(
            "update_market_data",
            {
                "parameters": {
                    "end_date": "2026-07-10",
                    "chunk_size": 500,
                    "sleep_seconds": 0.5,
                    "max_chunks": 2,
                    "include_existing": True,
                }
            },
            python_executable="python-test",
        )

        self.assertEqual(command[command.index("--end-date") + 1], "2026-07-10")
        self.assertEqual(command[command.index("--chunk-size") + 1], "500")
        self.assertEqual(command[command.index("--sleep-seconds") + 1], "0.5")
        self.assertEqual(command[command.index("--max-chunks") + 1], "2")
        self.assertIn("--include-existing", command)

    def test_workflow_parameters_use_safe_defaults(self) -> None:
        command, _label = build_dashboard_job_command("annual_router_grid", python_executable="python-test")

        self.assertEqual(command[command.index("--max-combinations") + 1], "20")
        self.assertNotIn("--force-rebuild-cache", command)

    def test_workflow_parameters_reject_unknown_or_out_of_range_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported workflow parameters: output"):
            build_dashboard_job_command("run_backtest", {"parameters": {"output": "outside.csv"}})
        with self.assertRaisesRegex(ValueError, "chunk_size must be <= 2000"):
            build_dashboard_job_command("update_market_data", {"parameters": {"chunk_size": 999999}})
        with self.assertRaisesRegex(ValueError, "end_date must use YYYY-MM-DD"):
            build_dashboard_job_command("run_backtest", {"parameters": {"end_date": "latest;rm -rf /"}})
        with self.assertRaisesRegex(ValueError, "thresholds has an invalid format"):
            build_dashboard_job_command("rebalance_drift_probe", {"parameters": {"thresholds": "0.1;whoami"}})

    def test_workflow_parameters_reject_wrong_types_and_non_finite_numbers(self) -> None:
        with self.assertRaisesRegex(ValueError, "full_grid must be a boolean"):
            build_dashboard_job_command("optimize_parameters", {"parameters": {"full_grid": "true"}})
        with self.assertRaisesRegex(ValueError, "max_seconds must be finite"):
            build_dashboard_job_command("risk_refine", {"parameters": {"max_seconds": "inf"}})

    def test_list_dashboard_jobs_skips_malformed_job_json(self) -> None:
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            job_dir = out_dir / "dashboard_jobs"
            job_dir.mkdir()
            (job_dir / "bad.json").write_text("{not-json", encoding="utf-8")
            with patch("src.dashboard_control._output_dir", return_value=out_dir):
                self.assertEqual(list_dashboard_jobs(), [])

    def test_list_dashboard_jobs_adds_auto_signal_progress_from_status(self) -> None:
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            job_dir = out_dir / "dashboard_jobs"
            log_dir = out_dir / "logs"
            job_dir.mkdir()
            log_dir.mkdir()
            (out_dir / "auto_run_status.json").write_text(
                json.dumps(
                    {
                        "status": "running",
                        "started_at": "2026-06-24T01:07:40",
                        "stages": [
                            {"name": "update_data", "state": "complete", "updated_at": "2026-06-24T01:08:00"},
                            {
                                "name": "compute_factors",
                                "state": "running",
                                "updated_at": "2026-06-24T01:08:30",
                                "message": "loading factor cache",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (job_dir / "job-1.json").write_text(
                json.dumps(
                    {
                        "id": "job-1",
                        "action": "run_auto_signal",
                        "label": "重跑自动信号（正常门槛输出）",
                        "status": "running",
                        "message": "任务已启动。",
                        "command": ["python", "scripts/run_auto_signal.py"],
                        "started_at": "2026-06-24T01:07:39",
                        "completed_at": None,
                        "return_code": None,
                        "log_path": str(log_dir / "job.log"),
                        "pid": 123,
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch("src.dashboard_control._output_dir", return_value=out_dir),
                patch("src.dashboard_control._pid_is_running", return_value=True),
                patch("src.dashboard_control._pid_matches_job", return_value=True),
            ):
                jobs = list_dashboard_jobs()

            self.assertEqual(jobs[0]["progress"]["active_step"], "compute_factors")
            self.assertIn("计算因子", jobs[0]["progress"]["summary"])
            factor_step = next(step for step in jobs[0]["progress"]["steps"] if step["id"] == "compute_factors")
            self.assertEqual(factor_step["status"], "running")

    def test_auto_signal_progress_includes_current_data_update_fraction(self) -> None:
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            job_dir = out_dir / "dashboard_jobs"
            log_dir = out_dir / "logs"
            job_dir.mkdir()
            log_dir.mkdir()
            (out_dir / "auto_run_status.json").write_text(
                json.dumps({"started_at": "2026-07-11T02:10:28", "stages": [{"name": "update_data", "state": "running", "updated_at": "2026-07-11T02:10:30"}]}),
                encoding="utf-8",
            )
            (out_dir / "data_update_progress.json").write_text(
                json.dumps({"updated_at": "2026-07-11T02:12:00", "target_symbols": 1000, "fresh_or_confirmed_symbols": 500}),
                encoding="utf-8",
            )
            (job_dir / "job-1.json").write_text(
                json.dumps({"id": "job-1", "action": "run_auto_signal", "status": "running", "message": "任务已启动。", "command": ["python", "scripts/run_auto_signal.py"], "started_at": "2026-07-11T02:10:28", "log_path": str(log_dir / "job.log"), "pid": 123}),
                encoding="utf-8",
            )
            with patch("src.dashboard_control._output_dir", return_value=out_dir), patch("src.dashboard_control._pid_is_running", return_value=True), patch("src.dashboard_control._pid_matches_job", return_value=True):
                job = list_dashboard_jobs()[0]

            update_step = next(step for step in job["progress"]["steps"] if step["id"] == "update_data")
            self.assertIn("500/1000", update_step["message"])
            self.assertEqual(job["progress"]["percent"], 5)

    def test_auto_signal_progress_ignores_stale_data_update_file(self) -> None:
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            job_dir = out_dir / "dashboard_jobs"
            log_dir = out_dir / "logs"
            job_dir.mkdir()
            log_dir.mkdir()
            (out_dir / "auto_run_status.json").write_text(json.dumps({"started_at": "2026-07-11T02:10:28", "stages": [{"name": "update_data", "state": "running"}]}), encoding="utf-8")
            (out_dir / "data_update_progress.json").write_text(json.dumps({"updated_at": "2026-07-10T02:00:00", "target_symbols": 1000, "fresh_or_confirmed_symbols": 900}), encoding="utf-8")
            (job_dir / "job-1.json").write_text(json.dumps({"id": "job-1", "action": "run_auto_signal", "status": "running", "message": "任务已启动。", "command": ["python", "scripts/run_auto_signal.py"], "started_at": "2026-07-11T02:10:28", "log_path": str(log_dir / "job.log"), "pid": 123}), encoding="utf-8")
            with patch("src.dashboard_control._output_dir", return_value=out_dir), patch("src.dashboard_control._pid_is_running", return_value=True), patch("src.dashboard_control._pid_matches_job", return_value=True):
                job = list_dashboard_jobs()[0]

            self.assertEqual(job["progress"]["percent"], 0)

    def test_list_dashboard_jobs_adds_point_in_time_progress_from_log_tail(self) -> None:
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            job_dir = out_dir / "dashboard_jobs"
            log_dir = out_dir / "logs"
            job_dir.mkdir()
            log_dir.mkdir()
            log_path = log_dir / "job.log"
            log_path.write_text(
                "INFO:__main__:daily_basic cache written to data/factors/daily_basic.parquet\n",
                encoding="utf-8",
            )
            (job_dir / "job-1.json").write_text(
                json.dumps(
                    {
                        "id": "job-1",
                        "action": "repair_point_in_time",
                        "label": "补齐 daily_basic 点时数据",
                        "status": "running",
                        "message": "任务已启动。",
                        "command": ["python", "scripts/run_update_point_in_time_data.py"],
                        "started_at": "2026-06-24T01:07:39",
                        "completed_at": None,
                        "return_code": None,
                        "log_path": str(log_path),
                        "pid": 123,
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch("src.dashboard_control._output_dir", return_value=out_dir),
                patch("src.dashboard_control._pid_is_running", return_value=True),
                patch("src.dashboard_control._pid_matches_job", return_value=True),
            ):
                jobs = list_dashboard_jobs()

            self.assertEqual(jobs[0]["progress"]["active_step"], "data_governance")
            daily_step = next(step for step in jobs[0]["progress"]["steps"] if step["id"] == "daily_basic")
            self.assertEqual(daily_step["status"], "complete")

            payload = json.loads((job_dir / "job-1.json").read_text(encoding="utf-8"))
            payload["action"] = "update_point_in_time_all"
            (job_dir / "job-1.json").write_text(json.dumps(payload), encoding="utf-8")
            with (
                patch("src.dashboard_control._output_dir", return_value=out_dir),
                patch("src.dashboard_control._pid_is_running", return_value=True),
                patch("src.dashboard_control._pid_matches_job", return_value=True),
            ):
                full_jobs = list_dashboard_jobs()
            full_daily_step = next(
                step for step in full_jobs[0]["progress"]["steps"] if step["id"] == "daily_basic"
            )
            self.assertEqual(full_daily_step["status"], "complete")

    def test_list_dashboard_jobs_tracks_historical_universe_substeps(self) -> None:
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            job_dir = out_dir / "dashboard_jobs"
            log_dir = out_dir / "logs"
            job_dir.mkdir()
            log_dir.mkdir()
            log_path = log_dir / "job.log"
            log_path.write_text(
                "INFO:index_weight cache updated for 000300.SH: cache.csv\n"
                "INFO:index_weight cache updated for 000905.SH: cache.csv\n",
                encoding="utf-8",
            )
            (job_dir / "job-1.json").write_text(
                json.dumps(
                    {
                        "id": "job-1",
                        "action": "build_historical_universe",
                        "label": "构建历史股票池",
                        "status": "running",
                        "message": "任务已启动。",
                        "command": ["python", "scripts/run_build_universe.py", "--skip-index-errors"],
                        "started_at": "2026-07-11T04:46:06",
                        "log_path": str(log_path),
                        "pid": 123,
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch("src.dashboard_control._output_dir", return_value=out_dir),
                patch("src.dashboard_control._pid_is_running", return_value=True),
                patch("src.dashboard_control._pid_matches_job", return_value=True),
            ):
                job = list_dashboard_jobs()[0]

            statuses = {step["id"]: step["status"] for step in job["progress"]["steps"]}
            self.assertEqual(statuses["000300.sh"], "complete")
            self.assertEqual(statuses["000905.sh"], "complete")
            self.assertEqual(statuses["000852.sh"], "running")
            self.assertEqual(statuses["historical_universe"], "pending")

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

    def test_dashboard_precheck_api_returns_precheck_payload(self) -> None:
        precheck = {
            "version": 1,
            "generated_at": "2026-06-24T09:00:00",
            "status": "pass",
            "summary": "运行前检查通过，可以重跑自动信号。",
            "can_run_normal": True,
            "target_date_resolution": {"target_date": "2026-06-23"},
            "items": [],
        }
        with patch("src.dashboard_api.build_dashboard_precheck", return_value=precheck):
            response = TestClient(create_dashboard_app()).get("/api/dashboard/precheck")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), precheck)

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

    def test_stop_dashboard_job_does_not_terminate_an_unverified_pid(self) -> None:
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
                patch("src.dashboard_control._pid_command_line", return_value=""),
                patch("src.dashboard_control._terminate_job_process") as terminate,
            ):
                with self.assertRaisesRegex(DashboardJobStopError, "not running"):
                    stop_dashboard_job("job-1")

            persisted = json.loads((job_dir / "job-1.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["status"], "stale")
            terminate.assert_not_called()

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
