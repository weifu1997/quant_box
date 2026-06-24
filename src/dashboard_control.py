"""Controlled dashboard actions for local repair and signal reruns."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any

from src.config_loader import PROJECT_ROOT, load_config, resolve_path


JOB_VERSION = 1
JOB_LIMIT = 8
LOG_TAIL_LINES = 120
STOP_TIMEOUT_SECONDS = 5
ACTIVE_JOB_STATUSES = {"running", "stopping"}
AUTO_SIGNAL_STEPS = [
    ("update_data", "更新行情数据"),
    ("convert_data", "转换价格面板"),
    ("compute_factors", "计算因子"),
    ("data_health", "检查数据健康"),
    ("adj_factor_meta", "构建复权元数据"),
    ("data_governance", "检查点时治理"),
    ("annual_state_router", "年度状态路由"),
    ("optimize_params", "优化参数"),
    ("backtest", "运行回测"),
    ("research_diagnostics", "生成研究诊断"),
    ("generate_signal", "生成信号与订单"),
]
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PROCESS_LOCK = threading.Lock()
_RUNNING_PROCESSES: dict[str, subprocess.Popen[bytes]] = {}


class DashboardJobConflictError(RuntimeError):
    """Raised when another dashboard job is already running."""


class DashboardJobStartError(RuntimeError):
    """Raised when a dashboard job cannot be started."""


class DashboardJobNotFoundError(RuntimeError):
    """Raised when a dashboard job cannot be found."""


class DashboardJobStopError(RuntimeError):
    """Raised when a dashboard job cannot be stopped."""


def start_dashboard_job(action: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Start a whitelisted local dashboard action in the background."""
    payload = dict(payload or {})
    command, label = build_dashboard_job_command(action, payload)
    _ensure_no_running_job()
    out_dir = _output_dir()
    job_dir = _job_dir(out_dir)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    job_id = _new_job_id(action)
    log_path = log_dir / f"dashboard_job_{job_id}.log"
    job = {
        "version": JOB_VERSION,
        "id": job_id,
        "action": action,
        "mode": payload.get("mode"),
        "label": label,
        "status": "running",
        "message": "任务已启动。",
        "command": command,
        "started_at": _now_text(),
        "completed_at": None,
        "return_code": None,
        "log_path": str(log_path.resolve()),
    }
    _write_job(job_dir, job)

    log_handle = log_path.open("ab", buffering=0)
    creationflags = 0
    popen_kwargs: dict[str, Any] = {}
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
            **popen_kwargs,
        )
    except OSError as exc:
        log_handle.write(f"Failed to start dashboard job: {exc}\n".encode("utf-8", errors="replace"))
        log_handle.close()
        job["status"] = "failed"
        job["message"] = f"任务启动失败：{exc}"
        job["completed_at"] = _now_text()
        _write_job(job_dir, job)
        raise DashboardJobStartError(f"Dashboard job failed to start: {exc}") from exc
    job["pid"] = process.pid
    _write_job(job_dir, job)
    with _PROCESS_LOCK:
        _RUNNING_PROCESSES[job_id] = process
    thread = threading.Thread(target=_wait_for_job, args=(job_dir, job_id, process, log_handle), daemon=True)
    thread.start()
    return _decorate_job(job)


def stop_dashboard_job(job_id: str | None = None) -> dict[str, Any]:
    """Stop a running dashboard job and its child process tree."""
    out_dir = _output_dir()
    job_dir = _job_dir(out_dir)
    job = _job_for_stop(job_dir, job_id)
    status = str(job.get("status") or "")
    if status not in ACTIVE_JOB_STATUSES:
        raise DashboardJobStopError(f"Dashboard job is not running: {job.get('label') or job.get('id')}")

    job_id = str(job.get("id") or "")
    if status != "stopping":
        job["status"] = "stopping"
        job["message"] = "正在停止任务，请稍候。"
        _write_job(job_dir, job)

    with _PROCESS_LOCK:
        process = _RUNNING_PROCESSES.get(job_id)
    pid = _int_value(job.get("pid"))
    if pid is None and process is not None:
        pid = process.pid
    if pid is None:
        return _mark_job_stale(job_dir, job, "任务没有可停止的进程 ID；请查看日志确认结果。")
    if process is None and _pid_is_running(pid) and not _pid_matches_job(pid, _list_value(job.get("command"))):
        return _mark_job_stale(job_dir, job, "任务进程 ID 已被其他进程占用，未执行停止操作。")

    stopped, return_code, message = _terminate_job_process(process, pid)
    if not stopped:
        job["message"] = message or "已发送停止请求，但进程仍在运行。"
        _write_job(job_dir, job)
        if _pid_is_running(pid):
            return _decorate_job(job)
        stopped = True

    with _PROCESS_LOCK:
        _RUNNING_PROCESSES.pop(job_id, None)
    latest = _read_job(job_dir / f"{job_id}.json") or job
    if message:
        latest["message"] = message
    return _decorate_job(_finalize_job(job_dir, latest, return_code))


def list_dashboard_jobs(limit: int = JOB_LIMIT) -> list[dict[str, Any]]:
    """List recent dashboard jobs with log tails."""
    job_dir = _job_dir(_output_dir())
    jobs: list[dict[str, Any]] = []
    for path in job_dir.glob("*.json"):
        job = _read_job(path)
        if not job:
            continue
        jobs.append(_decorate_job(_refresh_running_job(job_dir, job)))
    jobs.sort(key=lambda item: str(item.get("started_at") or ""), reverse=True)
    return jobs[:limit]


def build_dashboard_job_command(
    action: str,
    payload: Mapping[str, Any] | None = None,
    *,
    out_dir: str | Path | None = None,
    python_executable: str | None = None,
) -> tuple[list[str], str]:
    """Build a safe command for a whitelisted dashboard action."""
    payload = dict(payload or {})
    python = python_executable or sys.executable
    if action == "repair_point_in_time":
        start_date, end_date = _daily_basic_repair_window(out_dir=out_dir)
        return (
            [
                python,
                str(PROJECT_ROOT / "scripts" / "run_update_point_in_time_data.py"),
                "--start-date",
                start_date,
                "--end-date",
                end_date,
                "--skip-index-constituents",
                "--skip-st-calendar",
            ],
            f"补齐 daily_basic 点时数据 {start_date} 至 {end_date}",
        )
    if action == "run_auto_signal":
        mode = str(payload.get("mode") or "candidate")
        if mode not in {"candidate", "normal"}:
            raise ValueError("mode must be candidate or normal.")
        command = [python, str(PROJECT_ROOT / "scripts" / "run_auto_signal.py"), "--no-archive"]
        if mode == "candidate":
            command.append("--candidate-only")
            label = "重跑自动信号（候选输出）"
        else:
            label = "重跑自动信号（正常门槛输出）"
        return command, label
    raise ValueError(f"Unsupported dashboard action: {action}")


def _wait_for_job(job_dir: Path, job_id: str, process: subprocess.Popen[bytes], log_handle: Any) -> None:
    return_code = process.wait()
    log_handle.close()
    with _PROCESS_LOCK:
        _RUNNING_PROCESSES.pop(job_id, None)
    job = _read_job(job_dir / f"{job_id}.json")
    _finalize_job(job_dir, job, return_code)


def _ensure_no_running_job() -> None:
    running_jobs = [job for job in list_dashboard_jobs(limit=JOB_LIMIT) if job.get("status") in ACTIVE_JOB_STATUSES]
    if running_jobs:
        raise DashboardJobConflictError(f"Dashboard job already running: {running_jobs[0].get('label')}")


def _refresh_running_job(job_dir: Path, job: Mapping[str, Any]) -> dict[str, Any]:
    refreshed = dict(job)
    status = str(refreshed.get("status") or "")
    if status not in ACTIVE_JOB_STATUSES:
        return refreshed
    job_id = str(refreshed.get("id") or "")
    with _PROCESS_LOCK:
        process = _RUNNING_PROCESSES.get(job_id)
    if process is not None:
        return_code = process.poll()
        if return_code is None:
            return refreshed
        with _PROCESS_LOCK:
            _RUNNING_PROCESSES.pop(job_id, None)
        return _finalize_job(job_dir, refreshed, return_code)
    pid = _int_value(refreshed.get("pid"))
    if pid is not None and _pid_is_running(pid) and _pid_matches_job(pid, _list_value(refreshed.get("command"))):
        return refreshed
    if status == "stopping":
        return _finalize_job(job_dir, refreshed, -15)
    return _mark_job_stale(job_dir, refreshed, "仪表盘服务曾重启或断开，无法确认任务结果；请查看日志并刷新最新报告。")


def _job_for_stop(job_dir: Path, job_id: str | None) -> dict[str, Any]:
    if job_id:
        job = _refresh_running_job(job_dir, _read_job(job_dir / f"{job_id}.json"))
        if not job:
            raise DashboardJobNotFoundError(f"Dashboard job not found: {job_id}")
        return job
    active = next((job for job in list_dashboard_jobs(limit=JOB_LIMIT) if job.get("status") in ACTIVE_JOB_STATUSES), None)
    if active is None:
        raise DashboardJobNotFoundError("No running dashboard job.")
    return active


def _mark_job_stale(job_dir: Path, job: Mapping[str, Any], message: str) -> dict[str, Any]:
    refreshed = dict(job)
    refreshed["status"] = "stale"
    refreshed["message"] = message
    refreshed["completed_at"] = refreshed.get("completed_at") or _now_text()
    _write_job(job_dir, refreshed)
    return refreshed


def _daily_basic_repair_window(out_dir: str | Path | None = None) -> tuple[str, str]:
    output_dir = _output_dir(out_dir)
    governance = _read_json(output_dir / "data_governance_report.json")
    auto_report = _read_json(output_dir / "auto_signal_report.json")
    for source in [governance, _mapping_value(auto_report.get("data_governance"))]:
        for action in _list_value(source.get("repair_actions")):
            if not isinstance(action, Mapping) or action.get("component") != "daily_basic":
                continue
            start_date = _valid_date(action.get("start_date"))
            end_date = _valid_date(action.get("end_date"))
            if start_date and end_date:
                return start_date, end_date

    start = _valid_date(governance.get("daily_basic_start_date")) or _config_start_date()
    target_resolution = _mapping_value(auto_report.get("target_date_resolution"))
    end = (
        _valid_date(target_resolution.get("target_date"))
        or _valid_date(governance.get("factor_cache_meta_end_date"))
        or _valid_date(governance.get("daily_basic_end_date"))
        or "auto"
    )
    return start, end


def _output_dir(out_dir: str | Path | None = None) -> Path:
    if out_dir is not None:
        return resolve_path(out_dir)
    config = load_config()
    return resolve_path(_mapping_value(config.get("outputs")).get("dir", "outputs"))


def _job_dir(out_dir: Path) -> Path:
    path = out_dir / "dashboard_jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_job(job_dir: Path, job: Mapping[str, Any]) -> None:
    path = job_dir / f"{job['id']}.json"
    path.write_text(json.dumps(dict(job), indent=2, ensure_ascii=False), encoding="utf-8")


def _read_job(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(data) if isinstance(data, Mapping) else {}


def _decorate_job(job: Mapping[str, Any]) -> dict[str, Any]:
    decorated = dict(job)
    log_path = Path(str(decorated.get("log_path") or ""))
    decorated["log_tail"] = _read_log_tail(log_path)
    decorated["progress"] = _job_progress(decorated, decorated["log_tail"])
    return decorated


def _job_progress(job: Mapping[str, Any], log_tail: list[str]) -> dict[str, Any]:
    action = str(job.get("action") or "")
    if action == "run_auto_signal":
        return _auto_signal_progress(job)
    if action == "repair_point_in_time":
        return _point_in_time_progress(job, log_tail)
    return _generic_progress(job)


def _auto_signal_progress(job: Mapping[str, Any]) -> dict[str, Any]:
    out_dir = _job_output_dir(job)
    status = _read_json(out_dir / "auto_run_status.json")
    stage_rows = _list_value(status.get("stages"))
    latest_by_name: dict[str, dict[str, Any]] = {}
    if _timestamp_is_not_before(status.get("started_at"), job.get("started_at")):
        for stage in stage_rows:
            if not isinstance(stage, Mapping):
                continue
            name = str(stage.get("name") or "")
            latest_by_name[name] = dict(stage)
    steps = []
    for step_id, label in AUTO_SIGNAL_STEPS:
        stage = latest_by_name.get(step_id, {})
        steps.append(
            {
                "id": step_id,
                "label": label,
                "status": _progress_status(stage.get("state")),
                "message": str(stage.get("message") or ""),
                "updated_at": stage.get("updated_at"),
            }
        )
    return _progress_payload(job, steps)


def _point_in_time_progress(job: Mapping[str, Any], log_tail: list[str]) -> dict[str, Any]:
    log_text = "\n".join(log_tail).lower()
    daily_done = "daily_basic cache written" in log_text
    governance_done = "data governance report written" in log_text
    steps = [
        {
            "id": "daily_basic",
            "label": "补齐 daily_basic",
            "status": "complete" if daily_done else _first_step_status(job),
            "message": "daily_basic 缓存已写入。" if daily_done else "",
            "updated_at": None,
        },
        {
            "id": "data_governance",
            "label": "刷新点时治理报告",
            "status": "complete" if governance_done else ("running" if daily_done and _is_job_running(job) else "pending"),
            "message": "点时治理报告已刷新。" if governance_done else "",
            "updated_at": None,
        },
    ]
    return _progress_payload(job, steps)


def _generic_progress(job: Mapping[str, Any]) -> dict[str, Any]:
    return _progress_payload(
        job,
        [
            {
                "id": "job",
                "label": str(job.get("label") or "后台任务"),
                "status": "running" if _is_job_running(job) else _completed_progress_status(job),
                "message": str(job.get("message") or ""),
                "updated_at": job.get("completed_at") or job.get("started_at"),
            }
        ],
    )


def _progress_payload(job: Mapping[str, Any], steps: list[dict[str, Any]]) -> dict[str, Any]:
    adjusted = _apply_terminal_progress(job, steps)
    current = next((step for step in adjusted if step["status"] == "running"), None)
    if current is None:
        current = next((step for step in reversed(adjusted) if step["status"] in {"complete", "failed", "skipped"}), None)
    complete_count = sum(1 for step in adjusted if step["status"] in {"complete", "skipped"})
    percent = round((complete_count / len(adjusted)) * 100) if adjusted else 0
    if str(job.get("status")) == "succeeded":
        percent = 100
    return {
        "summary": _progress_summary(job, current),
        "percent": percent,
        "active_step": current["id"] if current else None,
        "steps": adjusted,
    }


def _apply_terminal_progress(job: Mapping[str, Any], steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    status = str(job.get("status") or "")
    if status not in {"succeeded", "failed", "cancelled", "stale"}:
        return steps
    updated = [dict(step) for step in steps]
    if status == "succeeded":
        for step in updated:
            if step["status"] in {"pending", "running"}:
                step["status"] = "complete"
        return updated
    terminal = "failed" if status in {"failed", "stale"} else "skipped"
    for step in updated:
        if step["status"] == "running":
            step["status"] = terminal
            return updated
    for step in reversed(updated):
        if step["status"] == "pending":
            step["status"] = terminal
            return updated
    return updated


def _progress_summary(job: Mapping[str, Any], current: Mapping[str, Any] | None) -> str:
    status = str(job.get("status") or "")
    if status == "succeeded":
        return "任务已完成。"
    if status == "failed":
        return "任务失败，请查看日志尾部。"
    if status == "cancelled":
        return "任务已停止。"
    if status == "stale":
        return "任务状态待确认，请查看日志尾部。"
    if current:
        label = str(current.get("label") or "")
        message = str(current.get("message") or "")
        return f"{label}：{message}" if message else label
    return str(job.get("message") or "")


def _progress_status(state: Any) -> str:
    text = str(state or "").lower()
    if text in {"complete", "completed", "succeeded"}:
        return "complete"
    if text in {"running", "in_progress", "planning"}:
        return "running"
    if text in {"skipped", "skip"}:
        return "skipped"
    if text in {"failed", "error", "timeout"}:
        return "failed"
    return "pending"


def _first_step_status(job: Mapping[str, Any]) -> str:
    return "running" if _is_job_running(job) else _completed_progress_status(job)


def _completed_progress_status(job: Mapping[str, Any]) -> str:
    status = str(job.get("status") or "")
    if status == "succeeded":
        return "complete"
    if status in {"failed", "stale"}:
        return "failed"
    if status == "cancelled":
        return "skipped"
    return "pending"


def _is_job_running(job: Mapping[str, Any]) -> bool:
    return str(job.get("status") or "") in ACTIVE_JOB_STATUSES


def _job_output_dir(job: Mapping[str, Any]) -> Path:
    log_path = Path(str(job.get("log_path") or ""))
    if log_path.parent.name == "logs":
        return log_path.parent.parent
    return _output_dir()


def _timestamp_is_not_before(value: Any, reference: Any) -> bool:
    if not value or not reference:
        return False
    try:
        return datetime.fromisoformat(str(value)) >= datetime.fromisoformat(str(reference))
    except ValueError:
        return str(value) >= str(reference)


def _read_log_tail(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-LOG_TAIL_LINES:]


def _new_job_id(action: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"{random.randrange(16**6):06x}"
    safe_action = re.sub(r"[^a-zA-Z0-9_]+", "_", action).strip("_") or "job"
    return f"{stamp}_{safe_action}_{suffix}"


def _now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _finalize_job(job_dir: Path, job: Mapping[str, Any], return_code: int | None) -> dict[str, Any]:
    updated = dict(job)
    if updated.get("status") in {"stopping", "cancelled"}:
        updated["status"] = "cancelled"
        updated["message"] = "任务已停止。"
    else:
        updated["status"] = "succeeded" if return_code == 0 else "failed"
        updated["message"] = "任务完成。" if return_code == 0 else f"任务失败，退出码 {return_code}。"
    updated["completed_at"] = _now_text()
    updated["return_code"] = return_code
    _write_job(job_dir, updated)
    return updated


def _terminate_job_process(process: subprocess.Popen[bytes] | None, pid: int) -> tuple[bool, int | None, str]:
    if os.name == "nt":
        return _taskkill_pid_tree(pid)
    if process is not None and process.poll() is not None:
        return True, process.returncode, "任务已停止。"
    try:
        if process is not None and getattr(process, "pid", None):
            os.killpg(process.pid, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        if not _pid_is_running(pid):
            return True, -15, "任务已停止。"
        return False, None, f"停止任务失败：{exc}"
    if process is not None:
        try:
            return_code = process.wait(timeout=STOP_TIMEOUT_SECONDS)
            return True, return_code, "任务已停止。"
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
                return True, process.wait(timeout=STOP_TIMEOUT_SECONDS), "任务已强制停止。"
            except (OSError, subprocess.SubprocessError) as exc:
                return False, None, f"强制停止任务失败：{exc}"
    time.sleep(0.5)
    if not _pid_is_running(pid):
        return True, -15, "任务已停止。"
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError as exc:
        return False, None, f"强制停止任务失败：{exc}"
    time.sleep(0.5)
    return (not _pid_is_running(pid), -9, "任务已强制停止。")


def _taskkill_pid_tree(pid: int) -> tuple[bool, int | None, str]:
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        result = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=creationflags,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, None, f"停止任务失败：{exc}"
    if result.returncode == 0 or not _pid_is_running(pid):
        return True, -15, "任务已停止。"
    detail = (result.stderr or result.stdout or "").strip()
    return False, None, f"停止任务失败：{detail or f'taskkill 退出码 {result.returncode}'}"


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                timeout=3,
                creationflags=creationflags,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        output = result.stdout.strip()
        return result.returncode == 0 and str(pid) in output and "No tasks" not in output
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _pid_matches_job(pid: int, command: list[Any]) -> bool:
    command_line = _pid_command_line(pid)
    if not command_line:
        return True
    normalized_command_line = _normalize_command_text(command_line)
    script_parts = [str(part) for part in command if str(part).lower().endswith(".py")]
    if script_parts:
        return any(_normalize_command_text(part) in normalized_command_line for part in script_parts)
    return bool(command) and _normalize_command_text(Path(str(command[0])).name) in normalized_command_line


def _pid_command_line(pid: int) -> str:
    if pid <= 0:
        return ""
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        command = (
            f"$p = Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}'; "
            "if ($p) { $p.CommandLine }"
        )
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                capture_output=True,
                text=True,
                timeout=3,
                creationflags=creationflags,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return result.stdout.strip() if result.returncode == 0 else ""
    cmdline = Path("/proc") / str(pid) / "cmdline"
    try:
        return cmdline.read_text(encoding="utf-8", errors="replace").replace("\x00", " ").strip()
    except OSError:
        return ""


def _normalize_command_text(value: str) -> str:
    return value.replace("\\", "/").replace('"', "").lower()


def _int_value(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(data) if isinstance(data, Mapping) else {}


def _config_start_date() -> str:
    config = load_config()
    data_config = _mapping_value(config.get("data"))
    value = data_config.get("history_start_date") or data_config.get("start_date") or "2015-01-01"
    return str(value)


def _valid_date(value: Any) -> str | None:
    text = str(value or "").strip()
    return text if _DATE_RE.match(text) else None


def _mapping_value(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
