from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config, resolve_path


JOB_FILE_NAME = "auto_signal_job.json"
STATUS_FILE_NAME = "auto_run_status.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Start and inspect long-running auto signal jobs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="Start scripts/run_auto_signal.py in the background.")
    start.add_argument("--python", dest="python_executable", help="Python executable to use. Defaults to .venv when available.")
    start.add_argument("--log-file", help="Optional log file path. Defaults to outputs/logs/auto_signal_*.log.")
    start.add_argument("run_args", nargs=argparse.REMAINDER, help="Arguments passed through after an optional -- separator.")

    status = subparsers.add_parser("status", help="Show the latest background job and auto_run_status.json.")
    status.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    tail = subparsers.add_parser("tail", help="Print the last lines from the latest job log.")
    tail.add_argument("-n", "--lines", type=int, default=80)

    args = parser.parse_args()
    if args.command == "start":
        job = start_background_run(
            _normalize_run_args(args.run_args),
            python_executable=args.python_executable,
            log_file=args.log_file,
        )
        print(f"Started auto signal job pid={job['pid']}")
        print(f"Log: {job['log_file']}")
        print(f"Status: {job['status_file']}")
        return
    if args.command == "status":
        info = background_status()
        if args.json:
            print(json.dumps(info, indent=2, ensure_ascii=False))
        else:
            print(_format_status(info))
        return
    if args.command == "tail":
        print(tail_latest_log(args.lines), end="")
        return


def start_background_run(
    run_args: list[str],
    python_executable: str | Path | None = None,
    log_file: str | Path | None = None,
) -> dict[str, Any]:
    config = load_config()
    out_dir = resolve_path(config.get("outputs", {}).get("dir", "outputs"))
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = resolve_path(log_file) if log_file else log_dir / f"auto_signal_{timestamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / STATUS_FILE_NAME
    job_path = out_dir / JOB_FILE_NAME

    command = build_auto_signal_command(run_args, python_executable=python_executable)
    header = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "cwd": str(ROOT),
        "command": command,
        "log_file": str(log_path),
        "status_file": str(status_path),
    }
    log_path.write_text(
        "# auto signal background job\n"
        + json.dumps(header, ensure_ascii=False)
        + "\n\n",
        encoding="utf-8",
    )
    log_handle = log_path.open("a", encoding="utf-8", buffering=1)
    try:
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=_background_creationflags(),
        )
    finally:
        log_handle.close()

    job = {
        **header,
        "pid": proc.pid,
        "run_args": run_args,
        "job_file": str(job_path),
    }
    job_path.write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")
    return job


def build_auto_signal_command(run_args: list[str], python_executable: str | Path | None = None) -> list[str]:
    python_path = str(python_executable or _default_python())
    return [python_path, str(ROOT / "scripts" / "run_auto_signal.py"), *run_args]


def background_status() -> dict[str, Any]:
    config = load_config()
    out_dir = resolve_path(config.get("outputs", {}).get("dir", "outputs"))
    job = _read_json(out_dir / JOB_FILE_NAME)
    status = _read_json(out_dir / STATUS_FILE_NAME)
    try:
        pid = int(job.get("pid", 0) or 0) if isinstance(job, dict) else 0
    except (TypeError, ValueError):
        pid = 0
    latest_stage = _latest_stage(status)
    return {
        "job": job,
        "process_running": _process_running(pid) if pid else False,
        "auto_run_status": status,
        "latest_stage": latest_stage,
    }


def tail_latest_log(lines: int = 80) -> str:
    info = background_status()
    job = info.get("job") or {}
    log_file = job.get("log_file") if isinstance(job, dict) else None
    if not log_file:
        return "No background job log found.\n"
    path = Path(str(log_file))
    if not path.exists():
        return f"Log file not found: {path}\n"
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    selected = content[-max(1, int(lines)) :]
    return "\n".join(selected) + ("\n" if selected else "")


def _normalize_run_args(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


def _default_python() -> Path:
    if os.name == "nt":
        venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        venv_python = ROOT / ".venv" / "bin" / "python"
    return venv_python if venv_python.exists() else Path(sys.executable)


def _background_creationflags() -> int:
    if os.name != "nt":
        return 0
    return int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)) | int(getattr(subprocess, "DETACHED_PROCESS", 0))


def _process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        rows = csv.reader(result.stdout.splitlines())
        return any(len(row) >= 2 and row[1] == str(pid) for row in rows)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _latest_stage(status: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(status, dict):
        return None
    stages = status.get("stages")
    if not isinstance(stages, list) or not stages:
        return None
    latest = stages[-1]
    return latest if isinstance(latest, dict) else None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _format_status(info: dict[str, Any]) -> str:
    job = info.get("job") or {}
    status = info.get("auto_run_status") or {}
    stage = info.get("latest_stage") or {}
    has_job = isinstance(job, dict) and bool(job)
    lines = [
        f"Supervisor job: {'found' if has_job else 'not found'}",
        f"Job running: {bool(info.get('process_running')) if has_job else ''}",
        f"PID: {job.get('pid', '') if isinstance(job, dict) else ''}",
        f"Log: {job.get('log_file', '') if isinstance(job, dict) else ''}",
        f"Run status: {status.get('status', '') if isinstance(status, dict) else ''}",
        f"Latest stage: {stage.get('name', '')}:{stage.get('state', '')} {stage.get('message', '')}".rstrip(),
    ]
    if isinstance(job, dict) and job.get("command"):
        command = job["command"]
        if isinstance(command, list):
            lines.append("Command: " + " ".join(shlex.quote(str(value)) for value in command))
    if isinstance(status, dict) and status.get("block_reasons"):
        lines.append("Block reasons:")
        lines.extend(f"- {reason}" for reason in status["block_reasons"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
