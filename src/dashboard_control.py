"""Cross-platform controlled dashboard actions for quant_box workflows."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
from tempfile import NamedTemporaryFile
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
WORKFLOW_CATALOG: tuple[dict[str, Any], ...] = (
    {"action": "check_tushare_config", "label": "检查 Tushare 配置", "category": "data", "description": "只读检查代理地址和 token 是否已配置，不发起网络请求。", "duration": "少于 1 分钟", "script": "check_tushare_config.py", "args": []},
    {"action": "update_market_data", "label": "增量更新行情", "category": "data", "description": "补齐缺失或过期的主板股票日线数据，支持中断后继续。", "duration": "5–60 分钟", "script": "run_update_data.py", "args": [], "parameters": [
        {"name": "end_date", "label": "目标日期", "type": "date", "flag": "--end-date", "default": "", "optional": True, "help": "留空使用配置目标日期。"},
        {"name": "chunk_size", "label": "每批股票数", "type": "integer", "flag": "--chunk-size", "default": 300, "min": 1, "max": 2000},
        {"name": "sleep_seconds", "label": "批次等待秒数", "type": "number", "flag": "--sleep-seconds", "default": 0, "min": 0, "max": 60},
        {"name": "max_chunks", "label": "最多批次", "type": "integer", "flag": "--max-chunks", "default": "", "optional": True, "min": 1, "max": 10000},
        {"name": "include_existing", "label": "同时刷新已有股票", "type": "boolean", "flag": "--include-existing", "default": False},
    ]},
    {"action": "update_point_in_time_all", "label": "更新点时治理数据", "category": "data", "description": "补齐 daily_basic、指数成分和 ST 历史日历并刷新治理报告。", "duration": "5–120 分钟", "script": "run_update_point_in_time_data.py", "args": [], "parameters": [
        {"name": "end_date", "label": "目标日期", "type": "date", "flag": "--end-date", "default": "", "optional": True},
        {"name": "max_dates", "label": "最多 daily_basic 日期", "type": "integer", "flag": "--max-dates", "default": "", "optional": True, "min": 1, "max": 10000},
        {"name": "max_index_windows", "label": "最多指数窗口", "type": "integer", "flag": "--max-index-windows", "default": "", "optional": True, "min": 1, "max": 10000},
        {"name": "sleep_seconds", "label": "请求等待秒数", "type": "number", "flag": "--sleep-seconds", "default": 0, "min": 0, "max": 60},
    ]},
    {"action": "update_fundamentals", "label": "更新财务与分红", "category": "data", "description": "按缺失项补齐财务指标和分红缓存。", "duration": "10–120 分钟", "script": "run_update_fundamentals.py", "args": ["--missing-only"], "parameters": [
        {"name": "end_date", "label": "目标日期", "type": "date", "flag": "--end-date", "default": "", "optional": True},
        {"name": "max_symbols", "label": "最多股票数", "type": "integer", "flag": "--max-symbols", "default": "", "optional": True, "min": 1, "max": 10000},
        {"name": "sleep_seconds", "label": "请求等待秒数", "type": "number", "flag": "--sleep-seconds", "default": 0, "min": 0, "max": 60},
    ]},
    {"action": "build_historical_universe", "label": "构建历史股票池", "category": "data", "description": "增量更新沪深300、中证500和中证1000成分；单窗口失败会保留警告，再构建点时股票池。", "duration": "5–60 分钟", "script": "run_build_universe.py", "args": ["--skip-index-errors"]},
    {"action": "convert_data", "label": "转换价格数据", "category": "pipeline", "description": "把原始 CSV 转换为 Qlib provider 和本地价格面板。", "duration": "2–15 分钟", "script": "run_convert_data.py", "args": []},
    {"action": "calculate_factors", "label": "计算 Alpha158 因子", "category": "pipeline", "description": "计算或复用 Alpha158 因子缓存。", "duration": "5–30 分钟", "script": "run_calc_factors.py", "args": [], "parameters": [
        {"name": "end_date", "label": "结束日期", "type": "date", "flag": "--end-date", "default": "", "optional": True},
        {"name": "force", "label": "强制重新计算", "type": "boolean", "flag": "--force", "default": False, "help": "忽略有效缓存，耗时显著增加。"},
    ]},
    {"action": "factor_diagnostics", "label": "运行因子诊断", "category": "research", "description": "生成 IC、年度稳定性和因子分组收益报告。", "duration": "2–15 分钟", "script": "run_factor_diagnostics.py", "args": []},
    {"action": "optimize_parameters", "label": "优化策略参数", "category": "research", "description": "运行默认有界 walk-forward 参数搜索。", "duration": "5–60 分钟", "script": "run_optimize.py", "args": [], "parameters": [
        {"name": "start_date", "label": "开始日期", "type": "date", "flag": "--start-date", "default": "", "optional": True},
        {"name": "end_date", "label": "结束日期", "type": "date", "flag": "--end-date", "default": "", "optional": True},
        {"name": "full_grid", "label": "完整参数网格", "type": "boolean", "flag": "--full-grid", "default": False, "help": "完整网格可能比快速基线慢数倍。"},
        {"name": "train_years", "label": "训练年数", "type": "integer", "flag": "--train-years", "default": 3, "min": 1, "max": 15},
        {"name": "test_months", "label": "测试月数", "type": "integer", "flag": "--test-months", "default": 12, "min": 1, "max": 60},
        {"name": "step_months", "label": "滚动步长月数", "type": "integer", "flag": "--step-months", "default": 12, "min": 1, "max": 60},
    ]},
    {"action": "run_backtest", "label": "运行真实化回测", "category": "research", "description": "使用当前配置运行包含成本、容量和涨跌停约束的回测。", "duration": "2–20 分钟", "script": "run_backtest.py", "args": [], "parameters": [
        {"name": "start_date", "label": "开始日期", "type": "date", "flag": "--start-date", "default": "", "optional": True},
        {"name": "end_date", "label": "结束日期", "type": "date", "flag": "--end-date", "default": "", "optional": True},
    ]},
    {"action": "quant_diagnostics", "label": "生成量化诊断", "category": "research", "description": "生成五层回测诊断和一致性检查报告。", "duration": "1–10 分钟", "script": "run_quant_diagnostics.py", "args": []},
    {"action": "optimization_review", "label": "生成优化复核", "category": "research", "description": "复核策略风格、风险和交易约束。", "duration": "1–10 分钟", "script": "run_optimization_review.py", "args": []},
    {"action": "evidence_optimizer", "label": "生成证据优化计划", "category": "research", "description": "依据诊断产物生成可追踪的优化建议。", "duration": "1–10 分钟", "script": "run_evidence_optimizer.py", "args": []},
    {"action": "fundamental_screen", "label": "生成基本面筛选", "category": "research", "description": "输出质量、分红、负债和估值筛选报告。", "duration": "1–10 分钟", "script": "run_fundamental_screen.py", "args": ["--date", "latest"]},
    {"action": "generate_candidate_signal", "label": "生成候选信号", "category": "signal", "description": "基于当前缓存生成最新候选信号，不覆盖正式持仓。", "duration": "1–10 分钟", "script": "run_daily_signal.py", "args": ["--date", "latest"]},
    {"action": "risk_refine", "label": "风险参数精炼", "category": "advanced", "description": "在有限时间内搜索流动性、止损、仓位与熔断组合。", "duration": "15–120 分钟", "script": "run_risk_refine.py", "args": [], "parameters": [
        {"name": "start_date", "label": "开始日期", "type": "date", "flag": "--start-date", "default": "", "optional": True},
        {"name": "end_date", "label": "结束日期", "type": "date", "flag": "--end-date", "default": "", "optional": True},
        {"name": "max_seconds", "label": "最长运行秒数", "type": "number", "flag": "--max-seconds", "default": 900, "min": 60, "max": 14400},
        {"name": "resume", "label": "继续已有结果", "type": "boolean", "flag": "--resume", "default": True},
    ]},
    {"action": "regime_blend_probe", "label": "市场状态混合探针", "category": "advanced", "description": "快速比较市场状态、流动性和防御权重组合。", "duration": "5–45 分钟", "script": "run_regime_blend_probe.py", "args": [], "parameters": [
        {"name": "start_date", "label": "开始日期", "type": "date", "flag": "--start-date", "default": "", "optional": True},
        {"name": "end_date", "label": "结束日期", "type": "date", "flag": "--end-date", "default": "", "optional": True},
        {"name": "max_symbols", "label": "最多股票数", "type": "integer", "flag": "--max-symbols", "default": 700, "min": 10, "max": 5000},
    ]},
    {"action": "rebalance_drift_probe", "label": "调仓漂移探针", "category": "advanced", "description": "快速比较不同调仓漂移阈值的效果。", "duration": "5–30 分钟", "script": "run_rebalance_drift_probe.py", "args": [], "parameters": [
        {"name": "start_date", "label": "开始日期", "type": "date", "flag": "--start-date", "default": "2024-01-01"},
        {"name": "end_date", "label": "结束日期", "type": "date", "flag": "--end-date", "default": "", "optional": True},
        {"name": "thresholds", "label": "漂移阈值", "type": "text", "flag": "--thresholds", "default": "0.0,0.02,0.05", "pattern": r"^\d+(?:\.\d+)?(?:,\d+(?:\.\d+)?)*$", "help": "英文逗号分隔的非负小数。"},
        {"name": "max_symbols", "label": "最多股票数", "type": "integer", "flag": "--max-symbols", "default": 500, "min": 10, "max": 5000},
    ]},
    {"action": "annual_router_grid", "label": "年度状态路由网格", "category": "advanced", "description": "以可恢复缓存运行年度状态路由组合搜索。", "duration": "30–240 分钟", "script": "run_annual_state_router_grid.py", "args": [], "parameters": [
        {"name": "end_date", "label": "结束日期", "type": "date", "flag": "--end-date", "default": "", "optional": True},
        {"name": "max_combinations", "label": "最多组合数", "type": "integer", "flag": "--max-combinations", "default": 20, "min": 1, "max": 1000},
        {"name": "force_rebuild_cache", "label": "强制重建分数缓存", "type": "boolean", "flag": "--force-rebuild-cache", "default": False},
        {"name": "include_exposure_diagnostics", "label": "包含暴露诊断", "type": "boolean", "flag": "--include-exposure-diagnostics", "default": False},
    ]},
)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PROCESS_LOCK = threading.Lock()
_START_LOCK = threading.Lock()
_RUNNING_PROCESSES: dict[str, subprocess.Popen[bytes]] = {}


class DashboardJobConflictError(RuntimeError):
    """Raised when another dashboard job is already running."""


class DashboardJobStartError(RuntimeError):
    """Raised when a dashboard job cannot be started."""


class DashboardJobNotFoundError(RuntimeError):
    """Raised when a dashboard job cannot be found."""


class DashboardJobStopError(RuntimeError):
    """Raised when a dashboard job cannot be stopped."""


def list_dashboard_workflows() -> list[dict[str, Any]]:
    """Return the fixed workflow catalog without exposing shell arguments."""
    workflows: list[dict[str, Any]] = []
    for workflow in WORKFLOW_CATALOG:
        public = {key: value for key, value in workflow.items() if key not in {"script", "args"}}
        public["parameters"] = [
            {key: value for key, value in parameter.items() if key not in {"flag", "pattern"}}
            for parameter in workflow.get("parameters", [])
        ]
        workflows.append(public)
    return workflows


def start_dashboard_job(action: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Start a whitelisted local dashboard action in the background."""
    payload = dict(payload or {})
    command, label = build_dashboard_job_command(action, payload)
    # Keep the persisted single-job check and process reservation together. A
    # request can otherwise pass the check just before another request writes
    # its running job record.
    with _START_LOCK:
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
            "parameters": payload.get("parameters"),
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
    workflow = next((item for item in WORKFLOW_CATALOG if item["action"] == action), None)
    if workflow is not None:
        command = [python, str(PROJECT_ROOT / "scripts" / str(workflow["script"]))]
        command.extend(str(value) for value in workflow.get("args", []))
        command.extend(_workflow_parameter_args(workflow, payload.get("parameters")))
        return command, str(workflow["label"])
    raise ValueError(f"Unsupported dashboard action: {action}")


def _workflow_parameter_args(workflow: Mapping[str, Any], raw_parameters: Any) -> list[str]:
    schemas = [dict(item) for item in workflow.get("parameters", []) if isinstance(item, Mapping)]
    if raw_parameters is None:
        parameters: dict[str, Any] = {}
    elif isinstance(raw_parameters, Mapping):
        parameters = dict(raw_parameters)
    else:
        raise ValueError("parameters must be an object")
    allowed = {str(schema.get("name")) for schema in schemas}
    unknown = sorted(str(key) for key in parameters if str(key) not in allowed)
    if unknown:
        raise ValueError("Unsupported workflow parameters: " + ",".join(unknown))

    args: list[str] = []
    for schema in schemas:
        name = str(schema.get("name") or "")
        flag = str(schema.get("flag") or "")
        value = parameters.get(name, schema.get("default"))
        optional = bool(schema.get("optional"))
        if optional and (value is None or value == ""):
            continue
        parameter_type = str(schema.get("type") or "text")
        if parameter_type == "boolean":
            if not isinstance(value, bool):
                raise ValueError(f"Workflow parameter {name} must be a boolean")
            if value:
                args.append(flag)
            continue
        normalized = _normalize_workflow_parameter(name, value, schema)
        args.extend([flag, normalized])
    return args


def _normalize_workflow_parameter(name: str, value: Any, schema: Mapping[str, Any]) -> str:
    parameter_type = str(schema.get("type") or "text")
    if isinstance(value, bool):
        raise ValueError(f"Workflow parameter {name} has an invalid type")
    if parameter_type == "integer":
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Workflow parameter {name} must be an integer") from exc
        if str(value).strip() not in {str(number), f"{number}.0"}:
            raise ValueError(f"Workflow parameter {name} must be an integer")
        _validate_workflow_range(name, float(number), schema)
        return str(number)
    if parameter_type == "number":
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Workflow parameter {name} must be a number") from exc
        if not math.isfinite(number):
            raise ValueError(f"Workflow parameter {name} must be finite")
        _validate_workflow_range(name, number, schema)
        return str(number)
    text = str(value).strip()
    if parameter_type == "date":
        if not _DATE_RE.match(text):
            raise ValueError(f"Workflow parameter {name} must use YYYY-MM-DD")
        return text
    pattern = schema.get("pattern")
    if pattern and not re.fullmatch(str(pattern), text):
        raise ValueError(f"Workflow parameter {name} has an invalid format")
    if len(text) > 200:
        raise ValueError(f"Workflow parameter {name} is too long")
    return text


def _validate_workflow_range(name: str, value: float, schema: Mapping[str, Any]) -> None:
    minimum = schema.get("min")
    maximum = schema.get("max")
    if minimum is not None and value < float(minimum):
        raise ValueError(f"Workflow parameter {name} must be >= {minimum}")
    if maximum is not None and value > float(maximum):
        raise ValueError(f"Workflow parameter {name} must be <= {maximum}")


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
    content = json.dumps(dict(job), indent=2, ensure_ascii=False)
    job_dir.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            suffix=path.suffix,
            prefix=f".{path.stem}-",
            dir=path.parent,
            delete=False,
            encoding="utf-8",
        ) as handle:
            handle.write(content)
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


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
    if action in {"repair_point_in_time", "update_point_in_time_all"}:
        return _point_in_time_progress(job, log_tail)
    if action == "build_historical_universe":
        return _historical_universe_progress(job, log_tail)
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
    data_fraction = _merge_data_update_progress(out_dir, job, steps)
    payload = _progress_payload(job, steps)
    if data_fraction is not None and payload.get("active_step") == "update_data":
        completed = sum(1 for step in payload["steps"] if step["status"] in {"complete", "skipped"})
        payload["percent"] = round(((completed + data_fraction) / len(payload["steps"])) * 100)
    return payload


def _merge_data_update_progress(out_dir: Path, job: Mapping[str, Any], steps: list[dict[str, Any]]) -> float | None:
    progress = _read_json(out_dir / "data_update_progress.json")
    if not _timestamp_is_not_before(progress.get("updated_at"), job.get("started_at")):
        return None
    target = _int_value(progress.get("target_symbols")) or _int_value(progress.get("pending_symbols"))
    completed = _int_value(progress.get("fresh_or_confirmed_symbols"))
    if completed is None:
        completed = _int_value(progress.get("completed_symbols"))
    if not target or completed is None:
        return None
    fraction = min(max(completed / target, 0.0), 1.0)
    step = next((item for item in steps if item.get("id") == "update_data"), None)
    if step is not None and step.get("status") == "running":
        step["message"] = f"已确认 {completed}/{target} 只股票（{fraction:.0%}）"
    return fraction


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


def _historical_universe_progress(job: Mapping[str, Any], log_tail: list[str]) -> dict[str, Any]:
    log_text = "\n".join(log_tail).lower()
    log_path = Path(str(job.get("log_path") or ""))
    if log_path.exists():
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            pass
    codes = ["000300.sh", "000905.sh", "000852.sh"]
    labels = ["更新沪深300成分", "更新中证500成分", "更新中证1000成分"]
    steps: list[dict[str, Any]] = []
    previous_complete = True
    for code, label in zip(codes, labels):
        complete = f"index_weight cache updated for {code}" in log_text
        running = previous_complete and _is_job_running(job) and not complete
        steps.append(
            {
                "id": code,
                "label": label,
                "status": "complete" if complete else ("running" if running else "pending"),
                "message": "指数成分缓存已更新。" if complete else "正在增量补齐指数窗口。" if running else "",
                "updated_at": None,
            }
        )
        previous_complete = previous_complete and complete
    universe_done = "historical universe snapshots written" in log_text
    steps.append(
        {
            "id": "historical_universe",
            "label": "生成历史股票池",
            "status": "complete" if universe_done else ("running" if previous_complete and _is_job_running(job) else "pending"),
            "message": "历史股票池已写入。" if universe_done else "",
            "updated_at": None,
        }
    )
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
    persisted = _read_job(job_dir / f"{job['id']}.json")
    updated = persisted or dict(job)
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
        # An unreadable command line is not evidence that the PID belongs to
        # this job. Treat it as unverified so stop/recovery cannot target an
        # unrelated process after PID reuse or a permission failure.
        return False
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
