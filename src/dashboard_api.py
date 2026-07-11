"""FastAPI app for the local signal review dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.config_loader import PROJECT_ROOT
from src.dashboard import build_dashboard_precheck, build_dashboard_snapshot, resolve_dashboard_artifact
from src.dashboard_account import apply_account_update, build_account_workspace, preview_account_update
from src.dashboard_control import (
    DashboardJobConflictError,
    DashboardJobNotFoundError,
    DashboardJobStartError,
    DashboardJobStopError,
    list_dashboard_jobs,
    list_dashboard_workflows,
    start_dashboard_job,
    stop_dashboard_job,
)
from src.dashboard_execution import apply_execution_feedback, build_execution_workspace, preview_execution_feedback
from src.dashboard_stock import build_stock_detail


def create_dashboard_app() -> FastAPI:
    """Create the local dashboard API app."""
    app = FastAPI(title="quant_box dashboard", version="1")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/dashboard/latest")
    def latest_dashboard() -> dict:
        return build_dashboard_snapshot()

    @app.get("/api/dashboard/stocks/{instrument}")
    def dashboard_stock(instrument: str) -> dict[str, Any]:
        try:
            return build_stock_detail(instrument)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/dashboard/precheck")
    def dashboard_precheck() -> dict:
        return build_dashboard_precheck()

    @app.get("/api/dashboard/jobs")
    def dashboard_jobs() -> dict[str, Any]:
        jobs = list_dashboard_jobs()
        active = next((job for job in jobs if job.get("status") in {"running", "stopping"}), None)
        return {"jobs": jobs, "active_job": active}

    @app.get("/api/dashboard/workflows")
    def dashboard_workflows() -> dict[str, Any]:
        return {"workflows": list_dashboard_workflows()}

    @app.get("/api/dashboard/account")
    def dashboard_account() -> dict[str, Any]:
        return build_account_workspace()

    @app.post("/api/dashboard/account/preview")
    def dashboard_account_preview(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            return preview_account_update(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/dashboard/account/apply")
    def dashboard_account_apply(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            return apply_account_update(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/dashboard/execution")
    def dashboard_execution() -> dict[str, Any]:
        return build_execution_workspace()

    @app.post("/api/dashboard/execution/preview")
    def dashboard_execution_preview(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            return preview_execution_feedback(payload)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/dashboard/execution/apply")
    def dashboard_execution_apply(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            return apply_execution_feedback(payload)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/dashboard/jobs")
    def start_dashboard_job_api(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
        request_payload = dict(payload or {})
        action = str(request_payload.get("action") or "")
        try:
            job = start_dashboard_job(action, request_payload)
        except DashboardJobConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except DashboardJobStartError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"job": job}

    @app.post("/api/dashboard/jobs/{job_id}/stop")
    def stop_dashboard_job_api(job_id: str) -> dict[str, Any]:
        try:
            job = stop_dashboard_job(job_id)
        except DashboardJobNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except DashboardJobStopError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"job": job}

    @app.get("/api/dashboard/artifacts/{artifact_id}")
    def dashboard_artifact(artifact_id: str) -> FileResponse:
        path = resolve_dashboard_artifact(artifact_id)
        if path is None:
            raise HTTPException(status_code=404, detail="Artifact not found or not downloadable.")
        return FileResponse(path, media_type=_media_type(path), filename=path.name)

    static_dir = PROJECT_ROOT / "web" / "dist"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="dashboard")
    else:

        @app.get("/")
        def root() -> dict[str, str]:
            return {
                "name": "quant_box dashboard",
                "message": "Start the Vite dev server or build web/dist to view the dashboard UI.",
            }

    return app


def _media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "application/json"
    if suffix == ".csv":
        return "text/csv"
    if suffix in {".md", ".markdown"}:
        return "text/markdown"
    return "application/octet-stream"


app = create_dashboard_app()
