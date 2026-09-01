"""FastAPI application entry point for Geochemistry Pi Online."""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from geochemistrypi._version import __version__

from .data_mining_service import DataMiningService
from .router import create_router
from .service import OnlineService
from .task_runner import TaskRunner


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DEVELOPMENT_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


def _configured_origins() -> list[str]:
    value = os.getenv("GEOCHEMISTRYPI_ALLOWED_ORIGINS")
    if value is None:
        return list(LOCAL_DEVELOPMENT_ORIGINS)
    return [origin.strip().rstrip("/") for origin in value.split(",") if origin.strip()]


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def create_app(runtime_dir: Path | None = None) -> FastAPI:
    api_docs_enabled = _env_flag("GEOCHEMISTRYPI_ENABLE_API_DOCS", True)
    app = FastAPI(
        title="Geochemistry Pi Online",
        version=__version__,
        description="Minimal Online API for Geochemistry Pi chemical modeling and data mining.",
        docs_url="/docs" if api_docs_enabled else None,
        redoc_url="/redoc" if api_docs_enabled else None,
        openapi_url="/openapi.json" if api_docs_enabled else None,
    )
    allowed_origins = _configured_origins()
    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    configured_runtime_dir = os.getenv("GEOCHEMISTRYPI_RUNTIME_DIR", "").strip()
    resolved_runtime_dir = runtime_dir or (
        Path(configured_runtime_dir).expanduser().resolve()
        if configured_runtime_dir
        else PROJECT_ROOT / "runtime"
    )
    service = OnlineService(resolved_runtime_dir)
    data_mining_service = DataMiningService(resolved_runtime_dir)
    task_runner = TaskRunner(resolved_runtime_dir / "online-task.lock")
    app.state.online_service = service
    app.state.data_mining_service = data_mining_service
    app.state.task_runner = task_runner
    app.include_router(create_router(service, data_mining_service, task_runner))

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "message": "Geochemistry Pi Online API",
            "docs": "/docs",
        }

    return app


app = create_app()
