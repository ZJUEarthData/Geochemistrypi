"""FastAPI application entry point for the first Online version."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .router import create_router
from .service import OnlineService


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_app(runtime_dir: Path | None = None) -> FastAPI:
    app = FastAPI(
        title="Geochemistry Pi Online",
        version="0.1.0",
        description="Minimal Online API for Geochemistry Pi chemical modeling.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    service = OnlineService(runtime_dir or PROJECT_ROOT / "runtime")
    app.state.online_service = service
    app.include_router(create_router(service))

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "message": "Geochemistry Pi Online API",
            "docs": "/docs",
        }

    return app


app = create_app()
