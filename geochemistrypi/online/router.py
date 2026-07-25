"""HTTP routes for the lightweight Online API."""

import asyncio

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from .schemas import CatalogResponse, HealthResponse, RunResponse
from .service import InvalidDatasetError, OnlineService, UploadTooLargeError


def create_router(service: OnlineService) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", service="geochemistrypi-online")

    @router.get(
        "/chemical-modeling/catalog",
        response_model=CatalogResponse,
        tags=["chemical-modeling"],
    )
    async def catalog() -> CatalogResponse:
        tasks = await asyncio.to_thread(service.build_catalog)
        return CatalogResponse(tasks=tasks)

    @router.post(
        "/chemical-modeling/run",
        response_model=RunResponse,
        tags=["chemical-modeling"],
    )
    async def run_chemical_model(
        task: str = Form(...),
        method: str = Form(...),
        element: str = Form(...),
        dataset: UploadFile = File(...),
    ) -> RunResponse:
        content = await dataset.read(service.max_upload_bytes + 1)
        try:
            return await asyncio.to_thread(
                service.run_job,
                task=task,
                method=method,
                element=element,
                filename=dataset.filename,
                content=content,
            )
        except UploadTooLargeError as exc:
            raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=str(exc)) from exc
        except (InvalidDatasetError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Calculation failed: {type(exc).__name__}: {exc}",
            ) from exc
        finally:
            await dataset.close()

    @router.get("/jobs/{job_id}/files/{file_path:path}", tags=["jobs"])
    async def download_artifact(job_id: str, file_path: str) -> FileResponse:
        try:
            artifact = service.resolve_artifact(job_id, file_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Result file not found") from exc
        return FileResponse(artifact, filename=artifact.name)

    return router
