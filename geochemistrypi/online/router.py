"""HTTP routes for the lightweight Online API."""

import asyncio
import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from geochemistrypi._version import __version__

from .data_mining_service import DataMiningService
from .schemas import (
    CatalogResponse,
    ClassificationResponse,
    ClusteringResponse,
    DataMiningCatalogResponse,
    DataPreprocessingResponse,
    DatasetProfileResponse,
    HealthResponse,
    RegressionResponse,
    RunResponse,
)
from .service import InvalidDatasetError, OnlineService, UploadTooLargeError


def create_router(
    service: OnlineService,
    data_mining_service: DataMiningService,
) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service="geochemistrypi-online",
            version=__version__,
        )

    @router.get(
        "/chemical-modeling/catalog",
        response_model=CatalogResponse,
        tags=["chemical-modeling"],
    )
    async def catalog() -> CatalogResponse:
        tasks = await asyncio.to_thread(service.build_catalog)
        return CatalogResponse(tasks=tasks)

    @router.get(
        "/data-mining/catalog",
        response_model=DataMiningCatalogResponse,
        tags=["data-mining"],
    )
    async def data_mining_catalog() -> DataMiningCatalogResponse:
        return data_mining_service.build_catalog()

    @router.post(
        "/data-mining/profile",
        response_model=DatasetProfileResponse,
        tags=["data-mining"],
    )
    async def profile_dataset(
        dataset: UploadFile = File(...),
    ) -> DatasetProfileResponse:
        content = await dataset.read(data_mining_service.max_upload_bytes + 1)
        try:
            return await asyncio.to_thread(
                data_mining_service.profile_dataset,
                filename=dataset.filename,
                content=content,
            )
        except UploadTooLargeError as exc:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=str(exc),
            ) from exc
        except (InvalidDatasetError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Data profiling failed: {type(exc).__name__}: {exc}",
            ) from exc
        finally:
            await dataset.close()

    @router.post(
        "/data-mining/preprocess",
        response_model=DataPreprocessingResponse,
        tags=["data-mining"],
    )
    async def preprocess_dataset(
        selected_columns: str = Form(...),
        missing_strategy: str = Form(...),
        dataset: UploadFile = File(...),
    ) -> DataPreprocessingResponse:
        content = await dataset.read(data_mining_service.max_upload_bytes + 1)
        try:
            try:
                parsed_columns = json.loads(selected_columns)
            except json.JSONDecodeError as exc:
                raise InvalidDatasetError(
                    "Selected columns must be a valid JSON list"
                ) from exc
            return await asyncio.to_thread(
                data_mining_service.preprocess_dataset,
                filename=dataset.filename,
                content=content,
                selected_columns=parsed_columns,
                missing_strategy=missing_strategy,
            )
        except UploadTooLargeError as exc:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=str(exc),
            ) from exc
        except (InvalidDatasetError, ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Data preprocessing failed: {type(exc).__name__}: {exc}",
            ) from exc
        finally:
            await dataset.close()

    @router.post(
        "/data-mining/regression",
        response_model=RegressionResponse,
        tags=["data-mining"],
    )
    async def run_regression(
        model: str = Form("linear_regression"),
        target_column: str = Form(...),
        feature_columns: str = Form(...),
        test_size: float = Form(0.2),
        dataset: UploadFile = File(...),
    ) -> RegressionResponse:
        content = await dataset.read(data_mining_service.max_upload_bytes + 1)
        try:
            try:
                parsed_features = json.loads(feature_columns)
            except json.JSONDecodeError as exc:
                raise InvalidDatasetError(
                    "Feature columns must be a valid JSON list"
                ) from exc
            return await asyncio.to_thread(
                data_mining_service.run_regression,
                filename=dataset.filename,
                content=content,
                target_column=target_column,
                feature_columns=parsed_features,
                test_size=test_size,
                model_name=model,
            )
        except UploadTooLargeError as exc:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=str(exc),
            ) from exc
        except (InvalidDatasetError, ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Regression failed: {type(exc).__name__}: {exc}",
            ) from exc
        finally:
            await dataset.close()

    @router.post(
        "/data-mining/classification",
        response_model=ClassificationResponse,
        tags=["data-mining"],
    )
    async def run_classification(
        target_column: str = Form(...),
        feature_columns: str = Form(...),
        test_size: float = Form(0.2),
        dataset: UploadFile = File(...),
    ) -> ClassificationResponse:
        content = await dataset.read(data_mining_service.max_upload_bytes + 1)
        try:
            try:
                parsed_features = json.loads(feature_columns)
            except json.JSONDecodeError as exc:
                raise InvalidDatasetError(
                    "Feature columns must be a valid JSON list"
                ) from exc
            return await asyncio.to_thread(
                data_mining_service.run_classification,
                filename=dataset.filename,
                content=content,
                target_column=target_column,
                feature_columns=parsed_features,
                test_size=test_size,
            )
        except UploadTooLargeError as exc:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=str(exc),
            ) from exc
        except (InvalidDatasetError, ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Classification failed: {type(exc).__name__}: {exc}",
            ) from exc
        finally:
            await dataset.close()

    @router.post(
        "/data-mining/clustering",
        response_model=ClusteringResponse,
        tags=["data-mining"],
    )
    async def run_clustering(
        feature_columns: str = Form(...),
        cluster_count: int = Form(3),
        dataset: UploadFile = File(...),
    ) -> ClusteringResponse:
        content = await dataset.read(data_mining_service.max_upload_bytes + 1)
        try:
            try:
                parsed_features = json.loads(feature_columns)
            except json.JSONDecodeError as exc:
                raise InvalidDatasetError(
                    "Feature columns must be a valid JSON list"
                ) from exc
            return await asyncio.to_thread(
                data_mining_service.run_clustering,
                filename=dataset.filename,
                content=content,
                feature_columns=parsed_features,
                cluster_count=cluster_count,
            )
        except UploadTooLargeError as exc:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=str(exc),
            ) from exc
        except (InvalidDatasetError, ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Clustering failed: {type(exc).__name__}: {exc}",
            ) from exc
        finally:
            await dataset.close()

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

    @router.get(
        "/data-mining/jobs/{job_id}/files/{file_path:path}",
        tags=["data-mining"],
    )
    async def download_data_mining_artifact(
        job_id: str,
        file_path: str,
    ) -> FileResponse:
        try:
            artifact = data_mining_service.resolve_artifact(job_id, file_path)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Data Mining result file not found",
            ) from exc
        return FileResponse(artifact, filename=artifact.name)

    return router
