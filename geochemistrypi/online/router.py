"""HTTP routes for the lightweight Online API."""

import asyncio
import json

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from geochemistrypi._version import __version__

from .data_mining_service import DataMiningService
from .identity import BUILD_ID, INSTANCE_ID, SOURCE_REVISION
from .limits import MAX_CONCURRENT_TASKS, MAX_UPLOAD_BYTES, TASK_TIMEOUT_SECONDS
from .schemas import (
    AnomalyDetectionResponse,
    CatalogResponse,
    ClassificationResponse,
    ClusteringResponse,
    DataMiningCatalogResponse,
    DataPreprocessingResponse,
    DatasetProfileResponse,
    DimensionalityReductionResponse,
    HealthResponse,
    ModelInferenceResponse,
    RegressionResponse,
    RunResponse,
    TaskStatusResponse,
    TimeSeriesResponse,
)
from .service import InvalidDatasetError, OnlineService, UploadTooLargeError
from .task_runner import TaskCancelledError, TaskRunner, TaskTimeoutError


def create_router(
    service: OnlineService,
    data_mining_service: DataMiningService,
    task_runner: TaskRunner,
) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service="geochemistrypi-online",
            version=__version__,
            instance_id=INSTANCE_ID,
            source_revision=SOURCE_REVISION,
            build_id=BUILD_ID,
            max_upload_bytes=MAX_UPLOAD_BYTES,
            task_timeout_seconds=TASK_TIMEOUT_SECONDS,
            max_concurrent_tasks=MAX_CONCURRENT_TASKS,
        )

    async def run_calculation(
        operation,
        tracking_id: str | None,
        task_label: str,
        **arguments,
    ):
        try:
            return await task_runner.run(
                operation,
                arguments=arguments,
                tracking_id=tracking_id,
                task_label=task_label,
            )
        except TaskCancelledError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        except TaskTimeoutError as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=str(exc),
            ) from exc

    def enforce_upload_limit(content: bytes, max_upload_bytes: int) -> None:
        if len(content) > max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"The uploaded file exceeds {max_upload_bytes} bytes",
            )

    @router.get(
        "/tasks/{task_id}",
        response_model=TaskStatusResponse,
        tags=["tasks"],
    )
    async def get_task_status(task_id: str) -> TaskStatusResponse:
        task = task_runner.get_status(task_id)
        if task is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
        return TaskStatusResponse(**task)

    @router.post(
        "/tasks/{task_id}/cancel",
        response_model=TaskStatusResponse,
        tags=["tasks"],
    )
    async def cancel_task(task_id: str) -> TaskStatusResponse:
        task = await asyncio.to_thread(task_runner.cancel, task_id)
        if task is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
        return TaskStatusResponse(**task)

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
        x_task_id: str | None = Header(None, alias="X-Task-ID"),
    ) -> DatasetProfileResponse:
        content = await dataset.read(data_mining_service.max_upload_bytes + 1)
        try:
            enforce_upload_limit(content, data_mining_service.max_upload_bytes)
            data_mining_service.validate_upload(dataset.filename, content)
            return await run_calculation(
                data_mining_service.profile_dataset,
                tracking_id=x_task_id,
                task_label="Dataset profile",
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
        except HTTPException:
            raise
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
        x_task_id: str | None = Header(None, alias="X-Task-ID"),
    ) -> DataPreprocessingResponse:
        content = await dataset.read(data_mining_service.max_upload_bytes + 1)
        try:
            enforce_upload_limit(content, data_mining_service.max_upload_bytes)
            data_mining_service.validate_upload(dataset.filename, content)
            try:
                parsed_columns = json.loads(selected_columns)
            except json.JSONDecodeError as exc:
                raise InvalidDatasetError(
                    "Selected columns must be a valid JSON list"
                ) from exc
            return await run_calculation(
                data_mining_service.preprocess_dataset,
                tracking_id=x_task_id,
                task_label="Data preprocessing",
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
        except HTTPException:
            raise
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
        x_task_id: str | None = Header(None, alias="X-Task-ID"),
    ) -> RegressionResponse:
        content = await dataset.read(data_mining_service.max_upload_bytes + 1)
        try:
            enforce_upload_limit(content, data_mining_service.max_upload_bytes)
            data_mining_service.validate_upload(dataset.filename, content)
            try:
                parsed_features = json.loads(feature_columns)
            except json.JSONDecodeError as exc:
                raise InvalidDatasetError(
                    "Feature columns must be a valid JSON list"
                ) from exc
            return await run_calculation(
                data_mining_service.run_regression,
                tracking_id=x_task_id,
                task_label=f"Regression: {model}",
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
        except HTTPException:
            raise
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
        model: str = Form("logistic_regression"),
        target_column: str = Form(...),
        feature_columns: str = Form(...),
        test_size: float = Form(0.2),
        dataset: UploadFile = File(...),
        x_task_id: str | None = Header(None, alias="X-Task-ID"),
    ) -> ClassificationResponse:
        content = await dataset.read(data_mining_service.max_upload_bytes + 1)
        try:
            enforce_upload_limit(content, data_mining_service.max_upload_bytes)
            data_mining_service.validate_upload(dataset.filename, content)
            try:
                parsed_features = json.loads(feature_columns)
            except json.JSONDecodeError as exc:
                raise InvalidDatasetError(
                    "Feature columns must be a valid JSON list"
                ) from exc
            return await run_calculation(
                data_mining_service.run_classification,
                tracking_id=x_task_id,
                task_label=f"Classification: {model}",
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
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Classification failed: {type(exc).__name__}: {exc}",
            ) from exc
        finally:
            await dataset.close()

    @router.post(
        "/data-mining/inference",
        response_model=ModelInferenceResponse,
        tags=["data-mining"],
    )
    async def run_model_inference(
        training_job_id: str = Form(...),
        dataset: UploadFile = File(...),
        x_task_id: str | None = Header(None, alias="X-Task-ID"),
    ) -> ModelInferenceResponse:
        content = await dataset.read(data_mining_service.max_upload_bytes + 1)
        try:
            enforce_upload_limit(content, data_mining_service.max_upload_bytes)
            data_mining_service.validate_upload(dataset.filename, content)
            return await run_calculation(
                data_mining_service.run_model_inference,
                tracking_id=x_task_id,
                task_label="Application Data inference",
                training_job_id=training_job_id,
                filename=dataset.filename,
                content=content,
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
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Application inference failed: {type(exc).__name__}: {exc}",
            ) from exc
        finally:
            await dataset.close()

    @router.post(
        "/data-mining/clustering",
        response_model=ClusteringResponse,
        tags=["data-mining"],
    )
    async def run_clustering(
        model: str = Form("kmeans"),
        feature_columns: str = Form(...),
        cluster_count: int = Form(3),
        dataset: UploadFile = File(...),
        x_task_id: str | None = Header(None, alias="X-Task-ID"),
    ) -> ClusteringResponse:
        content = await dataset.read(data_mining_service.max_upload_bytes + 1)
        try:
            enforce_upload_limit(content, data_mining_service.max_upload_bytes)
            data_mining_service.validate_upload(dataset.filename, content)
            try:
                parsed_features = json.loads(feature_columns)
            except json.JSONDecodeError as exc:
                raise InvalidDatasetError(
                    "Feature columns must be a valid JSON list"
                ) from exc
            return await run_calculation(
                data_mining_service.run_clustering,
                tracking_id=x_task_id,
                task_label=f"Clustering: {model}",
                filename=dataset.filename,
                content=content,
                feature_columns=parsed_features,
                cluster_count=cluster_count,
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
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Clustering failed: {type(exc).__name__}: {exc}",
            ) from exc
        finally:
            await dataset.close()

    @router.post(
        "/data-mining/dimensionality-reduction",
        response_model=DimensionalityReductionResponse,
        tags=["data-mining"],
    )
    async def run_dimensionality_reduction(
        model: str = Form("pca"),
        feature_columns: str = Form(...),
        component_count: int = Form(2),
        dataset: UploadFile = File(...),
        x_task_id: str | None = Header(None, alias="X-Task-ID"),
    ) -> DimensionalityReductionResponse:
        content = await dataset.read(data_mining_service.max_upload_bytes + 1)
        try:
            enforce_upload_limit(content, data_mining_service.max_upload_bytes)
            data_mining_service.validate_upload(dataset.filename, content)
            try:
                parsed_features = json.loads(feature_columns)
            except json.JSONDecodeError as exc:
                raise InvalidDatasetError(
                    "Feature columns must be a valid JSON list"
                ) from exc
            return await run_calculation(
                data_mining_service.run_dimensionality_reduction,
                tracking_id=x_task_id,
                task_label=f"Dimensionality reduction: {model}",
                filename=dataset.filename,
                content=content,
                feature_columns=parsed_features,
                component_count=component_count,
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
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "Dimensionality reduction failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            ) from exc
        finally:
            await dataset.close()

    @router.post(
        "/data-mining/anomaly-detection",
        response_model=AnomalyDetectionResponse,
        tags=["data-mining"],
    )
    async def run_anomaly_detection(
        model: str = Form("isolation_forest"),
        feature_columns: str = Form(...),
        dataset: UploadFile = File(...),
        x_task_id: str | None = Header(None, alias="X-Task-ID"),
    ) -> AnomalyDetectionResponse:
        content = await dataset.read(data_mining_service.max_upload_bytes + 1)
        try:
            enforce_upload_limit(content, data_mining_service.max_upload_bytes)
            data_mining_service.validate_upload(dataset.filename, content)
            try:
                parsed_features = json.loads(feature_columns)
            except json.JSONDecodeError as exc:
                raise InvalidDatasetError(
                    "Feature columns must be a valid JSON list"
                ) from exc
            return await run_calculation(
                data_mining_service.run_anomaly_detection,
                tracking_id=x_task_id,
                task_label=f"Anomaly detection: {model}",
                filename=dataset.filename,
                content=content,
                feature_columns=parsed_features,
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
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Anomaly detection failed: {type(exc).__name__}: {exc}",
            ) from exc
        finally:
            await dataset.close()

    @router.post(
        "/data-mining/time-series",
        response_model=TimeSeriesResponse,
        tags=["data-mining"],
    )
    async def run_time_series(
        age_column: str = Form(...),
        age_max_column: str = Form(...),
        probability_column: str = Form(...),
        latitude_column: str = Form(...),
        longitude_column: str = Form(...),
        age_unit: str = Form("Ma"),
        bin_width: float = Form(10.0),
        bootstrap_iterations: int = Form(100),
        dataset: UploadFile = File(...),
        x_task_id: str | None = Header(None, alias="X-Task-ID"),
    ) -> TimeSeriesResponse:
        content = await dataset.read(data_mining_service.max_upload_bytes + 1)
        try:
            enforce_upload_limit(content, data_mining_service.max_upload_bytes)
            data_mining_service.validate_upload(dataset.filename, content)
            return await run_calculation(
                data_mining_service.run_time_series,
                tracking_id=x_task_id,
                task_label="Time-series analysis",
                filename=dataset.filename,
                content=content,
                age_column=age_column,
                age_max_column=age_max_column,
                probability_column=probability_column,
                latitude_column=latitude_column,
                longitude_column=longitude_column,
                age_unit=age_unit,
                bin_width=bin_width,
                bootstrap_iterations=bootstrap_iterations,
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
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Time series analysis failed: {type(exc).__name__}: {exc}",
            ) from exc
        finally:
            await dataset.close()

    @router.post(
        "/data-mining/time-series/predict",
        response_model=TimeSeriesResponse,
        tags=["data-mining"],
    )
    async def run_predicted_time_series(
        age_column: str = Form(...),
        age_max_column: str = Form(...),
        latitude_column: str = Form(...),
        longitude_column: str = Form(...),
        age_unit: str = Form("Ma"),
        bin_width: float = Form(10.0),
        bootstrap_iterations: int = Form(100),
        dataset: UploadFile = File(...),
        x_task_id: str | None = Header(None, alias="X-Task-ID"),
    ) -> TimeSeriesResponse:
        content = await dataset.read(data_mining_service.max_upload_bytes + 1)
        try:
            enforce_upload_limit(content, data_mining_service.max_upload_bytes)
            data_mining_service.validate_upload(dataset.filename, content)
            return await run_calculation(
                data_mining_service.run_predicted_time_series,
                tracking_id=x_task_id,
                task_label="Model-predicted time series",
                filename=dataset.filename,
                content=content,
                age_column=age_column,
                age_max_column=age_max_column,
                latitude_column=latitude_column,
                longitude_column=longitude_column,
                age_unit=age_unit,
                bin_width=bin_width,
                bootstrap_iterations=bootstrap_iterations,
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
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "Model-predicted time series failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            ) from exc
        finally:
            await dataset.close()

    @router.post(
        "/data-mining/time-series/element-mean",
        response_model=TimeSeriesResponse,
        tags=["data-mining"],
    )
    async def run_element_time_series(
        age_column: str = Form(...),
        value_column: str = Form(...),
        age_unit: str = Form("Ma"),
        bin_width: float = Form(100.0),
        value_unit: str = Form("wt%"),
        filter_column: str | None = Form(None),
        filter_min: float | None = Form(None),
        filter_max: float | None = Form(None),
        dataset: UploadFile = File(...),
        x_task_id: str | None = Header(None, alias="X-Task-ID"),
    ) -> TimeSeriesResponse:
        content = await dataset.read(data_mining_service.max_upload_bytes + 1)
        try:
            enforce_upload_limit(content, data_mining_service.max_upload_bytes)
            data_mining_service.validate_upload(dataset.filename, content)
            return await run_calculation(
                data_mining_service.run_element_time_series,
                tracking_id=x_task_id,
                task_label="Element mean time series",
                filename=dataset.filename,
                content=content,
                age_column=age_column,
                value_column=value_column,
                age_unit=age_unit,
                bin_width=bin_width,
                value_unit=value_unit,
                filter_column=filter_column,
                filter_min=filter_min,
                filter_max=filter_max,
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
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Element time series failed: {type(exc).__name__}: {exc}",
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
        x_task_id: str | None = Header(None, alias="X-Task-ID"),
    ) -> RunResponse:
        content = await dataset.read(service.max_upload_bytes + 1)
        try:
            enforce_upload_limit(content, service.max_upload_bytes)
            service.validate_upload(dataset.filename, content)
            service.validate_selection(task, method, element)
            return await run_calculation(
                service.run_job,
                tracking_id=x_task_id,
                task_label=f"Chemical modeling: {method}",
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
        except HTTPException:
            raise
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
