"""Application service that adapts chemical-modeling functions for HTTP."""

from __future__ import annotations

import math
import sys
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from geochemistrypi.chemical_modeling.dispatcher import (
    discover_tasks,
    list_method_elements,
    list_task_methods,
    run_task_method,
)

from .method_metadata import get_method_metadata
from .schemas import ArtifactResponse, InputColumnItem, MethodCatalogItem, RunResponse, TaskCatalogItem


class InvalidDatasetError(ValueError):
    """Raised when an uploaded workbook cannot be used by the selected method."""


class UploadTooLargeError(ValueError):
    """Raised when an uploaded workbook exceeds the configured byte limit."""


class OnlineService:
    """Run lightweight Online jobs without importing the legacy web stack."""

    def __init__(self, runtime_dir: Path, max_upload_bytes: int = 10 * 1024 * 1024):
        self.runtime_dir = runtime_dir.resolve()
        self.max_upload_bytes = max_upload_bytes
        self.jobs_dir = self.runtime_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def build_catalog(self) -> list[TaskCatalogItem]:
        catalog: list[TaskCatalogItem] = []
        for task_name in discover_tasks():
            try:
                methods = list_task_methods(task_name)
                method_items: list[MethodCatalogItem] = []
                for method_name, description in methods.items():
                    metadata = get_method_metadata(task_name, method_name)
                    input_columns = [
                        InputColumnItem(
                            name=column.name,
                            label=column.label,
                            description=column.description,
                            data_type=column.data_type,
                            unit=column.unit,
                            example=column.example,
                            required=column.required,
                            minimum=column.minimum,
                            exclusive_minimum=column.exclusive_minimum,
                        )
                        for column in metadata.input_columns
                    ]
                    method_items.append(
                        MethodCatalogItem(
                            name=method_name,
                            description=description,
                            elements=list_method_elements(task_name, method_name),
                            status=metadata.status,
                            status_message=metadata.status_message,
                            formula=metadata.formula,
                            input_columns=input_columns,
                            input_notes=list(metadata.input_notes),
                            required_columns=[column.name for column in input_columns if column.required],
                        )
                    )
                catalog.append(TaskCatalogItem(name=task_name, available=True, methods=method_items))
            except Exception as exc:
                catalog.append(
                    TaskCatalogItem(
                        name=task_name,
                        available=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        return catalog

    def validate_selection(self, task: str, method: str, element: str) -> None:
        tasks = discover_tasks()
        if task not in tasks:
            raise ValueError(f"Unknown task: {task}")

        methods = list_task_methods(task)
        if method not in methods:
            raise ValueError(f"Unknown method '{method}' for task '{task}'")

        elements = list_method_elements(task, method)
        if element not in elements:
            raise ValueError(f"Unknown element '{element}' for method '{method}'")

        metadata = get_method_metadata(task, method)
        if metadata.status != "verified":
            raise ValueError(f"Method '{method}' has not completed Online verification")

    def run_job(
        self,
        *,
        task: str,
        method: str,
        element: str,
        filename: str | None,
        content: bytes,
    ) -> RunResponse:
        self.validate_selection(task, method, element)
        self._validate_upload(filename, content)
        self._validate_workbook(task, method, content)

        job_id = uuid4().hex
        job_dir = self.jobs_dir / job_id
        input_dir = job_dir / "input"
        output_dir = job_dir / "output"
        input_dir.mkdir(parents=True)
        output_dir.mkdir()

        input_path = input_dir / "input.xlsx"
        input_path.write_bytes(content)

        result: Any = run_task_method(
            task,
            method,
            element,
            str(input_path),
            str(output_dir),
        )

        artifacts = self._collect_artifacts(job_id, output_dir)
        if not artifacts:
            raise RuntimeError(f"The calculation returned no result files: {result!r}")

        status = result.get("status", "success") if isinstance(result, dict) else "success"
        return RunResponse(
            job_id=job_id,
            status=str(status),
            message="Calculation completed",
            artifacts=artifacts,
        )

    def resolve_artifact(self, job_id: str, file_path: str) -> Path:
        output_dir = (self.jobs_dir / job_id / "output").resolve()
        candidate = (output_dir / file_path).resolve()
        try:
            candidate.relative_to(output_dir)
        except ValueError as exc:
            raise FileNotFoundError(file_path) from exc
        if not candidate.is_file():
            raise FileNotFoundError(file_path)
        return candidate

    def _validate_upload(self, filename: str | None, content: bytes) -> None:
        suffix = Path(filename or "").suffix.lower()
        if suffix != ".xlsx":
            raise ValueError("Only .xlsx files are supported in the first Online version")
        if not content:
            raise ValueError("The uploaded file is empty")
        if len(content) > self.max_upload_bytes:
            raise UploadTooLargeError(f"The uploaded file exceeds {self.max_upload_bytes} bytes")

    @staticmethod
    def _validate_workbook(task: str, method: str, content: bytes) -> None:
        try:
            dataframe = pd.read_excel(BytesIO(content))
            columns = [str(column) for column in dataframe.columns]
        except Exception as exc:
            raise InvalidDatasetError("The uploaded file is not a readable .xlsx workbook") from exc

        metadata = get_method_metadata(task, method)
        required = [column.name for column in metadata.input_columns if column.required]
        missing = [column for column in required if column not in columns]
        if missing:
            raise InvalidDatasetError(f"Missing required Excel columns: {', '.join(missing)}")

        if dataframe.empty:
            raise InvalidDatasetError("The Excel workbook contains no data rows")

        for column in metadata.input_columns:
            if column.name not in dataframe.columns or column.data_type not in {"number", "integer"}:
                continue

            values = dataframe[column.name]
            if values.isna().any() or not pd.api.types.is_numeric_dtype(values):
                raise InvalidDatasetError(
                    f"Column '{column.name}' must contain numeric values without empty cells"
                )
            if not values.astype(float).map(math.isfinite).all():
                raise InvalidDatasetError(f"Column '{column.name}' must contain only finite values")
            if column.data_type == "integer" and not values.astype(float).map(float.is_integer).all():
                raise InvalidDatasetError(f"Column '{column.name}' must contain integer values")
            if column.minimum is None:
                continue

            invalid = values <= column.minimum if column.exclusive_minimum else values < column.minimum
            if invalid.any():
                relation = "greater than" if column.exclusive_minimum else "greater than or equal to"
                raise InvalidDatasetError(
                    f"Column '{column.name}' values must be {relation} {column.minimum:g}"
                )

        if task == "algo_thermodynamic" and method == "vanthoff":
            maximum_log = math.log(sys.float_info.max)
            for row in dataframe.itertuples(index=False):
                log_k2 = math.log(row.K1) - row.dH / 8.314 * (1 / row.T2 - 1 / row.T1)
                if not math.isfinite(log_k2) or log_k2 > maximum_log:
                    raise InvalidDatasetError(
                        "van't Hoff parameters produce a result outside the supported numeric range"
                    )

    @staticmethod
    def _collect_artifacts(job_id: str, output_dir: Path) -> list[ArtifactResponse]:
        artifacts: list[ArtifactResponse] = []
        for path in sorted(output_dir.rglob("*")):
            if not path.is_file():
                continue
            relative_path = path.relative_to(output_dir).as_posix()
            artifacts.append(
                ArtifactResponse(
                    name=relative_path,
                    download_url=f"/api/jobs/{job_id}/files/{relative_path}",
                    size_bytes=path.stat().st_size,
                )
            )
        return artifacts
