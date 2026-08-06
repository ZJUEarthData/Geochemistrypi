"""Application service that adapts chemical-modeling functions for HTTP."""

from __future__ import annotations

import json
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
    """Raised when an uploaded dataset cannot be used by the selected method."""


class UploadTooLargeError(ValueError):
    """Raised when an uploaded dataset exceeds the configured byte limit."""


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
        suffix = self._validate_upload(filename, content)
        dataframe = self._read_and_validate_dataset(task, method, suffix, content)

        job_id = uuid4().hex
        job_dir = self.jobs_dir / job_id
        input_dir = job_dir / "input"
        output_dir = job_dir / "output"
        input_dir.mkdir(parents=True)
        output_dir.mkdir()

        input_path = input_dir / "input.xlsx"
        if suffix == ".xlsx":
            input_path.write_bytes(content)
        else:
            sheet_name = (
                "3程序处理_输入常数"
                if task == "algo_fractionation" and method == "double_spike"
                else "Sheet1"
            )
            dataframe.to_excel(input_path, index=False, sheet_name=sheet_name)

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

    def _validate_upload(self, filename: str | None, content: bytes) -> str:
        suffix = Path(filename or "").suffix.lower()
        if suffix not in {".xlsx", ".csv"}:
            raise ValueError("Chemical Modeling supports only .xlsx and .csv files")
        if not content:
            raise ValueError("The uploaded file is empty")
        if len(content) > self.max_upload_bytes:
            raise UploadTooLargeError(f"The uploaded file exceeds {self.max_upload_bytes} bytes")
        return suffix

    @staticmethod
    def _read_and_validate_dataset(
        task: str,
        method: str,
        suffix: str,
        content: bytes,
    ) -> pd.DataFrame:
        try:
            if suffix == ".xlsx":
                sheet_name: int | str = (
                    "3程序处理_输入常数"
                    if task == "algo_fractionation" and method == "double_spike"
                    else 0
                )
                dataframe = pd.read_excel(BytesIO(content), sheet_name=sheet_name)
            else:
                dataframe = pd.read_csv(BytesIO(content), encoding="utf-8-sig")
            columns = [str(column) for column in dataframe.columns]
        except UnicodeDecodeError as exc:
            raise InvalidDatasetError("CSV files must use UTF-8 encoding") from exc
        except Exception as exc:
            if suffix == ".xlsx" and task == "algo_fractionation" and method == "double_spike":
                raise InvalidDatasetError(
                    "Mo double-spike requires worksheet '3程序处理_输入常数'"
                ) from exc
            if suffix == ".csv":
                raise InvalidDatasetError("The uploaded file is not a readable UTF-8 CSV dataset") from exc
            raise InvalidDatasetError("The uploaded file is not a readable .xlsx workbook") from exc

        metadata = get_method_metadata(task, method)
        required = [column.name for column in metadata.input_columns if column.required]
        missing = [column for column in required if column not in columns]
        if missing:
            raise InvalidDatasetError(f"Missing required dataset columns: {', '.join(missing)}")

        if dataframe.empty:
            raise InvalidDatasetError("The uploaded dataset contains no data rows")

        for column in metadata.input_columns:
            if column.name not in dataframe.columns:
                continue

            values = dataframe[column.name]
            if column.data_type == "string":
                if values.isna().any() or not values.map(lambda value: bool(str(value).strip())).all():
                    raise InvalidDatasetError(
                        f"Column '{column.name}' must contain non-empty text values"
                    )
                continue
            if column.data_type not in {"number", "integer"}:
                continue
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

        if task == "algo_equilibrium" and method == "mass_balance":
            species_columns = [column for column in dataframe.columns if column != "total_mass"]
            if not species_columns:
                raise InvalidDatasetError("Mass balance requires at least one species concentration column")
            for column in species_columns:
                values = dataframe[column]
                if values.isna().any() or not pd.api.types.is_numeric_dtype(values):
                    raise InvalidDatasetError(
                        f"Species column '{column}' must contain numeric values without empty cells"
                    )
                numeric = values.astype(float)
                if not numeric.map(math.isfinite).all():
                    raise InvalidDatasetError(f"Species column '{column}' must contain only finite values")
                if (numeric < 0).any():
                    raise InvalidDatasetError(
                        f"Species column '{column}' values must be greater than or equal to 0"
                    )

        if task == "algo_equilibrium" and method == "mass_action":
            from geochemistrypi.chemical_modeling.model.func.algo_equilibrium.mass_action import (
                law_of_mass_action,
            )

            for row_number, row in enumerate(dataframe.itertuples(index=False), start=2):
                try:
                    stoich = json.loads(row.stoich)
                    initial = json.loads(row.initial_concentrations)
                    law_of_mass_action(float(row.K), stoich, initial)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise InvalidDatasetError(f"Row {row_number}: {exc}") from exc

        if task == "algo_kinetic" and method == "adsorption_kinetics":
            invalid_models = sorted(
                {
                    str(value).strip().lower()
                    for value in dataframe["model"]
                    if str(value).strip().lower() not in {"first", "second"}
                }
            )
            if invalid_models:
                raise InvalidDatasetError("Column 'model' must contain either 'first' or 'second'")

        if task == "algo_fractionation" and method == "internal_standard":
            def normalize_label(value: object) -> str:
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    numeric = float(value)
                    if numeric.is_integer():
                        return str(int(numeric))
                return str(value).strip()

            labels = [normalize_label(value) for value in dataframe["Label"]]
            standard_positions = [index for index, label in enumerate(labels) if label == "3133"]
            if len(standard_positions) < 2:
                raise InvalidDatasetError(
                    "Hg internal standard requires at least two rows with Label '3133'"
                )

            for index, label in enumerate(labels):
                if label == "3133":
                    continue
                previous = [position for position in standard_positions if position < index]
                following = [position for position in standard_positions if position > index]
                if not previous or not following:
                    raise InvalidDatasetError(
                        f"Row {index + 2}: each sample must be bracketed by Label '3133' rows"
                    )

        if task == "algo_thermodynamic" and method == "gibbs_minimization":
            from geochemistrypi.chemical_modeling.model.func.algo_thermodynamic.gibbs_minimization import (
                gibbs_minimization,
            )

            for row_number, row in enumerate(dataframe.itertuples(index=False), start=2):
                try:
                    gibbs_minimization(
                        json.loads(row.gibbs_energies),
                        json.loads(row.stoichiometry),
                        json.loads(row.component_totals),
                    )
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise InvalidDatasetError(f"Row {row_number}: {exc}") from exc

        if task == "algo_solubility" and method == "ding":
            calibration_ranges = {
                "Pressure": (0.0001, 5.5),
                "T": (1473.15, 2073.15),
                "SiO2": (33.0, 55.0),
                "TiO2": (0.01, 15.0),
                "Al2O3": (5.0, 20.0),
                "FeO": (5.0, 30.0),
                "MgO": (6.0, 23.0),
                "CaO": (5.0, 19.0),
                "sulfide_Ni": (0.0, 50.0),
            }
            for column, (minimum, maximum) in calibration_ranges.items():
                values = dataframe[column].astype(float)
                if ((values < minimum) | (values > maximum)).any():
                    raise InvalidDatasetError(
                        f"Column '{column}' values must be between {minimum:g} and {maximum:g}"
                    )

            oxide_columns = [
                "SiO2",
                "TiO2",
                "Al2O3",
                "FeO",
                "MgO",
                "CaO",
                "Na2O",
                "K2O",
            ]
            oxide_totals = dataframe[oxide_columns].astype(float).sum(axis=1)
            if ((oxide_totals < 90) | (oxide_totals > 105)).any():
                raise InvalidDatasetError(
                    "Ding oxide totals must be between 90 and 105 wt.%"
                )

        if task == "algo_solubility" and method == "blanchard":
            calibration_ranges = {
                "Pressure": (0.0001, 24.0),
                "T": (1423.0, 2623.0),
                "SiO2": (0.0, 77.9),
                "TiO2": (0.0, 15.3),
                "Al2O3": (0.0, 34.1),
                "FeO": (0.5, 40.1),
                "MgO": (0.0, 53.3),
                "CaO": (0.0, 32.7),
                "Na2O": (0.0, 8.0),
                "K2O": (0.0, 8.4),
                "H2O": (0.0, 8.5),
            }
            for column, (minimum, maximum) in calibration_ranges.items():
                values = dataframe[column].astype(float)
                if ((values < minimum) | (values > maximum)).any():
                    raise InvalidDatasetError(
                        f"Column '{column}' values must be between {minimum:g} and {maximum:g}"
                    )

            if "P2O5" in dataframe.columns and (dataframe["P2O5"].astype(float) > 1.8).any():
                raise InvalidDatasetError("Column 'P2O5' values must be between 0 and 1.8")

            oxide_columns = [
                column
                for column in (
                    "SiO2",
                    "TiO2",
                    "Al2O3",
                    "FeO",
                    "MgO",
                    "CaO",
                    "Na2O",
                    "K2O",
                    "H2O",
                    "MnO",
                    "P2O5",
                    "Cr2O3",
                )
                if column in dataframe.columns
            ]
            oxide_totals = dataframe[oxide_columns].astype(float).sum(axis=1)
            if ((oxide_totals < 90) | (oxide_totals > 110)).any():
                raise InvalidDatasetError(
                    "Blanchard oxide totals must be between 90 and 110 wt.%"
                )

            sulfide_totals = dataframe[["Fe", "Ni", "Cu"]].astype(float).sum(axis=1)
            if (sulfide_totals > 100).any():
                raise InvalidDatasetError(
                    "Blanchard sulfide Fe + Ni + Cu must not exceed 100 wt.%"
                )

        if task == "algo_solubility" and method == "hybrid":
            calibration_ranges = {
                "Pressure": (0.0001, 24.0),
                "T": (1423.0, 2623.0),
                "SiO2": (27.712, 77.9),
                "TiO2": (0.0, 18.77),
                "Al2O3": (0.1397, 34.0438),
                "FeO": (0.00456, 40.9967),
                "MgO": (0.0, 53.212),
                "CaO": (0.35197, 32.673),
                "NiO": (0.0, 0.554),
                "Na2O": (0.0, 7.9595),
                "K2O": (0.0, 8.39),
                "H2O": (0.0, 8.5),
                "Fe": (0.0486, 91.054),
                "Ni+Cu+Co": (0.0, 77.48),
                "S": (8.672, 42.53),
                "O": (0.0, 7.136),
            }
            for column, (minimum, maximum) in calibration_ranges.items():
                values = dataframe[column].astype(float)
                if ((values < minimum) | (values > maximum)).any():
                    raise InvalidDatasetError(
                        f"Column '{column}' values must be between {minimum:g} and {maximum:g}"
                    )

            oxide_columns = [
                "SiO2",
                "TiO2",
                "Al2O3",
                "FeO",
                "MgO",
                "CaO",
                "NiO",
                "Na2O",
                "K2O",
                "H2O",
            ]
            oxide_totals = dataframe[oxide_columns].astype(float).sum(axis=1)
            if ((oxide_totals < 89) | (oxide_totals > 110)).any():
                raise InvalidDatasetError(
                    "Hybrid silicate oxide totals must be between 89 and 110 wt.%"
                )

            sulfide_columns = ["Fe", "Ni+Cu+Co", "S", "O"]
            sulfide_totals = dataframe[sulfide_columns].astype(float).sum(axis=1)
            if ((sulfide_totals < 75) | (sulfide_totals > 105)).any():
                raise InvalidDatasetError(
                    "Hybrid sulfide Fe + Ni+Cu+Co + S + O totals must be between 75 and 105 wt.%"
                )

        if task == "algo_thermodynamic" and method == "vanthoff":
            maximum_log = math.log(sys.float_info.max)
            for row in dataframe.itertuples(index=False):
                log_k2 = math.log(row.K1) - row.dH / 8.314 * (1 / row.T2 - 1 / row.T1)
                if not math.isfinite(log_k2) or log_k2 > maximum_log:
                    raise InvalidDatasetError(
                        "van't Hoff parameters produce a result outside the supported numeric range"
                    )

        return dataframe

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
