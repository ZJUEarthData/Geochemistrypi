"""Durable, non-blocking lifecycle control for local CLI subprocess runs."""

import json
import os
import re
import tempfile
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..api.schemas import (
    AnalysisValidationResponse,
    AnomalyDetectionRequest,
    CancelRunResponse,
    ClassificationRequest,
    ClusteringRequest,
    DatasetInspectionRequest,
    DecompositionRequest,
    RegressionRequest,
    RunResultResponse,
    RunStatusResponse,
    StartAnalysisResponse,
    TimeSeriesRequest,
)
from ..config.constants import ARTIFACT_INDEX_SCHEMA_VERSION, SERVER_VERSION
from ..config.settings import McpSettings
from ..contracts.anomaly_detection import MODEL_DISPLAY_NAMES as ANOMALY_DETECTION_MODEL_DISPLAY_NAMES
from ..contracts.anomaly_detection import MODEL_ORDER as ANOMALY_DETECTION_MODEL_ORDER
from ..contracts.classification import MODEL_DISPLAY_NAMES as CLASSIFICATION_MODEL_DISPLAY_NAMES
from ..contracts.classification import MODEL_ORDER as CLASSIFICATION_MODEL_ORDER
from ..contracts.clustering import MODEL_DISPLAY_NAMES as CLUSTERING_MODEL_DISPLAY_NAMES
from ..contracts.clustering import MODEL_ORDER as CLUSTERING_MODEL_ORDER
from ..contracts.decomposition import MODEL_DISPLAY_NAMES as DECOMPOSITION_MODEL_DISPLAY_NAMES
from ..contracts.decomposition import MODEL_ORDER as DECOMPOSITION_MODEL_ORDER
from ..contracts.regression import MODEL_DISPLAY_NAMES as REGRESSION_MODEL_DISPLAY_NAMES
from ..contracts.regression import MODEL_ORDER as REGRESSION_MODEL_ORDER
from ..data.catalog import DatasetCatalog, ResolvedDataset
from ..data.inspector import DatasetSnapshot
from ..data.inspector import inspect_dataset as inspect_local_dataset
from ..data.inspector import sha256_file, snapshot_dataset
from ..data.row_pairing import verify_original_row_pairing
from ..planning.interaction_plan import AnalysisPlanCompiler, DatasetCompilationContext, InteractionPlan
from ..tracking.experiments import ExperimentManager
from .artifacts import discover_artifacts, read_time_series_preprocessing_summary
from .cli_driver import CliInteractionDriver, CliRunCancelledError, validate_workspace_path

_RUN_ID = re.compile(r"^run-[0-9a-f]{16}$")
_TERMINAL_STATES = {"succeeded", "partial_failure", "failed", "cancelled"}
_ATOMIC_REPLACE_RETRY_DELAYS_SECONDS = (0.01, 0.02, 0.04)


def _selected_models(request: Any) -> tuple[str, ...]:
    if request.task == "time_series":
        return ("subaerial_proportion_bootstrap",)
    orders = {
        "classification": CLASSIFICATION_MODEL_ORDER,
        "regression": REGRESSION_MODEL_ORDER,
        "clustering": CLUSTERING_MODEL_ORDER,
        "decomposition": DECOMPOSITION_MODEL_ORDER,
        "anomaly_detection": ANOMALY_DETECTION_MODEL_ORDER,
    }
    return orders[request.task] if request.model_selection.mode == "all" else (request.model.type,)


def _selected_tuning(request: Any) -> str:
    if request.task == "time_series":
        return "not_applicable"
    if request.model_selection.mode == "all":
        return request.model_selection.tuning
    return getattr(request, "tuning", "not_applicable")


def _resolved_model_parameters(request: Any) -> dict[str, Any]:
    if request.task == "time_series" or request.model_selection.mode == "all" or _selected_tuning(request) == "automl":
        return {}
    model = getattr(request, "model", None)
    if model is None:
        return {}
    return model.model_dump(mode="python", exclude={"type"})


class RunNotFoundError(ValueError):
    """Raised when a run ID is not owned by this wrapper."""


class RunStateError(ValueError):
    """Raised when an operation is invalid for the run's current state."""


class InputIntegrityError(RuntimeError):
    """Raised when the user's source dataset changes during a run."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_run_id() -> str:
    return f"run-{uuid.uuid4().hex[:16]}"


def _replace_with_bounded_permission_retry(source: Path, destination: Path) -> None:
    """Retry only transient permission failures while publishing metadata."""
    for attempt in range(len(_ATOMIC_REPLACE_RETRY_DELAYS_SECONDS) + 1):
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            if attempt == len(_ATOMIC_REPLACE_RETRY_DELAYS_SECONDS):
                raise PermissionError(f"Atomic metadata replacement failed after {attempt + 1} attempts for {destination}: {exc}") from exc
            time.sleep(_ATOMIC_REPLACE_RETRY_DELAYS_SECONDS[attempt])


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        _replace_with_bounded_permission_retry(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RunStateError(f"Run metadata is unavailable or corrupt: {path}") from exc
    if not isinstance(value, dict):
        raise RunStateError(f"Run metadata must be a JSON object: {path}")
    return value


def _safe_error(exc: BaseException) -> str:
    message = " ".join(str(exc).split())
    return (message or type(exc).__name__)[:1000]


def _aggregate_children(output_directory: Path, task: str) -> tuple[str, tuple[dict[str, Any], ...]]:
    manifest_path = output_directory / "summary" / "Aggregate Model Results.json"
    value = _read_json(manifest_path)
    expected_fields = {
        "schema_version",
        "task",
        "selection_mode",
        "tuning",
        "state",
        "expected_model_count",
        "succeeded_count",
        "failed_count",
        "children",
    }
    if set(value) != expected_fields:
        raise RunStateError("Aggregate model manifest has unknown or missing fields.")
    if value["schema_version"] != 1 or value["task"] != task or value["selection_mode"] != "all" or value["state"] not in {"complete", "partial_failure"} or not isinstance(value["children"], list):
        raise RunStateError("Aggregate model manifest identity is invalid.")
    contracts = {
        "classification": (
            CLASSIFICATION_MODEL_ORDER,
            CLASSIFICATION_MODEL_DISPLAY_NAMES,
        ),
        "regression": (REGRESSION_MODEL_ORDER, REGRESSION_MODEL_DISPLAY_NAMES),
        "clustering": (CLUSTERING_MODEL_ORDER, CLUSTERING_MODEL_DISPLAY_NAMES),
        "decomposition": (
            DECOMPOSITION_MODEL_ORDER,
            DECOMPOSITION_MODEL_DISPLAY_NAMES,
        ),
        "anomaly_detection": (
            ANOMALY_DETECTION_MODEL_ORDER,
            ANOMALY_DETECTION_MODEL_DISPLAY_NAMES,
        ),
    }
    model_order, display_names = contracts[task]
    expected_models = [display_names[model] for model in model_order]
    children = []
    seen = set()
    for child in value["children"]:
        if not isinstance(child, dict) or set(child) != {
            "model",
            "state",
            "output_relative_path",
            "artifact_count",
            "error",
        }:
            raise RunStateError("Aggregate child metadata is invalid.")
        model = child.get("model")
        relative = child.get("output_relative_path")
        if not isinstance(model, str) or not model or model in seen:
            raise RunStateError("Aggregate child model names must be unique.")
        if not isinstance(relative, str):
            raise RunStateError("Aggregate child output path must be a string.")
        relative_path = Path(relative)
        if relative_path.is_absolute() or len(relative_path.parts) != 1:
            raise RunStateError("Aggregate child output must be one direct child directory.")
        child_directory = (output_directory / relative_path).resolve()
        try:
            child_directory.relative_to(output_directory.resolve())
        except ValueError as exc:
            raise RunStateError("Aggregate child output escapes the run directory.") from exc
        if not child_directory.is_dir():
            raise RunStateError("Aggregate child output directory is missing.")
        actual_count = sum(1 for path in child_directory.rglob("*") if path.is_file())
        if child.get("artifact_count") != actual_count:
            raise RunStateError(f"Aggregate child artifact count changed for {model!r}.")
        seen.add(model)
        children.append(child)
    succeeded = sum(child.get("state") == "succeeded" for child in children)
    failed = sum(child.get("state") == "failed" for child in children)
    aggregate_is_consistent = all(
        (
            [child["model"] for child in children] == expected_models,
            value["expected_model_count"] == len(expected_models),
            value["succeeded_count"] == succeeded,
            value["failed_count"] == failed,
            succeeded + failed == len(children),
            value["state"] == ("complete" if failed == 0 else "partial_failure"),
        )
    )
    if not aggregate_is_consistent:
        raise RunStateError("Aggregate model counts or state are inconsistent.")
    return value["state"], tuple(children)


@dataclass(frozen=True)
class RunPaths:
    root: Path
    wrapper: Path
    workspace: Path
    request: Path
    status: Path
    result: Path
    artifact_index: Path

    @classmethod
    def create(cls, runs_root: Path, run_id: str) -> "RunPaths":
        root = runs_root / run_id
        wrapper = root / "wrapper"
        return cls(
            root=root,
            wrapper=wrapper,
            workspace=root / "workspace",
            request=wrapper / "request.json",
            status=wrapper / "status.json",
            result=wrapper / "result.json",
            artifact_index=wrapper / "artifact-index.json",
        )


@dataclass
class _RunControl:
    cancellation: threading.Event
    future: Future[None] | None = None


class RunManager:
    """Own local run transitions while the real CLI owns scientific execution."""

    def __init__(
        self,
        settings: McpSettings,
        plan_compiler: AnalysisPlanCompiler | None = None,
        driver_factory: Callable[[], CliInteractionDriver] | None = None,
        cli_resolver: Callable[[], tuple[Path, str]] | None = None,
        dataset_catalog: DatasetCatalog | None = None,
        experiment_manager: ExperimentManager | None = None,
    ):
        self.settings = settings
        self.plan_compiler = plan_compiler or AnalysisPlanCompiler()
        self.driver_factory = driver_factory or (
            lambda: CliInteractionDriver(
                process_timeout_seconds=settings.maximum_process_seconds,
                automation_mode=True,
            )
        )
        self.cli_resolver = cli_resolver or settings.require_supported_cli
        self.dataset_catalog = dataset_catalog or DatasetCatalog(settings)
        self.experiment_manager = experiment_manager or ExperimentManager(settings)
        self._executor = ThreadPoolExecutor(max_workers=settings.concurrency, thread_name_prefix="geochemistrypi-run")
        self._lock = threading.RLock()
        self._active: dict[str, _RunControl] = {}
        self._initialized = False
        self._closed = False

    def _ensure_initialized(self) -> None:
        with self._lock:
            if self._initialized:
                return
            self.settings.runs_root.mkdir(parents=True, exist_ok=True)
            for run_directory in self.settings.runs_root.glob("run-*"):
                status_path = run_directory / "wrapper" / "status.json"
                if not status_path.is_file():
                    continue
                try:
                    status = _read_json(status_path)
                except RunStateError:
                    continue
                if status.get("state") in {"queued", "running"}:
                    status.update(
                        {
                            "state": "failed",
                            "stage": "failed",
                            "finished_at": _utc_now(),
                            "progress_message": "The previous MCP server stopped before this run finished.",
                            "error": "Run state recovered after an unclean server shutdown; no stale PID was terminated.",
                            "cli_pid": None,
                        }
                    )
                    _atomic_write_json(status_path, status)
            self._initialized = True

    def _paths(self, run_id: str) -> RunPaths:
        if not _RUN_ID.fullmatch(run_id):
            raise RunNotFoundError("Invalid GeochemistryPi run ID.")
        paths = RunPaths.create(self.settings.runs_root, run_id)
        if not paths.root.is_dir():
            raise RunNotFoundError(f"GeochemistryPi run does not exist: {run_id}")
        return paths

    def _write_status(self, paths: RunPaths, status: dict[str, Any]) -> None:
        _atomic_write_json(paths.status, status)

    def _status_response(self, status: dict[str, Any]) -> RunStatusResponse:
        public_status = {field: status.get(field) for field in RunStatusResponse.model_fields}
        return RunStatusResponse.model_validate(public_status)

    def validate(
        self,
        request: ClassificationRequest | RegressionRequest | ClusteringRequest | DecompositionRequest | AnomalyDetectionRequest | TimeSeriesRequest,
    ) -> AnalysisValidationResponse:
        """Resolve and compile an analysis without creating a run or CLI process."""
        cli_executable, _ = self.cli_resolver()
        existing_experiment_id = getattr(request, "existing_experiment_id", None)
        if existing_experiment_id:
            self.experiment_manager.require_matching_name(existing_experiment_id, request.experiment_name)
        training_reference = getattr(request, "training_dataset", None)
        resolved_training = (
            self.dataset_catalog.resolve(training_reference, task=request.task, role="training")
            if training_reference is not None
            else ResolvedDataset(
                path=request.training_dataset_path,
                expected_sha256=None,
                dataset_id=None,
                source="path",
            )
        )
        snapshot = snapshot_dataset(resolved_training.path, self.settings.maximum_dataset_bytes)
        if resolved_training.expected_sha256 is not None and snapshot.sha256 != resolved_training.expected_sha256:
            raise InputIntegrityError("The training dataset changed between source resolution and validation.")
        application_reference = getattr(request, "application_dataset", None)
        if application_reference is not None:
            resolved_application = self.dataset_catalog.resolve(application_reference, task=request.task, role="application")
        else:
            application_path = getattr(request, "application_dataset_path", None)
            resolved_application = ResolvedDataset(path=application_path, expected_sha256=None, dataset_id=None, source="path") if application_path is not None else None
        application_snapshot = snapshot_dataset(resolved_application.path, self.settings.maximum_dataset_bytes) if resolved_application is not None else None
        expected_application_sha256 = resolved_application.expected_sha256 if resolved_application is not None else None
        if application_snapshot is not None and expected_application_sha256 is not None:
            if application_snapshot.sha256 != expected_application_sha256:
                raise InputIntegrityError("The application dataset changed between source resolution and validation.")
        execution_request = request.model_copy(
            update={
                "training_dataset_path": snapshot.resolved_path,
                "training_dataset": None,
                **(
                    {
                        "application_dataset_path": application_snapshot.resolved_path if application_snapshot else None,
                        "application_dataset": None,
                    }
                    if hasattr(request, "application_dataset_path")
                    else {}
                ),
            }
        )
        dataset_context = DatasetCompilationContext(
            training_source=resolved_training.source,
            application_source=resolved_application.source if resolved_application is not None else None,
        )
        plan = self.plan_compiler.compile(
            execution_request,
            cli_executable=cli_executable,
            dataset_context=dataset_context,
        )
        validation_paths = RunPaths.create(self.settings.runs_root, "run-0000000000000000")
        validate_workspace_path(plan, validation_paths.workspace)
        inspection = inspect_local_dataset(
            DatasetInspectionRequest(dataset_path=snapshot.resolved_path, sample_rows=0),
            self.settings,
            allow_pandas_duplicate_mangling=dataset_context.allows_pandas_duplicate_mangling("training"),
        )
        models = _selected_models(request)
        tuning = _selected_tuning(request)
        warnings = []
        if len(models) > 1:
            warnings.append(f"This aggregate workload will execute {len(models)} child models in CLI order.")
        if tuning == "automl":
            warnings.append("AutoML can take substantially longer than a manual model run.")
        if getattr(request, "application_dataset_path", None) is None and getattr(request, "application_dataset", None) is None and request.task in {"classification", "regression"}:
            warnings.append("No application dataset was selected, so model inference will be skipped.")
        if getattr(request, "world_map", None) is not None and request.world_map.enabled:
            warnings.append("World-map rendering is enabled and may add platform-dependent image artifacts.")
        if existing_experiment_id:
            warnings.append("The new run will be attached to the verified existing MLflow experiment ID.")
        return AnalysisValidationResponse(
            task=request.task,
            models=models,
            estimated_model_count=len(models),
            tuning=tuning,
            training_source=resolved_training.source,
            training_dataset_path=str(snapshot.resolved_path),
            training_sha256=snapshot.sha256,
            training_size_bytes=snapshot.size_bytes,
            source_row_count=snapshot.row_lineage.source_row_count,
            row_identity_scheme=snapshot.row_lineage.scheme,
            row_identity_sha256=snapshot.row_lineage.ordered_identity_sha256,
            columns=tuple(column.name for column in inspection.columns),
            identifier_column=getattr(request, "identifier_column", None),
            feature_columns=tuple(getattr(request, "feature_columns", ())),
            selected_columns=tuple(getattr(request, "resolved_selected_columns", ())),
            target_column=getattr(request, "target_column", None),
            target_columns=(
                tuple(column.name for column in inspection.columns if column.name in set(request.resolved_target_columns))
                if isinstance(request, RegressionRequest)
                else ((request.target_column,) if isinstance(request, ClassificationRequest) else ())
            ),
            resolved_model_parameters=_resolved_model_parameters(request),
            application_source=resolved_application.source if resolved_application else None,
            application_dataset_path=str(application_snapshot.resolved_path) if application_snapshot else None,
            application_sha256=application_snapshot.sha256 if application_snapshot else None,
            application_source_row_count=(application_snapshot.row_lineage.source_row_count if application_snapshot else None),
            application_row_identity_sha256=(application_snapshot.row_lineage.ordered_identity_sha256 if application_snapshot else None),
            experiment_mode=("not_applicable" if request.task == "time_series" else "existing" if existing_experiment_id else "new"),
            experiment_name=request.experiment_name,
            existing_experiment_id=existing_experiment_id,
            interaction_plan=plan.name,
            warnings=tuple(warnings),
        )

    def start(
        self,
        request: ClassificationRequest | RegressionRequest | ClusteringRequest | DecompositionRequest | AnomalyDetectionRequest | TimeSeriesRequest,
    ) -> StartAnalysisResponse:
        """Validate synchronously, then queue the long-running CLI work."""
        self._ensure_initialized()
        with self._lock:
            if self._closed:
                raise RunStateError("The GeochemistryPi run manager is shutting down.")
        cli_executable, cli_version = self.cli_resolver()
        existing_experiment_id = getattr(request, "existing_experiment_id", None)
        if existing_experiment_id:
            self.experiment_manager.require_matching_name(existing_experiment_id, request.experiment_name)
        training_reference = getattr(request, "training_dataset", None)
        if training_reference is not None:
            resolved_training = self.dataset_catalog.resolve(
                training_reference,
                task=request.task,
                role="training",
            )
        else:
            resolved_training = ResolvedDataset(
                path=request.training_dataset_path,
                expected_sha256=None,
                dataset_id=None,
                source="path",
            )
        snapshot = snapshot_dataset(resolved_training.path, self.settings.maximum_dataset_bytes)
        if resolved_training.expected_sha256 is not None and snapshot.sha256 != resolved_training.expected_sha256:
            raise InputIntegrityError("The training dataset changed between source resolution and validation.")
        application_reference = getattr(request, "application_dataset", None)
        if application_reference is not None:
            resolved_application = self.dataset_catalog.resolve(
                application_reference,
                task=request.task,
                role="application",
            )
            application_dataset_path = resolved_application.path
        else:
            application_dataset_path = getattr(request, "application_dataset_path", None)
            resolved_application = (
                ResolvedDataset(
                    path=application_dataset_path,
                    expected_sha256=None,
                    dataset_id=None,
                    source="path",
                )
                if application_dataset_path is not None
                else None
            )
        application_snapshot = (
            snapshot_dataset(
                application_dataset_path,
                self.settings.maximum_dataset_bytes,
            )
            if application_dataset_path is not None
            else None
        )
        expected_application_sha256 = resolved_application.expected_sha256 if resolved_application is not None else None
        if application_snapshot is not None and expected_application_sha256 is not None:
            if application_snapshot.sha256 != expected_application_sha256:
                raise InputIntegrityError("The application dataset changed between source resolution and validation.")
        execution_request = request.model_copy(
            update={
                "training_dataset_path": snapshot.resolved_path,
                "training_dataset": None,
                **(
                    {
                        "application_dataset_path": (application_snapshot.resolved_path if application_snapshot is not None else None),
                        "application_dataset": None,
                    }
                    if hasattr(request, "application_dataset_path")
                    else {}
                ),
            }
        )
        dataset_context = DatasetCompilationContext(
            training_source=resolved_training.source,
            application_source=resolved_application.source if resolved_application is not None else None,
        )
        plan = self.plan_compiler.compile(
            execution_request,
            cli_executable=cli_executable,
            dataset_context=dataset_context,
        )
        if "data-mining" in plan.public_command:
            if self.settings.tracking_root is None:
                raise RunStateError("The installer-owned MLflow tracking root is not configured.")
            self.settings.tracking_root.mkdir(parents=True, exist_ok=True)
            plan = replace(
                plan,
                public_command=(
                    *plan.public_command,
                    "--tracking-root",
                    str(self.settings.tracking_root),
                ),
            )
        run_id = _new_run_id()
        paths = RunPaths.create(self.settings.runs_root, run_id)
        validate_workspace_path(plan, paths.workspace)
        created_at = _utc_now()
        request_record = {
            "schema_version": 1,
            "run_id": run_id,
            "request": request.model_dump(mode="json"),
            "input": {
                "source_path": str(snapshot.source_path),
                "resolved_path": str(snapshot.resolved_path),
                "size_bytes": snapshot.size_bytes,
                "sha256": snapshot.sha256,
                "format": snapshot.format,
                "source": resolved_training.source,
                "dataset_id": resolved_training.dataset_id,
                "row_identity": snapshot.row_lineage.as_record(),
            },
            "application_input": (
                {
                    "source_path": str(application_snapshot.source_path),
                    "resolved_path": str(application_snapshot.resolved_path),
                    "size_bytes": application_snapshot.size_bytes,
                    "sha256": application_snapshot.sha256,
                    "format": application_snapshot.format,
                    "source": resolved_application.source,
                    "dataset_id": resolved_application.dataset_id,
                    "row_identity": application_snapshot.row_lineage.as_record(),
                }
                if application_snapshot is not None
                else None
            ),
            "interaction_plan": {
                "name": plan.name,
                "schema_version": plan.schema_version,
            },
            "versions": {
                "geochemistrypi_mcp": SERVER_VERSION,
                "geochemistrypi_cli": cli_version,
            },
        }
        status = {
            "schema_version": 1,
            "run_id": run_id,
            "state": "queued",
            "stage": "queued",
            "created_at": created_at,
            "started_at": None,
            "finished_at": None,
            "cli_pid": None,
            "recorded_cli_pid": None,
            "recorded_process_create_time": None,
            "progress_message": "Waiting for the local GeochemistryPi CLI execution slot.",
            "error": None,
        }
        control = _RunControl(cancellation=threading.Event())
        with self._lock:
            if self._closed:
                raise RunStateError("The GeochemistryPi run manager is shutting down.")
            if len(self._active) >= self.settings.maximum_pending_runs:
                raise RunStateError(f"The local run queue is full ({self.settings.maximum_pending_runs} active or queued runs). " "Wait for a run to finish or cancel one before starting another.")
            paths.wrapper.mkdir(parents=True)
            paths.workspace.mkdir()
            _atomic_write_json(paths.request, request_record)
            self._write_status(paths, status)
            self._active[run_id] = control
            control.future = self._executor.submit(
                self._execute,
                run_id,
                execution_request,
                snapshot,
                application_snapshot,
                plan,
                cli_version,
                control,
            )
        return StartAnalysisResponse(
            run_id=run_id,
            state="queued",
            models=_selected_models(request),
            estimated_model_count=len(_selected_models(request)),
            status_hint=f"Poll get_run_status with run_id {run_id}; call get_run_result after state becomes succeeded or partial_failure.",
        )

    def _mark_running(self, paths: RunPaths, control: _RunControl, pid: int, create_time: float) -> None:
        with self._lock:
            if control.cancellation.is_set():
                raise CliRunCancelledError(
                    "The run was cancelled while the CLI process was starting.",
                    paths.workspace,
                )
            status = _read_json(paths.status)
            status.update(
                {
                    "state": "running",
                    "stage": "running_cli",
                    "started_at": status.get("started_at") or _utc_now(),
                    "cli_pid": pid,
                    "recorded_cli_pid": pid,
                    "recorded_process_create_time": create_time,
                    "progress_message": "The existing GeochemistryPi CLI is running in the managed workspace.",
                }
            )
            self._write_status(paths, status)

    def _finish(self, paths: RunPaths, state: str, message: str, error: str | None = None) -> None:
        with self._lock:
            status = _read_json(paths.status)
            status.update(
                {
                    "state": state,
                    "stage": ("completed" if state in {"succeeded", "partial_failure"} else "cancelled" if state == "cancelled" else "failed"),
                    "finished_at": _utc_now(),
                    "cli_pid": None,
                    "progress_message": message,
                    "error": error,
                }
            )
            self._write_status(paths, status)

    def _execute(
        self,
        run_id: str,
        request: ClassificationRequest | RegressionRequest | ClusteringRequest | DecompositionRequest | AnomalyDetectionRequest | TimeSeriesRequest,
        snapshot: DatasetSnapshot,
        application_snapshot: DatasetSnapshot | None,
        plan: InteractionPlan,
        cli_version: str,
        control: _RunControl,
    ) -> None:
        paths = RunPaths.create(self.settings.runs_root, run_id)
        try:
            if control.cancellation.is_set():
                raise CliRunCancelledError("The queued run was cancelled before execution.", paths.workspace)
            if sha256_file(snapshot.resolved_path) != snapshot.sha256:
                raise InputIntegrityError("The input dataset changed after validation and before CLI execution.")
            if application_snapshot is not None and sha256_file(application_snapshot.resolved_path) != application_snapshot.sha256:
                raise InputIntegrityError("The application dataset changed after validation and before CLI execution.")
            driver = self.driver_factory()
            cli_result = driver.run(
                plan,
                workspace=paths.workspace,
                capture_directory=paths.wrapper,
                cancellation_event=control.cancellation,
                process_started=lambda pid, create_time: self._mark_running(paths, control, pid, create_time),
                trace_metadata={
                    "geochemistrypi_mcp_version": SERVER_VERSION,
                    "geochemistrypi_cli_version": cli_version,
                },
            )
            if control.cancellation.is_set():
                raise CliRunCancelledError(
                    "The run was cancelled before its result was published.",
                    paths.workspace,
                )
            with self._lock:
                status = _read_json(paths.status)
                status.update(
                    {
                        "stage": "indexing_outputs",
                        "progress_message": "The CLI finished; original outputs are being indexed without recalculation.",
                    }
                )
                self._write_status(paths, status)
            final_hash = sha256_file(snapshot.resolved_path)
            if final_hash != snapshot.sha256:
                raise InputIntegrityError("The input dataset changed during CLI execution; the result was not published as valid.")
            application_hash_verified = None
            if application_snapshot is not None:
                application_hash_verified = sha256_file(application_snapshot.resolved_path) == application_snapshot.sha256
                if not application_hash_verified:
                    raise InputIntegrityError("The application dataset changed during CLI execution; the result was not published as valid.")
            output_directory = paths.workspace / "geopi_output" / request.experiment_name / request.run_name
            source_row_pairing = (
                verify_original_row_pairing(
                    snapshot.resolved_path,
                    output_directory,
                    request.identifier_column,
                    snapshot.row_lineage,
                )
                if plan.requires_source_row_pairing
                else None
            )
            discovered = discover_artifacts(output_directory, self.settings.maximum_artifact_references)
            preprocessing_summary = (
                read_time_series_preprocessing_summary(
                    output_directory,
                    source_row_count=snapshot.row_lineage.source_row_count,
                    indexed_relative_paths=tuple(entry["relative_path"] for entry in discovered.all_index_entries),
                )
                if request.task == "time_series"
                else None
            )
            is_aggregate = request.task != "time_series" and request.model_selection.mode == "all"
            aggregate_state, children = _aggregate_children(output_directory, request.task) if is_aggregate else (None, ())
            result_state = "partial_failure" if aggregate_state == "partial_failure" else "succeeded"
            artifact_index = {
                "schema_version": ARTIFACT_INDEX_SCHEMA_VERSION,
                "run_id": run_id,
                "output_directory": str(output_directory),
                "source_row_lineage": snapshot.row_lineage.as_record(),
                "source_row_pairing": source_row_pairing,
                "artifacts": discovered.all_index_entries,
            }
            _atomic_write_json(paths.artifact_index, artifact_index)
            result = RunResultResponse(
                run_id=run_id,
                state=result_state,
                task=request.task,
                model=("subaerial_proportion_bootstrap" if request.task == "time_series" else "all_models" if is_aggregate else request.model.type),
                tuning=(request.model_selection.tuning if is_aggregate else getattr(request, "tuning", "not_applicable")),
                output_directory=str(output_directory),
                interaction_trace=str(cli_result.trace_path),
                cli_stdout_log=str(cli_result.stdout_path),
                cli_stderr_log=str(cli_result.stderr_path),
                cli_exit_code=cli_result.returncode,
                cli_version=cli_version,
                input_sha256=snapshot.sha256,
                input_hash_verified=True,
                source_row_count=snapshot.row_lineage.source_row_count,
                row_identity_scheme=snapshot.row_lineage.scheme,
                row_identity_sha256=snapshot.row_lineage.ordered_identity_sha256,
                source_row_pairing_verified=(source_row_pairing["verified"] if source_row_pairing else None),
                source_row_pairing_sha256=(source_row_pairing["ordered_pairing_sha256"] if source_row_pairing else None),
                preprocessing_summary=preprocessing_summary,
                application_input_sha256=(application_snapshot.sha256 if application_snapshot is not None else None),
                application_input_hash_verified=application_hash_verified,
                reported_metrics=discovered.reported_metrics,
                artifact_count=len(discovered.all_index_entries),
                artifacts=discovered.response_references,
                artifacts_truncated=discovered.truncated,
                aggregate_state=aggregate_state,
                aggregate_summary=(
                    {
                        "expected_model_count": len(children),
                        "succeeded_count": sum(child["state"] == "succeeded" for child in children),
                        "failed_count": sum(child["state"] == "failed" for child in children),
                    }
                    if is_aggregate
                    else None
                ),
                children=children,
                limitations=(
                    "Metrics are read from original CLI files and are not recalculated by MCP.",
                    "The MCP wrapper does not recreate models, metrics, predictions, plots, or transformed datasets.",
                    {
                        "classification": "The public sample-balancing helper remains unwired in the CLI workflow.",
                        "regression": (
                            "Multiple-target regression exposes per-target holdout metrics, while cross-validation metrics remain uniformly "
                            "averaged across targets; univariate feature selection is rejected for this branch."
                        ),
                        "clustering": "Application inference, targets, unsupervised AutoML, and internal-only OPTICS remain outside the public workflow.",
                        "decomposition": "Application inference, targets, unsupervised AutoML, and unresolved missing values remain outside the public workflow.",
                        "anomaly_detection": "Application inference, targets, unsupervised AutoML, feature selection, and unresolved missing values remain outside the public workflow.",
                        "time_series": "The Time Series workflow uses the validated CLI numerical implementation and does not perform model inference.",
                    }[request.task],
                ),
            )
            _atomic_write_json(paths.result, result.model_dump(mode="json"))
            self._finish(
                paths,
                result_state,
                "The existing GeochemistryPi CLI completed and its original outputs were indexed."
                if result_state == "succeeded"
                else "The all-models CLI run completed with isolated child failures; successful and failed children were indexed.",
            )
        except CliRunCancelledError:
            self._finish(
                paths,
                "cancelled",
                "The recorded GeochemistryPi CLI process tree was cancelled.",
            )
        except Exception as exc:
            self._finish(
                paths,
                "failed",
                "The GeochemistryPi CLI run failed; inspect the bounded error and wrapper logs.",
                _safe_error(exc),
            )
        finally:
            with self._lock:
                self._active.pop(run_id, None)

    def get_status(self, run_id: str) -> RunStatusResponse:
        self._ensure_initialized()
        paths = self._paths(run_id)
        return self._status_response(_read_json(paths.status))

    def get_result(self, run_id: str) -> RunResultResponse:
        self._ensure_initialized()
        paths = self._paths(run_id)
        status = _read_json(paths.status)
        if status.get("state") not in {"succeeded", "partial_failure"}:
            raise RunStateError(f"Run {run_id} is {status.get('state')}; a result is available only after it succeeds or completes with partial failures.")
        return RunResultResponse.model_validate(_read_json(paths.result))

    def cancel(self, run_id: str) -> CancelRunResponse:
        self._ensure_initialized()
        paths = self._paths(run_id)
        with self._lock:
            status = _read_json(paths.status)
            state = status.get("state")
            if state in _TERMINAL_STATES:
                raise RunStateError(f"Run {run_id} is already {state} and cannot be cancelled.")
            control = self._active.get(run_id)
            if control is None:
                raise RunStateError(f"Run {run_id} has no live process handle and will not be terminated by PID alone.")
            control.cancellation.set()
            if state == "queued" and control.future is not None and control.future.cancel():
                self._active.pop(run_id, None)
                self._finish(
                    paths,
                    "cancelled",
                    "The queued run was cancelled before the CLI process started.",
                )
                return CancelRunResponse(
                    run_id=run_id,
                    state="cancelled",
                    message="No CLI process was started.",
                )
            status["progress_message"] = "Cancellation requested for this run's recorded CLI process tree."
            self._write_status(paths, status)
            return CancelRunResponse(
                run_id=run_id,
                state="cancellation_requested",
                message="Cancellation is in progress; poll get_run_status until the state becomes cancelled.",
            )

    def close(self) -> None:
        """Cancel live owned runs and wait for their drivers to release subprocesses."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            active = list(self._active.items())
            for _, control in active:
                control.cancellation.set()
        self._executor.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            for run_id, _ in active:
                paths = RunPaths.create(self.settings.runs_root, run_id)
                if not paths.status.is_file():
                    continue
                status = _read_json(paths.status)
                if status.get("state") not in _TERMINAL_STATES:
                    self._finish(
                        paths,
                        "cancelled",
                        "The run was cancelled because the MCP server stopped.",
                    )
            self._active.clear()
