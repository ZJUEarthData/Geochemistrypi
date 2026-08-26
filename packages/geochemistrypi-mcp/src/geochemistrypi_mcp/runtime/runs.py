"""Durable, non-blocking lifecycle control for local CLI subprocess runs."""

import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..api.schemas import (
    AnalysisRequest,
    AnalysisValidationResponse,
    AnomalyDetectionRequest,
    ArtifactReference,
    CancelRunResponse,
    ClassificationRequest,
    ClusteringRequest,
    DatasetInspectionRequest,
    DatasetPreparationContract,
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
from ..data.headers import source_allows_pandas_duplicate_mangling
from ..data.inspector import DatasetSnapshot
from ..data.inspector import inspect_dataset as inspect_local_dataset
from ..data.inspector import sha256_file, snapshot_dataset
from ..data.preparation import DatasetPreparationError, PreparedDataset, prepare_dataset_view
from ..data.row_pairing import verify_original_row_pairing
from ..planning.interaction_plan import AnalysisPlanCompiler, DatasetCompilationContext, InteractionPlan, PlanCompilationError
from ..planning.scientific_contract import assess_scientific_compatibility, canonical_scientific_contract, canonical_sha256, planned_artifact_requirements, resolved_environment_profile
from ..tracking.experiments import ExperimentManager
from .artifacts import discover_artifacts, read_time_series_preprocessing_summary
from .cli_capabilities import probe_cli_capabilities
from .cli_driver import CliInteractionDriver, CliRunCancelledError, validate_workspace_path
from .environment import EnvironmentSnapshot, inspect_cli_environment

_RUN_ID = re.compile(r"^run-[0-9a-f]{16}$")
_VALIDATION_ID = re.compile(r"^val-[0-9a-f]{32}$")
_TERMINAL_STATES = {"succeeded", "partial_failure", "failed", "cancelled"}
_ATOMIC_REPLACE_RETRY_DELAYS_SECONDS = (0.01, 0.02, 0.04)
_VALIDATION_TTL_SECONDS = 1800
_VALIDATION_RECEIPT_FIELDS = {
    "schema_version",
    "validation_id",
    "request_hash",
    "canonical_contract",
    "canonical_contract_hash",
    "compiled_plan_hash",
    "dataset_hash",
    "adapter_identity",
    "artifact_requirements",
    "execution_readiness",
    "request",
    "task",
    "created_at_epoch",
    "expires_at_epoch",
    "training",
    "application",
    "cli",
    "interaction_plan",
    "environment",
    "integrity_hmac_sha256",
}


def _selected_models(request: Any) -> tuple[str, ...]:
    if request.task == "time_series":
        return (
            {
                "subaerial_proportion": "subaerial_proportion_bootstrap",
                "continuous": "spatiotemporal_weighted_continuous_bootstrap",
                "element_mean": "element_mean",
                "reference_anomaly_series": "reference_label_event_overlay",
            }[request.mode],
        )
    if request.task == "decomposition" and request.mode == "embedding_label_overlay":
        return ("embedding_label_overlay",)
    orders = {
        "classification": CLASSIFICATION_MODEL_ORDER,
        "regression": REGRESSION_MODEL_ORDER,
        "clustering": CLUSTERING_MODEL_ORDER,
        "decomposition": DECOMPOSITION_MODEL_ORDER,
        "anomaly_detection": ANOMALY_DETECTION_MODEL_ORDER,
    }
    return orders[request.task] if request.model_selection.mode == "all" else (request.model.type,)


def _selected_tuning(request: Any) -> str:
    if request.task == "time_series" or (request.task == "decomposition" and request.mode == "embedding_label_overlay"):
        return "not_applicable"
    if request.model_selection.mode == "all":
        return request.model_selection.tuning
    return getattr(request, "tuning", "not_applicable")


def _resolved_model_parameters(request: Any) -> dict[str, Any]:
    if request.task == "time_series":
        return {
            "mode": request.mode,
            "bin_width": request.bin_width,
            **(
                {
                    "iterations": request.iterations,
                    "seed": request.seed,
                    "age_unit": request.age_unit,
                    "fit_curve": request.fit_curve,
                }
                if request.mode == "subaerial_proportion"
                else {
                    "aggregation": request.aggregation,
                    "uncertainty": request.uncertainty,
                    "minimum_samples_per_bin": request.minimum_samples_per_bin,
                    "filter_minimum": request.filter_minimum,
                    "filter_maximum": request.filter_maximum,
                }
            ),
        }
    if request.task == "decomposition" and request.mode == "embedding_label_overlay":
        return {
            "mode": request.mode,
            "join_policy": "exact_identifier_set_one_to_one",
            "positive_label_values": list(request.positive_label_values),
        }
    if request.model_selection.mode == "all" or _selected_tuning(request) == "automl":
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


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _validation_request_value(request: Any) -> dict[str, Any]:
    """Serialize a strict request without reintroducing mutually exclusive defaults."""

    value = request.model_dump(mode="json")

    def normalize(item: Any) -> None:
        if isinstance(item, dict):
            if item.get("header_row_indices"):
                item.pop("header_row_index", None)
            for child in item.values():
                normalize(child)
        elif isinstance(item, list):
            for child in item:
                normalize(child)

    normalize(value)
    return value


def _plan_identity(plan: InteractionPlan) -> dict[str, Any]:
    value = asdict(plan)
    return {
        "name": plan.name,
        "schema_version": plan.schema_version,
        "environment_profile_id": plan.environment_profile_id,
        "environment_profile_identity_sha256": plan.environment_profile_identity_sha256,
        "sha256": _json_sha256(value),
    }


def _materialize_tracking_root(
    plan: InteractionPlan,
    tracking_root: Path | None,
) -> InteractionPlan:
    """Bind installer-owned runtime paths before a plan receives an identity."""

    if "data-mining" not in plan.public_command:
        return plan
    if tracking_root is None:
        raise RunStateError("The installer-owned MLflow tracking root is not configured.")
    if "--tracking-root" in plan.public_command:
        return plan
    return replace(
        plan,
        public_command=(
            *plan.public_command,
            "--tracking-root",
            str(tracking_root),
        ),
    )


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
    provenance_manifest: Path

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
            provenance_manifest=wrapper / "scientific-run-manifest.json",
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
        environment_resolver: Callable[[Path], EnvironmentSnapshot] | None = None,
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
        self.environment_resolver = environment_resolver or inspect_cli_environment
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

    def _prepare_dataset(
        self,
        request: Any,
        role: str,
        resolved: ResolvedDataset,
        source_snapshot: DatasetSnapshot,
    ) -> PreparedDataset:
        reference = getattr(request, f"{role}_dataset", None)
        if role == "application" and reference is None:
            reference, _, _ = self._secondary_dataset(request)
        contract = reference.preparation if reference is not None else DatasetPreparationContract()
        requested_sheet = getattr(request, "sheet", "0")
        if request.task == "time_series" and role == "training" and source_snapshot.resolved_path.suffix.lower() == ".xlsx" and (requested_sheet != "0" or contract.worksheet is None):
            try:
                from openpyxl import load_workbook

                with source_snapshot.resolved_path.open("rb") as stream:
                    workbook = load_workbook(stream, read_only=True, data_only=True)
                    try:
                        if str(requested_sheet).isdigit():
                            sheet_index = int(requested_sheet)
                            if sheet_index >= len(workbook.sheetnames):
                                raise RunStateError(f"Excel sheet index {sheet_index} is out of range.")
                            resolved_sheet = workbook.sheetnames[sheet_index]
                        else:
                            resolved_sheet = str(requested_sheet)
                            if resolved_sheet not in workbook.sheetnames:
                                raise RunStateError(f"Excel sheet {resolved_sheet!r} does not exist.")
                    finally:
                        workbook.close()
            except OSError as exc:
                raise RunStateError(f"Unable to resolve Time Series worksheet {requested_sheet!r}.") from exc
            if contract.worksheet is not None and contract.worksheet != resolved_sheet:
                raise RunStateError("Time Series sheet conflicts with training_dataset.preparation.worksheet.")
            contract = contract.model_copy(update={"worksheet": resolved_sheet})
        state_root = self.settings.service_state_root
        if state_root is None:
            raise RunStateError("The MCP service-state root is not configured.")
        try:
            return prepare_dataset_view(
                source_snapshot,
                contract,
                state_root,
                self.settings.maximum_dataset_bytes,
                self.settings.maximum_columns,
                allow_pandas_duplicate_mangling=source_allows_pandas_duplicate_mangling(resolved.source),
            )
        except DatasetPreparationError as exc:
            raise PlanCompilationError(str(exc)) from exc

    @staticmethod
    def _secondary_dataset(request: Any) -> tuple[Any | None, Path | None, bool]:
        """Resolve ordinary application or nested external-evaluation input."""

        application_reference = getattr(request, "application_dataset", None)
        application_path = getattr(request, "application_dataset_path", None)
        if application_reference is not None or application_path is not None:
            return application_reference, application_path, False
        evaluation = getattr(request, "evaluation", None)
        if evaluation is not None and evaluation.mode == "external_labeled":
            return (
                evaluation.evaluation_dataset,
                evaluation.evaluation_dataset_path,
                True,
            )
        return None, None, False

    @staticmethod
    def _execution_dataset_updates(
        request: Any,
        training_path: Path,
        secondary_path: Path | None,
        secondary_is_evaluation: bool,
    ) -> dict[str, Any]:
        updates: dict[str, Any] = {
            "training_dataset_path": training_path,
            "training_dataset": None,
        }
        if secondary_is_evaluation:
            updates["evaluation"] = request.evaluation.model_copy(
                update={
                    "evaluation_dataset_path": secondary_path,
                    "evaluation_dataset": None,
                }
            )
        elif hasattr(request, "application_dataset_path"):
            updates.update(
                {
                    "application_dataset_path": secondary_path,
                    "application_dataset": None,
                }
            )
        return updates

    @staticmethod
    def _prepared_identity(prepared: PreparedDataset, resolved: ResolvedDataset) -> dict[str, Any]:
        snapshot = prepared.snapshot
        return {
            "resolved_path": str(snapshot.resolved_path),
            "size_bytes": snapshot.size_bytes,
            "sha256": snapshot.sha256,
            "source": resolved.source,
            "dataset_id": resolved.dataset_id,
            "source_file": prepared.record["source_file"],
            "preparation": prepared.record,
        }

    def _validation_secret(self) -> bytes:
        state_root = self.settings.service_state_root
        if state_root is None:  # McpSettings normalizes this in __post_init__.
            raise RunStateError("The MCP validation state root is not configured.")
        secret_path = state_root / "validation-receipt.key"
        secret_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                secret_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
                0o600,
            )
        except FileExistsError:
            descriptor = None
        if descriptor is not None:
            try:
                secret = secrets.token_bytes(32)
                remaining = memoryview(secret)
                while remaining:
                    written = os.write(descriptor, remaining)
                    if written <= 0:
                        raise OSError("Validation receipt integrity key write made no progress.")
                    remaining = remaining[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        try:
            secret = secret_path.read_bytes()
        except OSError as exc:
            raise RunStateError("The validation receipt integrity key is unavailable.") from exc
        if len(secret) != 32:
            raise RunStateError("The validation receipt integrity key is corrupt.")
        return secret

    def _receipt_integrity(self, receipt_without_integrity: dict[str, Any]) -> str:
        return hmac.new(
            self._validation_secret(),
            _canonical_json_bytes(receipt_without_integrity),
            hashlib.sha256,
        ).hexdigest()

    def _write_validation_receipt(
        self,
        request: Any,
        prepared_training: PreparedDataset,
        resolved_training: ResolvedDataset,
        prepared_application: PreparedDataset | None,
        resolved_application: ResolvedDataset | None,
        cli_executable: Path,
        cli_version: str,
        plan: InteractionPlan,
        environment_snapshot: EnvironmentSnapshot,
    ) -> dict[str, Any]:
        request_value = _validation_request_value(request)
        request_hash = _json_sha256(request_value)
        training = self._prepared_identity(prepared_training, resolved_training)
        application = self._prepared_identity(prepared_application, resolved_application) if prepared_application is not None and resolved_application is not None else None
        cli = {
            "executable": str(cli_executable.resolve()),
            "executable_sha256": sha256_file(cli_executable),
            "version": cli_version,
        }
        plan_identity = _plan_identity(plan)
        artifact_requirements = planned_artifact_requirements(request, plan)
        canonical_contract = canonical_scientific_contract(request, plan)
        canonical_contract_hash = canonical_sha256(canonical_contract)
        assessment = assess_scientific_compatibility(
            request,
            plan,
            artifact_requirements,
            environment_snapshot,
        )
        environment = {
            "identity_sha256": environment_snapshot.identity_sha256,
            "record": environment_snapshot.record,
        }
        stable_identity = {
            "schema_version": 1,
            "request_hash": request_hash,
            "request": request_value,
            "task": request.task,
            "training": training,
            "application": application,
            "cli": cli,
            "interaction_plan": plan_identity,
            "canonical_contract_hash": canonical_contract_hash,
            "environment": environment,
        }
        validation_id = f"val-{_json_sha256(stable_identity)[:32]}"
        created_at_epoch = int(time.time())
        receipt_without_integrity = {
            "schema_version": 1,
            "validation_id": validation_id,
            "request_hash": request_hash,
            "canonical_contract": canonical_contract,
            "canonical_contract_hash": canonical_contract_hash,
            "compiled_plan_hash": plan_identity["sha256"],
            "dataset_hash": training["sha256"],
            "adapter_identity": {
                "id": plan.adapter_id,
                "version": plan.adapter_version,
                "status": assessment.adapter_status,
            },
            "artifact_requirements": [requirement.model_dump(mode="json") for requirement in artifact_requirements],
            "execution_readiness": {
                "execution_ready": assessment.execution_ready,
                "comparison_ready": assessment.comparison_ready,
                "claim_ready": assessment.claim_ready,
                "scientific_status": assessment.scientific_status,
                "adapter_status": assessment.adapter_status,
                "artifact_status": assessment.artifact_status,
                "environment_status": assessment.environment_status,
                "blocking_issues": list(assessment.blocking_issues),
            },
            "request": request_value,
            "task": request.task,
            "created_at_epoch": created_at_epoch,
            "expires_at_epoch": created_at_epoch + _VALIDATION_TTL_SECONDS,
            "training": training,
            "application": application,
            "cli": cli,
            "interaction_plan": plan_identity,
            "environment": environment,
        }
        receipt = {
            **receipt_without_integrity,
            "integrity_hmac_sha256": self._receipt_integrity(receipt_without_integrity),
        }
        state_root = self.settings.service_state_root
        if state_root is None:
            raise RunStateError("The MCP validation state root is not configured.")
        _atomic_write_json(state_root / "validations" / f"{validation_id}.json", receipt)
        return receipt

    def _load_validation_receipt(
        self,
        validation_id: str,
        request_hash: str,
        *,
        expected_task: str | None,
    ) -> tuple[Any, dict[str, Any]]:
        if not _VALIDATION_ID.fullmatch(validation_id):
            raise RunStateError("Invalid validation ID.")
        if not re.fullmatch(r"[0-9a-f]{64}", request_hash):
            raise RunStateError("Invalid validation request hash.")
        state_root = self.settings.service_state_root
        if state_root is None:
            raise RunStateError("The MCP validation state root is not configured.")
        receipt_path = state_root / "validations" / f"{validation_id}.json"
        try:
            receipt = _read_json(receipt_path)
        except RunStateError as exc:
            raise RunStateError(f"Validation receipt does not exist or is unreadable: {validation_id}") from exc
        if set(receipt) != _VALIDATION_RECEIPT_FIELDS:
            raise RunStateError("Validation receipt integrity check failed: unknown or missing fields.")
        recorded_integrity = receipt.pop("integrity_hmac_sha256")
        if not isinstance(recorded_integrity, str) or not hmac.compare_digest(
            recorded_integrity,
            self._receipt_integrity(receipt),
        ):
            raise RunStateError("Validation receipt integrity check failed.")
        receipt["integrity_hmac_sha256"] = recorded_integrity
        if receipt["schema_version"] != 1 or receipt["validation_id"] != validation_id:
            raise RunStateError("Validation receipt identity is invalid.")
        if not hmac.compare_digest(str(receipt["request_hash"]), request_hash):
            raise RunStateError("Validation request hash does not match the receipt.")
        if not isinstance(receipt["expires_at_epoch"], int) or time.time() > receipt["expires_at_epoch"]:
            raise RunStateError("Validation receipt expired; call validate_analysis again.")
        if expected_task is not None and receipt["task"] != expected_task:
            raise RunStateError(f"Validation receipt task must be {expected_task!r} in this scoped session.")
        if _json_sha256(receipt["request"]) != request_hash:
            raise RunStateError("Validation request integrity check failed.")
        if canonical_sha256(receipt["canonical_contract"]) != receipt["canonical_contract_hash"]:
            raise RunStateError("Validation canonical scientific contract integrity check failed.")
        try:
            request = AnalysisRequest.model_validate(receipt["request"]).root
        except Exception as exc:
            raise RunStateError("Validated analysis request is no longer readable by the strict protocol.") from exc
        if request.task != receipt["task"]:
            raise RunStateError("Validation receipt task identity is invalid.")
        return request, receipt

    def _assert_validation_still_matches(
        self,
        receipt: dict[str, Any],
        prepared_training: PreparedDataset,
        resolved_training: ResolvedDataset,
        prepared_application: PreparedDataset | None,
        resolved_application: ResolvedDataset | None,
        cli_executable: Path,
        cli_version: str,
        plan: InteractionPlan,
        environment_snapshot: EnvironmentSnapshot,
    ) -> None:
        current_training = self._prepared_identity(prepared_training, resolved_training)
        if receipt["training"] != current_training:
            raise RunStateError("The training dataset changed since validation; validate the request again.")
        current_application = self._prepared_identity(prepared_application, resolved_application) if prepared_application is not None and resolved_application is not None else None
        if receipt["application"] != current_application:
            raise RunStateError("The application dataset changed since validation; validate the request again.")
        current_cli = {
            "executable": str(cli_executable.resolve()),
            "executable_sha256": sha256_file(cli_executable),
            "version": cli_version,
        }
        if receipt["cli"] != current_cli:
            raise RunStateError("The configured GeochemistryPi CLI changed since validation; validate the request again.")
        if receipt["interaction_plan"] != _plan_identity(plan):
            raise RunStateError("The compiled interaction plan changed since validation; validate the request again.")
        current_environment = {
            "identity_sha256": environment_snapshot.identity_sha256,
            "record": environment_snapshot.record,
        }
        if receipt["environment"] != current_environment:
            raise RunStateError("The isolated CLI environment changed since validation; validate the request again.")

    def validate(
        self,
        request: ClassificationRequest | RegressionRequest | ClusteringRequest | DecompositionRequest | AnomalyDetectionRequest | TimeSeriesRequest,
    ) -> AnalysisValidationResponse:
        """Resolve and compile an analysis without creating a run or CLI process."""
        cli_executable, cli_version = self.cli_resolver()
        environment_snapshot = self.environment_resolver(cli_executable)
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
        source_snapshot = snapshot_dataset(resolved_training.path, self.settings.maximum_dataset_bytes)
        if resolved_training.expected_sha256 is not None and source_snapshot.sha256 != resolved_training.expected_sha256:
            raise InputIntegrityError("The training dataset changed between source resolution and validation.")
        prepared_training = self._prepare_dataset(request, "training", resolved_training, source_snapshot)
        snapshot = prepared_training.snapshot
        application_reference, application_path, secondary_is_evaluation = self._secondary_dataset(request)
        if application_reference is not None:
            resolved_application = self.dataset_catalog.resolve(application_reference, task=request.task, role="application")
        else:
            resolved_application = ResolvedDataset(path=application_path, expected_sha256=None, dataset_id=None, source="path") if application_path is not None else None
        application_source_snapshot = snapshot_dataset(resolved_application.path, self.settings.maximum_dataset_bytes) if resolved_application is not None else None
        expected_application_sha256 = resolved_application.expected_sha256 if resolved_application is not None else None
        if application_source_snapshot is not None and expected_application_sha256 is not None:
            if application_source_snapshot.sha256 != expected_application_sha256:
                raise InputIntegrityError("The application dataset changed between source resolution and validation.")
        prepared_application = (
            self._prepare_dataset(request, "application", resolved_application, application_source_snapshot) if resolved_application is not None and application_source_snapshot is not None else None
        )
        application_snapshot = prepared_application.snapshot if prepared_application is not None else None
        execution_request = request.model_copy(
            update=self._execution_dataset_updates(
                request,
                snapshot.resolved_path,
                application_snapshot.resolved_path if application_snapshot else None,
                secondary_is_evaluation,
            )
        )
        dataset_context = DatasetCompilationContext(
            training_source=(resolved_training.source if snapshot.resolved_path == source_snapshot.resolved_path else "path"),
            application_source=(
                resolved_application.source
                if resolved_application is not None
                and application_source_snapshot is not None
                and application_snapshot is not None
                and application_snapshot.resolved_path == application_source_snapshot.resolved_path
                else "path"
                if resolved_application is not None
                else None
            ),
        )
        plan = self.plan_compiler.compile(
            execution_request,
            cli_executable=cli_executable,
            dataset_context=dataset_context,
        )
        plan = AnalysisPlanCompiler.bind_scientific_adapter(plan, execution_request)
        capability_probe = probe_cli_capabilities(
            cli_executable,
            plan.required_cli_capabilities,
        )
        if capability_probe.missing:
            missing = ", ".join(capability_probe.missing)
            plan = replace(
                plan,
                adapter_status="unavailable",
                execution_ready=False,
                blocking_issues=tuple(
                    dict.fromkeys(
                        (
                            *plan.blocking_issues,
                            f"The configured CLI is missing required public capabilities: {missing}.",
                        )
                    )
                ),
            )
        plan = _materialize_tracking_root(plan, self.settings.tracking_root)
        artifact_requirements = planned_artifact_requirements(request, plan)
        assessment = assess_scientific_compatibility(request, plan, artifact_requirements, environment_snapshot)
        if plan.execution_ready:
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
        if application_reference is None and application_path is None and request.task in {"classification", "regression"}:
            warnings.append("No application dataset was selected, so model inference will be skipped.")
        if getattr(request, "world_map", None) is not None and request.world_map.enabled:
            warnings.append("World-map rendering is enabled and may add platform-dependent image artifacts.")
        if existing_experiment_id:
            warnings.append("The new run will be attached to the verified existing MLflow experiment ID.")
        if not assessment.execution_ready:
            warnings.append("The request is scientifically representable but the exact validated contract is not execution-ready.")
        receipt = self._write_validation_receipt(
            request,
            prepared_training,
            resolved_training,
            prepared_application,
            resolved_application,
            cli_executable,
            cli_version,
            plan,
            environment_snapshot,
        )
        return AnalysisValidationResponse(
            validation_id=receipt["validation_id"],
            request_hash=receipt["request_hash"],
            canonical_contract_hash=receipt["canonical_contract_hash"],
            compiled_plan_hash=receipt["compiled_plan_hash"],
            validation_expires_at=datetime.fromtimestamp(
                receipt["expires_at_epoch"],
                timezone.utc,
            ).isoformat(),
            execution_ready=assessment.execution_ready,
            comparison_ready=assessment.comparison_ready,
            claim_ready=assessment.claim_ready,
            scientific_status=assessment.scientific_status,
            adapter_status=assessment.adapter_status,
            artifact_status=assessment.artifact_status,
            environment_status=assessment.environment_status,
            workflow_family=plan.workflow_family,
            workflow_mode=plan.workflow_mode,
            method=plan.method,
            scientific_contract_id=plan.scientific_contract_id,
            adapter_id=plan.adapter_id,
            adapter_version=plan.adapter_version,
            adapter_identity=(f"{plan.adapter_id}@{plan.adapter_version}" if plan.adapter_id and plan.adapter_version else plan.adapter_id),
            artifact_requirements=artifact_requirements,
            blocking_issues=assessment.blocking_issues,
            task=request.task,
            models=models,
            estimated_model_count=len(models),
            tuning=tuning,
            training_source=resolved_training.source,
            training_dataset_path=str(snapshot.resolved_path),
            training_sha256=snapshot.sha256,
            training_size_bytes=snapshot.size_bytes,
            source_dataset_path=str(source_snapshot.resolved_path),
            source_dataset_sha256=source_snapshot.sha256,
            dataset_preparation=prepared_training.record,
            dataset_preparation_sha256=prepared_training.record["contract_hash"],
            environment_identity_sha256=environment_snapshot.identity_sha256,
            environment_profile={
                "requested": resolved_environment_profile(request),
                "observed": environment_snapshot.record,
            },
            environment_profile_identity_sha256=plan.environment_profile_identity_sha256,
            requested_seeds=dict(plan.requested_seeds),
            effective_seeds=dict(plan.effective_seeds),
            parameter_binding={
                "requested_model": {name: json.loads(value) for name, value in plan.requested_model_parameters},
                "effective_model": {name: json.loads(value) for name, value in plan.effective_model_parameters},
                "model_binding": plan.model_parameter_binding,
                "requested_preprocessing": {name: json.loads(value) for name, value in plan.requested_preprocessing_parameters},
                "effective_preprocessing": {name: json.loads(value) for name, value in plan.effective_preprocessing_parameters},
                "preprocessing_binding": plan.preprocessing_parameter_binding,
            },
            adapter_artifact_mappings=tuple(
                {
                    "mapping_id": mapping.mapping_id,
                    "scientific_type": mapping.scientific_type,
                    "output_role": mapping.output_role,
                    "relative_path": mapping.relative_path,
                    "availability": mapping.availability,
                    "reason": mapping.reason,
                }
                for mapping in plan.artifact_mappings
            ),
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
            application_source_sha256=(application_source_snapshot.sha256 if application_source_snapshot else None),
            application_preparation=(prepared_application.record if prepared_application else None),
            application_source_row_count=(application_snapshot.row_lineage.source_row_count if application_snapshot else None),
            application_row_identity_sha256=(application_snapshot.row_lineage.ordered_identity_sha256 if application_snapshot else None),
            experiment_mode=("not_applicable" if request.task == "time_series" else "existing" if existing_experiment_id else "new"),
            experiment_name=request.experiment_name,
            existing_experiment_id=existing_experiment_id,
            interaction_plan=plan.name,
            warnings=tuple(warnings),
        )

    def start_validated(
        self,
        validation_id: str,
        request_hash: str,
        *,
        expected_task: str | None = None,
    ) -> StartAnalysisResponse:
        """Start only the immutable request and inputs covered by a validation receipt."""
        request, receipt = self._load_validation_receipt(
            validation_id,
            request_hash,
            expected_task=expected_task,
        )
        return self._start(request, validation=receipt)

    def start(
        self,
        request: ClassificationRequest | RegressionRequest | ClusteringRequest | DecompositionRequest | AnomalyDetectionRequest | TimeSeriesRequest,
    ) -> StartAnalysisResponse:
        """Retain the strict legacy full-request start path for compatibility."""
        return self._start(request, validation=None)

    def _start(
        self,
        request: ClassificationRequest | RegressionRequest | ClusteringRequest | DecompositionRequest | AnomalyDetectionRequest | TimeSeriesRequest,
        *,
        validation: dict[str, Any] | None,
    ) -> StartAnalysisResponse:
        """Validate synchronously, then queue the long-running CLI work."""
        self._ensure_initialized()
        with self._lock:
            if self._closed:
                raise RunStateError("The GeochemistryPi run manager is shutting down.")
        cli_executable, cli_version = self.cli_resolver()
        environment_snapshot = self.environment_resolver(cli_executable)
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
        source_snapshot = snapshot_dataset(resolved_training.path, self.settings.maximum_dataset_bytes)
        if resolved_training.expected_sha256 is not None and source_snapshot.sha256 != resolved_training.expected_sha256:
            raise InputIntegrityError("The training dataset changed between source resolution and validation.")
        prepared_training = self._prepare_dataset(request, "training", resolved_training, source_snapshot)
        snapshot = prepared_training.snapshot
        application_reference, application_dataset_path, secondary_is_evaluation = self._secondary_dataset(request)
        if application_reference is not None:
            resolved_application = self.dataset_catalog.resolve(
                application_reference,
                task=request.task,
                role="application",
            )
            application_dataset_path = resolved_application.path
        else:
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
        application_source_snapshot = (
            snapshot_dataset(
                application_dataset_path,
                self.settings.maximum_dataset_bytes,
            )
            if application_dataset_path is not None
            else None
        )
        expected_application_sha256 = resolved_application.expected_sha256 if resolved_application is not None else None
        if application_source_snapshot is not None and expected_application_sha256 is not None:
            if application_source_snapshot.sha256 != expected_application_sha256:
                raise InputIntegrityError("The application dataset changed between source resolution and validation.")
        prepared_application = (
            self._prepare_dataset(request, "application", resolved_application, application_source_snapshot) if resolved_application is not None and application_source_snapshot is not None else None
        )
        application_snapshot = prepared_application.snapshot if prepared_application is not None else None
        execution_request = request.model_copy(
            update=self._execution_dataset_updates(
                request,
                snapshot.resolved_path,
                application_snapshot.resolved_path if application_snapshot else None,
                secondary_is_evaluation,
            )
        )
        dataset_context = DatasetCompilationContext(
            training_source=(resolved_training.source if snapshot.resolved_path == source_snapshot.resolved_path else "path"),
            application_source=(
                resolved_application.source
                if resolved_application is not None
                and application_source_snapshot is not None
                and application_snapshot is not None
                and application_snapshot.resolved_path == application_source_snapshot.resolved_path
                else "path"
                if resolved_application is not None
                else None
            ),
        )
        plan = self.plan_compiler.compile(
            execution_request,
            cli_executable=cli_executable,
            dataset_context=dataset_context,
        )
        plan = AnalysisPlanCompiler.bind_scientific_adapter(plan, execution_request)
        capability_probe = probe_cli_capabilities(
            cli_executable,
            plan.required_cli_capabilities,
        )
        if capability_probe.missing:
            missing = ", ".join(capability_probe.missing)
            plan = replace(
                plan,
                adapter_status="unavailable",
                execution_ready=False,
                blocking_issues=tuple(
                    dict.fromkeys(
                        (
                            *plan.blocking_issues,
                            f"The configured CLI is missing required public capabilities: {missing}.",
                        )
                    )
                ),
            )
        plan = _materialize_tracking_root(plan, self.settings.tracking_root)
        if validation is not None:
            self._assert_validation_still_matches(
                validation,
                prepared_training,
                resolved_training,
                prepared_application,
                resolved_application,
                cli_executable,
                cli_version,
                plan,
                environment_snapshot,
            )
        artifact_requirements = planned_artifact_requirements(request, plan)
        assessment = assess_scientific_compatibility(request, plan, artifact_requirements, environment_snapshot)
        if not assessment.execution_ready:
            explanation = "; ".join(assessment.blocking_issues) or "The exact scientific contract has no executable CLI adapter."
            raise RunStateError(f"Validated scientific contract is not execution-ready: {explanation}")
        if "data-mining" in plan.public_command:
            assert self.settings.tracking_root is not None
            self.settings.tracking_root.mkdir(parents=True, exist_ok=True)
        run_id = _new_run_id()
        paths = RunPaths.create(self.settings.runs_root, run_id)
        validate_workspace_path(plan, paths.workspace)
        created_at = _utc_now()
        request_value = _validation_request_value(request)
        request_hash = _json_sha256(request_value)
        if validation is not None and not hmac.compare_digest(
            request_hash,
            str(validation["request_hash"]),
        ):
            raise RunStateError("Validated request serialization changed before run creation.")
        request_record = {
            "schema_version": 1,
            "run_id": run_id,
            "request_hash": request_hash,
            "canonical_contract_hash": canonical_sha256(canonical_scientific_contract(request, plan)),
            "request": request_value,
            "validation": (
                {
                    "validation_id": validation["validation_id"],
                    "request_hash": validation["request_hash"],
                }
                if validation is not None
                else None
            ),
            "input": {
                "source_path": str(snapshot.source_path),
                "resolved_path": str(snapshot.resolved_path),
                "size_bytes": snapshot.size_bytes,
                "sha256": snapshot.sha256,
                "format": snapshot.format,
                "source": resolved_training.source,
                "dataset_id": resolved_training.dataset_id,
                "row_identity": snapshot.row_lineage.as_record(),
                "source_file": prepared_training.record["source_file"],
                "preparation": prepared_training.record,
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
                    "source_file": prepared_application.record["source_file"],
                    "preparation": prepared_application.record,
                }
                if application_snapshot is not None
                else None
            ),
            "interaction_plan": {
                "name": plan.name,
                "schema_version": plan.schema_version,
                "sha256": _plan_identity(plan)["sha256"],
                "adapter_id": plan.adapter_id,
                "adapter_version": plan.adapter_version,
                "workflow_family": plan.workflow_family,
                "workflow_mode": plan.workflow_mode,
                "method": plan.method,
                "environment_profile_id": plan.environment_profile_id,
                "environment_profile_identity_sha256": plan.environment_profile_identity_sha256,
                "requested_seeds": dict(plan.requested_seeds),
                "effective_seeds": dict(plan.effective_seeds),
                "seed_binding": plan.seed_binding,
                "requested_model_parameters": {name: json.loads(value) for name, value in plan.requested_model_parameters},
                "effective_model_parameters": {name: json.loads(value) for name, value in plan.effective_model_parameters},
                "requested_preprocessing_parameters": {name: json.loads(value) for name, value in plan.requested_preprocessing_parameters},
                "effective_preprocessing_parameters": {name: json.loads(value) for name, value in plan.effective_preprocessing_parameters},
                "artifact_mappings": [
                    {
                        "mapping_id": mapping.mapping_id,
                        "scientific_type": mapping.scientific_type,
                        "output_role": mapping.output_role,
                        "relative_path": mapping.relative_path,
                        "availability": mapping.availability,
                        "reason": mapping.reason,
                    }
                    for mapping in plan.artifact_mappings
                ],
            },
            "versions": {
                "geochemistrypi_mcp": SERVER_VERSION,
                "geochemistrypi_cli": cli_version,
                "environment_identity_sha256": environment_snapshot.identity_sha256,
                "environment": environment_snapshot.record,
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
            status_hint=(f"Call get_run_result once with run_id {run_id} and wait_seconds up to 300; " "use get_run_status only when progress detail is needed."),
            request_hash=(validation["request_hash"] if validation is not None else None),
            started_from_validation=validation is not None,
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
            artifact_requirements = planned_artifact_requirements(request, plan)
            discovered = discover_artifacts(
                output_directory,
                self.settings.maximum_artifact_references,
                artifact_requirements,
                workflow_family=plan.workflow_family,
                artifact_mappings=plan.artifact_mappings,
            )
            preprocessing_summary = (
                read_time_series_preprocessing_summary(
                    output_directory,
                    source_row_count=snapshot.row_lineage.source_row_count,
                    indexed_relative_paths=tuple(entry["relative_path"] for entry in discovered.all_index_entries),
                )
                if request.task == "time_series" and request.mode in {"subaerial_proportion", "continuous"}
                else None
            )
            is_aggregate = request.task != "time_series" and request.model_selection.mode == "all"
            aggregate_state, children = _aggregate_children(output_directory, request.task) if is_aggregate else (None, ())
            result_state = "partial_failure" if aggregate_state == "partial_failure" else "succeeded"
            request_record = _read_json(paths.request)
            validation_identity = request_record["validation"] or {
                "mode": "legacy_full_request",
                "validation_id": None,
                "request_hash": request_record["request_hash"],
            }
            missing_artifact_requirement_ids = discovered.missing_requirement_ids
            artifact_index = {
                "schema_version": ARTIFACT_INDEX_SCHEMA_VERSION,
                "run_id": run_id,
                "request_identity": {
                    "request_hash": request_record["request_hash"],
                    "canonical_contract_hash": request_record["canonical_contract_hash"],
                },
                "validation_identity": validation_identity,
                "plan_identity": request_record["interaction_plan"],
                "output_directory": str(output_directory),
                "source_row_lineage": snapshot.row_lineage.as_record(),
                "source_row_pairing": source_row_pairing,
                "artifact_contract": {
                    "required": [item.model_dump(mode="json") for item in artifact_requirements],
                    "missing_required_ids": list(missing_artifact_requirement_ids),
                    "matches": {key: list(value) for key, value in discovered.requirement_matches.items()},
                    "failures": discovered.requirement_failures,
                },
                "artifacts": discovered.all_index_entries,
            }
            _atomic_write_json(paths.artifact_index, artifact_index)
            provenance_manifest = {
                "schema_version": 1,
                "request_identity": {
                    "request_hash": request_record["request_hash"],
                    "canonical_contract_hash": request_record["canonical_contract_hash"],
                },
                "validation_identity": validation_identity,
                "run_identity": {"run_id": run_id},
                "plan_identity": request_record["interaction_plan"],
                "runtime": request_record["versions"],
                "input": request_record["input"],
                "application_input": request_record["application_input"],
                "artifact_index": {
                    "path": str(paths.artifact_index),
                    "sha256": sha256_file(paths.artifact_index),
                },
                "artifacts": list(discovered.all_index_entries),
                "artifact_contract": artifact_index["artifact_contract"],
                "metadata": {
                    "task": request.task,
                    "workflow_family": plan.workflow_family,
                    "workflow_mode": plan.workflow_mode,
                    "method": plan.method,
                    "adapter_id": plan.adapter_id,
                    "adapter_version": plan.adapter_version,
                },
            }
            _atomic_write_json(paths.provenance_manifest, provenance_manifest)
            provenance_manifest_sha256 = sha256_file(paths.provenance_manifest)
            result = RunResultResponse(
                run_id=run_id,
                request_hash=request_record["request_hash"],
                validation_id=(request_record["validation"] or {}).get("validation_id"),
                canonical_contract_hash=request_record["canonical_contract_hash"],
                compiled_plan_hash=request_record["interaction_plan"]["sha256"],
                provenance_manifest_path=str(paths.provenance_manifest),
                provenance_manifest_sha256=provenance_manifest_sha256,
                contract_status=("incomplete" if missing_artifact_requirement_ids else "complete"),
                missing_artifact_requirement_ids=missing_artifact_requirement_ids,
                state=result_state,
                task=request.task,
                model=("all_models" if is_aggregate else _selected_models(request)[0]),
                tuning=(request.model_selection.tuning if is_aggregate else _selected_tuning(request)),
                output_directory=str(output_directory),
                interaction_trace=str(cli_result.trace_path),
                cli_stdout_log=str(cli_result.stdout_path),
                cli_stderr_log=str(cli_result.stderr_path),
                cli_exit_code=cli_result.returncode,
                cli_version=cli_version,
                input_sha256=snapshot.sha256,
                input_hash_verified=True,
                source_input_sha256=request_record["input"]["source_file"]["sha256"],
                dataset_preparation=request_record["input"]["preparation"],
                environment_identity_sha256=request_record["versions"]["environment_identity_sha256"],
                effective_seeds=request_record["interaction_plan"]["effective_seeds"],
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
                artifact_offset=0,
                returned_artifact_count=len(discovered.response_references),
                next_artifact_offset=(len(discovered.response_references) if discovered.truncated else None),
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

    def _wait_for_run(self, run_id: str, wait_seconds: float) -> None:
        if wait_seconds < 0 or wait_seconds > 300:
            raise RunStateError("wait_seconds must be between 0 and 300.")
        if wait_seconds == 0:
            return
        with self._lock:
            control = self._active.get(run_id)
            future = control.future if control is not None else None
        if future is None:
            return
        try:
            future.result(timeout=wait_seconds)
        except FutureTimeoutError:
            return

    def get_status(self, run_id: str, *, wait_seconds: float = 0) -> RunStatusResponse:
        self._ensure_initialized()
        paths = self._paths(run_id)
        self._wait_for_run(run_id, wait_seconds)
        return self._status_response(_read_json(paths.status))

    def get_result(
        self,
        run_id: str,
        *,
        wait_seconds: float = 0,
        artifact_offset: int = 0,
        artifact_limit: int | None = None,
    ) -> RunResultResponse:
        self._ensure_initialized()
        paths = self._paths(run_id)
        self._wait_for_run(run_id, wait_seconds)
        status = _read_json(paths.status)
        if status.get("state") not in {"succeeded", "partial_failure"}:
            raise RunStateError(f"Run {run_id} is {status.get('state')}; a result is available only after it succeeds or completes with partial failures.")
        if artifact_offset < 0:
            raise RunStateError("artifact_offset must be non-negative.")
        if artifact_limit is not None and (artifact_limit < 1 or artifact_limit > 200):
            raise RunStateError("artifact_limit must be between 1 and 200.")
        response = RunResultResponse.model_validate(_read_json(paths.result))
        artifact_index = _read_json(paths.artifact_index)
        if artifact_index.get("schema_version") != ARTIFACT_INDEX_SCHEMA_VERSION or artifact_index.get("run_id") != run_id or not isinstance(artifact_index.get("artifacts"), list):
            raise RunStateError("The complete artifact index identity is invalid.")
        entries = artifact_index["artifacts"]
        if len(entries) != response.artifact_count:
            raise RunStateError("The complete artifact index count is inconsistent with the run result.")
        if artifact_offset > len(entries):
            raise RunStateError("artifact_offset is beyond the complete artifact index.")
        page_limit = artifact_limit or self.settings.maximum_artifact_references
        page_end = min(len(entries), artifact_offset + page_limit)
        page = tuple(ArtifactReference.model_validate(entry) for entry in entries[artifact_offset:page_end])
        next_offset = page_end if page_end < len(entries) else None
        return response.model_copy(
            update={
                "artifact_offset": artifact_offset,
                "returned_artifact_count": len(page),
                "next_artifact_offset": next_offset,
                "artifacts": page,
                "artifacts_truncated": next_offset is not None,
            }
        )

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
