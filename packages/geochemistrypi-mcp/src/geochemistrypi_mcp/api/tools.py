"""Strict low-level MCP tool definitions and dispatch."""

import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

import anyio
from mcp import MCPError
from mcp.server import ServerRequestContext
from mcp.types import INVALID_PARAMS, CallToolRequestParams, CallToolResult, ListToolsResult, PaginatedRequestParams, TextContent, Tool
from pydantic import BaseModel, ConfigDict, ValidationError

from ..config.clients import SUPPORTED_CLIENTS
from ..config.constants import (
    ARTIFACT_INDEX_SCHEMA_VERSION,
    CLI_AUTOMATION_CONTRACT_VERSION,
    CLI_PYTHON_REQUIRES,
    COMPATIBILITY_POLICY_VERSION,
    INTERACTION_PLAN_VERSION,
    MCP_PYTHON_REQUIRES,
    MCP_SDK_REQUIRES,
    PENDING_RELEASE_GATES,
    PUBLIC_RELEASE_READY,
    RELEASE_CHANNEL,
    SERVER_NAME,
    SERVER_VERSION,
    SUPPORTED_CLI_VERSIONS,
    TARGET_OPERATING_SYSTEMS,
)
from ..config.settings import McpSettings, SettingsError
from ..contracts.anomaly_detection import MISSING_VALUE_METHODS as ANOMALY_DETECTION_MISSING_VALUE_METHODS
from ..contracts.anomaly_detection import MODEL_ORDER as ANOMALY_DETECTION_MODEL_ORDER
from ..contracts.anomaly_detection import SCALING_METHODS as ANOMALY_DETECTION_SCALING_METHODS
from ..contracts.anomaly_detection import UNSUPPORTED_INTERACTIONS as ANOMALY_DETECTION_UNSUPPORTED_INTERACTIONS
from ..contracts.classification import FEATURE_SELECTION_METHODS, LABEL_STRATEGIES, MISSING_VALUE_METHODS
from ..contracts.classification import MODEL_ORDER as CLASSIFICATION_MODEL_ORDER
from ..contracts.classification import SCALING_METHODS, TUNING_MODES
from ..contracts.classification import UNSUPPORTED_INTERACTIONS as CLASSIFICATION_UNSUPPORTED_INTERACTIONS
from ..contracts.clustering import MISSING_VALUE_METHODS as CLUSTERING_MISSING_VALUE_METHODS
from ..contracts.clustering import MODEL_ORDER as CLUSTERING_MODEL_ORDER
from ..contracts.clustering import SCALING_METHODS as CLUSTERING_SCALING_METHODS
from ..contracts.clustering import UNSUPPORTED_INTERACTIONS as CLUSTERING_UNSUPPORTED_INTERACTIONS
from ..contracts.decomposition import MISSING_VALUE_METHODS as DECOMPOSITION_MISSING_VALUE_METHODS
from ..contracts.decomposition import MODEL_ORDER as DECOMPOSITION_MODEL_ORDER
from ..contracts.decomposition import SCALING_METHODS as DECOMPOSITION_SCALING_METHODS
from ..contracts.decomposition import UNSUPPORTED_INTERACTIONS as DECOMPOSITION_UNSUPPORTED_INTERACTIONS
from ..contracts.manifest import known_gap_ids, load_capability_manifest, public_capabilities
from ..contracts.regression import MODEL_ORDER as REGRESSION_MODEL_ORDER
from ..contracts.regression import MODELS_WITHOUT_AUTOML
from ..contracts.regression import UNSUPPORTED_INTERACTIONS as REGRESSION_UNSUPPORTED_INTERACTIONS
from ..data.catalog import DatasetCatalogError
from ..data.headers import source_allows_pandas_duplicate_mangling
from ..data.inspector import DatasetInspectionError
from ..data.inspector import inspect_dataset as inspect_local_dataset
from ..planning.interaction_plan import PlanCompilationError
from ..runtime.cli_driver import CliDriverError
from ..runtime.runs import RunManager, RunNotFoundError, RunStateError
from ..tracking.experiments import ExperimentManager, ExperimentStoreError
from ..tracking.ui import MlflowUiError, MlflowUiManager
from .schemas import (
    AnalysisRequest,
    AnalysisValidationResponse,
    CancelRunResponse,
    CapabilitiesResponse,
    CompatibilityPolicy,
    DatasetInspectionRequest,
    DatasetInspectionResponse,
    GetExperimentRequest,
    GetExperimentResponse,
    ListDatasetsRequest,
    ListDatasetsResponse,
    ListExperimentsRequest,
    ListExperimentsResponse,
    MlflowUiStatusResponse,
    ResourceLimits,
    RunLookupRequest,
    RunResultResponse,
    RunStatusResponse,
    StartAnalysisResponse,
    StartMlflowUiRequest,
)

LOGGER = logging.getLogger(__name__)
_MAX_MODEL_TEXT = 4000
_MAX_ACTIONABLE_ERROR_TEXT = 3000
_MAX_VALIDATION_ERRORS = 8
_MAX_VALIDATION_ISSUE_TEXT = 360
_SAFE_FIELD_PART = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")
_WINDOWS_LOCAL_PATH = re.compile(r"(?i)(?:[A-Z]:[\\/]|\\\\)[^\r\n,;]+")
_POSIX_LOCAL_PATH = re.compile(r"(?<![\w:])/(?:[^\s'\",;]+/)*[^\s'\",;]*")


class EmptyRequest(BaseModel):
    """Strict empty arguments for capability discovery."""

    model_config = ConfigDict(extra="forbid")


_ToolFunction = Callable[[BaseModel], BaseModel]
_PUBLIC_ERRORS = (
    CliDriverError,
    DatasetCatalogError,
    DatasetInspectionError,
    PlanCompilationError,
    RunNotFoundError,
    RunStateError,
    ExperimentStoreError,
    MlflowUiError,
    SettingsError,
    ValidationError,
)


def _tool(name: str, description: str, request: type[BaseModel], response: type[BaseModel]) -> Tool:
    return Tool(
        name=name,
        description=description,
        input_schema=request.model_json_schema(),
        output_schema=response.model_json_schema(),
    )


def _compact_text(name: str, response: BaseModel) -> str:
    data = response.model_dump(mode="json")
    if name == "get_capabilities":
        text = f"GeochemistryPi MCP {data['server_version']} supports {', '.join(data['supported_tasks'])} with {', '.join(data['supported_models'])}."
        return text[:_MAX_MODEL_TEXT]
    if name == "inspect_dataset":
        columns = ", ".join(column["name"] for column in data["columns"])
        text = f"Dataset: {data['row_count']} rows, {data['column_count']} columns. Columns: {columns}. SHA-256: {data['sha256']}."
        return text[:_MAX_MODEL_TEXT]
    if name == "list_datasets":
        text = f"Found {len(data['datasets'])} safe GeochemistryPi datasets for " f"source {data['source_filter']}."
        return text[:_MAX_MODEL_TEXT]
    if name == "list_experiments":
        return f"Found {data['experiment_count']} active persistent MLflow experiments."[:_MAX_MODEL_TEXT]
    if name == "get_experiment":
        return (f"Experiment {data['experiment']['experiment_id']} is " f"{data['experiment']['name']} with {data['run_count']} returned runs.")[:_MAX_MODEL_TEXT]
    if name in {"start_mlflow_ui", "mlflow_ui_status", "stop_mlflow_ui"}:
        location = f" at {data['url']}" if data.get("url") else ""
        return f"Managed MLflow UI is {data['state']}{location}. {data['message']}"[:_MAX_MODEL_TEXT]
    if name == "start_analysis":
        return (f"Queued GeochemistryPi run {data['run_id']} for {data['estimated_model_count']} model(s): " f"{', '.join(data['models'])}. {data['status_hint']}")[:_MAX_MODEL_TEXT]
    if name == "validate_analysis":
        return (f"Analysis is valid for {data['task']} with {data['estimated_model_count']} model(s): " f"{', '.join(data['models'])}. No analysis process was started.")[:_MAX_MODEL_TEXT]
    if name == "get_run_status":
        return f"Run {data['run_id']} is {data['state']} at stage {data['stage']}: {data['progress_message']}"[:_MAX_MODEL_TEXT]
    if name == "get_run_result":
        metrics = json.dumps(data["reported_metrics"], ensure_ascii=False, separators=(",", ":"))
        preprocessing = data.get("preprocessing_summary")
        row_summary = (
            " Preprocessing rows: " f"input={preprocessing['input_row_count']}, " f"analysis={preprocessing['analysis_row_count']}, " f"dropped={preprocessing['dropped_row_count']}."
            if preprocessing is not None
            else ""
        )
        text = f"Run {data['run_id']} is {data['state']}. Original output: {data['output_directory']}. Artifacts: {data['artifact_count']}.{row_summary} CLI-reported metrics: {metrics}"
        return text[:_MAX_MODEL_TEXT]
    return f"Run {data['run_id']}: {data['message']}"[:_MAX_MODEL_TEXT]


def _bounded_actual(value: Any) -> str:
    if isinstance(value, dict):
        return f"object with {len(value)} field(s)"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__} with {len(value)} item(s)"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return f"string with {len(value)} character(s)"
    return "value of an unsupported type"


def _safe_error_field(error: dict[str, Any]) -> str:
    parts = []
    for part in error.get("loc", ()):
        if isinstance(part, int):
            parts.append(str(part))
            continue
        text = str(part)
        parts.append(text if _SAFE_FIELD_PART.fullmatch(text) else "unknown_field")
    return ".".join(parts) or "request"


def _validation_error_kind(error_type: str) -> str:
    if error_type == "missing":
        return "missing"
    if error_type == "extra_forbidden":
        return "extra_forbidden"
    if error_type == "string_pattern_mismatch":
        return "pattern"
    if error_type in {"greater_than", "greater_than_equal", "less_than", "less_than_equal"}:
        return "range"
    if error_type in {"literal_error", "union_tag_invalid", "union_tag_not_found"}:
        return "literal"
    if error_type.endswith("_type") or error_type.endswith("_parsing"):
        return "type"
    return "value"


def _validation_problem(kind: str) -> str:
    return {
        "missing": "Required field is missing",
        "extra_forbidden": "Extra input is not permitted",
        "pattern": "String does not match the declared pattern",
        "range": "Value is outside the declared range",
        "literal": "Value is not one of the declared literals",
        "type": "Value has the wrong JSON type",
        "value": "Value does not satisfy the declared field contract",
    }[kind]


def _validation_alternative(field: str, kind: str) -> str:
    if field.startswith("time_series.") and kind == "extra_forbidden":
        replacement = {
            "dataset": "remove it and use top-level 'training_dataset'",
            "model": "remove it because the Time Series workflow has a fixed model",
            "model_parameters": "remove it and place supported fields such as 'bin_width', 'iterations', and 'seed' at the top level",
            "bin_width_ma": "remove it and use top-level 'bin_width'",
            "bootstrap_iterations": "remove it and use top-level 'iterations'",
            "random_seed": "remove it and use top-level 'seed'",
        }.get(field.rsplit(".", 1)[-1])
        if replacement is not None:
            return replacement
    if kind == "missing":
        return "provide this required field at the location shown"
    if kind == "extra_forbidden":
        return "remove this unsupported field"
    if kind == "pattern" and field.endswith("dataset_id"):
        return "use the exact 'builtin:<id>' value returned by list_datasets, including its prefix"
    if kind == "pattern":
        return "use a value matching the field's declared pattern"
    if kind == "range":
        return "use a value within the field's declared minimum and maximum"
    if kind == "literal":
        return "use one of the field's declared literal values"
    if kind == "type":
        return "use the field's declared JSON type"
    return "inspect the field description and supported capabilities"


def _validation_issue(error: dict[str, Any]) -> tuple[tuple[str, str], str]:
    field = _safe_error_field(error)
    error_type = str(error.get("type", ""))
    kind = _validation_error_kind(error_type)
    text = (
        f"[{kind}] Invalid field '{field}'. Actual value: {_bounded_actual(error.get('input'))}. "
        f"Problem: {_validation_problem(kind)}. "
        f"Valid alternatives: {_validation_alternative(field, kind)}."
    )
    return (field, kind), text[:_MAX_VALIDATION_ISSUE_TEXT]


def _redacted_public_error(exc: BaseException) -> str:
    message = " ".join(str(exc).split())
    message = _WINDOWS_LOCAL_PATH.sub("<local-path>", message)
    message = _POSIX_LOCAL_PATH.sub("<local-path>", message)
    return message[:700] or type(exc).__name__


def _actionable_error(exc: BaseException) -> str:
    if isinstance(exc, ValidationError):
        unique: dict[tuple[str, str], str] = {}
        for error in exc.errors(include_url=False):
            key, issue = _validation_issue(error)
            unique.setdefault(key, issue)
        ordered = [unique[key] for key in sorted(unique)]
        shown = ordered[:_MAX_VALIDATION_ERRORS]
        issue_text = " ".join(f"{index}. {issue}" for index, issue in enumerate(shown, start=1))
        omitted = len(ordered) - len(shown)
        omitted_text = f" Additional issues omitted: {omitted}." if omitted else ""
        next_action = (
            "Next action: Resolve it from the conversation or earlier tool results when possible, "
            "apply every listed fix in one request, and validate once. Only if an unresolved scientific "
            "choice belongs to the user, ask one plain-language question; never ask the user for a schema "
            "field name or JSON, and do not ask the user to edit JSON."
        )
        return (f"Validation failed with {len(ordered)} independent issue(s): {issue_text}{omitted_text} {next_action}")[:_MAX_ACTIONABLE_ERROR_TEXT]
    message = _redacted_public_error(exc)
    return (
        f"Invalid analysis configuration. Problem: {message}. "
        "Valid alternatives: inspect the dataset for exact columns and check the supported scientific workflows. "
        "Next action: resolve safe defaults automatically, ask the user one plain-language scientific question "
        "if needed, and preview the corrected analysis before starting it."
    )[:_MAX_ACTIONABLE_ERROR_TEXT]


def build_tool_handlers(
    settings: McpSettings,
    runs: RunManager,
    experiments: ExperimentManager | None = None,
    mlflow_ui: MlflowUiManager | None = None,
) -> tuple[Callable[[ServerRequestContext[Any], PaginatedRequestParams | None], Awaitable[ListToolsResult],], Callable[[ServerRequestContext[Any], CallToolRequestParams], Awaitable[CallToolResult]],]:
    """Build strict schemas and a sanitized dispatcher for one server."""
    experiment_store = experiments or getattr(runs, "experiment_manager", None) or ExperimentManager(settings)
    ui_manager = mlflow_ui or MlflowUiManager(settings)
    definitions = (
        _tool(
            "get_capabilities",
            "Discover supported scientific workflows and limits before planning an analysis.",
            EmptyRequest,
            CapabilitiesResponse,
        ),
        _tool(
            "list_datasets",
            "List built-in datasets and safe immediate files in Desktop/geopi_input without modifying them.",
            ListDatasetsRequest,
            ListDatasetsResponse,
        ),
        _tool(
            "inspect_dataset",
            "Inspect one explicit, built-in, or Desktop CSV/XLSX dataset read-only; return only bounded rows and inferred types.",
            DatasetInspectionRequest,
            DatasetInspectionResponse,
        ),
        _tool(
            "list_experiments",
            "List active experiments in the installer-owned persistent MLflow tracking store using stable IDs.",
            ListExperimentsRequest,
            ListExperimentsResponse,
        ),
        _tool(
            "get_experiment",
            "Read one persistent MLflow experiment and a bounded list of its newest runs by stable experiment ID.",
            GetExperimentRequest,
            GetExperimentResponse,
        ),
        _tool(
            "start_mlflow_ui",
            "Explicitly start the installer-owned MLflow UI on 127.0.0.1; it is never started automatically.",
            StartMlflowUiRequest,
            MlflowUiStatusResponse,
        ),
        _tool(
            "mlflow_ui_status",
            "Inspect and recover durable managed MLflow UI state without starting or stopping any process.",
            EmptyRequest,
            MlflowUiStatusResponse,
        ),
        _tool(
            "stop_mlflow_ui",
            "Stop only the MLflow UI process tree whose PID, creation time, command, and tracking root are verified.",
            EmptyRequest,
            MlflowUiStatusResponse,
        ),
        _tool(
            "validate_analysis",
            "Validate without starting science. Time Series uses top-level training_dataset, bin_width, iterations, seed, and role fields; do not send dataset, model, or model_parameters.",
            AnalysisRequest,
            AnalysisValidationResponse,
        ),
        _tool(
            "start_analysis",
            "Queue one validated CLI run. Time Series uses top-level training_dataset, bin_width, iterations, seed, and role fields; do not send dataset, model, or model_parameters.",
            AnalysisRequest,
            StartAnalysisResponse,
        ),
        _tool(
            "get_run_status",
            "Read the durable state of one wrapper-owned run without blocking on the CLI.",
            RunLookupRequest,
            RunStatusResponse,
        ),
        _tool(
            "get_run_result",
            "Return bounded metrics and references to original CLI outputs after a run succeeds.",
            RunLookupRequest,
            RunResultResponse,
        ),
        _tool(
            "cancel_run",
            "Cancel only the live CLI process tree recorded for the specified wrapper-owned run.",
            RunLookupRequest,
            CancelRunResponse,
        ),
    )
    request_models: dict[str, type[BaseModel]] = {
        "get_capabilities": EmptyRequest,
        "list_datasets": ListDatasetsRequest,
        "inspect_dataset": DatasetInspectionRequest,
        "list_experiments": ListExperimentsRequest,
        "get_experiment": GetExperimentRequest,
        "validate_analysis": AnalysisRequest,
        "start_mlflow_ui": StartMlflowUiRequest,
        "mlflow_ui_status": EmptyRequest,
        "stop_mlflow_ui": EmptyRequest,
        "start_analysis": AnalysisRequest,
        "get_run_status": RunLookupRequest,
        "get_run_result": RunLookupRequest,
        "cancel_run": RunLookupRequest,
    }

    def capabilities(_: BaseModel) -> CapabilitiesResponse:
        manifest = load_capability_manifest()
        return CapabilitiesResponse(
            server_name=SERVER_NAME,
            server_version=SERVER_VERSION,
            supported_cli_versions=SUPPORTED_CLI_VERSIONS,
            supported_tasks=(
                "classification",
                "regression",
                "clustering",
                "decomposition",
                "anomaly_detection",
                "time_series",
            ),
            supported_models=tuple(
                dict.fromkeys(
                    (
                        *CLASSIFICATION_MODEL_ORDER,
                        *REGRESSION_MODEL_ORDER,
                        *CLUSTERING_MODEL_ORDER,
                        *DECOMPOSITION_MODEL_ORDER,
                        *ANOMALY_DETECTION_MODEL_ORDER,
                    )
                )
            ),
            supported_dataset_formats=("csv", "xlsx"),
            maximum_dataset_bytes=settings.maximum_dataset_bytes,
            default_concurrency=settings.concurrency,
            capability_manifest_schema_version=manifest["schema_version"],
            capability_manifest_id=manifest["manifest_id"],
            cli_automation_contract_version=CLI_AUTOMATION_CONTRACT_VERSION,
            capabilities=public_capabilities(),
            known_gaps=known_gap_ids(),
            supported_data_sources=("path", "builtin", "desktop"),
            supported_clients=SUPPORTED_CLIENTS,
            compatibility=CompatibilityPolicy(
                schema_version=COMPATIBILITY_POLICY_VERSION,
                release_channel=RELEASE_CHANNEL,
                public_release_ready=PUBLIC_RELEASE_READY,
                mcp_python_requires=MCP_PYTHON_REQUIRES,
                cli_python_requires=CLI_PYTHON_REQUIRES,
                mcp_sdk_requires=MCP_SDK_REQUIRES,
                supported_cli_versions=SUPPORTED_CLI_VERSIONS,
                interaction_plan_version=INTERACTION_PLAN_VERSION,
                cli_automation_contract_version=CLI_AUTOMATION_CONTRACT_VERSION,
                artifact_index_schema_version=ARTIFACT_INDEX_SCHEMA_VERSION,
                target_operating_systems=TARGET_OPERATING_SYSTEMS,
                pending_release_gates=PENDING_RELEASE_GATES,
            ),
            resource_limits=ResourceLimits(
                maximum_dataset_bytes=settings.maximum_dataset_bytes,
                maximum_columns=settings.maximum_columns,
                maximum_artifact_references=settings.maximum_artifact_references,
                maximum_concurrent_runs=settings.concurrency,
                maximum_pending_runs=settings.maximum_pending_runs,
                maximum_process_seconds=settings.maximum_process_seconds,
            ),
            classification_options={
                "label_customization": LABEL_STRATEGIES,
                "missing_values": MISSING_VALUE_METHODS,
                "scaling": SCALING_METHODS,
                "feature_selection": FEATURE_SELECTION_METHODS,
                "tuning": TUNING_MODES,
                "application_data": ("disabled", "enabled"),
                "sample_balancing": ("none",),
                "model_selection": ("single", "all"),
            },
            regression_options={
                "missing_values": MISSING_VALUE_METHODS,
                "scaling": SCALING_METHODS,
                "feature_selection": FEATURE_SELECTION_METHODS,
                "tuning": TUNING_MODES,
                "automl_models": tuple(model for model in REGRESSION_MODEL_ORDER if model not in MODELS_WITHOUT_AUTOML),
                "application_data": ("disabled", "enabled"),
                "target_columns": ("single_numeric", "multiple_numeric"),
                "model_selection": ("single", "all"),
            },
            clustering_options={
                "missing_values": CLUSTERING_MISSING_VALUE_METHODS,
                "scaling": CLUSTERING_SCALING_METHODS,
                "application_data": ("disabled",),
                "target_columns": ("not_applicable",),
                "tuning": ("not_applicable",),
                "model_selection": ("single", "all"),
            },
            decomposition_options={
                "missing_values": DECOMPOSITION_MISSING_VALUE_METHODS,
                "scaling": DECOMPOSITION_SCALING_METHODS,
                "application_data": ("disabled",),
                "target_columns": ("not_applicable",),
                "tuning": ("not_applicable",),
                "transformed_data": ("enabled",),
                "model_selection": ("single", "all"),
            },
            anomaly_detection_options={
                "missing_values": ANOMALY_DETECTION_MISSING_VALUE_METHODS,
                "scaling": ANOMALY_DETECTION_SCALING_METHODS,
                "application_data": ("disabled",),
                "target_columns": ("not_applicable",),
                "tuning": ("not_applicable",),
                "detection_labels": ("1_inlier", "-1_outlier"),
                "model_selection": ("single", "all"),
            },
            time_series_options={
                "model": ("subaerial_proportion_bootstrap",),
                "age_units": ("Ma", "Ga"),
                "fit_curve": ("disabled", "enabled"),
                "randomness": ("explicit_seed",),
            },
            supported_models_by_task={
                "classification": CLASSIFICATION_MODEL_ORDER,
                "regression": REGRESSION_MODEL_ORDER,
                "clustering": CLUSTERING_MODEL_ORDER,
                "decomposition": DECOMPOSITION_MODEL_ORDER,
                "anomaly_detection": ANOMALY_DETECTION_MODEL_ORDER,
                "time_series": ("subaerial_proportion_bootstrap",),
            },
            unsupported_interactions=(
                *CLASSIFICATION_UNSUPPORTED_INTERACTIONS,
                *REGRESSION_UNSUPPORTED_INTERACTIONS,
                *CLUSTERING_UNSUPPORTED_INTERACTIONS,
                *DECOMPOSITION_UNSUPPORTED_INTERACTIONS,
                *ANOMALY_DETECTION_UNSUPPORTED_INTERACTIONS,
            ),
            notes=(
                "PR9 covers semantic world maps, seeded Time Series analysis, and isolated "
                "all-models aggregates in addition to every single-model family offered by "
                "the GeochemistryPi 0.8.1 public CLI.",
                "The stable compatibility policy is published only by the protected workflow after every required release gate passes.",
                "The existing GeochemistryPi CLI creates every scientific result and original output file.",
                "Unsupported or scientifically invalid combinations are rejected before the CLI starts.",
                "Use inspect_dataset when column roles are not already known.",
            ),
        )

    def inspect_dataset_request(request: DatasetInspectionRequest) -> DatasetInspectionResponse:
        if request.dataset is None:
            return inspect_local_dataset(request, settings)
        resolved = runs.dataset_catalog.resolve(request.dataset)
        response = inspect_local_dataset(
            request.model_copy(update={"dataset_path": resolved.path, "dataset": None}),
            settings,
            allow_pandas_duplicate_mangling=source_allows_pandas_duplicate_mangling(resolved.source),
        )
        if resolved.expected_sha256 is not None and response.sha256 != resolved.expected_sha256:
            raise DatasetCatalogError("The dataset changed between source resolution and inspection.")
        return response

    functions: dict[str, _ToolFunction] = {
        "get_capabilities": capabilities,
        "list_datasets": lambda request: runs.dataset_catalog.list(request),
        "inspect_dataset": inspect_dataset_request,
        "list_experiments": lambda request: experiment_store.list(request),
        "get_experiment": lambda request: experiment_store.get(request),
        "validate_analysis": lambda request: runs.validate(request.root),
        "start_mlflow_ui": lambda request: ui_manager.start(request),
        "mlflow_ui_status": lambda request: ui_manager.status(),
        "stop_mlflow_ui": lambda request: ui_manager.stop(),
        "start_analysis": lambda request: runs.start(request.root),
        "get_run_status": lambda request: runs.get_status(request.run_id),
        "get_run_result": lambda request: runs.get_result(request.run_id),
        "cancel_run": lambda request: runs.cancel(request.run_id),
    }

    async def list_tools(_: ServerRequestContext[Any], __: PaginatedRequestParams | None) -> ListToolsResult:
        return ListToolsResult(tools=list(definitions))

    async def call_tool(_: ServerRequestContext[Any], params: CallToolRequestParams) -> CallToolResult:
        if params.name not in functions:
            raise MCPError(INVALID_PARAMS, f"Unknown tool: {params.name}")
        try:
            request = request_models[params.name].model_validate(params.arguments or {})
            response = await anyio.to_thread.run_sync(functions[params.name], request)
            structured = response.model_dump(mode="json")
            return CallToolResult(
                content=[TextContent(type="text", text=_compact_text(params.name, response))],
                structured_content=structured,
            )
        except _PUBLIC_ERRORS as exc:
            message = _actionable_error(exc)
            return CallToolResult(
                content=[TextContent(type="text", text=f"GeochemistryPi request rejected: {message}")],
                is_error=True,
            )
        except Exception:
            LOGGER.exception("Unexpected GeochemistryPi MCP tool failure in %s", params.name)
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="GeochemistryPi encountered an internal wrapper error. Check the server's stderr log.",
                    )
                ],
                is_error=True,
            )

    return list_tools, call_tool
