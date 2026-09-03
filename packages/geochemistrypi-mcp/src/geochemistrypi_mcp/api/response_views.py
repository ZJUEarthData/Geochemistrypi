"""Deterministic compact response views over complete protocol records."""

import hashlib
import json

from .schemas import (
    AnalysisTaskName,
    CapabilitiesNotModifiedResponse,
    CapabilitiesRequest,
    CapabilitiesResponse,
    CompactCapabilitiesResponse,
    CompactCapabilityBoundary,
    CompactDatasetInspectionResponse,
    DatasetInspectionRequest,
    DatasetInspectionResponse,
    StartReadyCapabilitiesResponse,
    TaskValidationRequestContract,
)

_TASK_NAMES = (
    "classification",
    "regression",
    "clustering",
    "decomposition",
    "anomaly_detection",
    "time_series",
)
_UNSUPERVISED_TASKS = frozenset({"clustering", "decomposition", "anomaly_detection"})
_CAPABILITY_VIEW_SCHEMA_VERSION = 1


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def capabilities_sha256(response: CapabilitiesResponse) -> str:
    """Hash the complete observed capability snapshot, independent of response view."""
    payload = response.model_dump(
        mode="json",
        exclude={"response_detail", "capabilities_sha256", "capability_view_sha256"},
    )
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _capability_view_sha256(
    response: CapabilitiesResponse
    | CompactCapabilitiesResponse
    | StartReadyCapabilitiesResponse,
) -> str:
    """Hash only the fields delivered by one capability projection."""
    payload = response.model_dump(
        mode="json",
        exclude={"capabilities_sha256", "capability_view_sha256"},
    )
    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "view_schema_version": _CAPABILITY_VIEW_SCHEMA_VERSION,
                "payload": payload,
            }
        )
    ).hexdigest()


def capability_projection_sha256(
    response: CapabilitiesResponse
    | CompactCapabilitiesResponse
    | StartReadyCapabilitiesResponse,
) -> str:
    """Return the stable identity of any public capability projection."""
    return _capability_view_sha256(response)


def _capability_is_relevant(capability_id: str, task: AnalysisTaskName | None) -> bool:
    if task is None:
        return True
    for candidate in _TASK_NAMES:
        if capability_id.startswith(f"task.{candidate}") or capability_id.startswith(f"model.{candidate}.") or capability_id.startswith(f"branch.{candidate}_"):
            return candidate == task
    if capability_id.startswith("branch.unsupervised_"):
        return task in _UNSUPERVISED_TASKS
    if capability_id == "workflow.reference_anomaly_time_series":
        return task == "time_series"
    if capability_id == "workflow.embedding_label_overlay":
        return task == "decomposition"
    return True


def _unsupported_is_relevant(value: str, task: AnalysisTaskName | None) -> bool:
    if task is None:
        return True
    boundary_id = value.partition(":")[0]
    for candidate in _TASK_NAMES:
        if boundary_id.startswith(f"{candidate}."):
            return candidate == task
    if boundary_id in {"sample_balancing", "previous_experiment"}:
        return task == "classification"
    return True


def _task_options(response: CapabilitiesResponse, task: AnalysisTaskName | None) -> dict[str, tuple[str, ...]]:
    if task is None:
        return {}
    return {
        "classification": response.classification_options,
        "regression": response.regression_options,
        "clustering": response.clustering_options,
        "decomposition": response.decomposition_options,
        "anomaly_detection": response.anomaly_detection_options,
        "time_series": response.time_series_options,
    }[task]


def compact_capabilities_view(
    response: CapabilitiesResponse,
    task: AnalysisTaskName | None = None,
    validation_request_contract: TaskValidationRequestContract | None = None,
) -> CompactCapabilitiesResponse:
    """Project one complete snapshot into a bounded planning response."""
    if validation_request_contract is not None and (task is None or validation_request_contract.task != task):
        raise ValueError("a validation request contract must match the selected task filter")
    identity = capabilities_sha256(response)
    boundaries = tuple(
        CompactCapabilityBoundary(
            id=capability.id,
            category=capability.category,
            status=capability.status,
            cli_public=capability.cli_public,
            mcp_supported=capability.mcp_supported,
        )
        for capability in response.capabilities
        if (capability.status in {"known_gap", "not_public"} or not capability.mcp_supported) and _capability_is_relevant(capability.id, task)
    )
    relevant_boundary_ids = {boundary.id for boundary in boundaries}
    known_gaps = tuple(gap for gap in response.known_gaps if gap in relevant_boundary_ids or _capability_is_relevant(gap, task))
    models_by_task = response.supported_models_by_task if task is None else {task: response.supported_models_by_task.get(task, ())}
    unsupported = tuple(item for item in response.unsupported_interactions if _unsupported_is_relevant(item, task))
    compact = CompactCapabilitiesResponse(
        capabilities_sha256=identity,
        capability_view_sha256="0" * 64,
        task_filter=task,
        server_name=response.server_name,
        server_version=response.server_version,
        supported_cli_versions=response.supported_cli_versions,
        supported_tasks=response.supported_tasks,
        analysis_schema_task_scope=response.analysis_schema_task_scope,
        analysis_start_modes=response.analysis_start_modes,
        capability_manifest_schema_version=response.capability_manifest_schema_version,
        capability_manifest_id=response.capability_manifest_id,
        cli_automation_contract_version=response.cli_automation_contract_version,
        supported_dataset_formats=response.supported_dataset_formats,
        supported_data_sources=response.supported_data_sources,
        compatibility=response.compatibility,
        resource_limits=response.resource_limits,
        supported_models_by_task=models_by_task,
        scientific_attestation=response.scientific_attestation,
        task_options=_task_options(response, task),
        known_gaps=known_gaps,
        capability_boundaries=boundaries,
        unsupported_interactions=unsupported,
        validation_request_contract=validation_request_contract,
        next_action=(
            "Choose one supported task, then request its compact capability view before validation."
            if task is None
            else ("Construct one validate_analysis request from validation_request_contract; " "do not infer field placement from another task or retry guessed schemas.")
        ),
    )
    return compact.model_copy(update={"capability_view_sha256": _capability_view_sha256(compact)})


def capability_view_sha256(
    response: CapabilitiesResponse,
    detail: str,
    task: AnalysisTaskName | None,
    validation_request_contract: TaskValidationRequestContract | None = None,
) -> str:
    """Bind a conditional receipt to the actual fields in one capability view."""
    if detail == "compact":
        return compact_capabilities_view(
            response,
            task,
            validation_request_contract,
        ).capability_view_sha256
    full = response.model_copy(
        update={
            "capabilities_sha256": capabilities_sha256(response),
            "capability_view_sha256": "0" * 64,
        }
    )
    return _capability_view_sha256(full)


def capabilities_response_view(
    response: CapabilitiesResponse,
    request: CapabilitiesRequest,
    validation_request_contract: TaskValidationRequestContract | None = None,
) -> CapabilitiesResponse | CompactCapabilitiesResponse | CapabilitiesNotModifiedResponse:
    """Return full, compact, or unchanged capability delivery without altering source data."""
    identity = capabilities_sha256(response)
    projected = (
        compact_capabilities_view(
            response,
            request.task,
            validation_request_contract,
        )
        if request.detail == "compact"
        else response.model_copy(
            update={
                "capabilities_sha256": identity,
                "capability_view_sha256": "0" * 64,
            }
        )
    )
    view_identity = _capability_view_sha256(projected)
    legacy_unchanged = request.if_capabilities_sha256 == identity
    view_unchanged = request.if_capability_view_sha256 == view_identity
    if legacy_unchanged or view_unchanged:
        return CapabilitiesNotModifiedResponse(
            capabilities_sha256=identity,
            capability_view_sha256=view_identity,
            task_filter=request.task,
            server_name=response.server_name,
            server_version=response.server_version,
            capability_manifest_id=response.capability_manifest_id,
        )
    return projected.model_copy(update={"capability_view_sha256": view_identity})


def dataset_inspection_response_view(
    response: DatasetInspectionResponse,
    request: DatasetInspectionRequest,
) -> DatasetInspectionResponse | CompactDatasetInspectionResponse:
    """Keep full inspection intact or omit names-mode preparation duplication by default."""
    if request.detail == "full":
        return response
    preparation = response.dataset_preparation or {}
    contract_hash = preparation.get("contract_hash")
    if not isinstance(contract_hash, str) or len(contract_hash) != 64:
        contract_hash = None
    include_samples = request.sample_rows > 0
    return CompactDatasetInspectionResponse(
        format=response.format,
        size_bytes=response.size_bytes,
        source_sha256=response.source_sha256,
        prepared_view_sha256=response.prepared_view_sha256,
        prepared_view_is_source=response.source_sha256 == response.prepared_view_sha256,
        preparation_contract_sha256=contract_hash,
        row_count=response.row_count,
        row_count_exact=response.row_count_exact,
        column_count=response.column_count,
        column_names=response.column_names or tuple(column.name for column in response.columns),
        header_warnings=response.header_warnings,
        sample_rows=response.sample_rows if include_samples else (),
        sample_truncated=response.sample_truncated if include_samples else response.row_count > 0,
    )
