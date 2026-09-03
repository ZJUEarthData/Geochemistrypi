import json

import pytest
from geochemistrypi_mcp.api.response_views import capabilities_response_view, capabilities_sha256, capability_view_sha256, dataset_inspection_response_view
from geochemistrypi_mcp.api.schemas import (
    CapabilitiesRequest,
    CapabilitiesResponse,
    CapabilitySummary,
    CompatibilityPolicy,
    DatasetInspectionRequest,
    DatasetInspectionResponse,
    ResourceLimits,
    ScientificAttestationCapabilities,
)
from pydantic import ValidationError


def _capabilities() -> CapabilitiesResponse:
    return CapabilitiesResponse(
        server_version="0.2.1",
        supported_cli_versions=("0.8.1",),
        supported_tasks=(
            "classification",
            "regression",
            "clustering",
            "decomposition",
            "anomaly_detection",
            "time_series",
        ),
        supported_models=("xgboost", "ridge_regression", "pca"),
        supported_dataset_formats=("csv", "xlsx"),
        maximum_dataset_bytes=1024,
        default_concurrency=1,
        capability_manifest_schema_version=1,
        capability_manifest_id="test-capabilities",
        cli_automation_contract_version=1,
        capabilities=(
            CapabilitySummary(
                id="task.classification",
                category="task",
                status="verified",
                cli_public=True,
                mcp_supported=True,
                evidence=("maintainer-only-evidence" * 100,),
            ),
            CapabilitySummary(
                id="branch.classification_inference",
                category="inference",
                status="known_gap",
                cli_public=True,
                mcp_supported=False,
                evidence=("classification-evidence",),
            ),
            CapabilitySummary(
                id="branch.regression_inference",
                category="inference",
                status="known_gap",
                cli_public=True,
                mcp_supported=False,
                evidence=("regression-evidence",),
            ),
            CapabilitySummary(
                id="format.xls",
                category="dataset_format",
                status="not_public",
                cli_public=False,
                mcp_supported=False,
                evidence=("format-evidence",),
            ),
        ),
        known_gaps=(
            "branch.classification_inference",
            "branch.regression_inference",
        ),
        supported_data_sources=("path", "builtin", "desktop"),
        supported_clients=("codex",),
        compatibility=CompatibilityPolicy(
            release_channel="stable",
            public_release_ready=True,
            mcp_python_requires=">=3.10,<4",
            cli_python_requires=">=3.9,<3.10",
            mcp_sdk_requires="==2.0.0",
            supported_cli_versions=("0.8.1",),
            interaction_plan_version=1,
            cli_automation_contract_version=1,
            artifact_index_schema_version=1,
            target_operating_systems=("windows",),
            pending_release_gates=(),
        ),
        resource_limits=ResourceLimits(
            maximum_dataset_bytes=1024,
            maximum_columns=256,
            maximum_artifact_references=200,
            maximum_concurrent_runs=1,
            maximum_pending_runs=8,
            maximum_process_seconds=900,
        ),
        classification_options={"model_selection": ("single", "all")},
        regression_options={"model_selection": ("single", "all")},
        supported_models_by_task={
            "classification": ("xgboost",),
            "regression": ("ridge_regression",),
            "clustering": (),
            "decomposition": ("pca",),
            "anomaly_detection": (),
            "time_series": ("subaerial_proportion_bootstrap",),
        },
        scientific_attestation=ScientificAttestationCapabilities(
            v4_attested_methods_by_task={"classification": ("xgboost",)},
            legacy_methods_without_v4_attestation_by_task={
                "regression": ("ridge_regression",),
            },
        ),
        unsupported_interactions=(
            "sample_balancing: not public",
            "regression.multiple_targets.feature_selection: not public",
        ),
        notes=("maintainer note",),
    )


def test_capabilities_compact_default_keeps_all_task_and_model_indexes() -> None:
    full = _capabilities()
    compact = capabilities_response_view(full, CapabilitiesRequest())

    assert compact.response_detail == "compact"
    assert compact.supported_tasks == full.supported_tasks
    assert compact.supported_models_by_task == full.supported_models_by_task
    assert compact.scientific_attestation == full.scientific_attestation
    assert compact.task_options == {}
    assert compact.capabilities_sha256 == capabilities_sha256(full)
    assert compact.capability_view_sha256 == capability_view_sha256(
        full,
        "compact",
        None,
    )
    assert {item.id for item in compact.capability_boundaries} == {
        "branch.classification_inference",
        "branch.regression_inference",
        "format.xls",
    }
    payload = compact.model_dump(mode="json")
    assert "capabilities" not in payload
    assert "evidence" not in json.dumps(payload)
    assert len(json.dumps(payload)) < len(json.dumps(full.model_dump(mode="json")))


def test_capabilities_task_filter_keeps_task_options_and_relevant_boundaries() -> None:
    compact = capabilities_response_view(
        _capabilities(),
        CapabilitiesRequest(task="classification"),
    )

    assert compact.supported_tasks == _capabilities().supported_tasks
    assert compact.supported_models_by_task == {"classification": ("xgboost",)}
    assert compact.task_options == {"model_selection": ("single", "all")}
    assert compact.known_gaps == ("branch.classification_inference",)
    assert {item.id for item in compact.capability_boundaries} == {
        "branch.classification_inference",
        "format.xls",
    }
    assert compact.unsupported_interactions == ("sample_balancing: not public",)


def test_capabilities_full_and_conditional_views_are_lossless_and_stable() -> None:
    source = _capabilities()
    full = capabilities_response_view(source, CapabilitiesRequest(detail="full"))
    identity = capabilities_sha256(source)

    assert full.response_detail == "full"
    assert full.capabilities_sha256 == identity
    assert full.capability_view_sha256 == capability_view_sha256(
        source,
        "full",
        None,
    )
    assert full.capabilities == source.capabilities
    assert full.unsupported_interactions == source.unsupported_interactions
    assert capabilities_sha256(full) == identity

    classification = capabilities_response_view(
        source,
        CapabilitiesRequest(task="classification"),
    )
    unchanged = capabilities_response_view(
        source,
        CapabilitiesRequest(
            task="classification",
            if_capability_view_sha256=classification.capability_view_sha256,
        ),
    )
    assert unchanged.response_detail == "not_modified"
    assert unchanged.not_modified is True
    assert unchanged.capabilities_sha256 == identity
    assert unchanged.capability_view_sha256 == classification.capability_view_sha256
    assert unchanged.task_filter == "classification"
    assert "capabilities" not in unchanged.model_dump(mode="json")

    regression = capabilities_response_view(
        source,
        CapabilitiesRequest(
            task="regression",
            if_capability_view_sha256=classification.capability_view_sha256,
        ),
    )
    assert regression.response_detail == "compact"
    assert regression.task_filter == "regression"
    assert regression.capability_view_sha256 != classification.capability_view_sha256

    legacy_unchanged = capabilities_response_view(
        source,
        CapabilitiesRequest(
            detail="full",
            if_capabilities_sha256=identity,
        ),
    )
    assert legacy_unchanged.response_detail == "not_modified"

    changed = source.model_copy(update={"server_version": "0.2.2"})
    assert capabilities_sha256(changed) != identity


def test_legacy_capability_snapshot_condition_cannot_suppress_a_new_projection() -> None:
    identity = capabilities_sha256(_capabilities())

    with pytest.raises(ValidationError, match="only for the compact"):
        CapabilitiesRequest(detail="full", task="classification")

    with pytest.raises(ValidationError, match="safe only for the unfiltered full view"):
        CapabilitiesRequest(
            task="classification",
            if_capabilities_sha256=identity,
        )

    with pytest.raises(ValidationError, match="provide only one"):
        CapabilitiesRequest(
            detail="full",
            if_capabilities_sha256=identity,
            if_capability_view_sha256=identity,
        )


def test_compact_capability_identity_ignores_fields_omitted_from_that_view() -> None:
    source = _capabilities()
    first = capabilities_response_view(
        source,
        CapabilitiesRequest(task="classification"),
    )
    changed_full_only = source.model_copy(
        update={"notes": ("a different maintainer-only note",)},
    )

    unchanged = capabilities_response_view(
        changed_full_only,
        CapabilitiesRequest(
            task="classification",
            if_capability_view_sha256=first.capability_view_sha256,
        ),
    )

    assert capabilities_sha256(changed_full_only) != capabilities_sha256(source)
    assert unchanged.response_detail == "not_modified"
    assert unchanged.capability_view_sha256 == first.capability_view_sha256
    assert unchanged.capabilities_sha256 == capabilities_sha256(changed_full_only)


def test_names_inspection_defaults_to_no_samples_and_exposes_both_hashes() -> None:
    request = DatasetInspectionRequest(
        dataset_path="C:/data/rocks.xlsx",
        detail="names",
    )
    assert request.sample_rows == 0
    assert (
        DatasetInspectionRequest(
            dataset_path="C:/data/rocks.xlsx",
            detail="names",
            sample_rows=2,
        ).sample_rows
        == 2
    )

    response = DatasetInspectionResponse(
        source_path="C:/cache/prepared.xlsx",
        resolved_path="C:/cache/prepared.xlsx",
        original_source_path="C:/data/rocks.xlsx",
        original_source_sha256="1" * 64,
        dataset_preparation={"contract_hash": "3" * 64, "contract": {"selected_columns": ["SiO2"]}},
        format="xlsx",
        size_bytes=100,
        sha256="2" * 64,
        row_count=10,
        row_count_exact=True,
        column_count=2,
        detail="names",
        column_names=("Sample", "SiO2"),
        sample_rows=({"Sample": "A", "SiO2": 50.0},),
        sample_truncated=True,
    )
    assert response.source_sha256 == "1" * 64
    assert response.prepared_view_sha256 == "2" * 64

    compact = dataset_inspection_response_view(response, request)
    payload = compact.model_dump(mode="json")
    assert compact.source_sha256 == "1" * 64
    assert compact.prepared_view_sha256 == "2" * 64
    assert compact.preparation_contract_sha256 == "3" * 64
    assert compact.sample_rows == ()
    assert compact.sample_truncated is True
    assert "dataset_preparation" not in payload
    assert "source_path" not in payload
    assert "resolved_path" not in payload

    full_request = DatasetInspectionRequest(
        dataset_path="C:/data/rocks.xlsx",
        detail="full",
    )
    assert dataset_inspection_response_view(response, full_request) is response
    full_payload = response.model_dump(mode="json")
    assert full_payload["dataset_preparation"] == response.dataset_preparation
    assert full_payload["sample_rows"] == [{"Sample": "A", "SiO2": 50.0}]
    assert full_payload["source_sha256"] == "1" * 64
    assert full_payload["prepared_view_sha256"] == "2" * 64
