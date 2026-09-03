"""Unit tests for token-bounded analysis-validation projections."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from geochemistrypi_mcp.api.schemas import (
    AnalysisValidationResponse,
    AnomalyDetectionRequest,
    ArtifactRequirement,
    ClassificationRequest,
    ClusteringRequest,
    DecompositionRequest,
    RegressionRequest,
    TimeSeriesArtifactRequirement,
    TimeSeriesRequest,
)
from geochemistrypi_mcp.api.validation_views import (
    CompactAnalysisValidationResponse,
    CompactMappingReceipt,
    CompactSequenceReceipt,
    CompactTextReceipt,
    _compact_execution_decisions,
    compact_analysis_validation,
    full_analysis_validation_detail,
)
from geochemistrypi_mcp.runtime.runs import _resolved_model_parameters, _validation_execution_decisions
from pydantic import BaseModel, ValidationError


def _sha(character: str) -> str:
    return character * 64


def _complete_receipt_value(value: object) -> object:
    """Restore an untruncated receipt for semantic-equivalence assertions."""
    if isinstance(value, CompactTextReceipt):
        assert value.truncated is False
        return value.text
    if isinstance(value, CompactSequenceReceipt):
        assert value.truncated is False
        assert len(value.prefix) == value.total_count
        return tuple(_complete_receipt_value(item) for item in value.prefix)
    if isinstance(value, CompactMappingReceipt):
        assert value.truncated is False
        assert len(value.prefix) == value.total_count
        return {_complete_receipt_value(entry.key): _complete_receipt_value(entry.value) for entry in value.prefix}
    if isinstance(value, BaseModel):
        return {field: _complete_receipt_value(getattr(value, field)) for field in value.__class__.model_fields}
    if isinstance(value, tuple):
        return tuple(_complete_receipt_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _complete_receipt_value(item) for key, item in value.items()}
    return value


def test_complete_literal_sequence_receipt_rejects_a_stale_sha256() -> None:
    with pytest.raises(
        ValidationError,
        match="complete compact literal sequence must match its SHA-256",
    ):
        CompactSequenceReceipt[str](
            prefix=("application/json",),
            total_count=1,
            truncated=False,
            sha256="0" * 64,
        )


def _canonical_json_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, tuple):
        return [_canonical_json_value(item) for item in value]
    if isinstance(value, list):
        return [_canonical_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _canonical_json_value(item) for key, item in value.items()}
    return value


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        _canonical_json_value(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _full_validation_response() -> AnalysisValidationResponse:
    requirements = tuple(
        ArtifactRequirement(
            requirement_id=f"artifact-{index:02d}",
            scientific_type="classification_evidence",
            output_role=f"native-output-{index:02d}",
            required=True,
            category="artifacts",
            media_types=(
                "application/json",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
            expected_relative_path=f"artifacts/model-{index:02d}/" + ("native-output-" * 18) + ".xlsx",
            path_pattern=f"artifacts/model-{index:02d}/**/*.xlsx",
            minimum_count=1,
            maximum_count=2,
            required_json_keys=tuple(f"required_key_{item:02d}" for item in range(16)),
        )
        for index in range(16)
    )
    return AnalysisValidationResponse(
        validation_id="val-0123456789abcdef0123456789abcdef",
        request_hash=_sha("1"),
        canonical_contract_hash=_sha("2"),
        compiled_plan_hash=_sha("3"),
        validation_expires_at="2026-08-29T12:34:56+00:00",
        execution_ready=False,
        comparison_ready=False,
        scientific_status="requirements_unmet",
        adapter_status="requirements_unmet",
        artifact_status="requirements_unmet",
        environment_status="MISMATCH",
        workflow_family="supervised_learning",
        workflow_mode="classification",
        method="xgboost",
        scientific_contract_id="scientific-contract-v1/classification",
        adapter_id="classification-automation-adapter",
        adapter_version="2.1.0",
        adapter_identity="classification-automation-adapter@2.1.0",
        artifact_requirements=requirements,
        blocking_issues=(
            "environment identity does not match the requested profile",
            "one or more native artifacts cannot be planned",
        ),
        task="classification",
        models=("xgboost", "random_forest"),
        estimated_model_count=2,
        tuning="manual",
        training_source="path",
        training_dataset_path=r"D:\private\training\Data_Classification.xlsx",
        training_sha256=_sha("5"),
        training_size_bytes=123_456,
        source_dataset_path=r"D:\private\source\Data_Classification.xlsx",
        source_dataset_sha256=_sha("4"),
        dataset_preparation={
            "contract_hash": _sha("7"),
            "table": {
                "input_row_count": 2011,
                "source_row_count": 2000,
                "filtered_row_count": 11,
            },
            "resolved_paths": [rf"D:\private\preparation\stage-{index:03d}" for index in range(80)],
            "provenance": [{"operation": "prepare", "record": "training-preparation-record-" * 12} for _ in range(40)],
        },
        dataset_preparation_sha256=_sha("6"),
        environment_identity_sha256=_sha("d"),
        environment_profile={
            "requested": {
                "profile_id": "publication-runtime-v1",
                "expected_identity_sha256": _sha("e"),
                "records": ["requested-environment-record-" * 12 for _ in range(30)],
            },
            "observed": {
                "executable": r"D:\private\environment\python.exe",
                "packages": [{"name": f"package-{index:03d}", "record": "installed-package-record-" * 14} for index in range(60)],
            },
        },
        environment_profile_identity_sha256=_sha("f"),
        requested_seeds={"split": 2025, "model": 2025},
        effective_seeds={"split": 2025, "model": 2025, "cross_validation": 2025},
        execution_decisions={
            "evaluation": {
                "requested_mode": "holdout",
                "effective_mode": "internal_holdout",
                "requested_test_ratio": 0.2,
                "effective_test_ratio": 0.2,
                "requested_split_strategy": "stratified_holdout",
                "effective_split_strategy": "stratified_holdout",
                "requested_cross_validation_folds": None,
                "effective_cross_validation_folds": 10,
                "requested_metrics": ("accuracy", "precision", "recall", "f1"),
                "metric_artifact_bindings": {},
                "required_artifact_ids": (),
                "class_order": ("0", "1"),
                "requested_confusion_matrix_normalization": "none",
                "effective_confusion_matrix_normalization": "none",
                "requested_metric_average": "binary",
                "effective_metric_average": "binary",
                "requested_positive_label": {"type": "integer", "value": 1},
                "effective_positive_label": {"type": "integer", "value": 1},
            },
            "preprocessing": {
                "missing_values": {
                    "method": "impute",
                    "strategy": "constant",
                    "fill_value": -999.0,
                },
                "scaling": "none",
                "feature_selection": {
                    "method": "select_k_best",
                    "retain_count": 2,
                },
                "engineered_features": ({"name": "oxide_ratio", "formula": "`SIO2(WT%)` / `TIO2(WT%)`"},),
                "label_customization": {
                    "strategy": "map",
                    "mapping": {"0": "basalt", "1": "other"},
                },
                "world_map": {
                    "enabled": True,
                    "longitude_column": "LONGITUDE",
                    "latitude_column": "LATITUDE",
                    "value_columns": ("SIO2(WT%)",),
                },
                "target_transformations": {},
                "sample_balancing": "none",
                "metadata_columns": (),
                "feature_engineering": None,
            },
            "application": {
                "enabled": True,
                "role": "inference",
                "training_identifier_column": "SAMPLE NAME",
                "secondary_identifier_column": "SAMPLE NAME",
                "target_columns": (),
                "label_used_as_feature": False,
            },
            "bindings": {
                "model": "interaction_plan",
                "preprocessing": "interaction_plan",
                "scientific_execution_contract_bound": True,
                "workflow_specific_contract": None,
            },
        },
        parameter_binding={"bindings": [{"parameter": f"parameter-{index:03d}", "evidence": "binding-evidence-" * 14} for index in range(40)]},
        adapter_artifact_mappings=tuple(
            {
                "requirement_id": f"artifact-{index:02d}",
                "native_paths": [rf"D:\private\outputs\model-{index:02d}\artifact-{item:02d}.xlsx" for item in range(8)],
                "mapping_record": "complete-adapter-artifact-mapping-" * 12,
            }
            for index in range(16)
        ),
        source_row_count=2000,
        row_identity_scheme="sha256(canonical-row-json)",
        row_identity_sha256=_sha("b"),
        columns=("SAMPLE NAME", "SIO2(WT%)", "TIO2(WT%)", "AL2O3(WT%)", "Label"),
        identifier_column="SAMPLE NAME",
        feature_columns=("SIO2(WT%)", "TIO2(WT%)", "AL2O3(WT%)"),
        selected_columns=("SAMPLE NAME", "SIO2(WT%)", "TIO2(WT%)", "AL2O3(WT%)", "Label"),
        target_column="Label",
        target_columns=("Label",),
        resolved_model_parameters={
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 4,
            "tree_method": "hist",
            "use_label_encoder": False,
        },
        application_source="path",
        application_dataset_path=r"D:\private\application\ApplicationData_Classification.xlsx",
        application_sha256=_sha("9"),
        application_source_sha256=_sha("8"),
        application_preparation={
            "contract_hash": _sha("a"),
            "table": {
                "input_row_count": 1006,
                "source_row_count": 1000,
                "filtered_row_count": 6,
            },
            "resolved_paths": [rf"D:\private\application-preparation\stage-{index:03d}" for index in range(80)],
            "provenance": [{"operation": "prepare", "record": "application-preparation-record-" * 12} for _ in range(40)],
        },
        application_source_row_count=1000,
        application_row_identity_sha256=_sha("c"),
        experiment_mode="existing",
        experiment_name="classification-audit",
        existing_experiment_id="42",
        interaction_plan=r"D:\private\plans\classification-interaction-plan.json",
        warnings=("native outputs remain unverified until execution",),
    )


def test_compact_validation_preserves_start_stop_evidence_and_removes_bulk() -> None:
    full = _full_validation_response()

    compact = compact_analysis_validation(full)
    payload = compact.model_dump(mode="json")

    assert payload["response_detail"] == "compact"
    assert payload["validation_id"] == full.validation_id
    assert payload["request_hash"] == full.request_hash
    assert payload["contains_truncated_content"] is False
    assert payload["truncated_sections"] == []
    assert payload["start_relevant_content_complete"] is True
    assert payload["full_detail_request"] == {
        "validation_id": full.validation_id,
        "request_hash": full.request_hash,
        "detail": "full",
    }
    assert payload["canonical_contract_hash"] == full.canonical_contract_hash
    assert payload["compiled_plan_hash"] == full.compiled_plan_hash
    assert payload["validation_expires_at"] == full.validation_expires_at
    assert payload["readiness"] == {
        "valid": True,
        "execution_ready": False,
        "comparison_ready": False,
        "claim_ready": False,
        "schema_status": "valid",
        "scientific_status": "requirements_unmet",
        "adapter_status": "requirements_unmet",
        "artifact_status": "requirements_unmet",
        "environment_status": "MISMATCH",
    }
    assert _complete_receipt_value(compact.blocking_issues) == full.blocking_issues
    assert compact.blocking_issues.sha256 == _canonical_json_sha256(full.blocking_issues)
    assert _complete_receipt_value(compact.warnings) == full.warnings
    assert (compact.task, compact.workflow_family, compact.workflow_mode, compact.method) == (
        "classification",
        "supervised_learning",
        "classification",
        "xgboost",
    )
    assert (compact.adapter_id, compact.adapter_version, compact.adapter_identity) == (
        full.adapter_id,
        full.adapter_version,
        full.adapter_identity,
    )
    assert compact.models == full.models
    assert compact.estimated_model_count == 2
    assert compact.tuning == "manual"
    assert compact.training.model_dump(mode="json") == {
        "source": "path",
        "source_sha256": _sha("4"),
        "prepared_sha256": _sha("5"),
        "prepared_size_bytes": 123456,
        "source_row_count": 2011,
        "prepared_row_count": 2000,
        "dropped_row_count": 11,
        "preparation_sha256": _sha("6"),
        "row_identity_scheme": "sha256(canonical-row-json)",
        "row_identity_sha256": _sha("b"),
    }
    assert compact.application is not None
    assert compact.application.model_dump(mode="json") == {
        "source": "path",
        "source_sha256": _sha("8"),
        "prepared_sha256": _sha("9"),
        "prepared_size_bytes": None,
        "source_row_count": 1006,
        "prepared_row_count": 1000,
        "dropped_row_count": 6,
        "preparation_sha256": _sha("a"),
        "row_identity_scheme": None,
        "row_identity_sha256": _sha("c"),
    }
    assert compact.event is None
    assert _complete_receipt_value(compact.column_roles.columns) == full.columns
    assert compact.column_roles.identifier_column == full.identifier_column
    assert _complete_receipt_value(compact.column_roles.feature_columns) == full.feature_columns
    assert _complete_receipt_value(compact.column_roles.selected_columns) == full.selected_columns
    assert compact.column_roles.target_column == full.target_column
    assert _complete_receipt_value(compact.column_roles.target_columns) == full.target_columns
    assert compact.requested_seeds == full.requested_seeds
    assert compact.effective_seeds == full.effective_seeds
    assert _complete_receipt_value(compact.execution_decisions) == full.execution_decisions
    assert compact.resolved_model_parameters == full.resolved_model_parameters
    assert compact.artifact_requirement_count == 16
    assert compact.artifact_requirements.total_count == 16
    assert compact.artifact_requirements.sha256 == _canonical_json_sha256(full.artifact_requirements)
    assert compact.artifact_requirements.truncated is False
    assert len(compact.artifact_requirements.prefix) == 16
    first_requirement = compact.artifact_requirements.prefix[0]
    first_requirement_payload = first_requirement.model_dump(mode="json")
    media_types = first_requirement_payload.pop("media_types")
    required_json_keys = first_requirement_payload.pop("required_json_keys")
    assert first_requirement_payload == {
        "requirement_id": "artifact-00",
        "scientific_type": "classification_evidence",
        "output_role": "native-output-00",
        "required": True,
        "category": "artifacts",
        "expected_relative_path": "artifacts/model-00/" + ("native-output-" * 18) + ".xlsx",
        "path_pattern": "artifacts/model-00/**/*.xlsx",
        "minimum_count": 1,
        "maximum_count": 2,
    }
    assert _complete_receipt_value(first_requirement.media_types) == (
        "application/json",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    assert first_requirement.media_types.prefix == (
        "application/json",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    assert _complete_receipt_value(first_requirement.required_json_keys) == tuple(f"required_key_{item:02d}" for item in range(16))
    assert first_requirement.required_json_keys.prefix == tuple(f"required_key_{item:02d}" for item in range(16))
    assert media_types["sha256"] == _canonical_json_sha256(
        (
            "application/json",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    )
    assert required_json_keys["sha256"] == _canonical_json_sha256(tuple(f"required_key_{item:02d}" for item in range(16)))
    assert compact.environment.model_dump(mode="json") == {
        "status": "MISMATCH",
        "observed_identity_sha256": _sha("d"),
        "requested_identity_sha256": _sha("e"),
        "profile_id": "publication-runtime-v1",
        "profile_identity_sha256": _sha("f"),
    }
    assert compact.experiment.model_dump(mode="json") == {
        "mode": "existing",
        "name": "classification-audit",
        "existing_experiment_id": "42",
    }
    assert compact.analysis_process_started is False

    for omitted_field in (
        "training_dataset_path",
        "source_dataset_path",
        "dataset_preparation",
        "environment_profile",
        "parameter_binding",
        "adapter_artifact_mappings",
        "application_dataset_path",
        "application_preparation",
        "interaction_plan",
    ):
        assert omitted_field not in payload

    compact_json = compact.model_dump_json()
    full_size = len(full.model_dump_json().encode("utf-8"))
    compact_size = len(compact_json.encode("utf-8"))
    assert compact_size <= 64 * 1024
    # The compact view still removes bulk execution records and paths, while
    # sequence-level receipts retain each artifact's MIME/JSON-key contract.
    assert compact_size < full_size * 0.55
    assert "D:\\\\private" not in compact_json
    assert "complete-adapter-artifact-mapping" not in compact_json
    assert "installed-package-record" not in compact_json
    assert "preparation-record" not in compact_json


def test_time_series_requirement_projects_to_exact_cardinality() -> None:
    full = _full_validation_response().model_copy(
        update={
            "task": "time_series",
            "workflow_family": "time_series",
            "workflow_mode": "subaerial_proportion",
            "method": "subaerial_proportion_bootstrap",
            "models": ("subaerial_proportion_bootstrap",),
            "estimated_model_count": 1,
            "tuning": "not_applicable",
            "artifact_requirements": (
                TimeSeriesArtifactRequirement(
                    requirement_id="subaerial-pdf",
                    scientific_type="subaerial_proportion_pdf",
                    path_pattern="Subaerial Proportion.pdf",
                    count=3,
                    required_json_keys=("rows", "columns"),
                ),
            ),
        }
    )

    requirement = compact_analysis_validation(full).artifact_requirements.prefix[0]

    assert requirement.required is True
    assert requirement.minimum_count == 3
    assert requirement.maximum_count == 3
    assert _complete_receipt_value(requirement.media_types) == ()
    assert _complete_receipt_value(requirement.required_json_keys) == (
        "rows",
        "columns",
    )


def test_full_validation_detail_recovers_every_compact_decision_sequence() -> None:
    full = _full_validation_response()
    compact = compact_analysis_validation(full)
    detail = full_analysis_validation_detail(full)

    assert detail.validation_id == compact.validation_id
    assert detail.request_hash == compact.request_hash
    assert detail.blocking_issues == full.blocking_issues
    assert detail.blocking_issues_sha256 == compact.blocking_issues.sha256
    assert detail.warnings == full.warnings
    assert detail.warnings_sha256 == compact.warnings.sha256
    assert detail.artifact_requirements == full.artifact_requirements
    assert detail.artifact_requirements_sha256 == compact.artifact_requirements.sha256
    assert detail.complete_validation_sha256 == _canonical_json_sha256(full)


def test_extreme_compact_validation_is_deterministic_hash_bound_and_under_64_kib() -> None:
    columns = tuple(f"COLUMN_{index:03d}_" + (chr(65 + index % 26) * 4_000) for index in range(256))
    diagnostics = tuple(f"diagnostic-{index:03d}:" + (chr(97 + index % 26) * 4_000) for index in range(256))
    requirements = tuple(
        ArtifactRequirement(
            requirement_id=f"artifact-{index:03d}",
            scientific_type="classification_evidence",
            output_role=f"native-output-{index:03d}",
            required=True,
            category="artifacts",
            expected_relative_path=(f"artifacts/model-{index:03d}/" + ("x" * 450) + ".json"),
            path_pattern=f"artifacts/model-{index:03d}/" + ("y" * 450) + "*.json",
            minimum_count=1,
            maximum_count=2,
            required_json_keys=tuple(f"key-{item:02d}" for item in range(64)),
        )
        for index in range(146)
    )
    large_mapping = {f"key-{index:03d}-" + ("k" * 500): f"value-{index:03d}-" + ("v" * 4_000) for index in range(256)}
    full = _full_validation_response().model_copy(
        update={
            "columns": columns,
            "feature_columns": columns,
            "selected_columns": columns,
            "target_columns": columns,
            "blocking_issues": diagnostics,
            "warnings": tuple(reversed(diagnostics)),
            "artifact_requirements": requirements,
            "execution_decisions": {
                "evaluation": {
                    "requested_mode": "holdout",
                    "effective_mode": "internal_holdout",
                    "requested_test_ratio": 0.2,
                    "effective_test_ratio": 0.2,
                    "requested_split_strategy": "stratified_holdout",
                    "effective_split_strategy": "stratified_holdout",
                    "requested_cross_validation_folds": 10,
                    "effective_cross_validation_folds": 10,
                    "requested_metrics": diagnostics,
                    "metric_artifact_bindings": large_mapping,
                    "required_artifact_ids": diagnostics,
                    "class_order": diagnostics,
                    "requested_confusion_matrix_normalization": "none",
                    "effective_confusion_matrix_normalization": "none",
                    "requested_metric_average": "binary",
                    "effective_metric_average": "binary",
                    "requested_positive_label": {"type": "integer", "value": 1},
                    "effective_positive_label": {"type": "integer", "value": 1},
                },
                "preprocessing": {
                    "missing_values": {"method": "drop_rows", "columns": columns},
                    "scaling": "none",
                    "feature_selection": {"method": "none"},
                    "engineered_features": tuple(
                        {
                            "name": f"feature_{index:03d}",
                            "formula": "x" * 500,
                        }
                        for index in range(256)
                    ),
                    "label_customization": {
                        "strategy": "map",
                        "mapping": large_mapping,
                    },
                    "world_map": {
                        "enabled": True,
                        "longitude_column": "LONGITUDE",
                        "latitude_column": "LATITUDE",
                        "value_columns": columns,
                    },
                    "target_transformations": {column: {"scale": 1.0, "offset": float(index)} for index, column in enumerate(columns)},
                    "sample_balancing": "none",
                    "metadata_columns": columns,
                    "feature_engineering": "none",
                },
                "application": {
                    "enabled": True,
                    "role": "inference",
                    "training_identifier_column": "SAMPLE",
                    "secondary_identifier_column": "SAMPLE",
                    "target_columns": columns,
                    "label_used_as_feature": False,
                },
                "bindings": {
                    "model": "interaction_plan",
                    "preprocessing": "interaction_plan",
                    "scientific_execution_contract_bound": True,
                    "workflow_specific_contract": None,
                },
            },
        }
    )
    full_json_before = full.model_dump_json()

    first = compact_analysis_validation(full)
    second = compact_analysis_validation(full)
    first_json = first.model_dump_json()

    assert len(first_json.encode("utf-8")) <= 64 * 1024
    assert first_json == second.model_dump_json()
    assert full.model_dump_json() == full_json_before
    assert columns[-1] in full_json_before
    assert diagnostics[-1] in full_json_before

    assert first.column_roles.columns.total_count == 256
    assert first.column_roles.columns.truncated is True
    assert first.column_roles.columns.sha256 == _canonical_json_sha256(columns)
    assert len(first.column_roles.columns.prefix) < 16
    assert first.contains_truncated_content is True
    assert first.start_relevant_content_complete is False
    assert "column_roles.columns" in first.truncated_sections
    assert any(section != "column_roles.columns" for section in first.truncated_sections)
    assert first.blocking_issues.total_count == 256
    assert first.blocking_issues.sha256 == _canonical_json_sha256(diagnostics)
    assert first.blocking_issues.prefix[0].truncated is True
    assert first.blocking_issues.prefix[0].sha256 == _canonical_json_sha256(diagnostics[0])
    assert first.artifact_requirements.total_count == 146
    assert first.artifact_requirements.truncated is True
    assert first.artifact_requirements.sha256 == _canonical_json_sha256(requirements)
    assert first.execution_decisions.evaluation.requested_metrics.total_count == 256
    assert first.execution_decisions.preprocessing.target_transformations.total_count == 256


def test_s06_compact_validation_keeps_complete_scientific_decisions_and_outputs() -> None:
    observed_columns = tuple(f"OBSERVED_{index:02d}" for index in range(50)) + (
        "ROCK NAME",
        "R_AGE",
        "R_MAX_AGE",
        "Estimated Proportion of Subaerial Basalts",
        "LATITUDE",
        "LONGITUDE",
        "REFERENCE",
        "SOURCE",
        "NOTES",
    )
    selected_columns = (
        "ROCK NAME",
        "R_AGE",
        "R_MAX_AGE",
        "Estimated Proportion of Subaerial Basalts",
        "LATITUDE",
        "LONGITUDE",
        "REFERENCE",
        "SOURCE",
        "NOTES",
    )
    decisions = {
        "evaluation": {
            "requested_mode": "not_applicable",
            "effective_mode": "not_applicable",
            "requested_test_ratio": None,
            "effective_test_ratio": None,
            "requested_split_strategy": None,
            "effective_split_strategy": None,
            "requested_cross_validation_folds": None,
            "effective_cross_validation_folds": None,
            "requested_metrics": (),
            "metric_artifact_bindings": {},
            "required_artifact_ids": ("subaerial-pdf", "subaerial-csv"),
            "class_order": (),
            "requested_confusion_matrix_normalization": None,
            "effective_confusion_matrix_normalization": None,
            "requested_metric_average": None,
            "effective_metric_average": None,
            "requested_positive_label": None,
            "effective_positive_label": None,
        },
        "preprocessing": {
            "missing_values": {
                "method": "drop_rows",
                "columns": selected_columns,
            },
            "scaling": "none",
            "feature_selection": {"method": "none"},
            "engineered_features": (),
            "label_customization": None,
            "world_map": {"enabled": False},
            "target_transformations": {},
            "sample_balancing": "none",
            "metadata_columns": (),
            "feature_engineering": "none",
        },
        "application": {
            "enabled": False,
            "role": "none",
            "training_identifier_column": "ROCK NAME",
            "secondary_identifier_column": None,
            "target_columns": (),
            "label_used_as_feature": False,
        },
        "bindings": {
            "model": "interaction_plan",
            "preprocessing": "interaction_plan",
            "scientific_execution_contract_bound": True,
            "workflow_specific_contract": {
                "contract_type": "time_series_subaerial_proportion",
                "identifier_column": "ROCK NAME",
                "selected_columns": selected_columns,
                "sheet": "Sheet1",
                "age_column": "R_AGE",
                "maximum_age_column": "R_MAX_AGE",
                "probability_column": "Estimated Proportion of Subaerial Basalts",
                "latitude_column": "LATITUDE",
                "longitude_column": "LONGITUDE",
            },
        },
    }
    full = _full_validation_response().model_copy(
        update={
            "execution_ready": True,
            "scientific_status": "valid",
            "adapter_status": "available",
            "artifact_status": "planned",
            "environment_status": "READY",
            "blocking_issues": (),
            "warnings": (),
            "task": "time_series",
            "workflow_family": "time_series",
            "workflow_mode": "subaerial_proportion",
            "method": "subaerial_proportion_bootstrap",
            "models": ("subaerial_proportion_bootstrap",),
            "estimated_model_count": 1,
            "tuning": "not_applicable",
            "columns": observed_columns,
            "identifier_column": "ROCK NAME",
            "feature_columns": (),
            "selected_columns": selected_columns,
            "target_column": None,
            "target_columns": (),
            "execution_decisions": decisions,
            "resolved_model_parameters": {
                "bin_width": 100,
                "iterations": 100,
                "seed": 2025,
                "curve_fitting": False,
            },
            "artifact_requirements": (
                TimeSeriesArtifactRequirement(
                    requirement_id="subaerial-pdf",
                    scientific_type="subaerial_proportion_pdf",
                    path_pattern="Subaerial Proportion.pdf",
                    count=1,
                ),
                TimeSeriesArtifactRequirement(
                    requirement_id="subaerial-csv",
                    scientific_type="subaerial_proportion_csv",
                    path_pattern="Subaerial Proportion.csv",
                    count=1,
                ),
            ),
        }
    )

    compact = compact_analysis_validation(full)
    workflow = compact.execution_decisions.bindings.workflow_specific_contract

    assert len(compact.model_dump_json().encode("utf-8")) <= 64 * 1024
    assert compact.contains_truncated_content is True
    assert compact.truncated_sections == ("column_roles.columns",)
    assert compact.start_relevant_content_complete is True
    assert compact.full_detail_request.validation_id == full.validation_id
    assert compact.full_detail_request.request_hash == full.request_hash
    assert compact.column_roles.columns.total_count == 59
    assert compact.column_roles.columns.truncated is True
    assert _complete_receipt_value(compact.column_roles.selected_columns) == selected_columns
    assert compact.artifact_requirements.total_count == 2
    assert compact.artifact_requirements.truncated is False
    assert tuple(requirement.path_pattern for requirement in compact.artifact_requirements.prefix) == ("Subaerial Proportion.pdf", "Subaerial Proportion.csv")
    assert all(_complete_receipt_value(requirement.media_types) == () and _complete_receipt_value(requirement.required_json_keys) == () for requirement in compact.artifact_requirements.prefix)
    assert workflow is not None
    assert workflow.contract_type == "time_series_subaerial_proportion"
    assert _complete_receipt_value(workflow.selected_columns) == selected_columns
    assert workflow.identifier_column == "ROCK NAME"
    assert workflow.age_column == "R_AGE"
    assert workflow.maximum_age_column == "R_MAX_AGE"
    assert workflow.probability_column == "Estimated Proportion of Subaerial Basalts"
    assert compact.resolved_model_parameters == {
        "bin_width": 100,
        "iterations": 100,
        "seed": 2025,
        "curve_fitting": False,
    }


def test_compact_validation_model_forbids_undeclared_fields() -> None:
    payload = compact_analysis_validation(_full_validation_response()).model_dump(mode="json")
    payload["dataset_preparation"] = {"unexpected": "bulk record"}

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CompactAnalysisValidationResponse.model_validate(payload)


def _decision_plan(contract: dict[str, object] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        scientific_execution_contract_json=(json.dumps(contract, sort_keys=True) if contract is not None else None),
        model_parameter_binding="interaction_plan",
        preprocessing_parameter_binding="interaction_plan",
    )


def test_classification_decisions_preserve_effective_evaluation_and_preprocessing_contract() -> None:
    request = ClassificationRequest(
        training_dataset_path=r"D:\data\classification.csv",
        application_dataset_path=r"D:\data\classification-application.csv",
        experiment_name="classification-audit",
        run_name="classification-run",
        identifier_column="SAMPLE",
        feature_columns=("F1", "F2"),
        target_column="Label",
        evaluation={
            "mode": "holdout",
            "metrics": ("accuracy", "f1"),
            "split_strategy": "stratified_holdout",
            "class_order": ("0", "1"),
            "confusion_matrix_normalization": "none",
        },
        missing_values={
            "method": "impute",
            "strategy": "constant",
            "fill_value": -999.0,
        },
        scaling="min_max",
        feature_selection={"method": "select_k_best", "retain_count": 1},
        engineered_features=({"name": "F_RATIO", "formula": "F1 / F2"},),
        label_customization={
            "strategy": "map",
            "mapping": {"0": "negative", "1": "positive"},
        },
        world_map={
            "enabled": True,
            "longitude_column": "LONGITUDE",
            "latitude_column": "LATITUDE",
            "value_columns": ("F1",),
        },
        metric_average="binary",
        positive_label=1,
    )
    contract = {
        "evaluation_mode": "internal_holdout",
        "split_strategy": "stratified_holdout",
        "cross_validation_folds": 10,
        "confusion_matrix_normalization": None,
        "classification_metric_average": "binary",
        "classification_positive_label": {"type": "integer", "value": 1},
    }

    decisions = _validation_execution_decisions(
        request,
        _decision_plan(contract),
        application_enabled=True,
        secondary_is_evaluation=False,
    )
    _compact_execution_decisions(SimpleNamespace(execution_decisions=decisions))

    assert decisions["evaluation"] == {
        "requested_mode": "holdout",
        "effective_mode": "internal_holdout",
        "requested_test_ratio": 0.2,
        "effective_test_ratio": 0.2,
        "requested_split_strategy": "stratified_holdout",
        "effective_split_strategy": "stratified_holdout",
        "requested_cross_validation_folds": None,
        "effective_cross_validation_folds": 10,
        "requested_metrics": ("accuracy", "f1"),
        "metric_artifact_bindings": {},
        "required_artifact_ids": (),
        "class_order": ("0", "1"),
        "requested_confusion_matrix_normalization": "none",
        "effective_confusion_matrix_normalization": "none",
        "requested_metric_average": "binary",
        "effective_metric_average": "binary",
        "requested_positive_label": {"type": "integer", "value": 1},
        "effective_positive_label": {"type": "integer", "value": 1},
    }
    preprocessing = decisions["preprocessing"]
    assert preprocessing["missing_values"] == {
        "method": "impute",
        "strategy": "constant",
        "fill_value": -999.0,
    }
    assert preprocessing["feature_selection"] == {
        "method": "select_k_best",
        "retain_count": 1,
    }
    assert preprocessing["engineered_features"] == ({"name": "F_RATIO", "formula": "F1 / F2"},)
    assert preprocessing["label_customization"] == {
        "strategy": "map",
        "mapping": {"0": "negative", "1": "positive"},
    }
    assert preprocessing["world_map"] == {
        "enabled": True,
        "longitude_column": "LONGITUDE",
        "latitude_column": "LATITUDE",
        "value_columns": ["F1"],
    }
    assert decisions["application"]["role"] == "inference"
    assert decisions["application"]["secondary_identifier_column"] == "SAMPLE"


def test_regression_external_evaluation_reports_no_effective_holdout_or_cv() -> None:
    request = RegressionRequest(
        training_dataset_path=r"D:\data\regression.csv",
        experiment_name="regression-audit",
        run_name="regression-run",
        identifier_column="SAMPLE",
        feature_columns=("F1", "F2"),
        target_column="Age",
        target_transformations={"Age": {"scale": 0.001, "offset": 1.5}},
        evaluation={
            "mode": "external_labeled",
            "evaluation_dataset_path": r"D:\data\regression-evaluation.csv",
            "external_identifier_column": "EVAL_ID",
            "metrics": ("r2", "rmse"),
        },
    )
    contract = {
        "evaluation_mode": "external_labeled",
        "split_strategy": None,
        "cross_validation_folds": 10,
    }

    decisions = _validation_execution_decisions(
        request,
        _decision_plan(contract),
        application_enabled=True,
        secondary_is_evaluation=True,
    )
    _compact_execution_decisions(SimpleNamespace(execution_decisions=decisions))

    evaluation = decisions["evaluation"]
    assert evaluation["requested_test_ratio"] == 0.2
    assert evaluation["effective_test_ratio"] is None
    assert evaluation["requested_split_strategy"] == "cli_default"
    assert evaluation["effective_split_strategy"] is None
    assert evaluation["effective_cross_validation_folds"] is None
    assert decisions["preprocessing"]["target_transformations"] == {"Age": {"scale": 0.001, "offset": 1.5}}
    assert decisions["application"] == {
        "enabled": True,
        "role": "external_evaluation",
        "training_identifier_column": "SAMPLE",
        "secondary_identifier_column": "EVAL_ID",
        "target_columns": ("Age",),
        "label_used_as_feature": False,
    }


def test_clustering_decisions_preserve_unsupervised_preprocessing_values() -> None:
    request = ClusteringRequest(
        training_dataset_path=r"D:\data\clustering.csv",
        experiment_name="clustering-audit",
        run_name="clustering-run",
        identifier_column="SAMPLE",
        feature_columns=("F1", "F2"),
        evaluation={"mode": "quality_report", "metrics": ("silhouette",)},
        missing_values={
            "method": "impute",
            "strategy": "constant",
            "fill_value": 0.25,
        },
        scaling="mean_normalization",
        engineered_features=({"name": "F_SUM", "formula": "F1 + F2"},),
    )

    decisions = _validation_execution_decisions(
        request,
        _decision_plan(),
        application_enabled=False,
        secondary_is_evaluation=False,
    )
    _compact_execution_decisions(SimpleNamespace(execution_decisions=decisions))

    assert decisions["evaluation"]["effective_mode"] == "quality_report"
    assert decisions["evaluation"]["effective_split_strategy"] is None
    assert decisions["evaluation"]["effective_cross_validation_folds"] is None
    assert decisions["preprocessing"]["missing_values"]["fill_value"] == 0.25
    assert decisions["preprocessing"]["scaling"] == "mean_normalization"
    assert decisions["preprocessing"]["engineered_features"] == ({"name": "F_SUM", "formula": "F1 + F2"},)


def test_decomposition_overlay_reports_label_binding_not_inference() -> None:
    request = DecompositionRequest(
        mode="embedding_label_overlay",
        training_dataset_path=r"D:\data\coordinates.xlsx",
        application_dataset_path=r"D:\data\labels.xlsx",
        experiment_name="overlay-audit",
        run_name="overlay-run",
        identifier_column="COORD_ID",
        feature_columns=("X", "Y"),
        scaling="none",
        coordinate_sheet="coordinates",
        label_sheet="labels",
        label_identifier_column="LABEL_ID",
        label_column="Class",
        positive_label_values=("ore", "mineralized"),
    )

    decisions = _validation_execution_decisions(
        request,
        _decision_plan(),
        application_enabled=True,
        secondary_is_evaluation=False,
    )
    _compact_execution_decisions(SimpleNamespace(execution_decisions=decisions))

    assert decisions["evaluation"]["effective_mode"] == "not_applicable"
    assert decisions["application"] == {
        "enabled": True,
        "role": "artifact_overlay",
        "training_identifier_column": "COORD_ID",
        "secondary_identifier_column": "LABEL_ID",
        "target_columns": ("Class",),
        "label_used_as_feature": False,
    }
    assert decisions["bindings"]["workflow_specific_contract"] == {
        "contract_type": "decomposition_embedding_label_overlay",
        "coordinate_sheet": "coordinates",
        "label_sheet": "labels",
        "coordinate_identifier_column": "COORD_ID",
        "label_identifier_column": "LABEL_ID",
        "label_column": "Class",
        "positive_label_values": ("ore", "mineralized"),
        "join_policy": "exact_identifier_set_one_to_one",
    }
    assert _resolved_model_parameters(request) == {
        "mode": "embedding_label_overlay",
        "join_policy": "exact_identifier_set_one_to_one",
    }


def test_anomaly_detection_does_not_report_irrelevant_contract_cv_folds() -> None:
    request = AnomalyDetectionRequest(
        training_dataset_path=r"D:\data\anomaly.csv",
        experiment_name="anomaly-audit",
        run_name="anomaly-run",
        identifier_column="SAMPLE",
        feature_columns=("F1", "F2"),
        evaluation={"mode": "quality_report", "metrics": ("outlier_count",)},
        model={
            "type": "local_outlier_factor",
            "detection_mode": "novelty_detection",
        },
    )
    contract = {
        "evaluation_mode": "novelty_detection",
        "split_strategy": None,
        "cross_validation_folds": 10,
    }

    decisions = _validation_execution_decisions(
        request,
        _decision_plan(contract),
        application_enabled=False,
        secondary_is_evaluation=False,
    )
    _compact_execution_decisions(SimpleNamespace(execution_decisions=decisions))

    assert decisions["evaluation"]["requested_mode"] == "quality_report"
    assert decisions["evaluation"]["effective_mode"] == "quality_report"
    assert decisions["evaluation"]["effective_cross_validation_folds"] is None
    assert decisions["evaluation"]["effective_split_strategy"] is None


def test_time_series_reference_contract_preserves_roles_event_and_active_parameters() -> None:
    request = TimeSeriesRequest(
        mode="reference_anomaly_series",
        training_dataset_path=r"D:\data\observations.xlsx",
        experiment_name="reference-audit",
        run_name="reference-run",
        identifier_column="SAMPLE",
        selected_columns=("DATE", "SIGNAL_A", "SIGNAL_B", "REFERENCE", "COMPARISON"),
        sheet="observations",
        time_column="DATE",
        signal_columns=("SIGNAL_A", "SIGNAL_B"),
        reference_label_column="REFERENCE",
        reference_positive_values=("1", "yes"),
        reference_label_provenance="expert_review",
        comparison_label_column="COMPARISON",
        comparison_positive_values=("flag",),
        comparison_label_provenance="calculated",
        event_dataset_path=r"D:\data\events.xlsx",
        event_sheet="events",
        event_time_column="EVENT_DATE",
        event_identifier_column="EVENT_ID",
        event_filter_column="EVENT_TYPE",
        event_filter_values=("eruption", "impact"),
        association_window_days=30.0,
        association_direction="symmetric",
        evaluation={"mode": "reference_comparison"},
    )

    decisions = _validation_execution_decisions(
        request,
        _decision_plan(),
        application_enabled=False,
        secondary_is_evaluation=False,
    )
    _compact_execution_decisions(SimpleNamespace(execution_decisions=decisions))

    workflow = decisions["bindings"]["workflow_specific_contract"]
    assert workflow == {
        "contract_type": "time_series_reference_anomaly_series",
        "identifier_column": "SAMPLE",
        "selected_columns": (
            "DATE",
            "SIGNAL_A",
            "SIGNAL_B",
            "REFERENCE",
            "COMPARISON",
        ),
        "sheet": "observations",
        "time_column": "DATE",
        "signal_columns": ("SIGNAL_A", "SIGNAL_B"),
        "reference_label_column": "REFERENCE",
        "reference_positive_values": ("1", "yes"),
        "comparison_label_column": "COMPARISON",
        "comparison_positive_values": ("flag",),
        "event_sheet": "events",
        "event_time_column": "EVENT_DATE",
        "event_identifier_column": "EVENT_ID",
        "event_filter_column": "EVENT_TYPE",
        "event_filter_values": ("eruption", "impact"),
    }
    assert _resolved_model_parameters(request) == {
        "mode": "reference_anomaly_series",
        "sheet": "observations",
        "reference_label_provenance": "expert_review",
        "comparison_label_provenance": "calculated",
        "event_sheet": "events",
        "association_window_days": 30.0,
        "association_direction": "symmetric",
    }


def test_compact_validation_reports_event_identity_without_path() -> None:
    full = _full_validation_response().model_copy(
        update={
            "task": "time_series",
            "workflow_family": "time_series",
            "workflow_mode": "reference_anomaly_series",
            "event_dataset_path": r"D:\private\events.xlsx",
            "event_source_sha256": _sha("0"),
            "event_size_bytes": 42_123,
        }
    )

    compact = compact_analysis_validation(full)

    assert compact.event is not None
    assert compact.event.model_dump(mode="json") == {
        "source": "path",
        "source_sha256": _sha("0"),
        "size_bytes": 42_123,
    }
    assert "events.xlsx" not in compact.model_dump_json()


def test_legacy_decisions_remain_unknown_instead_of_being_invented() -> None:
    full = _full_validation_response().model_copy(update={"execution_decisions": {}})

    decisions = compact_analysis_validation(full).execution_decisions

    assert decisions.evaluation.requested_mode == "not_reported"
    assert decisions.evaluation.effective_mode == "not_reported"
    assert decisions.evaluation.effective_test_ratio is None
    assert decisions.preprocessing.missing_values is None
    assert decisions.preprocessing.scaling is None
    assert decisions.application.enabled is None
    assert decisions.application.role == "not_reported"
    assert decisions.application.label_used_as_feature is None
    assert decisions.bindings.model == "not_reported"
    assert decisions.bindings.scientific_execution_contract_bound is None
