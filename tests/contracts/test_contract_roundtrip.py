import json
from typing import Any, Dict

import pytest
from geochemistrypi_contracts import (
    CONTRACT_VERSION,
    ArtifactRef,
    ClassificationExperimentSpec,
    ContractError,
    DatasetRef,
    ErrorResponse,
    EvaluationSpec,
    ExperimentResult,
    ModelSpec,
    PreprocessingSpec,
    SchemaName,
    SplitSpec,
    load_schema,
)
from jsonschema import Draft202012Validator
from referencing import Registry, Resource


def _schema_registry() -> Registry:
    resources = []
    for name in SchemaName:
        schema = load_schema(name)
        resources.append((schema["$id"], Resource.from_contents(schema)))
    return Registry().with_resources(resources)


def _validate(schema_name: SchemaName, payload: Dict[str, Any]) -> None:
    validator = Draft202012Validator(load_schema(schema_name), registry=_schema_registry())
    validator.validate(payload)


def _json_round_trip(payload: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(payload, allow_nan=False))


@pytest.fixture
def classification_spec() -> ClassificationExperimentSpec:
    return ClassificationExperimentSpec(
        schema_version=CONTRACT_VERSION,
        client_request_id="paper-benchmark-001",
        dataset=DatasetRef(
            kind="local_file",
            path="D:/data/geochemistry.csv",
            format="csv",
            id_column="Sample_ID",
            snapshot_policy="copy",
            expected_sha256="a" * 64,
        ),
        target_column="Deposit_Type",
        feature_columns=("SiO2", "Fe2O3", "MgO"),
        group_column=None,
        preprocessing=PreprocessingSpec(
            missing_values="median",
            scaling="standard",
            class_balance="none",
        ),
        split=SplitSpec(
            strategy="stratified_random",
            test_size=0.2,
            group_column=None,
            random_seed=42,
        ),
        model=ModelSpec(
            name="random_forest",
            mode="manual",
            parameters={
                "n_estimators": 300,
                "max_depth": 10,
                "class_weight": {"ore": 2.0, "waste": 1.0},
                "thresholds": (0.25, 0.5),
            },
        ),
        evaluation=EvaluationSpec(
            primary_metric="macro_f1",
            positive_label=None,
            cross_validation_folds=5,
        ),
    )


def test_classification_spec_round_trips_through_json_schema(classification_spec: ClassificationExperimentSpec) -> None:
    payload = classification_spec.to_dict()

    _validate(SchemaName.CLASSIFICATION_EXPERIMENT_SPEC, payload)
    wire_payload = _json_round_trip(payload)
    restored = ClassificationExperimentSpec.from_dict(wire_payload)

    assert restored == classification_spec
    assert restored.to_dict() == payload
    assert payload["model"]["parameters"]["thresholds"] == [0.25, 0.5]


def test_group_split_round_trip_preserves_group_column() -> None:
    spec = ClassificationExperimentSpec(
        schema_version=CONTRACT_VERSION,
        dataset=DatasetRef(
            kind="local_file",
            path="D:/data/grouped.xlsx",
            format="xlsx",
            id_column="Sample_ID",
            snapshot_policy="copy",
        ),
        target_column="Deposit_Type",
        feature_columns=None,
        group_column="Drillhole_ID",
        preprocessing=PreprocessingSpec(),
        split=SplitSpec(strategy="group", group_column="Drillhole_ID"),
        model=ModelSpec(name="random_forest"),
        evaluation=EvaluationSpec(),
    )

    payload = spec.to_dict()
    _validate(SchemaName.CLASSIFICATION_EXPERIMENT_SPEC, payload)

    assert ClassificationExperimentSpec.from_dict(_json_round_trip(payload)) == spec


def test_experiment_result_round_trips_through_json_schema() -> None:
    result = ExperimentResult(
        schema_version=CONTRACT_VERSION,
        run_id="20260725-143011-a82c4f",
        request_hash="b" * 64,
        status="completed",
        metrics={
            "accuracy": 0.91,
            "balanced_accuracy": 0.88,
            "macro_f1": 0.87,
            "weighted_f1": 0.89,
        },
        artifacts=(
            ArtifactRef(
                artifact_id="trained-pipeline",
                role="trained_pipeline",
                media_type="application/x-joblib",
                relative_path="artifacts/model/pipeline.joblib",
                size_bytes=123456,
                sha256="c" * 64,
            ),
        ),
        warnings=(),
        manifest_path="manifest.json",
        provenance_path="provenance.json",
    )

    payload = result.to_dict()
    _validate(SchemaName.EXPERIMENT_RESULT, payload)
    restored = ExperimentResult.from_dict(_json_round_trip(payload))

    assert restored == result
    assert restored.to_dict() == payload


def test_error_response_round_trips_through_json_schema() -> None:
    response = ErrorResponse(
        error=ContractError(
            code="INVALID_TARGET_COLUMN",
            message="Target column was not found.",
            stage="validation",
            run_id=None,
            retryable=False,
            details={"target_column": "Deposit_Type", "available_column_count": 12},
        )
    )

    payload = response.to_dict()
    _validate(SchemaName.ERROR_RESPONSE, payload)
    restored = ErrorResponse.from_dict(_json_round_trip(payload))

    assert restored == response
    assert restored.to_dict() == payload
