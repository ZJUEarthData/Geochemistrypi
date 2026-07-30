import copy
import hashlib
from typing import Any, Dict

import pytest
from geochemistrypi_contracts import (
    CONTRACT_VERSION,
    SCHEMA_BASE_URI,
    SCHEMA_FILENAMES,
    ClassificationExperimentSpec,
    DatasetRef,
    EvaluationSpec,
    ModelSpec,
    PreprocessingSpec,
    SchemaName,
    SplitSpec,
    load_schema,
    schema_bytes,
    schema_id,
    schema_sha256,
)
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from referencing import Registry, Resource


def _registry() -> Registry:
    schemas = [load_schema(name) for name in SchemaName]
    return Registry().with_resources((schema["$id"], Resource.from_contents(schema)) for schema in schemas)


def _validator(name: SchemaName) -> Draft202012Validator:
    return Draft202012Validator(load_schema(name), registry=_registry())


def _minimal_dataset() -> Dict[str, Any]:
    return {
        "kind": "local_file",
        "path": "D:/data/geochemistry.csv",
        "format": "csv",
        "id_column": None,
        "snapshot_policy": "copy",
    }


def _minimal_classification_request() -> Dict[str, Any]:
    return {
        "schema_version": CONTRACT_VERSION,
        "dataset": _minimal_dataset(),
        "target_column": "Deposit_Type",
        "preprocessing": {},
        "split": {},
        "model": {"name": "random_forest"},
        "evaluation": {},
    }


def test_all_schemas_use_draft_2020_12_stable_ids_and_contract_version() -> None:
    expected_ids = {
        SchemaName.DATASET_REF: f"{SCHEMA_BASE_URI}dataset-ref.schema.json",
        SchemaName.CLASSIFICATION_EXPERIMENT_SPEC: f"{SCHEMA_BASE_URI}classification-experiment-spec.schema.json",
        SchemaName.EXPERIMENT_RESULT: f"{SCHEMA_BASE_URI}experiment-result.schema.json",
        SchemaName.ERROR_RESPONSE: f"{SCHEMA_BASE_URI}error-response.schema.json",
    }

    assert set(SCHEMA_FILENAMES) == set(SchemaName)
    for name in SchemaName:
        schema = load_schema(name)
        Draft202012Validator.check_schema(schema)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["x-contract-version"] == CONTRACT_VERSION
        assert schema_id(name) == expected_ids[name]
        assert schema_sha256(name) == hashlib.sha256(schema_bytes(name)).hexdigest()
        assert len(schema_sha256(name)) == 64


def test_dataset_schema_distinguishes_missing_null_and_defaulted_fields() -> None:
    validator = _validator(SchemaName.DATASET_REF)
    payload = _minimal_dataset()

    validator.validate(payload)
    assert "read_options" not in payload
    assert payload["id_column"] is None

    canonical = DatasetRef.from_dict(payload).to_dict()
    assert canonical["id_column"] is None
    assert canonical["read_options"]["encoding"] == "utf-8"
    assert canonical["expected_sha256"] is None

    missing_required = copy.deepcopy(payload)
    del missing_required["path"]
    with pytest.raises(ValidationError):
        validator.validate(missing_required)


@pytest.mark.parametrize(
    "change",
    [
        {"unexpected": True},
        {"format": "csv", "path": "D:/data/geochemistry.xlsx"},
        {"format": "parquet"},
        {"expected_sha256": "not-a-sha256"},
    ],
)
def test_dataset_schema_rejects_unknown_or_inconsistent_values(change: Dict[str, Any]) -> None:
    payload = _minimal_dataset()
    payload.update(change)

    with pytest.raises(ValidationError):
        _validator(SchemaName.DATASET_REF).validate(payload)


def test_classification_schema_rejects_unknown_fields_and_versions() -> None:
    validator = _validator(SchemaName.CLASSIFICATION_EXPERIMENT_SPEC)
    payload = _minimal_classification_request()
    validator.validate(payload)

    bad_version = copy.deepcopy(payload)
    bad_version["schema_version"] = "2.0"
    with pytest.raises(ValidationError):
        validator.validate(bad_version)

    unknown_field = copy.deepcopy(payload)
    unknown_field["execute_python"] = "print('unsafe')"
    with pytest.raises(ValidationError):
        validator.validate(unknown_field)


def test_group_split_requires_group_columns_in_schema_and_dataclass() -> None:
    payload = _minimal_classification_request()
    payload["split"] = {"strategy": "group", "group_column": "Drillhole_ID"}

    with pytest.raises(ValidationError):
        _validator(SchemaName.CLASSIFICATION_EXPERIMENT_SPEC).validate(payload)

    with pytest.raises(ValueError, match="must match"):
        ClassificationExperimentSpec(
            schema_version=CONTRACT_VERSION,
            dataset=DatasetRef.from_dict(_minimal_dataset()),
            target_column="Deposit_Type",
            group_column="Site_ID",
            preprocessing=PreprocessingSpec(),
            split=SplitSpec(strategy="group", group_column="Drillhole_ID"),
            model=ModelSpec(name="random_forest"),
            evaluation=EvaluationSpec(),
        )


def test_dataclasses_reject_unknown_fields_and_cross_field_leakage() -> None:
    payload = _minimal_classification_request()
    payload["unknown"] = 1
    with pytest.raises(ValueError, match="unknown fields"):
        ClassificationExperimentSpec.from_dict(payload)

    valid = _minimal_classification_request()
    valid["feature_columns"] = ["SiO2", "Deposit_Type"]
    with pytest.raises(ValueError, match="target_column"):
        ClassificationExperimentSpec.from_dict(valid)


def test_artifact_paths_cannot_escape_run_directory() -> None:
    payload = {
        "schema_version": CONTRACT_VERSION,
        "run_id": "safe-run",
        "request_hash": "a" * 64,
        "status": "completed",
        "metrics": {},
        "artifacts": [
            {
                "artifact_id": "model",
                "role": "trained_pipeline",
                "media_type": "application/x-joblib",
                "relative_path": "../outside.joblib",
                "size_bytes": 1,
                "sha256": "b" * 64,
            }
        ],
        "warnings": [],
        "manifest_path": "manifest.json",
        "provenance_path": "provenance.json",
    }

    with pytest.raises(ValidationError):
        _validator(SchemaName.EXPERIMENT_RESULT).validate(payload)
