import hashlib
import json
from pathlib import Path

import pytest
from geochemistrypi_mcp.api.schemas import ArtifactRequirement
from geochemistrypi_mcp.runtime.artifacts import SCIENTIFIC_ESTIMATOR_IDENTITIES, ArtifactDiscoveryError, discover_artifacts, read_time_series_preprocessing_summary

_LIU_PARAMETERS_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "liu_time_series"


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _attestation_source(*, workflow_mode: str = "classification") -> dict:
    return {
        "schema_version": 4,
        "workflow_family": "data_mining",
        "workflow_mode": workflow_mode,
        "method": "xgboost" if workflow_mode == "classification" else "decision_tree",
        "split_seed": 2025,
        "split_strategy": "stratified_holdout" if workflow_mode == "classification" else "random_holdout",
        "model_seed": 2025,
        "cross_validation_folds": 10,
        "evaluation_mode": "internal_holdout",
        "confusion_matrix_normalization": None,
        "external_evaluation_identifier_column": None,
        "external_evaluation_target_columns": [],
        "target_transformations": {},
        "classification_metric_average": "auto" if workflow_mode == "classification" else None,
        "classification_positive_label": None,
        "model_parameters": {"max_depth": 4},
    }


def _attestation_record(
    source: dict,
    source_sha256: str,
    *,
    multi_output: bool = False,
) -> dict:
    expected_identity = {"module_root": "sklearn", "class_name": "DecisionTreeRegressor"} if source["workflow_mode"] == "regression" else {"module_root": "xgboost", "class_name": "XGBClassifier"}
    observed_identity = (
        {"module": "sklearn.tree._classes", "qualname": "DecisionTreeRegressor"} if source["workflow_mode"] == "regression" else {"module": "xgboost.sklearn", "qualname": "XGBClassifier"}
    )
    if multi_output:
        observed_identity["wrapper"] = {
            "module": "sklearn.multioutput",
            "qualname": "MultiOutputRegressor",
            "fitted_estimator_count": 2,
        }
    metric_semantics = None
    if source["workflow_mode"] == "classification":
        positive = {"type": "integer", "value": 1}
        metric_semantics = {
            "schema_version": 2,
            "requested_average": "auto",
            "effective_average": "binary",
            "requested_positive_label": None,
            "aggregate_semantic_positive_label": positive,
            "aggregate_encoded_positive_label": 1,
            "curve_semantic_positive_label": positive,
            "curve_encoded_positive_label": 1,
            "curve_probability_column_index": 1,
            "consumers": {
                "holdout_score": {
                    "consumer_kind": "aggregate_metric",
                    "effective_average": "binary",
                    "aggregate_encoded_positive_label": 1,
                },
                "cross_validation": {
                    "consumer_kind": "aggregate_metric",
                    "effective_average": "binary",
                    "aggregate_encoded_positive_label": 1,
                },
                "precision_recall": {
                    "consumer_kind": "binary_curve",
                    "curve_encoded_positive_label": 1,
                    "probability_column_index": 1,
                },
                "precision_recall_threshold": {
                    "consumer_kind": "binary_curve",
                    "curve_encoded_positive_label": 1,
                    "probability_column_index": 1,
                },
                "roc": {
                    "consumer_kind": "binary_curve",
                    "curve_encoded_positive_label": 1,
                    "probability_column_index": 1,
                },
            },
        }
    record = {
        "schema_version": 2,
        "contract": {**source, "source_sha256": source_sha256},
        "effective_model_parameters": {"max_depth": 4, "random_state": 2025},
        "verified_parameter_names": ["max_depth", "random_state"],
        "estimator_identity": {
            "expected": expected_identity,
            "observed": observed_identity,
        },
        "classification_metric_semantics": metric_semantics,
        "verification_status": "matched",
    }
    record["attestation_sha256"] = _canonical_sha256(record)
    return record


def _write_attestation(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")


def test_artifact_discovery_indexes_parent_and_all_model_children(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    for directory in ("artifacts", "metrics", "parameters", "summary"):
        (output / directory).mkdir(parents=True)
    (output / "summary" / "Aggregate Model Results.json").write_text("{}", encoding="utf-8")
    for model, score in (("Model A", 0.8), ("Model B", 0.7)):
        for directory in ("artifacts", "metrics", "parameters", "summary"):
            (output / model / directory).mkdir(parents=True)
        (output / model / "artifacts" / "model.joblib").write_bytes(model.encode())
        (output / model / "metrics" / "Score.json").write_text(json.dumps({"score": score}), encoding="utf-8")

    discovered = discover_artifacts(output, maximum_response_references=100)
    relative_paths = {item.relative_path for item in discovered.response_references}

    assert "summary/Aggregate Model Results.json" in relative_paths
    assert "Model A/artifacts/model.joblib" in relative_paths
    assert "Model B/metrics/Score.json" in relative_paths
    assert discovered.reported_metrics == {
        "Model A/metrics/Score.json": {"score": 0.8},
        "Model B/metrics/Score.json": {"score": 0.7},
    }
    assert {item.category for item in discovered.response_references} == {
        "artifacts",
        "metrics",
        "summary",
    }


@pytest.mark.parametrize(
    "workflow_family",
    (
        "classification",
        "regression",
        "clustering",
        "embedding",
        "outlier_detection",
        "time_series",
    ),
)
def test_artifact_index_marks_only_unambiguous_unbound_summary_mirrors_across_workflows(
    tmp_path: Path,
    workflow_family: str,
) -> None:
    output = tmp_path / workflow_family
    for directory in ("artifacts", "metrics", "parameters", "summary"):
        (output / directory).mkdir(parents=True)
    (output / "artifacts" / "Unique.json").write_text('{"value": 1}', encoding="utf-8")
    (output / "summary" / "Unique.json").write_text('{"value": 1}', encoding="utf-8")
    (output / "artifacts" / "Ambiguous.json").write_text('{"value": 2}', encoding="utf-8")
    (output / "metrics" / "Ambiguous.json").write_text('{"value": 2}', encoding="utf-8")
    (output / "summary" / "Ambiguous.json").write_text('{"value": 2}', encoding="utf-8")
    (output / "parameters" / "Bound.json").write_text('{"value": 3}', encoding="utf-8")
    (output / "summary" / "Bound.json").write_text('{"value": 3}', encoding="utf-8")
    requirement = ArtifactRequirement(
        requirement_id="summary.bound",
        scientific_type="structured_record",
        output_role="scientific.output",
        category="summary",
        expected_relative_path="summary/Bound.json",
    )

    discovered = discover_artifacts(
        output,
        maximum_response_references=100,
        requirements=(requirement,),
        workflow_family=workflow_family,
    )
    entries = {entry["relative_path"]: entry for entry in discovered.all_index_entries}

    unique_source = entries["artifacts/Unique.json"]
    unique_summary = entries["summary/Unique.json"]
    assert unique_summary["metadata"]["summary_mirror"] is True
    assert unique_summary["metadata"]["mirror_of_artifact_id"] == unique_source["artifact_id"]
    assert "summary_mirror" not in entries["summary/Ambiguous.json"]["metadata"]
    assert "mirror_of_artifact_id" not in entries["summary/Ambiguous.json"]["metadata"]
    assert entries["summary/Bound.json"]["requirement_ids"] == [requirement.requirement_id]
    assert "summary_mirror" not in entries["summary/Bound.json"]["metadata"]
    assert "mirror_of_artifact_id" not in entries["summary/Bound.json"]["metadata"]


def test_parameter_attestation_is_semantically_validated_and_bound_to_its_source(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    for directory in ("artifacts", "metrics", "parameters", "summary"):
        (output / directory).mkdir(parents=True)
    attestation_path = output / "parameters" / "Scientific Execution Attestation.json"
    source = _attestation_source()
    source_path = tmp_path / "scientific-execution.json"
    source_path.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    requirement = ArtifactRequirement(
        requirement_id="parameters.execution-attestation",
        scientific_type="parameter_attestation",
        output_role="provenance.parameters.attested",
        category="parameters",
        expected_relative_path="parameters/Scientific Execution Attestation.json",
    )

    valid = _attestation_record(source, source_sha256)
    _write_attestation(attestation_path, valid)
    complete = discover_artifacts(
        output,
        20,
        (requirement,),
        expected_attestation_source_sha256=source_sha256,
        expected_attestation_source_contract=source,
    )
    assert complete.missing_requirement_ids == ()

    cases = []
    wrong_status = {**valid, "verification_status": "unverified"}
    wrong_status["attestation_sha256"] = _canonical_sha256({key: value for key, value in wrong_status.items() if key != "attestation_sha256"})
    cases.append((wrong_status, "verification_status"))
    cases.append(({**valid, "attestation_sha256": "0" * 64}, "self-hash"))
    wrong_identity = json.loads(json.dumps(valid))
    wrong_identity["estimator_identity"]["observed"]["qualname"] = "RandomForestClassifier"
    wrong_identity["attestation_sha256"] = _canonical_sha256({key: value for key, value in wrong_identity.items() if key != "attestation_sha256"})
    cases.append((wrong_identity, "estimator identities"))
    forged_identity = json.loads(json.dumps(valid))
    forged_identity["estimator_identity"] = {
        "expected": {
            "module_root": "sklearn",
            "class_name": "RandomForestClassifier",
        },
        "observed": {
            "module": "sklearn.ensemble._forest",
            "qualname": "RandomForestClassifier",
        },
    }
    forged_identity["attestation_sha256"] = _canonical_sha256({key: value for key, value in forged_identity.items() if key != "attestation_sha256"})
    cases.append((forged_identity, "trusted method registry"))
    wrong_source = json.loads(json.dumps(valid))
    wrong_source["contract"]["source_sha256"] = "f" * 64
    wrong_source["attestation_sha256"] = _canonical_sha256({key: value for key, value in wrong_source.items() if key != "attestation_sha256"})
    cases.append((wrong_source, "source hash"))
    wrong_contract = json.loads(json.dumps(valid))
    wrong_contract["contract"]["method"] = "random_forest"
    wrong_contract["attestation_sha256"] = _canonical_sha256({key: value for key, value in wrong_contract.items() if key != "attestation_sha256"})
    cases.append((wrong_contract, "contract does not match"))
    empty_verified = json.loads(json.dumps(valid))
    empty_verified["verified_parameter_names"] = []
    empty_verified["attestation_sha256"] = _canonical_sha256({key: value for key, value in empty_verified.items() if key != "attestation_sha256"})
    cases.append((empty_verified, "exactly cover"))
    wrong_parameter = json.loads(json.dumps(valid))
    wrong_parameter["effective_model_parameters"]["random_state"] = 42
    wrong_parameter["attestation_sha256"] = _canonical_sha256({key: value for key, value in wrong_parameter.items() if key != "attestation_sha256"})
    cases.append((wrong_parameter, "random_state"))
    wrong_metrics = json.loads(json.dumps(valid))
    wrong_metrics["classification_metric_semantics"]["consumers"].pop("roc")
    wrong_metrics["attestation_sha256"] = _canonical_sha256({key: value for key, value in wrong_metrics.items() if key != "attestation_sha256"})
    cases.append((wrong_metrics, "required consumers"))
    missing_metrics = json.loads(json.dumps(valid))
    missing_metrics["classification_metric_semantics"] = None
    missing_metrics["attestation_sha256"] = _canonical_sha256({key: value for key, value in missing_metrics.items() if key != "attestation_sha256"})
    cases.append((missing_metrics, "missing metric semantics"))

    for record, expected_failure in cases:
        _write_attestation(attestation_path, record)
        incomplete = discover_artifacts(
            output,
            20,
            (requirement,),
            expected_attestation_source_sha256=source_sha256,
            expected_attestation_source_contract=source,
        )
        assert incomplete.missing_requirement_ids == (requirement.requirement_id,)
        assert expected_failure in incomplete.requirement_failures[requirement.requirement_id]


def test_parameter_attestation_accepts_valid_multi_output_wrapper_identity(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    for directory in ("artifacts", "metrics", "parameters", "summary"):
        (output / directory).mkdir(parents=True)
    source = _attestation_source(workflow_mode="regression")
    source_path = tmp_path / "scientific-execution.json"
    source_path.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    attestation = output / "parameters" / "Scientific Execution Attestation.json"
    _write_attestation(
        attestation,
        _attestation_record(source, source_sha256, multi_output=True),
    )
    requirement = ArtifactRequirement(
        requirement_id="parameters.multi-output-attestation",
        scientific_type="parameter_attestation",
        output_role="provenance.parameters.attested",
        category="parameters",
        expected_relative_path="parameters/Scientific Execution Attestation.json",
    )

    discovered = discover_artifacts(
        output,
        20,
        (requirement,),
        expected_attestation_source_sha256=source_sha256,
        expected_attestation_source_contract=source,
    )

    assert discovered.missing_requirement_ids == ()


def test_parameter_attestation_accepts_empty_verified_names_only_for_a_real_deterministic_metric_only_estimator(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    for directory in ("artifacts", "metrics", "parameters", "summary"):
        (output / directory).mkdir(parents=True)
    source = _attestation_source()
    source.update(
        {
            "method": "logistic_regression",
            "model_seed": None,
            "model_parameters": {},
        }
    )
    source_path = tmp_path / "scientific-execution.json"
    source_path.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    record = _attestation_record(source, source_sha256)
    record["effective_model_parameters"] = {
        "solver": "lbfgs",
        "random_state": None,
    }
    record["verified_parameter_names"] = []
    record["estimator_identity"] = {
        "expected": {
            "module_root": "sklearn",
            "class_name": "LogisticRegression",
        },
        "observed": {
            "module": "sklearn.linear_model._logistic",
            "qualname": "LogisticRegression",
        },
    }
    record["attestation_sha256"] = _canonical_sha256({key: value for key, value in record.items() if key != "attestation_sha256"})
    attestation = output / "parameters" / "Scientific Execution Attestation.json"
    _write_attestation(attestation, record)
    requirement = ArtifactRequirement(
        requirement_id="parameters.deterministic-attestation",
        scientific_type="parameter_attestation",
        output_role="provenance.parameters.attested",
        category="parameters",
        expected_relative_path="parameters/Scientific Execution Attestation.json",
    )

    discovered = discover_artifacts(
        output,
        20,
        (requirement,),
        expected_attestation_source_sha256=source_sha256,
        expected_attestation_source_contract=source,
    )

    assert discovered.missing_requirement_ids == ()


def test_trusted_estimator_identity_registry_covers_every_sidecar_method() -> None:
    expected_methods = {
        ("classification", method)
        for method in (
            "logistic_regression",
            "support_vector_machine",
            "decision_tree",
            "random_forest",
            "extra_trees",
            "xgboost",
            "multi_layer_perceptron",
            "gradient_boosting",
            "k_nearest_neighbors",
            "stochastic_gradient_descent",
            "adaboost",
        )
    }
    expected_methods.update(
        ("regression", method)
        for method in (
            "decision_tree",
            "random_forest",
            "extra_trees",
            "gradient_boosting",
            "xgboost",
            "multi_layer_perceptron",
            "lasso_regression",
            "elastic_net",
            "stochastic_gradient_descent",
        )
    )
    expected_methods.update(
        {
            ("clustering", "kmeans"),
            ("clustering", "affinity_propagation"),
            ("embedding", "pca"),
            ("embedding", "tsne"),
            ("embedding", "mds"),
            ("outlier_detection", "isolation_forest"),
            ("outlier_detection", "local_outlier_factor"),
        }
    )

    assert set(SCIENTIFIC_ESTIMATOR_IDENTITIES) == expected_methods


def test_time_series_preprocessing_summary_uses_indexed_cli_parameters() -> None:
    summary = read_time_series_preprocessing_summary(
        _LIU_PARAMETERS_FIXTURE_ROOT,
        source_row_count=22640,
        indexed_relative_paths=("parameters/Time Series Parameters.json",),
    )

    assert summary.model_dump() == {
        "input_row_count": 22640,
        "analysis_row_count": 22623,
        "dropped_row_count": 17,
    }


@pytest.mark.parametrize(
    ("payload", "source_row_count"),
    [
        ({"preprocessing": {"input_row_count": 4, "analysis_row_count": 5, "dropped_row_count": 0}}, 4),
        ({"preprocessing": {"input_row_count": 4, "analysis_row_count": 3, "dropped_row_count": 0}}, 4),
        ({"preprocessing": {"input_row_count": 4, "analysis_row_count": 3, "dropped_row_count": 1}}, 5),
        ({"preprocessing": {"input_row_count": "4", "analysis_row_count": 3, "dropped_row_count": 1}}, 4),
        ({"preprocessing": {"input_row_count": True, "analysis_row_count": 1, "dropped_row_count": 0}}, 1),
    ],
    ids=("analysis-too-large", "bad-difference", "source-mismatch", "string-count", "boolean-count"),
)
def test_time_series_preprocessing_summary_rejects_inconsistent_or_untyped_counts(
    tmp_path: Path,
    payload: dict,
    source_row_count: int,
) -> None:
    parameters = tmp_path / "parameters" / "Time Series Parameters.json"
    parameters.parent.mkdir()
    parameters.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtifactDiscoveryError):
        read_time_series_preprocessing_summary(
            tmp_path,
            source_row_count=source_row_count,
            indexed_relative_paths=("parameters/Time Series Parameters.json",),
        )


@pytest.mark.parametrize("case", ("missing", "malformed", "not-indexed"))
def test_time_series_preprocessing_summary_fails_closed_when_unavailable(
    tmp_path: Path,
    case: str,
) -> None:
    parameters = tmp_path / "parameters" / "Time Series Parameters.json"
    parameters.parent.mkdir()
    indexed = ("parameters/Time Series Parameters.json",)
    if case == "malformed":
        parameters.write_text("{", encoding="utf-8")
    elif case == "not-indexed":
        parameters.write_text(
            json.dumps(
                {
                    "preprocessing": {
                        "input_row_count": 4,
                        "analysis_row_count": 3,
                        "dropped_row_count": 1,
                    }
                }
            ),
            encoding="utf-8",
        )
        indexed = ()

    with pytest.raises(ArtifactDiscoveryError):
        read_time_series_preprocessing_summary(
            tmp_path,
            source_row_count=4,
            indexed_relative_paths=indexed,
        )
