import json
from pathlib import Path

import pandas as pd
import pytest

from geochemistrypi.scientific_execution import ScientificExecutionContract, ScientificExecutionContractError, save_scientific_execution_attestation, scientific_execution_context


def _contract(**updates: object) -> dict:
    value = {
        "schema_version": 2,
        "workflow_family": "supervised_learning",
        "workflow_mode": "regression",
        "method": "xgboost",
        "split_seed": 99,
        "model_seed": 0,
        "cross_validation_folds": 5,
        "evaluation_mode": "internal_holdout",
        "confusion_matrix_normalization": None,
        "external_evaluation_identifier_column": None,
        "external_evaluation_target_columns": [],
        "target_transformations": {"pressure": {"scale": 10.0, "offset": 0.0}},
        "model_parameters": {
            "n_estimators": 890,
            "max_depth": 19,
            "min_child_weight": 130.0,
            "base_score": 0.2,
        },
    }
    value.update(updates)
    return value


def _write_contract(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "scientific-execution.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path.resolve()


def test_scientific_contract_preserves_zero_seed_and_target_transform(
    tmp_path: Path,
) -> None:
    contract = ScientificExecutionContract.load(_write_contract(tmp_path, _contract()))

    assert contract.split_seed == 99
    assert contract.model_seed == 0
    assert contract.cross_validation_folds == 5
    assert contract.constructor_parameters("xgboost", {"random_state": 42})["random_state"] == 0
    transformed = contract.transform_targets(pd.DataFrame({"pressure": [1.2, 2.3]}))
    assert transformed["pressure"].tolist() == pytest.approx([12.0, 23.0])


def test_scientific_contract_rejects_unknown_fields(tmp_path: Path) -> None:
    value = _contract(unregistered=True)
    with pytest.raises(ScientificExecutionContractError, match="Unknown"):
        ScientificExecutionContract.load(_write_contract(tmp_path, value))


def test_lof_contract_requires_explicit_native_evaluation_semantics(
    tmp_path: Path,
) -> None:
    value = _contract(
        workflow_family="anomaly_detection",
        workflow_mode="outlier_detection",
        method="local_outlier_factor",
        split_seed=None,
        model_seed=None,
        evaluation_mode="external_labeled",
        external_evaluation_target_columns=["label"],
        target_transformations={},
        model_parameters={"n_neighbors": 20, "contamination": 0.08},
    )
    with pytest.raises(
        ScientificExecutionContractError,
        match="training_outlier or novelty_detection",
    ):
        ScientificExecutionContract.load(_write_contract(tmp_path, value))


def test_classification_only_confusion_normalization_is_fail_closed(
    tmp_path: Path,
) -> None:
    value = _contract(confusion_matrix_normalization="true")
    with pytest.raises(
        ScientificExecutionContractError,
        match="only for classification",
    ):
        ScientificExecutionContract.load(_write_contract(tmp_path, value))


def test_external_labeled_regression_has_no_internal_split_seed(
    tmp_path: Path,
) -> None:
    value = _contract(
        split_seed=None,
        evaluation_mode="external_labeled",
        external_evaluation_identifier_column="ExternalSampleID",
        external_evaluation_target_columns=["pressure"],
    )
    contract = ScientificExecutionContract.load(_write_contract(tmp_path, value))

    assert contract.workflow_mode == "regression"
    assert contract.evaluation_mode == "external_labeled"
    assert contract.external_evaluation_identifier_column == "ExternalSampleID"
    assert contract.split_seed is None

    value["split_seed"] = 99
    with pytest.raises(
        ScientificExecutionContractError,
        match="complete training cohort.*split_seed",
    ):
        ScientificExecutionContract.load(_write_contract(tmp_path, value))


class _Estimator:
    def __init__(self, parameters: dict):
        self.parameters = parameters

    def get_params(self, deep: bool = False) -> dict:
        assert deep is False
        return self.parameters


def test_attestation_verifies_actual_estimator_parameters(tmp_path: Path) -> None:
    contract_path = _write_contract(tmp_path, _contract())
    output = tmp_path / "parameters"
    with scientific_execution_context(contract_path):
        save_scientific_execution_attestation(
            _Estimator(
                {
                    "n_estimators": 890,
                    "max_depth": 19,
                    "min_child_weight": 130.0,
                    "base_score": 0.2,
                    "random_state": 0,
                }
            ),
            str(output),
        )
    record = json.loads((output / "Scientific Execution Attestation.json").read_text(encoding="utf-8"))
    assert record["verification_status"] == "matched"
    assert record["contract"]["model_seed"] == 0
    assert record["attestation_sha256"]


def test_attestation_rejects_effective_parameter_mismatch(tmp_path: Path) -> None:
    contract_path = _write_contract(tmp_path, _contract())
    with scientific_execution_context(contract_path):
        with pytest.raises(
            ScientificExecutionContractError,
            match="does not match",
        ):
            save_scientific_execution_attestation(
                _Estimator(
                    {
                        "n_estimators": 100,
                        "max_depth": 19,
                        "min_child_weight": 130.0,
                        "base_score": 0.2,
                        "random_state": 0,
                    }
                ),
                str(tmp_path / "parameters"),
            )
