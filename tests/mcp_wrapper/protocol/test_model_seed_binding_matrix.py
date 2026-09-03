"""Static coverage for truthful primary-estimator seed binding."""

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from geochemistrypi_mcp import AnalysisPlanCompiler, AnomalyDetectionRequest, ClassificationRequest, ClusteringRequest, DecompositionRequest, RegressionRequest
from geochemistrypi_mcp.contracts.scientific_execution import SCIENTIFIC_EXECUTION_METHOD_COUNT, SCIENTIFIC_EXECUTION_METHODS_BY_TASK
from geochemistrypi_mcp.planning.scientific_contract import assess_scientific_compatibility, planned_artifact_requirements

from geochemistrypi.scientific_execution import _WORKFLOW_METHODS, ScientificExecutionContract

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests" / "cli_contract" / "fixtures"

_RUNTIME_WORKFLOW_BY_TASK = {
    "classification": ("supervised_learning", "classification"),
    "regression": ("supervised_learning", "regression"),
    "clustering": ("clustering", "clustering"),
    "decomposition": ("dimension_reduction", "embedding"),
    "anomaly_detection": ("anomaly_detection", "outlier_detection"),
}


def _request(
    task: str,
    model_type: str,
    *,
    model_overrides: dict[str, Any] | None = None,
    reproducibility: dict[str, Any] | None = None,
    tuning: str = "manual",
    all_models: bool = False,
):
    model = {"type": model_type, **(model_overrides or {})}
    common = {
        "experiment_name": "SeedAudit",
        "run_name": "R",
        "reproducibility": reproducibility or {},
    }
    selection = {"model_selection": {"mode": "all", "tuning": tuning}} if all_models else {"model": model}
    if task == "classification":
        return ClassificationRequest(
            training_dataset_path=FIXTURES / "classification_baseline.csv",
            identifier_column="SampleID",
            feature_columns=("SIO2(WT%)", "TIO2(WT%)", "AL2O3(WT%)", "FEOT(WT%)"),
            target_column="Label",
            **({"tuning": tuning} if not all_models else {}),
            **selection,
            **common,
        )
    if task == "regression":
        return RegressionRequest(
            training_dataset_path=FIXTURES / "regression_baseline.csv",
            identifier_column="SampleID",
            feature_columns=("SIO2", "TIO2"),
            target_column="Target",
            **({"tuning": tuning} if not all_models else {}),
            **selection,
            **common,
        )
    request_type, file_name = {
        "clustering": (ClusteringRequest, "clustering_baseline.csv"),
        "decomposition": (DecompositionRequest, "decomposition_baseline.csv"),
        "anomaly_detection": (AnomalyDetectionRequest, "anomaly_detection_baseline.csv"),
    }[task]
    return request_type(
        training_dataset_path=FIXTURES / file_name,
        identifier_column="SampleID",
        feature_columns=("FeatureA", "FeatureB", "FeatureC"),
        **selection,
        **common,
    )


REGISTERED_SCIENTIFIC_EXECUTION_CASES = tuple((task, method) for task, methods in SCIENTIFIC_EXECUTION_METHODS_BY_TASK.items() for method in methods)


@pytest.mark.parametrize(
    "task,method",
    REGISTERED_SCIENTIFIC_EXECUTION_CASES,
)
def test_every_registered_method_compiles_one_canonical_v4_sidecar_and_attestation_contract(
    tmp_path: Path,
    task: str,
    method: str,
) -> None:
    assert SCIENTIFIC_EXECUTION_METHOD_COUNT == 27
    assert len(REGISTERED_SCIENTIFIC_EXECUTION_CASES) == 27
    workflow_family, workflow_mode = _RUNTIME_WORKFLOW_BY_TASK[task]
    assert set(SCIENTIFIC_EXECUTION_METHODS_BY_TASK[task]) == set(_WORKFLOW_METHODS[(workflow_family, workflow_mode)])

    request = _request(task, method)
    plan = AnalysisPlanCompiler().compile(
        request,
        cli_executable=Path(sys.executable),
    )

    assert plan.workflow_family == workflow_family
    assert plan.workflow_mode == workflow_mode
    assert plan.method == method
    assert plan.scientific_contract_id == (f"scientific-contract-v4/{workflow_family}/{workflow_mode}/{method}")
    assert plan.scientific_execution_contract_json is not None
    assert any(path.replace("\\", "/").endswith("/parameters/Scientific Execution Attestation.json") for path in plan.expected_output_relative_paths)

    sidecar = json.loads(plan.scientific_execution_contract_json)
    assert sidecar["schema_version"] == 4
    assert (
        sidecar["workflow_family"],
        sidecar["workflow_mode"],
        sidecar["method"],
    ) == (workflow_family, workflow_mode, method)

    contract_path = tmp_path / "scientific-execution.json"
    contract_path.write_text(
        plan.scientific_execution_contract_json,
        encoding="utf-8",
    )
    contract = ScientificExecutionContract.load(contract_path)
    assert (
        contract.workflow_family,
        contract.workflow_mode,
        contract.method,
    ) == (workflow_family, workflow_mode, method)


BOUND_CASES = (
    *(
        ("classification", model, {})
        for model in (
            "support_vector_machine",
            "decision_tree",
            "random_forest",
            "extra_trees",
            "xgboost",
            "multi_layer_perceptron",
            "gradient_boosting",
            "stochastic_gradient_descent",
            "adaboost",
        )
    ),
    ("classification", "logistic_regression", {"solver": "saga"}),
    *(
        ("regression", model, {})
        for model in (
            "decision_tree",
            "random_forest",
            "extra_trees",
            "gradient_boosting",
            "xgboost",
            "multi_layer_perceptron",
            "stochastic_gradient_descent",
        )
    ),
    ("regression", "lasso_regression", {"selection": "random"}),
    ("regression", "elastic_net", {"selection": "random"}),
    ("clustering", "kmeans", {}),
    ("clustering", "affinity_propagation", {}),
    ("decomposition", "pca", {"svd_solver": "randomized"}),
    ("decomposition", "tsne", {}),
    ("decomposition", "mds", {}),
    ("anomaly_detection", "isolation_forest", {}),
)


@pytest.mark.parametrize("task,model_type,model_overrides", BOUND_CASES)
def test_bound_manual_models_emit_an_attestable_requested_seed(
    task: str,
    model_type: str,
    model_overrides: dict[str, Any],
) -> None:
    request = _request(
        task,
        model_type,
        model_overrides=model_overrides,
        reproducibility={"model_seed": 2025},
    )
    plan = AnalysisPlanCompiler().compile(request, cli_executable=Path(sys.executable))
    assessment = assess_scientific_compatibility(request, plan, planned_artifact_requirements(request, plan))

    execution = json.loads(plan.scientific_execution_contract_json)
    assert dict(plan.effective_seeds)["model"] == 2025
    assert execution["model_seed"] == 2025
    assert (
        execution["evaluation_mode"]
        == {
            "classification": "internal_holdout",
            "regression": "internal_holdout",
            "clustering": "training_clustering",
            "decomposition": "fit_transform",
            "anomaly_detection": "training_outlier",
        }[task]
    )
    assert plan.seed_binding != "unbound"
    assert assessment.execution_ready is True


NOT_APPLICABLE_CASES = (
    ("classification", "logistic_regression", {}),
    ("classification", "k_nearest_neighbors", {}),
    (
        "classification",
        "stochastic_gradient_descent",
        {"shuffle": False, "early_stopping": False},
    ),
    *(
        ("regression", model, {})
        for model in (
            "linear_regression",
            "polynomial_regression",
            "k_nearest_neighbors",
            "support_vector_machine",
            "bayesian_ridge",
            "ridge_regression",
        )
    ),
    ("regression", "lasso_regression", {"selection": "cyclic"}),
    ("regression", "elastic_net", {"selection": "cyclic"}),
    ("regression", "stochastic_gradient_descent", {"shuffle": False}),
    ("clustering", "dbscan", {}),
    ("clustering", "agglomerative", {}),
    ("clustering", "mean_shift", {}),
    ("decomposition", "pca", {"svd_solver": "full"}),
    ("anomaly_detection", "local_outlier_factor", {}),
)


@pytest.mark.parametrize("task,model_type,model_overrides", NOT_APPLICABLE_CASES)
def test_non_random_manual_models_never_claim_an_effective_model_seed(
    task: str,
    model_type: str,
    model_overrides: dict[str, Any],
) -> None:
    request = _request(
        task,
        model_type,
        model_overrides=model_overrides,
        reproducibility={"model_seed": 2025},
    )
    plan = AnalysisPlanCompiler().compile(request, cli_executable=Path(sys.executable))
    assessment = assess_scientific_compatibility(request, plan, planned_artifact_requirements(request, plan))

    assert "model" not in dict(plan.effective_seeds)
    assert assessment.execution_ready is False
    assert "The CLI adapter does not expose an effective model seed for attestation." in assessment.blocking_issues


def test_pca_auto_keeps_v4_but_does_not_claim_fixed_seed_readiness() -> None:
    request = _request(
        "decomposition",
        "pca",
        reproducibility={"deterministic_policy": "fixed_seed_required"},
    )
    plan = AnalysisPlanCompiler().compile(request, cli_executable=Path(sys.executable))
    assessment = assess_scientific_compatibility(request, plan, planned_artifact_requirements(request, plan))

    assert plan.seed_binding == "unbound"
    assert dict(plan.effective_seeds) == {}
    execution = json.loads(plan.scientific_execution_contract_json)
    assert execution["method"] == "pca"
    assert execution["model_seed"] is None
    assert plan.scientific_contract_id.endswith("/dimension_reduction/embedding/pca")
    assert assessment.execution_ready is False
    assert any("stochastic CLI stages do not expose an effective seed" in issue for issue in assessment.blocking_issues)


@pytest.mark.parametrize(
    ("row_count", "feature_count"),
    (
        (40, 3),
        (600, 100),
    ),
    ids=("auto-deterministic-shape", "auto-potentially-randomized-shape"),
)
def test_pca_auto_fixed_seed_policy_blocks_both_solver_shape_classes(
    tmp_path: Path,
    row_count: int,
    feature_count: int,
) -> None:
    feature_names = tuple(f"F{index}" for index in range(feature_count))
    dataset = tmp_path / "pca-auto.csv"
    dataset.write_text(
        "SampleID," + ",".join(feature_names) + "\n" + "\n".join(f"S{row}," + ",".join(str((row + 1) * (column + 1) / 1000) for column in range(feature_count)) for row in range(row_count)) + "\n",
        encoding="utf-8",
    )
    request = DecompositionRequest(
        training_dataset_path=dataset,
        experiment_name="SeedAudit",
        run_name="PCA auto shape",
        identifier_column="SampleID",
        feature_columns=feature_names,
        model={
            "type": "pca",
            "number_of_components": 2,
            "svd_solver": "auto",
        },
        reproducibility={"deterministic_policy": "fixed_seed_required"},
    )
    plan = AnalysisPlanCompiler().compile(
        request,
        cli_executable=Path(sys.executable),
    )
    assessment = assess_scientific_compatibility(
        request,
        plan,
        planned_artifact_requirements(request, plan),
    )

    assert plan.scientific_execution_contract_json is not None
    assert plan.seed_binding == "unbound"
    assert assessment.execution_ready is False


@pytest.mark.parametrize(
    "analysis_request",
    (
        _request("classification", "random_forest", tuning="automl"),
        _request("regression", "random_forest", tuning="automl"),
        _request("classification", "logistic_regression", all_models=True),
        _request("clustering", "kmeans", all_models=True),
    ),
)
def test_automl_and_all_model_plans_do_not_fabricate_model_or_tuning_seeds(analysis_request: Any) -> None:
    plan = AnalysisPlanCompiler().compile(analysis_request, cli_executable=Path(sys.executable))

    assert plan.seed_binding == "unbound"
    assert "model" not in dict(plan.effective_seeds)
    assert "tuning" not in dict(plan.effective_seeds)
    assert plan.scientific_execution_contract_json is None


@pytest.mark.parametrize(
    "analysis_request",
    (
        _request(
            "classification",
            "random_forest",
            tuning="automl",
            reproducibility={"model_seed": 2025},
        ),
        _request(
            "regression",
            "random_forest",
            tuning="automl",
            reproducibility={"model_seed": 2025},
        ),
        _request(
            "classification",
            "logistic_regression",
            all_models=True,
            reproducibility={"model_seed": 2025},
        ),
        _request(
            "clustering",
            "kmeans",
            all_models=True,
            reproducibility={"model_seed": 2025},
        ),
    ),
)
def test_automl_and_all_model_explicit_model_seed_is_a_validation_blocker(
    analysis_request: Any,
) -> None:
    plan = AnalysisPlanCompiler().compile(
        analysis_request,
        cli_executable=Path(sys.executable),
    )
    assessment = assess_scientific_compatibility(
        analysis_request,
        plan,
        planned_artifact_requirements(analysis_request, plan),
    )

    assert plan.seed_binding == "unbound"
    assert dict(plan.effective_seeds) in ({"split": 42}, {})
    assert assessment.execution_ready is False
    assert "The CLI adapter does not expose an effective model seed for attestation." in assessment.blocking_issues


@pytest.mark.parametrize(
    "analysis_request",
    (
        _request(
            "classification",
            "random_forest",
            tuning="automl",
            reproducibility={"deterministic_policy": "fixed_seed_required"},
        ),
        _request(
            "clustering",
            "kmeans",
            all_models=True,
            reproducibility={"deterministic_policy": "fixed_seed_required"},
        ),
    ),
)
def test_automl_and_all_model_fixed_seed_policy_is_a_validation_blocker(
    analysis_request: Any,
) -> None:
    plan = AnalysisPlanCompiler().compile(
        analysis_request,
        cli_executable=Path(sys.executable),
    )
    assessment = assess_scientific_compatibility(
        analysis_request,
        plan,
        planned_artifact_requirements(analysis_request, plan),
    )

    assert plan.seed_binding == "unbound"
    assert assessment.execution_ready is False
    assert any("stochastic CLI stages do not expose an effective seed" in issue for issue in assessment.blocking_issues)


def test_zero_is_preserved_for_a_newly_bound_manual_model() -> None:
    request = _request(
        "classification",
        "random_forest",
        reproducibility={"model_seed": 0},
    )
    plan = AnalysisPlanCompiler().compile(request, cli_executable=Path(sys.executable))

    assert dict(plan.effective_seeds)["model"] == 0
    assert json.loads(plan.scientific_execution_contract_json)["model_seed"] == 0
