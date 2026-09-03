import json
import sys
from pathlib import Path

import pytest
from geochemistrypi_mcp.api.schemas import (
    AgglomerativeClusteringSettings,
    AnomalyDetectionRequest,
    ClassificationRequest,
    ClusteringRequest,
    DBSCANClusteringSettings,
    DecompositionRequest,
    EvaluationContract,
    IsolationForestAnomalyDetectionSettings,
    KNearestNeighborsSettings,
    LocalOutlierFactorAnomalyDetectionSettings,
    PCADecompositionSettings,
    RegressionRequest,
    TimeSeriesRequest,
    TSNEDecompositionSettings,
    XGBoostSettings,
)
from geochemistrypi_mcp.config.constants import CLI_VERSION
from geochemistrypi_mcp.config.settings import McpSettings
from geochemistrypi_mcp.planning.interaction_plan import AnalysisPlanCompiler
from geochemistrypi_mcp.planning.scientific_contract import assess_scientific_compatibility, canonical_scientific_contract, canonical_sha256, planned_artifact_requirements
from geochemistrypi_mcp.runtime.runs import RunManager, RunStateError


def _learning_dataset(tmp_path: Path) -> Path:
    path = tmp_path / "learning.csv"
    rows = ["SampleID,F1,F2,Label,Target"]
    for index in range(40):
        rows.append(f"S{index},{index + 1},{(index % 7) + 0.5},{'A' if index % 2 == 0 else 'B'},{index * 1.25 + 3}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _time_series_dataset(tmp_path: Path) -> Path:
    path = tmp_path / "time-series.csv"
    path.write_text(
        "SampleID,Age,MaximumAge,Probability,Latitude,Longitude,SiO2,Filter\n" "A,10,12,0.2,10,20,49.1,1\n" "B,20,22,0.4,11,21,51.2,1\n" "C,30,31,0.6,12,22,53.4,2\n" "D,40,43,0.8,13,23,55.6,2\n",
        encoding="utf-8",
    )
    return path


def _supervised_request(task: str, dataset: Path):
    common = {
        "training_dataset_path": dataset,
        "experiment_name": "Compatibility",
        "run_name": task,
        "identifier_column": "SampleID",
        "feature_columns": ("F1", "F2"),
    }
    if task == "classification":
        return ClassificationRequest(**common, target_column="Label")
    return RegressionRequest(**common, target_column="Target")


def _unsupervised_request(task: str, dataset: Path, model):
    request_type = {
        "clustering": ClusteringRequest,
        "decomposition": DecompositionRequest,
        "anomaly_detection": AnomalyDetectionRequest,
    }[task]
    return request_type(
        training_dataset_path=dataset,
        experiment_name="Compatibility",
        run_name=f"{task}-{model.type}",
        identifier_column="SampleID",
        feature_columns=("F1", "F2"),
        model=model,
    )


def test_native_unsupervised_processes_publish_hyper_parameter_records() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    workflow_base = (repository_root / "geochemistrypi" / "data_mining" / "model" / "_base.py").read_text(encoding="utf-8")
    assert 'save_text(hyper_parameters_str, f"Hyper Parameters - {model_name}"' in workflow_base

    for process_name in ("cluster.py", "decompose.py", "detect.py"):
        process_source = (repository_root / "geochemistrypi" / "data_mining" / "process" / process_name).read_text(encoding="utf-8")
        assert ".save_hyper_parameters(" in process_source


@pytest.mark.parametrize(
    ("case_id", "request_factory", "expected_family", "expected_method"),
    (
        (
            "supervised-regression",
            lambda learning, time_series: _supervised_request("regression", learning),
            "supervised_learning",
            "linear_regression",
        ),
        (
            "supervised-classification-a",
            lambda learning, time_series: _supervised_request("classification", learning),
            "supervised_learning",
            "logistic_regression",
        ),
        (
            "supervised-classification-b",
            lambda learning, time_series: _supervised_request("classification", learning),
            "supervised_learning",
            "logistic_regression",
        ),
        (
            "hierarchical-clustering",
            lambda learning, time_series: _unsupervised_request("clustering", learning, AgglomerativeClusteringSettings(number_of_clusters=2)),
            "clustering",
            "agglomerative",
        ),
        (
            "density-clustering",
            lambda learning, time_series: _unsupervised_request("clustering", learning, DBSCANClusteringSettings(minimum_samples=3)),
            "clustering",
            "dbscan",
        ),
        (
            "principal-components",
            lambda learning, time_series: _unsupervised_request("decomposition", learning, PCADecompositionSettings(number_of_components=2)),
            "dimension_reduction",
            "pca",
        ),
        (
            "stochastic-neighbor-embedding",
            lambda learning, time_series: _unsupervised_request("decomposition", learning, TSNEDecompositionSettings(perplexity=5)),
            "dimension_reduction",
            "tsne",
        ),
        (
            "isolation-anomaly-detection",
            lambda learning, time_series: _unsupervised_request("anomaly_detection", learning, IsolationForestAnomalyDetectionSettings()),
            "anomaly_detection",
            "isolation_forest",
        ),
        (
            "local-density-anomaly-detection",
            lambda learning, time_series: _unsupervised_request(
                "anomaly_detection",
                learning,
                LocalOutlierFactorAnomalyDetectionSettings(number_of_neighbors=5),
            ),
            "anomaly_detection",
            "local_outlier_factor",
        ),
        (
            "subaerial-proportion-time-series",
            lambda learning, time_series: TimeSeriesRequest(
                training_dataset_path=time_series,
                mode="subaerial_proportion",
                experiment_name="Compatibility",
                run_name="Subaerial",
                bin_width=10,
                age_column="Age",
                maximum_age_column="MaximumAge",
                probability_column="Probability",
                latitude_column="Latitude",
                longitude_column="Longitude",
                identifier_column="SampleID",
            ),
            "time_series",
            "subaerial_proportion_bootstrap",
        ),
    ),
)
def test_cli_backed_scientific_contracts_compile_without_execution(
    tmp_path: Path,
    case_id: str,
    request_factory,
    expected_family: str,
    expected_method: str,
) -> None:
    learning = _learning_dataset(tmp_path)
    time_series = _time_series_dataset(tmp_path)
    request = request_factory(learning, time_series)

    plan = AnalysisPlanCompiler().compile(request, cli_executable=Path(sys.executable))
    requirements = planned_artifact_requirements(request, plan)
    canonical = canonical_scientific_contract(request, plan)
    assessment = assess_scientific_compatibility(request, plan, requirements)

    assert case_id
    assert plan.execution_ready is True
    assert plan.adapter_status == "available"
    assert plan.public_command
    assert plan.workflow_family == expected_family
    assert plan.method == expected_method
    assert requirements
    normalized_outputs = {item.replace("\\", "/") for item in plan.expected_output_relative_paths}
    assert all(item.expected_relative_path in normalized_outputs for item in requirements)
    available_mappings = tuple(mapping for mapping in plan.artifact_mappings if mapping.availability == "available")
    assert normalized_outputs <= {mapping.relative_path for mapping in available_mappings}
    mapped_roles = {mapping.output_role for mapping in available_mappings}
    if plan.workflow_family == "supervised_learning" and plan.workflow_mode == "regression":
        assert {"evaluation.holdout", "evaluation.predictions", "evaluation.residuals"} <= mapped_roles
    if plan.workflow_family == "time_series":
        assert {"time_series.bins", "time_series.uncertainty", "time_series.figure"} <= mapped_roles
    assert canonical["workflow"] == {
        "family": expected_family,
        "mode": plan.workflow_mode,
        "method": expected_method,
    }
    assert len(canonical_sha256(canonical)) == 64
    assert assessment.execution_ready is True
    assert assessment.artifact_status == "planned"


def test_element_mean_is_representable_but_cannot_be_silently_mapped_to_another_cli_workflow(
    tmp_path: Path,
) -> None:
    request = TimeSeriesRequest(
        training_dataset_path=_time_series_dataset(tmp_path),
        mode="element_mean",
        experiment_name="Compatibility",
        run_name="Element Mean",
        bin_width=10,
        age_column="Age",
        element_columns=("SiO2",),
        filter_column="Filter",
        filter_minimum=1,
        filter_maximum=2,
        identifier_column="SampleID",
    )

    plan = AnalysisPlanCompiler().compile(request, cli_executable=Path(sys.executable))
    requirements = planned_artifact_requirements(request, plan)
    canonical = canonical_scientific_contract(request, plan)
    assessment = assess_scientific_compatibility(request, plan, requirements)

    assert canonical["workflow"] == {
        "family": "time_series",
        "mode": "element_mean",
        "method": "binned_arithmetic_mean",
    }
    assert canonical["column_roles"]["values"] == ["SiO2"]
    assert canonical["parameters"]["uncertainty"] == "standard_error"
    assert plan.public_command == ()
    assert plan.adapter_id is None
    assert plan.execution_ready is False
    assert assessment.execution_ready is False
    assert assessment.adapter_status == "unavailable"
    assert "no element_mean" in " ".join(assessment.blocking_issues)


def test_reference_comparison_is_post_run_and_does_not_block_native_execution(
    tmp_path: Path,
) -> None:
    request = TimeSeriesRequest(
        training_dataset_path=_time_series_dataset(tmp_path),
        mode="subaerial_proportion",
        experiment_name="Compatibility",
        run_name="Post-run comparison",
        bin_width=10,
        age_column="Age",
        maximum_age_column="MaximumAge",
        probability_column="Probability",
        latitude_column="Latitude",
        longitude_column="Longitude",
        identifier_column="SampleID",
        evaluation={"mode": "reference_comparison"},
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

    assert assessment.execution_ready is True
    assert assessment.comparison_ready is False
    assert assessment.claim_ready is False
    assert "reference_comparison" not in " ".join(assessment.blocking_issues)


def test_reference_comparison_metrics_do_not_require_native_cli_metric_bindings(
    tmp_path: Path,
) -> None:
    request = _unsupervised_request("anomaly_detection", _learning_dataset(tmp_path), LocalOutlierFactorAnomalyDetectionSettings(number_of_neighbors=5),).model_copy(
        update={
            "evaluation": EvaluationContract(
                mode="reference_comparison",
                metrics=("anomaly_count", "jaccard"),
            )
        }
    )
    request = AnomalyDetectionRequest.model_validate(request.model_dump())

    plan = AnalysisPlanCompiler().compile(
        request,
        cli_executable=Path(sys.executable),
    )
    assessment = assess_scientific_compatibility(
        request,
        plan,
        planned_artifact_requirements(request, plan),
    )

    assert assessment.execution_ready is True
    assert assessment.comparison_ready is False
    assert "metric artifact" not in " ".join(assessment.blocking_issues)
    assert "not bound" not in " ".join(assessment.blocking_issues)


def test_structured_xgboost_controls_are_bound_and_attested(
    tmp_path: Path,
) -> None:
    request = ClassificationRequest(
        training_dataset_path=_learning_dataset(tmp_path),
        experiment_name="Compatibility",
        run_name="XGBoost controls",
        identifier_column="SampleID",
        feature_columns=("F1", "F2"),
        target_column="Label",
        evaluation={
            "mode": "holdout",
            "split_strategy": "random_holdout",
            "confusion_matrix_normalization": "none",
            "folds": 5,
        },
        model=XGBoostSettings(
            gamma=0.3,
            tree_method="hist",
            objective="binary:logistic",
            importance_type="gain",
        ),
        reproducibility={
            "model_seed": 0,
            "model_parameter_assertions": {
                "gamma": 0.3,
                "importance_type": "gain",
                "objective": "binary:logistic",
                "random_state": 0,
                "tree_method": "hist",
            },
        },
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

    assert dict(plan.requested_model_parameters)["gamma"] == "0.3"
    assert dict(plan.effective_model_parameters)["gamma"] == "0.3"
    assert dict(plan.effective_model_parameters)["random_state"] == "0"
    execution = json.loads(plan.scientific_execution_contract_json)
    assert execution["model_parameters"]["gamma"] == 0.3
    assert execution["model_parameters"]["objective"] == "binary:logistic"
    assert execution["model_parameters"]["importance_type"] == "gain"
    assert execution["split_strategy"] == "random_holdout"
    assert execution["model_seed"] == 0
    assert execution["cross_validation_folds"] == 5
    assert not any("Normalized Confusion Matrix (none)" in path for path in plan.expected_output_relative_paths)
    assert plan.adapter_version == "3"
    assert assessment.execution_ready is True
    assert "gamma" not in " ".join(assessment.blocking_issues)


def test_knn_assertions_require_a_parameter_consumed_by_the_selected_cli_branch(
    tmp_path: Path,
) -> None:
    dataset = _learning_dataset(tmp_path)
    auto_request = ClassificationRequest(
        training_dataset_path=dataset,
        experiment_name="Compatibility",
        run_name="KNN auto assertion",
        identifier_column="SampleID",
        feature_columns=("F1", "F2"),
        target_column="Label",
        model=KNearestNeighborsSettings(),
        reproducibility={"model_parameter_assertions": {"leaf_size": 30}},
    )

    auto_plan = AnalysisPlanCompiler().compile(
        auto_request,
        cli_executable=Path(sys.executable),
    )
    auto_assessment = assess_scientific_compatibility(
        auto_request,
        auto_plan,
        planned_artifact_requirements(auto_request, auto_plan),
    )

    assert "leaf_size" not in dict(auto_plan.requested_model_parameters)
    assert "leaf_size" not in dict(auto_plan.effective_model_parameters)
    assert not any(step.id == "leaf_size" for step in auto_plan.steps)
    assert auto_assessment.execution_ready is False
    assert "leaf_size" in " ".join(auto_assessment.blocking_issues)

    tree_request = auto_request.model_copy(
        update={
            "run_name": "KNN tree assertion",
            "model": KNearestNeighborsSettings(
                algorithm="kd_tree",
                leaf_size=99,
                metric="euclidean",
            ),
            "reproducibility": auto_request.reproducibility.model_copy(update={"model_parameter_assertions": {"leaf_size": 99}}),
        }
    )
    tree_plan = AnalysisPlanCompiler().compile(
        tree_request,
        cli_executable=Path(sys.executable),
    )
    tree_assessment = assess_scientific_compatibility(
        tree_request,
        tree_plan,
        planned_artifact_requirements(tree_request, tree_plan),
    )

    assert dict(tree_plan.requested_model_parameters)["leaf_size"] == "99"
    assert dict(tree_plan.effective_model_parameters)["leaf_size"] == "99"
    assert "power" not in dict(tree_plan.effective_model_parameters)
    assert any(step.id == "leaf_size" for step in tree_plan.steps)
    assert tree_assessment.execution_ready is True


def test_validation_returns_blocked_readiness_and_start_refuses_element_mean(
    tmp_path: Path,
) -> None:
    request = TimeSeriesRequest(
        training_dataset_path=_time_series_dataset(tmp_path),
        mode="element_mean",
        experiment_name="Compatibility",
        run_name="Element Mean",
        bin_width=10,
        age_column="Age",
        element_columns=("SiO2",),
    )
    manager = RunManager(
        McpSettings(
            runs_root=tmp_path / "runs",
            cli_executable=Path(sys.executable),
            maximum_dataset_bytes=1024 * 1024,
        ),
        cli_resolver=lambda: (Path(sys.executable), CLI_VERSION),
    )
    try:
        preview = manager.validate(request)
        assert preview.valid is True
        assert preview.execution_ready is False
        assert preview.workflow_family == "time_series"
        assert preview.workflow_mode == "element_mean"
        assert preview.adapter_status == "unavailable"
        assert len(preview.canonical_contract_hash) == 64
        assert len(preview.compiled_plan_hash) == 64
        with pytest.raises(RunStateError, match="not execution-ready"):
            manager.start_validated(preview.validation_id, preview.request_hash)
        assert not list(manager.settings.runs_root.glob("run-*"))
    finally:
        manager.close()
