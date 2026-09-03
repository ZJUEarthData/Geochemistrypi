from pathlib import Path

import pytest
from geochemistrypi_mcp import AnalysisPlanCompiler
from geochemistrypi_mcp.api.schemas import AnomalyDetectionRequest, ClassificationRequest, ClusteringRequest, DecompositionRequest, RegressionRequest
from geochemistrypi_mcp.contracts.anomaly_detection import MODEL_ORDER as ANOMALY_MODEL_ORDER
from geochemistrypi_mcp.contracts.classification import MODEL_ORDER as CLASSIFICATION_MODEL_ORDER
from geochemistrypi_mcp.contracts.clustering import MODEL_ORDER as CLUSTERING_MODEL_ORDER
from geochemistrypi_mcp.contracts.decomposition import MODEL_ORDER as DECOMPOSITION_MODEL_ORDER
from geochemistrypi_mcp.contracts.regression import MODEL_ORDER as REGRESSION_MODEL_ORDER
from pydantic import ValidationError


def _dataset(tmp_path: Path) -> Path:
    path = tmp_path / "all-models.csv"
    rows = ["SampleID,F1,F2,F3,Label,Target,TargetB"]
    for index in range(50):
        rows.append(f"S{index:02d},{index + 1},{(index % 7) + 2},{(index % 11) + 3},{index % 2},{index * 0.5 + 1},{index * 0.25 + 3}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _common(path: Path, task: str) -> dict:
    return {
        "task": task,
        "training_dataset_path": path,
        "experiment_name": "All Models Contract",
        "run_name": task.replace("_", " ").title(),
        "identifier_column": "SampleID",
        "feature_columns": ("F1", "F2", "F3"),
        "model_selection": {"mode": "all", "tuning": "manual"},
    }


@pytest.mark.parametrize(
    "request_factory",
    [
        lambda path: ClassificationRequest(**_common(path, "classification"), target_column="Label"),
        lambda path: RegressionRequest(**_common(path, "regression"), target_column="Target"),
        lambda path: ClusteringRequest(**_common(path, "clustering")),
        lambda path: DecompositionRequest(**_common(path, "decomposition")),
        lambda path: AnomalyDetectionRequest(**_common(path, "anomaly_detection")),
    ],
)
def test_all_five_task_families_compile_complete_unique_child_plans(tmp_path: Path, request_factory) -> None:
    request = request_factory(_dataset(tmp_path))
    plan = AnalysisPlanCompiler().compile(request, cli_executable=Path(__file__))
    ids = [step.id for step in plan.steps]

    assert plan.name == f"{request.task}-all-models-manual-v1"
    assert len(ids) == len(set(ids))
    expected_response = {
        "classification": "12",
        "regression": "16",
        "clustering": "6",
        "decomposition": "4",
        "anomaly_detection": "3",
    }[request.task]
    assert next(step for step in plan.steps if step.id == "all_models").response == expected_response
    assert any(step_id.startswith(f"{request.model.type}.") for step_id in ids)
    assert not any(
        step_id.endswith(
            (
                ".continue_after_training",
                ".continue_after_transform_pipeline",
            )
        )
        for step_id in ids
    )
    assert plan.expected_output_relative_paths[0].endswith("summary/Aggregate Model Results.json")
    output_names = [Path(path).name for path in plan.expected_output_relative_paths]
    if request.task == "classification":
        assert output_names.count("Target Label Mapping.xlsx") == len(CLASSIFICATION_MODEL_ORDER)
    else:
        expected_hyper_parameter_count = {
            "regression": len(REGRESSION_MODEL_ORDER),
            "clustering": len(CLUSTERING_MODEL_ORDER),
            "decomposition": len(DECOMPOSITION_MODEL_ORDER),
            "anomaly_detection": len(ANOMALY_MODEL_ORDER),
        }[request.task]
        assert sum(name.startswith("Hyper Parameters - ") for name in output_names) == expected_hyper_parameter_count


def test_regression_all_models_preserves_multi_target_selection_across_children(tmp_path: Path) -> None:
    request = RegressionRequest(
        **_common(_dataset(tmp_path), "regression"),
        target_columns=("TargetB", "Target"),
    )

    plan = AnalysisPlanCompiler().compile(request, cli_executable=Path(__file__))
    responses = {step.id: step.response for step in plan.steps}

    assert plan.name == "regression-all-models-manual-multi-output-v1"
    assert responses["target_columns"] == "[4,5]"
    assert responses["all_models"] == "16"
    assert any(step_id.startswith("random_forest.") for step_id in responses)


@pytest.mark.parametrize(
    ("request_type", "task", "target"),
    [
        (ClassificationRequest, "classification", "Label"),
        (RegressionRequest, "regression", "Target"),
    ],
)
def test_supervised_all_models_automl_is_one_parent_choice_with_manual_fallbacks(tmp_path: Path, request_type, task: str, target: str) -> None:
    values = _common(_dataset(tmp_path), task)
    values["target_column"] = target
    values["model_selection"] = {"mode": "all", "tuning": "automl"}
    request = request_type(**values)

    plan = AnalysisPlanCompiler().compile(request, cli_executable=Path(__file__))
    ids = [step.id for step in plan.steps]

    assert ids.count("enable_automl") == 1
    assert not any(step_id.endswith(".enable_automl") for step_id in ids)
    if task == "regression":
        assert any(step_id.startswith("linear_regression.") and "continue_after_hyperparameters" not in step_id for step_id in ids)


def test_all_models_projects_manifest_and_children_under_the_real_parent_run(
    tmp_path: Path,
) -> None:
    request = AnomalyDetectionRequest(**_common(_dataset(tmp_path), "anomaly_detection"))

    plan = AnalysisPlanCompiler().compile(request, cli_executable=Path(__file__))
    paths = set(plan.expected_output_relative_paths)

    assert ("geopi_output/All Models Contract/Anomaly Detection/summary/" "Aggregate Model Results.json") in paths
    assert any(path.startswith("geopi_output/All Models Contract/Anomaly Detection/" "Isolation Forest/") for path in paths)
    assert not any(path.startswith("geopi_output/All Models Contract/Isolation Forest/" "Anomaly Detection/") for path in paths)


def test_all_models_rejects_ignored_legacy_fields_and_unsupervised_automl(
    tmp_path: Path,
) -> None:
    path = _dataset(tmp_path)
    with pytest.raises(ValidationError, match="replaces explicit legacy fields"):
        ClassificationRequest(
            **_common(path, "classification"),
            target_column="Label",
            tuning="manual",
        )
    values = _common(path, "clustering")
    values["model_selection"] = {"mode": "all", "tuning": "automl"}
    with pytest.raises(ValidationError, match="AutoML is not public"):
        ClusteringRequest(**values)
