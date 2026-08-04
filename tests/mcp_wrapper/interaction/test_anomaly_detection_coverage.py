import ast
import json
import sys
from pathlib import Path

import pytest
from geochemistrypi_mcp import AnomalyDetectionPlanCompiler, AnomalyDetectionRequest, PlanCompilationError
from geochemistrypi_mcp.anomaly_detection_contract import MODEL_DISPLAY_NAMES, MODEL_ORDER
from pydantic import ValidationError

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = REPOSITORY_ROOT / "tests" / "cli_contract" / "fixtures" / "anomaly_detection_baseline.csv"
CAPABILITY_FIXTURE = REPOSITORY_ROOT / "tests" / "mcp_wrapper" / "parity" / "fixtures" / "anomaly_detection_capability_matrix_v1.json"


def _request(path: Path = DATASET_PATH, **overrides) -> AnomalyDetectionRequest:
    values = {
        "task": "anomaly_detection",
        "training_dataset_path": path,
        "experiment_name": "PR8A Coverage",
        "run_name": "Anomaly Detection",
        "identifier_column": "SampleID",
        "feature_columns": ("FeatureA", "FeatureB", "FeatureC"),
    }
    values.update(overrides)
    return AnomalyDetectionRequest(**values)


def _assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Assignment {name} not found")


def test_versioned_anomaly_detection_matrix_matches_public_cli_constants() -> None:
    fixture = json.loads(CAPABILITY_FIXTURE.read_text(encoding="utf-8"))
    cli_models = _assignment(
        REPOSITORY_ROOT / "geochemistrypi" / "data_mining" / "constants.py",
        "ANOMALYDETECTION_MODELS",
    )

    assert tuple(item["id"] for item in fixture["models"]) == MODEL_ORDER
    assert [item["cli_name"] for item in fixture["models"]] == cli_models
    assert [MODEL_DISPLAY_NAMES[model] for model in MODEL_ORDER] == cli_models


@pytest.mark.parametrize("model_name", MODEL_ORDER)
def test_every_public_anomaly_detection_family_compiles(model_name: str) -> None:
    plan = AnomalyDetectionPlanCompiler().compile(
        _request(model={"type": model_name}),
        cli_executable=Path(sys.executable),
    )
    responses = {step.id: step.response for step in plan.steps}

    assert responses["anomaly_detection_mode"] == "5"
    assert responses[model_name] == str(MODEL_ORDER.index(model_name) + 1)
    supervised_steps = {
        "target_column",
        "default_test_ratio",
        "enable_automl",
        "continue_after_inference",
    }
    assert not supervised_steps.intersection(responses)
    assert any(Path(path).name == f"{MODEL_DISPLAY_NAMES[model_name]}.joblib" for path in plan.expected_output_relative_paths)
    assert any(Path(path).name == "X Abnormal Detection.xlsx" for path in plan.expected_output_relative_paths)


@pytest.mark.parametrize(
    ("model", "expected", "absent"),
    [
        (
            {
                "type": "isolation_forest",
                "number_of_estimators": 150,
                "contamination": 0.2,
                "maximum_features": 2,
                "bootstrap": False,
            },
            {
                "number_of_estimators": "150",
                "contamination": "0.2",
                "maximum_features": "2",
                "bootstrap": "2",
            },
            {"maximum_samples"},
        ),
        (
            {
                "type": "isolation_forest",
                "bootstrap": True,
                "maximum_samples": 24,
            },
            {"bootstrap": "1", "maximum_samples": "24"},
            set(),
        ),
        (
            {
                "type": "local_outlier_factor",
                "number_of_neighbors": 10,
                "leaf_size": 40,
                "power": 1.5,
                "contamination": 0.25,
                "number_of_jobs": -1,
            },
            {
                "number_of_neighbors": "10",
                "leaf_size": "40",
                "power": "1.5",
                "contamination": "0.25",
                "number_of_jobs": "-1",
            },
            set(),
        ),
    ],
)
def test_model_parameters_compile_exact_cli_responses(
    model: dict,
    expected: dict[str, str],
    absent: set[str],
) -> None:
    plan = AnomalyDetectionPlanCompiler().compile(
        _request(model=model),
        cli_executable=Path(sys.executable),
    )
    responses = {step.id: step.response for step in plan.steps}
    assert responses.items() >= expected.items()
    assert not absent & responses.keys()


def test_plot_selection_and_lof_specific_outputs_follow_cli_branches() -> None:
    three_features = AnomalyDetectionPlanCompiler().compile(
        _request(model={"type": "local_outlier_factor"}),
        cli_executable=Path(sys.executable),
    )
    assert {
        "anomaly_plot_2d_feature_1",
        "anomaly_plot_2d_feature_2",
        "anomaly_plot_3d_feature_1",
        "anomaly_plot_3d_feature_2",
        "anomaly_plot_3d_feature_3",
    } <= {step.id for step in three_features.steps}
    assert any(Path(path).name == "Lof Score Diagram - Local Outlier Factor.xlsx" for path in three_features.expected_output_relative_paths)

    two_features = AnomalyDetectionPlanCompiler().compile(
        _request(feature_columns=("FeatureA", "FeatureB")),
        cli_executable=Path(sys.executable),
    )
    assert not any(step.id.startswith("anomaly_plot_") for step in two_features.steps)
    assert not any("Two-Dimensional" in path or "Three-Dimensional" in path for path in two_features.expected_output_relative_paths)


def test_anomaly_detection_rejects_invalid_data_and_model_dimensions(
    tmp_path: Path,
) -> None:
    with pytest.raises(PlanCompilationError, match="maximum_features=4"):
        AnomalyDetectionPlanCompiler().compile(
            _request(model={"type": "isolation_forest", "maximum_features": 4}),
            cli_executable=Path(sys.executable),
        )
    with pytest.raises(PlanCompilationError, match="maximum_samples=31"):
        AnomalyDetectionPlanCompiler().compile(
            _request(
                model={
                    "type": "isolation_forest",
                    "bootstrap": True,
                    "maximum_samples": 31,
                }
            ),
            cli_executable=Path(sys.executable),
        )
    with pytest.raises(PlanCompilationError, match="number_of_neighbors=30"):
        AnomalyDetectionPlanCompiler().compile(
            _request(
                model={
                    "type": "local_outlier_factor",
                    "number_of_neighbors": 30,
                }
            ),
            cli_executable=Path(sys.executable),
        )

    non_numeric = tmp_path / "non-numeric.csv"
    non_numeric.write_text(
        DATASET_PATH.read_text(encoding="utf-8").replace("0.12", "invalid", 1),
        encoding="utf-8",
    )
    with pytest.raises(PlanCompilationError, match="non-numeric"):
        AnomalyDetectionPlanCompiler().compile(
            _request(non_numeric),
            cli_executable=Path(sys.executable),
        )

    missing = tmp_path / "missing.csv"
    missing.write_text(
        DATASET_PATH.read_text(encoding="utf-8").replace("0.12", "", 1),
        encoding="utf-8",
    )
    with pytest.raises(PlanCompilationError, match="choose drop_rows or impute"):
        AnomalyDetectionPlanCompiler().compile(
            _request(missing),
            cli_executable=Path(sys.executable),
        )
    imputed = AnomalyDetectionPlanCompiler().compile(
        _request(
            missing,
            missing_values={"method": "impute", "strategy": "mean"},
        ),
        cli_executable=Path(sys.executable),
    )
    assert {step.id: step.response for step in imputed.steps}["imputation_method"] == "1"


def test_anomaly_detection_schema_rejects_unavailable_inputs() -> None:
    for field, value in (
        ("target_column", "Label"),
        ("application_dataset_path", DATASET_PATH),
        ("tuning", "manual"),
        ("feature_selection", {"method": "none"}),
    ):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            _request(**{field: value})

    with pytest.raises(ValidationError, match="no anomaly-detection models"):
        _request(missing_values={"method": "keep"})
    with pytest.raises(ValidationError, match="maximum_samples is required"):
        _request(model={"type": "isolation_forest", "bootstrap": True})
    with pytest.raises(ValidationError, match="only used when bootstrap"):
        _request(
            model={
                "type": "isolation_forest",
                "bootstrap": False,
                "maximum_samples": 10,
            }
        )
    with pytest.raises(ValidationError, match="positive integer"):
        _request(model={"type": "local_outlier_factor", "number_of_jobs": 0})
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        _request(model={"type": "one_class_svm"})
