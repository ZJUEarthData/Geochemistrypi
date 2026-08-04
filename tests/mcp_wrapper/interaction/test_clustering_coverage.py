import ast
import json
import sys
from pathlib import Path

import pytest
from geochemistrypi_mcp import (
    ClusteringPlanCompiler,
    ClusteringRequest,
    PlanCompilationError,
)
from geochemistrypi_mcp.clustering_contract import MODEL_DISPLAY_NAMES, MODEL_ORDER
from pydantic import ValidationError

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = (
    REPOSITORY_ROOT
    / "tests"
    / "cli_contract"
    / "fixtures"
    / "clustering_baseline.csv"
)
CAPABILITY_FIXTURE = (
    REPOSITORY_ROOT
    / "tests"
    / "mcp_wrapper"
    / "parity"
    / "fixtures"
    / "clustering_capability_matrix_v1.json"
)


def _request(path: Path = DATASET_PATH, **overrides) -> ClusteringRequest:
    values = {
        "task": "clustering",
        "training_dataset_path": path,
        "experiment_name": "PR6 Coverage",
        "run_name": "Clustering",
        "identifier_column": "SampleID",
        "feature_columns": ("FeatureA", "FeatureB", "FeatureC"),
    }
    values.update(overrides)
    return ClusteringRequest(**values)


def _assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Assignment {name} not found")


def test_versioned_clustering_capability_matrix_matches_public_cli_constants() -> None:
    fixture = json.loads(CAPABILITY_FIXTURE.read_text(encoding="utf-8"))
    cli_models = _assignment(
        REPOSITORY_ROOT / "geochemistrypi" / "data_mining" / "constants.py",
        "CLUSTERING_MODELS",
    )

    assert tuple(item["id"] for item in fixture["models"]) == MODEL_ORDER
    assert [item["cli_name"] for item in fixture["models"]] == cli_models
    assert [MODEL_DISPLAY_NAMES[model] for model in MODEL_ORDER] == cli_models
    assert "OPTICS" not in cli_models


@pytest.mark.parametrize("model_name", MODEL_ORDER)
def test_every_public_cli_clustering_family_compiles_a_plan(
    model_name: str,
) -> None:
    plan = ClusteringPlanCompiler().compile(
        _request(model={"type": model_name}),
        cli_executable=Path(sys.executable),
    )
    responses = {step.id: step.response for step in plan.steps}

    assert responses["clustering_mode"] == "3"
    assert responses[model_name] == str(MODEL_ORDER.index(model_name) + 1)
    assert any(
        MODEL_DISPLAY_NAMES[model_name] + " - Hyper-parameters Specification"
        in step.output_anchors
        for step in plan.steps
    )
    assert not {
        "target_column",
        "default_test_ratio",
        "enable_automl",
        "continue_after_inference",
    } & responses.keys()
    assert Path(plan.expected_output_relative_paths[0]).parts[-3:] == (
        "artifacts",
        "model",
        f"{MODEL_DISPLAY_NAMES[model_name]}.joblib",
    )
    assert any(
        Path(path).name == f"Cluster Labels - {MODEL_DISPLAY_NAMES[model_name]}.xlsx"
        for path in plan.expected_output_relative_paths
    )


@pytest.mark.parametrize(
    ("model", "responses", "absent"),
    [
        (
            {"type": "kmeans", "initialization": "random", "algorithm": "auto"},
            {"initialization": "2", "algorithm": "1"},
            set(),
        ),
        (
            {
                "type": "dbscan",
                "algorithm": "kd_tree",
                "metric": "minkowski",
                "power": 2,
            },
            {"algorithm": "3", "metric": "3", "power": "2"},
            set(),
        ),
        (
            {"type": "agglomerative", "linkage": "single"},
            {"linkage": "4"},
            set(),
        ),
        (
            {"type": "affinity_propagation", "affinity": "euclidean"},
            {"affinity": "1"},
            set(),
        ),
        (
            {
                "type": "mean_shift",
                "bandwidth": None,
                "cluster_all": False,
                "bin_seeding": True,
            },
            {"bandwidth": "0", "cluster_all": "2", "bin_seeding": "1"},
            set(),
        ),
        (
            {"type": "dbscan", "algorithm": "auto", "metric": "euclidean"},
            {"metric": "1"},
            {"power"},
        ),
    ],
)
def test_clustering_conditional_model_prompt_branches(
    model: dict,
    responses: dict[str, str],
    absent: set[str],
) -> None:
    plan = ClusteringPlanCompiler().compile(
        _request(model=model),
        cli_executable=Path(sys.executable),
    )
    compiled = {step.id: step.response for step in plan.steps}
    assert compiled.items() >= responses.items()
    assert not absent & compiled.keys()


def test_clustering_preprocessing_and_plot_dimension_branches() -> None:
    no_scaling = ClusteringPlanCompiler().compile(
        _request(
            feature_columns=("FeatureA", "FeatureB"),
            scaling="none",
            model={"type": "dbscan", "minimum_samples": 2},
        ),
        cli_executable=Path(sys.executable),
    )
    no_scaling_steps = {step.id: step.response for step in no_scaling.steps}
    assert no_scaling_steps["skip_feature_scaling"] == "2"
    assert "clustering_plot_2d_feature_1" not in no_scaling_steps
    assert not any(
        Path(path).name == "Transform Pipeline.joblib"
        for path in no_scaling.expected_output_relative_paths
    )

    engineered = ClusteringPlanCompiler().compile(
        _request(
            feature_columns=("FeatureA", "FeatureB"),
            engineered_features=(
                {"name": "FeatureRatio", "formula": "{FeatureA} / ({FeatureB} + 1)"},
            ),
            scaling="mean_normalization",
            model={"type": "dbscan", "minimum_samples": 2},
        ),
        cli_executable=Path(sys.executable),
    )
    engineered_steps = {step.id: step.response for step in engineered.steps}
    assert engineered_steps["engineered_feature_1_formula"] == "a / (b + 1)"
    assert engineered_steps["mean_normalization"] == "3"
    assert engineered_steps["clustering_plot_2d_feature_1"] == "1"
    assert engineered_steps["clustering_plot_2d_feature_2"] == "2"
    assert engineered_steps["clustering_plot_3d_feature_1"] == "1"
    assert engineered_steps["clustering_plot_3d_feature_2"] == "2"
    assert engineered_steps["clustering_plot_3d_feature_3"] == "3"
    assert any(
        Path(path).name == "Transform Pipeline.joblib"
        for path in engineered.expected_output_relative_paths
    )


def test_clustering_rejects_invalid_data_before_cli_execution(tmp_path: Path) -> None:
    non_numeric = tmp_path / "non-numeric.csv"
    non_numeric.write_text(
        DATASET_PATH.read_text(encoding="utf-8").replace("0.12", "not-a-number", 1),
        encoding="utf-8",
    )
    with pytest.raises(PlanCompilationError, match="non-numeric"):
        ClusteringPlanCompiler().compile(
            _request(non_numeric),
            cli_executable=Path(sys.executable),
        )

    missing = tmp_path / "missing.csv"
    missing.write_text(
        DATASET_PATH.read_text(encoding="utf-8").replace("0.12", "", 1),
        encoding="utf-8",
    )
    with pytest.raises(PlanCompilationError, match="choose drop_rows or impute"):
        ClusteringPlanCompiler().compile(
            _request(missing),
            cli_executable=Path(sys.executable),
        )
    imputed = ClusteringPlanCompiler().compile(
        _request(missing, missing_values={"method": "impute", "strategy": "mean"}),
        cli_executable=Path(sys.executable),
    )
    assert {step.id: step.response for step in imputed.steps}["imputation_method"] == "1"

    too_small = tmp_path / "too-small.csv"
    too_small.write_text(
        "SampleID,FeatureA,FeatureB,FeatureC\n"
        + "\n".join(f"S-{index},{index},{index + 1},{index + 2}" for index in range(10))
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PlanCompilationError, match="k=2 through k=10"):
        ClusteringPlanCompiler().compile(
            _request(too_small),
            cli_executable=Path(sys.executable),
        )


def test_clustering_schema_rejects_supervised_and_internal_only_inputs() -> None:
    for field, value in (
        ("target_column", "Label"),
        ("application_dataset_path", DATASET_PATH),
        ("tuning", "manual"),
        ("feature_selection", {"method": "none"}),
    ):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            _request(**{field: value})

    with pytest.raises(ValidationError, match="no clustering models"):
        _request(missing_values={"method": "keep"})
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        _request(model={"type": "optics"})


def test_affinity_precomputed_requires_square_retained_matrix() -> None:
    with pytest.raises(PlanCompilationError, match="square feature matrix"):
        ClusteringPlanCompiler().compile(
            _request(
                model={
                    "type": "affinity_propagation",
                    "affinity": "precomputed",
                }
            ),
            cli_executable=Path(sys.executable),
        )
