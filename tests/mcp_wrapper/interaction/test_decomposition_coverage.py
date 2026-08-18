import ast
import json
import sys
from pathlib import Path

import pytest
from geochemistrypi_mcp import DecompositionPlanCompiler, DecompositionRequest, PlanCompilationError
from geochemistrypi_mcp.contracts.decomposition import MODEL_DISPLAY_NAMES, MODEL_ORDER
from pydantic import ValidationError

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = REPOSITORY_ROOT / "tests" / "cli_contract" / "fixtures" / "decomposition_baseline.csv"
CAPABILITY_FIXTURE = REPOSITORY_ROOT / "tests" / "mcp_wrapper" / "parity" / "fixtures" / "decomposition_capability_matrix_v1.json"


def _request(path: Path = DATASET_PATH, **overrides) -> DecompositionRequest:
    values = {
        "task": "decomposition",
        "training_dataset_path": path,
        "experiment_name": "PR7 Coverage",
        "run_name": "Decomposition",
        "identifier_column": "SampleID",
        "feature_columns": ("FeatureA", "FeatureB", "FeatureC"),
    }
    values.update(overrides)
    return DecompositionRequest(**values)


def _assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Assignment {name} not found")


def test_versioned_decomposition_matrix_matches_public_cli_constants() -> None:
    fixture = json.loads(CAPABILITY_FIXTURE.read_text(encoding="utf-8"))
    cli_models = _assignment(
        REPOSITORY_ROOT / "geochemistrypi" / "data_mining" / "constants.py",
        "DECOMPOSITION_MODELS",
    )

    assert tuple(item["id"] for item in fixture["models"]) == MODEL_ORDER
    assert [item["cli_name"] for item in fixture["models"]] == cli_models
    assert [MODEL_DISPLAY_NAMES[model] for model in MODEL_ORDER] == cli_models


@pytest.mark.parametrize("model_name", MODEL_ORDER)
def test_every_public_decomposition_family_compiles(model_name: str) -> None:
    model = {"type": model_name}
    if model_name == "tsne":
        model["perplexity"] = 10
    plan = DecompositionPlanCompiler().compile(
        _request(model=model),
        cli_executable=Path(sys.executable),
    )
    responses = {step.id: step.response for step in plan.steps}

    assert responses["decomposition_mode"] == "4"
    assert responses[model_name] == str(MODEL_ORDER.index(model_name) + 1)
    supervised_steps = {
        "target_column",
        "default_test_ratio",
        "enable_automl",
        "continue_after_inference",
    }
    assert not supervised_steps.intersection(responses)
    assert any(Path(path).name == f"{MODEL_DISPLAY_NAMES[model_name]}.joblib" for path in plan.expected_output_relative_paths)
    assert any(Path(path).name == "X Reduced.xlsx" for path in plan.expected_output_relative_paths)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (
            {"type": "pca", "number_of_components": 2, "svd_solver": "randomized"},
            {"number_of_components": "2", "svd_solver": "4"},
        ),
        (
            {
                "type": "tsne",
                "number_of_components": 2,
                "perplexity": 10,
                "learning_rate": 125.5,
                "number_of_iterations": 750,
                "early_exaggeration": 8.5,
            },
            {
                "number_of_components": "2",
                "perplexity": "10",
                "learning_rate": "125.5",
                "number_of_iterations": "750",
                "early_exaggeration": "8.5",
            },
        ),
        (
            {
                "type": "mds",
                "number_of_components": 2,
                "metric": False,
                "number_of_initializations": 6,
                "maximum_iterations": 450,
            },
            {
                "number_of_components": "2",
                "metric": "2",
                "number_of_initializations": "6",
                "maximum_iterations": "450",
            },
        ),
    ],
)
def test_method_specific_parameters_compile_exact_responses(model: dict, expected: dict[str, str]) -> None:
    plan = DecompositionPlanCompiler().compile(
        _request(model=model),
        cli_executable=Path(sys.executable),
    )
    responses = {step.id: step.response for step in plan.steps}
    assert responses.items() >= expected.items()


def test_pca_component_selection_branches_match_public_cli() -> None:
    two = DecompositionPlanCompiler().compile(
        _request(model={"type": "pca", "number_of_components": 2}),
        cli_executable=Path(sys.executable),
    )
    assert not any(step.id.startswith("pca_") for step in two.steps)

    three = DecompositionPlanCompiler().compile(
        _request(model={"type": "pca", "number_of_components": 3}),
        cli_executable=Path(sys.executable),
    )
    three_ids = {step.id for step in three.steps}
    assert {"pca_biplot_component_1", "pca_biplot_component_2"} <= three_ids
    assert not any(step_id.startswith("pca_triplot_component") for step_id in three_ids)

    four = DecompositionPlanCompiler().compile(
        _request(
            engineered_features=({"name": "FeatureSum", "formula": "{FeatureA} + {FeatureB}"},),
            model={"type": "pca", "number_of_components": 4},
        ),
        cli_executable=Path(sys.executable),
    )
    four_ids = {step.id for step in four.steps}
    assert {
        "pca_biplot_component_1",
        "pca_biplot_component_2",
        "pca_triplot_component_1",
        "pca_triplot_component_2",
        "pca_triplot_component_3",
    } <= four_ids


def test_decomposition_rejects_invalid_data_and_model_dimensions(tmp_path: Path) -> None:
    with pytest.raises(PlanCompilationError, match="exceeds min"):
        DecompositionPlanCompiler().compile(
            _request(model={"type": "pca", "number_of_components": 4}),
            cli_executable=Path(sys.executable),
        )
    with pytest.raises(PlanCompilationError, match="strictly less"):
        DecompositionPlanCompiler().compile(
            _request(model={"type": "pca", "number_of_components": 3, "svd_solver": "arpack"}),
            cli_executable=Path(sys.executable),
        )
    with pytest.raises(PlanCompilationError, match="must be less"):
        DecompositionPlanCompiler().compile(
            _request(model={"type": "tsne", "perplexity": 36}),
            cli_executable=Path(sys.executable),
        )

    non_numeric = tmp_path / "non-numeric.csv"
    non_numeric.write_text(
        DATASET_PATH.read_text(encoding="utf-8").replace("0.12", "not-a-number", 1),
        encoding="utf-8",
    )
    with pytest.raises(PlanCompilationError, match="non-numeric"):
        DecompositionPlanCompiler().compile(
            _request(non_numeric),
            cli_executable=Path(sys.executable),
        )

    missing = tmp_path / "missing.csv"
    missing.write_text(
        DATASET_PATH.read_text(encoding="utf-8").replace("0.12", "", 1),
        encoding="utf-8",
    )
    with pytest.raises(PlanCompilationError, match="choose drop_rows or impute"):
        DecompositionPlanCompiler().compile(
            _request(missing),
            cli_executable=Path(sys.executable),
        )
    imputed = DecompositionPlanCompiler().compile(
        _request(missing, missing_values={"method": "impute", "strategy": "mean"}),
        cli_executable=Path(sys.executable),
    )
    assert {step.id: step.response for step in imputed.steps}["imputation_method"] == "1"


def test_decomposition_schema_rejects_supervised_and_unavailable_inputs() -> None:
    for field, value in (
        ("target_column", "Label"),
        ("application_dataset_path", DATASET_PATH),
        ("tuning", "manual"),
        ("feature_selection", {"method": "none"}),
    ):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            _request(**{field: value})

    with pytest.raises(ValidationError, match="no decomposition models"):
        _request(missing_values={"method": "keep"})
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        _request(model={"type": "umap"})
