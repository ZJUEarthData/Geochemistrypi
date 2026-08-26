import ast
import json
import sys
from pathlib import Path

import pytest
from geochemistrypi_mcp import PlanCompilationError, RegressionPlanCompiler, RegressionRequest
from geochemistrypi_mcp.contracts.regression import MODEL_DISPLAY_NAMES, MODEL_ORDER, MODELS_WITHOUT_AUTOML
from geochemistrypi_mcp.planning.interaction_plan import AnalysisPlanCompiler
from pydantic import ValidationError

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CAPABILITY_FIXTURE = REPOSITORY_ROOT / "tests" / "mcp_wrapper" / "parity" / "fixtures" / "regression_capability_matrix_v1.json"


def _dataset(tmp_path: Path, *, missing: bool = False, non_numeric_target: bool = False) -> Path:
    path = tmp_path / ("regression-missing.csv" if missing else "regression.csv")
    rows = ["SampleID,Target,SIO2,TIO2"]
    for index in range(1, 21):
        target = "high" if non_numeric_target and index == 4 else str(10 + index * 1.5)
        sio2 = "" if missing and index == 3 else str(48 + index)
        rows.append(f"S-{index},{target},{sio2},{0.5 + index / 10}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _multi_target_dataset(tmp_path: Path, *, invalid_second_target: str | None = None) -> Path:
    path = tmp_path / "regression-multi-target.csv"
    rows = ["SampleID,Target,TargetB,SIO2,TIO2"]
    for index in range(1, 21):
        target_b = invalid_second_target if invalid_second_target is not None and index == 4 else str(4 + index * 0.75)
        rows.append(f"S-{index},{10 + index * 1.5},{target_b},{48 + index},{0.5 + index / 10}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _request(path: Path, **overrides) -> RegressionRequest:
    values = {
        "task": "regression",
        "training_dataset_path": path,
        "experiment_name": "PR5 Coverage",
        "run_name": "Regression",
        "identifier_column": "SampleID",
        "feature_columns": ("SIO2", "TIO2"),
        "target_column": "Target",
    }
    values.update(overrides)
    return RegressionRequest(**values)


def _assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Assignment {name} not found")


def test_versioned_regression_capability_matrix_matches_public_cli_constants() -> None:
    fixture = json.loads(CAPABILITY_FIXTURE.read_text(encoding="utf-8"))
    cli_models = _assignment(REPOSITORY_ROOT / "geochemistrypi" / "data_mining" / "constants.py", "REGRESSION_MODELS")

    assert tuple(item["id"] for item in fixture["models"]) == MODEL_ORDER
    assert [item["cli_name"] for item in fixture["models"]] == cli_models
    assert [MODEL_DISPLAY_NAMES[model] for model in MODEL_ORDER] == cli_models
    assert tuple(item["id"] for item in fixture["models"] if not item["automl"]) == MODELS_WITHOUT_AUTOML
    assert fixture["target_contract"] == "one_or_more_numeric"
    assert "multiple_targets" not in fixture["unsupported"]
    assert "multiple_targets_with_feature_selection" in fixture["unsupported"]


def test_regression_public_contract_accepts_legacy_single_and_plural_multi_targets(tmp_path: Path) -> None:
    compiler = RegressionPlanCompiler()
    legacy = _request(_dataset(tmp_path))
    assert legacy.resolved_target_columns == ("Target",)
    legacy_plan = compiler.compile(legacy, cli_executable=Path(sys.executable))
    assert legacy_plan.name == "regression-linear_regression-v1"
    assert {step.id: step.response for step in legacy_plan.steps}["target_column"] == "1"

    training = _multi_target_dataset(tmp_path)
    request = _request(
        training,
        target_column=None,
        target_columns=("TargetB", "Target"),
    )
    assert request.resolved_target_columns == ("TargetB", "Target")
    plan = compiler.compile(request, cli_executable=Path(sys.executable))
    responses = {step.id: step.response for step in plan.steps}

    assert plan.name == "regression-linear_regression-multi-output-v1"
    assert responses["selected_data_columns"] == "[2,5]"
    assert responses["target_columns"] == "[1,2]"
    assert responses["feature_columns"] == "[3,4]"
    assert "target_column" not in responses


def test_regression_multi_target_contract_rejects_ambiguous_or_unsafe_requests(tmp_path: Path) -> None:
    training = _multi_target_dataset(tmp_path)

    with pytest.raises(ValidationError, match="exactly one of target_column or target_columns"):
        _request(training, target_column=None)
    with pytest.raises(ValidationError, match="exactly one of target_column or target_columns"):
        _request(training, target_columns=("TargetB",))
    with pytest.raises(ValidationError, match="duplicate column names"):
        _request(training, target_column=None, target_columns=("Target", "Target"))
    with pytest.raises(ValidationError, match="must not also be features"):
        _request(training, target_column=None, target_columns=("Target", "TargetB"), feature_columns=("TargetB", "SIO2"))
    with pytest.raises(ValidationError, match="requires feature_selection.method='none'"):
        _request(
            training,
            target_column=None,
            target_columns=("Target", "TargetB"),
            feature_selection={"method": "select_k_best", "retain_count": 1},
        )


@pytest.mark.parametrize("invalid_value", ["high", "", "nan", "inf"])
def test_regression_multi_target_scans_every_target_before_execution(tmp_path: Path, invalid_value: str) -> None:
    request = _request(
        _multi_target_dataset(tmp_path, invalid_second_target=invalid_value),
        target_column=None,
        target_columns=("Target", "TargetB"),
    )
    expected = "missing" if invalid_value in {"", "nan"} else "non-numeric" if invalid_value == "high" else "non-finite"
    with pytest.raises(PlanCompilationError, match=rf"TargetB.*{expected}"):
        RegressionPlanCompiler().compile(request, cli_executable=Path(sys.executable))


def test_regression_multi_target_rejects_target_leakage_and_poisson_negatives(tmp_path: Path) -> None:
    training = _multi_target_dataset(tmp_path)
    with pytest.raises(PlanCompilationError, match="must not use the target column"):
        RegressionPlanCompiler().compile(
            _request(
                training,
                target_column=None,
                target_columns=("Target", "TargetB"),
                engineered_features=({"name": "Leak", "formula": "{TargetB} / {SIO2}"},),
            ),
            cli_executable=Path(sys.executable),
        )

    negative = _multi_target_dataset(tmp_path).read_text(encoding="utf-8").replace(",4.75,", ",-4.75,", 1)
    training.write_text(negative, encoding="utf-8")
    with pytest.raises(PlanCompilationError, match="poisson.*non-negative"):
        RegressionPlanCompiler().compile(
            _request(
                training,
                target_column=None,
                target_columns=("Target", "TargetB"),
                model={"type": "decision_tree", "criterion": "poisson"},
            ),
            cli_executable=Path(sys.executable),
        )


@pytest.mark.parametrize("model_name", MODEL_ORDER)
def test_every_public_cli_regression_family_compiles_a_manual_plan(tmp_path: Path, model_name: str) -> None:
    plan = RegressionPlanCompiler().compile(
        _request(_dataset(tmp_path), model={"type": model_name}),
        cli_executable=Path(sys.executable),
    )
    responses = {step.id: step.response for step in plan.steps}

    assert responses["regression_mode"] == "1"
    assert responses[model_name] == str(MODEL_ORDER.index(model_name) + 1)
    assert any(MODEL_DISPLAY_NAMES[model_name] + " - Hyper-parameters Specification" in step.output_anchors for step in plan.steps)
    assert Path(plan.expected_output_relative_paths[0]).parts[-3:] == (
        "artifacts",
        "model",
        f"{MODEL_DISPLAY_NAMES[model_name]}.joblib",
    )
    assert Path(plan.expected_output_relative_paths[2]).name == f"Predicted vs. Actual Diagram - {MODEL_DISPLAY_NAMES[model_name]}.png"


@pytest.mark.parametrize("model_name", tuple(model for model in MODEL_ORDER if model not in MODELS_WITHOUT_AUTOML))
def test_supported_regression_automl_plans_omit_manual_prompts(tmp_path: Path, model_name: str) -> None:
    plan = RegressionPlanCompiler().compile(
        _request(_dataset(tmp_path), tuning="automl", model={"type": model_name}),
        cli_executable=Path(sys.executable),
    )

    step_ids = {step.id for step in plan.steps}
    assert {step.id: step.response for step in plan.steps}["enable_automl"] == "1"
    assert "continue_after_hyperparameters" not in step_ids
    assert not any("Hyper-parameters Specification" in anchor for step in plan.steps for anchor in step.output_anchors)


@pytest.mark.parametrize("model_name", MODELS_WITHOUT_AUTOML)
def test_cli_models_without_automl_are_rejected_before_execution(tmp_path: Path, model_name: str) -> None:
    with pytest.raises(ValidationError, match="does not offer AutoML"):
        _request(_dataset(tmp_path), tuning="automl", model={"type": model_name})


@pytest.mark.parametrize(
    ("model", "required_steps", "absent_steps"),
    [
        ({"type": "polynomial_regression"}, {"degree", "interaction_only", "include_bias"}, {"disable_automl"}),
        ({"type": "support_vector_machine", "kernel": "poly"}, {"kernel", "degree", "gamma"}, set()),
        (
            {"type": "random_forest", "bootstrap": False, "maximum_samples": None, "out_of_bag_score": False},
            {"bootstrap", "out_of_bag_score"},
            {"maximum_samples"},
        ),
        ({"type": "k_nearest_neighbors", "algorithm": "kd_tree", "metric": "euclidean"}, {"leaf_size", "metric"}, {"power"}),
        ({"type": "stochastic_gradient_descent", "penalty": "elasticnet"}, {"penalty", "l1_ratio", "power"}, set()),
        ({"type": "bayesian_ridge"}, {"alpha_1", "lambda_2", "compute_score", "copy_x", "verbose"}, set()),
    ],
)
def test_regression_conditional_model_prompt_branches(
    tmp_path: Path,
    model: dict,
    required_steps: set[str],
    absent_steps: set[str],
) -> None:
    plan = RegressionPlanCompiler().compile(_request(_dataset(tmp_path), model=model), cli_executable=Path(sys.executable))
    step_ids = {step.id for step in plan.steps}
    assert required_steps <= step_ids
    assert not absent_steps & step_ids


def test_regression_training_application_and_preprocessing_parity(tmp_path: Path) -> None:
    training = _dataset(tmp_path)
    application = tmp_path / "regression-application.csv"
    application.write_text("SampleID,SIO2,TIO2\nA-1,55,1.1\nA-2,57,1.2\n", encoding="utf-8")
    request = _request(
        training,
        application_dataset_path=application,
        engineered_features=({"name": "SiTi", "formula": "{SIO2} / {TIO2}"},),
        scaling="mean_normalization",
        feature_selection={"method": "select_k_best", "retain_count": 2},
        model={"type": "ridge_regression"},
    )
    plan = RegressionPlanCompiler().compile(request, cli_executable=Path(sys.executable))
    responses = {step.id: step.response for step in plan.steps}

    assert plan.public_command[:6] == (
        str(Path(sys.executable).resolve()),
        "data-mining",
        "--training",
        str(training.resolve()),
        "--application",
        str(application.resolve()),
    )
    assert plan.public_command[6] == "--world-map-config"
    assert json.loads(plan.public_command[7])["enabled"] is False
    assert responses["engineered_feature_1_formula"] == "b / c"
    assert responses["feature_columns"] == "[2,4]"
    assert responses["mean_normalization"] == "3"
    assert responses["feature_selection_method"] == "2"
    assert responses["feature_selection_retain_count"] == "2"
    assert any(step.id == "continue_after_inference" for step in plan.steps)


def test_external_labeled_regression_fits_the_complete_training_cohort(
    tmp_path: Path,
) -> None:
    training = _dataset(tmp_path)
    evaluation = tmp_path / "external-evaluation.csv"
    evaluation.write_text(
        "ExternalID,Target,SIO2,TIO2\n" "E-1,92.5,55,1.1\n" "E-2,101.0,57,1.2\n",
        encoding="utf-8",
    )
    request = _request(
        training,
        scaling="standardization",
        model={
            "type": "extra_trees",
            "number_of_estimators": 550,
            "maximum_depth": None,
            "maximum_features": 2,
            "bootstrap": False,
            "maximum_samples": None,
            "out_of_bag_score": False,
        },
        evaluation={
            "mode": "external_labeled",
            "evaluation_dataset_path": evaluation,
            "external_identifier_column": "ExternalID",
        },
        reproducibility={"model_seed": 280},
        target_transformations={"Target": {"scale": 10.0, "offset": 0.0}},
    )

    plan = AnalysisPlanCompiler().compile(
        request,
        cli_executable=Path(sys.executable),
    )
    step_ids = {step.id for step in plan.steps}
    execution = json.loads(plan.scientific_execution_contract_json)
    output_names = {Path(path).name for path in plan.expected_output_relative_paths}
    mapped_artifacts = {Path(mapping.relative_path).name: (mapping.scientific_type, mapping.output_role) for mapping in plan.artifact_mappings if mapping.relative_path is not None}

    assert "default_test_ratio" not in step_ids
    assert "continue_after_split" not in step_ids
    assert "continue_after_external_training_scope" in step_ids
    assert execution["evaluation_mode"] == "external_labeled"
    assert execution["model_parameters"]["max_depth"] is None
    assert execution["split_seed"] is None
    assert execution["model_seed"] == 280
    assert dict(plan.effective_seeds) == {"model": 280}
    assert dict(plan.effective_model_parameters)["random_state"] == "280"
    assert next(step.response for step in plan.steps if step.id == "maximum_depth") == ""
    assert "Y Test Predict.xlsx" not in output_names
    assert "Model Score - Extra-Trees.txt" not in output_names
    assert "Predicted vs. Actual Diagram - Extra-Trees.png" not in output_names
    assert "Predicted vs. Actual Density - Extra-Trees.png" not in output_names
    assert "Y Train Predict.xlsx" in output_names
    assert "Training Model Score - Extra-Trees.txt" in output_names
    assert "Cross Validation - Extra-Trees.txt" in output_names
    assert "External Evaluation Predictions - Extra-Trees.xlsx" in output_names
    assert "External Evaluation Residuals - Extra-Trees.xlsx" in output_names
    assert "External Evaluation Model Score - Extra-Trees.txt" in output_names
    assert "Feature Importance Diagram - Extra-Trees.xlsx" in output_names
    assert mapped_artifacts["External Evaluation Model Score - Extra-Trees.txt"] == (
        "external_regression_metrics",
        "evaluation.external",
    )
    assert mapped_artifacts["External Predicted vs. Actual - Extra-Trees.png"] == (
        "observed_predicted_figure",
        "evaluation.figure",
    )


def test_full_structured_xgboost_contract_preserves_constructor_controls(
    tmp_path: Path,
) -> None:
    model = {
        "type": "xgboost",
        "number_of_estimators": 890,
        "learning_rate": 0.11,
        "maximum_depth": 19,
        "subsample": 1.0,
        "column_subsample": 0.9,
        "gamma": 0.0,
        "tree_method": "auto",
        "l1_regularization": 0.0,
        "l2_regularization": 1.0,
        "base_score": 0.2,
        "booster": "gbtree",
        "column_subsample_by_level": 1.0,
        "column_subsample_by_node": 1.0,
        "importance_type": "gain",
        "maximum_delta_step": 0.0,
        "minimum_child_weight": 130.0,
        "number_of_jobs": 1,
        "verbosity": 1,
    }
    request = _request(
        _dataset(tmp_path),
        scaling="none",
        missing_values={"method": "error"},
        model=model,
        evaluation={
            "mode": "holdout",
            "split_strategy": "random_holdout",
            "folds": 5,
        },
        reproducibility={
            "split_seed": 99,
            "model_seed": 0,
            "model_parameter_assertions": {**model, "random_state": 0},
        },
    )

    plan = AnalysisPlanCompiler().compile(
        request,
        cli_executable=Path(sys.executable),
    )
    execution = json.loads(plan.scientific_execution_contract_json)

    assert dict(plan.effective_seeds) == {"split": 99, "model": 0}
    assert execution["cross_validation_folds"] == 5
    assert execution["split_seed"] == 99
    assert execution["model_seed"] == 0
    assert execution["model_parameters"] == {
        "base_score": 0.2,
        "booster": "gbtree",
        "colsample_bylevel": 1.0,
        "colsample_bynode": 1.0,
        "colsample_bytree": 0.9,
        "gamma": 0.0,
        "importance_type": "gain",
        "learning_rate": 0.11,
        "max_delta_step": 0.0,
        "max_depth": 19,
        "min_child_weight": 130.0,
        "n_estimators": 890,
        "n_jobs": 1,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "subsample": 1.0,
        "tree_method": "auto",
        "verbosity": 1,
    }
    assert dict(plan.effective_model_parameters)["random_state"] == "0"


def test_linear_family_plot_dimension_prompts_follow_final_feature_count(tmp_path: Path) -> None:
    two_features = RegressionPlanCompiler().compile(
        _request(_dataset(tmp_path), model={"type": "linear_regression"}),
        cli_executable=Path(sys.executable),
    )
    two_feature_steps = {step.id: step.response for step in two_features.steps}
    assert two_feature_steps["one_dimensional_plot_feature"] == "1"
    assert "two_dimensional_plot_feature_1" not in two_feature_steps

    three_features = RegressionPlanCompiler().compile(
        _request(
            _dataset(tmp_path),
            engineered_features=({"name": "SiTi", "formula": "{SIO2} / {TIO2}"},),
            model={"type": "ridge_regression"},
        ),
        cli_executable=Path(sys.executable),
    )
    three_feature_steps = {step.id: step.response for step in three_features.steps}
    assert three_feature_steps["one_dimensional_plot_feature"] == "1"
    assert three_feature_steps["two_dimensional_plot_feature_1"] == "1"
    assert three_feature_steps["two_dimensional_plot_feature_2"] == "2"


def test_regression_rejects_non_numeric_targets_and_unsupported_missing_value_models(tmp_path: Path) -> None:
    with pytest.raises(PlanCompilationError, match="target column.*non-numeric"):
        RegressionPlanCompiler().compile(_request(_dataset(tmp_path, non_numeric_target=True)), cli_executable=Path(sys.executable))

    missing_path = _dataset(tmp_path, missing=True)
    with pytest.raises(PlanCompilationError, match="choose keep, drop_rows, or impute explicitly"):
        RegressionPlanCompiler().compile(_request(missing_path), cli_executable=Path(sys.executable))
    with pytest.raises(PlanCompilationError, match="only offers XGBoost"):
        RegressionPlanCompiler().compile(
            _request(missing_path, missing_values={"method": "keep"}, model={"type": "ridge_regression"}),
            cli_executable=Path(sys.executable),
        )

    xgboost = RegressionPlanCompiler().compile(
        _request(missing_path, missing_values={"method": "keep"}, model={"type": "xgboost"}),
        cli_executable=Path(sys.executable),
    )
    assert {step.id: step.response for step in xgboost.steps}["xgboost"] == "1"

    too_small = tmp_path / "too-small.csv"
    too_small.write_text("SampleID,Target,SIO2,TIO2\n" + "\n".join(f"S-{index},{index * 2},{index},{index / 10}" for index in range(1, 11)) + "\n", encoding="utf-8")
    with pytest.raises(PlanCompilationError, match="fixed 10-fold cross-validation"):
        RegressionPlanCompiler().compile(_request(too_small), cli_executable=Path(sys.executable))

    negative_target = tmp_path / "negative-target.csv"
    negative_target.write_text(_dataset(tmp_path).read_text(encoding="utf-8").replace("11.5", "-11.5", 1), encoding="utf-8")
    with pytest.raises(PlanCompilationError, match="poisson.*non-negative"):
        RegressionPlanCompiler().compile(
            _request(negative_target, model={"type": "decision_tree", "criterion": "poisson"}),
            cli_executable=Path(sys.executable),
        )


def test_regression_request_rejects_classification_only_and_manual_automl_fields(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _request(_dataset(tmp_path), label_customization={"strategy": "encode_original"})
    with pytest.raises(ValidationError, match="manual model settings are not used"):
        _request(
            _dataset(tmp_path),
            tuning="automl",
            model={"type": "ridge_regression", "alpha": 0.5},
        )
