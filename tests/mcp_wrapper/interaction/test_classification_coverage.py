import ast
import json
import sys
from pathlib import Path

import pytest
from geochemistrypi_mcp import ClassificationPlanCompiler, ClassificationRequest, DatasetCompilationContext, PlanCompilationError
from geochemistrypi_mcp.api.schemas import ArtifactRequirement
from geochemistrypi_mcp.contracts.classification import MODEL_DISPLAY_NAMES, MODEL_ORDER
from geochemistrypi_mcp.planning.interaction_plan import AnalysisPlanCompiler
from geochemistrypi_mcp.planning.scientific_contract import assess_scientific_compatibility, canonical_scientific_contract, describe_scientific_output, planned_artifact_requirements

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CAPABILITY_FIXTURE = REPOSITORY_ROOT / "tests" / "mcp_wrapper" / "parity" / "fixtures" / "classification_capability_matrix_v1.json"


def _dataset(tmp_path: Path, *, missing: bool = False, numeric_target: bool = False) -> Path:
    path = tmp_path / ("classification-missing.csv" if missing else "classification.csv")
    targets = (10, 20, 30, 40, 50, 60, 70, 80) if numeric_target else ("basalt", "granite") * 4
    rows = ["SampleID,Label,SIO2,TIO2"]
    for index, target in enumerate(targets, start=1):
        sio2 = "" if missing and index == 3 else str(48 + index)
        rows.append(f"S-{index},{target},{sio2},{0.5 + index / 10}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _request(path: Path, **overrides) -> ClassificationRequest:
    values = {
        "training_dataset_path": path,
        "experiment_name": "PR3 Coverage",
        "run_name": "Classification",
        "identifier_column": "SampleID",
        "feature_columns": ("SIO2", "TIO2"),
        "target_column": "Label",
    }
    values.update(overrides)
    return ClassificationRequest(**values)


def _assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Assignment {name} not found")


def test_classification_metric_request_is_typed_and_fail_closed() -> None:
    assert _request(Path("train.csv")).metric_average == "auto"
    with pytest.raises(ValueError, match="requires an explicit positive_label"):
        _request(Path("train.csv"), metric_average="binary")
    with pytest.raises(ValueError, match="valid only"):
        _request(Path("train.csv"), metric_average="weighted", positive_label="basalt")

    numeric = _request(Path("train.csv"), metric_average="binary", positive_label=1)
    textual = _request(Path("train.csv"), metric_average="binary", positive_label="1")
    assert numeric.model_dump()["positive_label"] == 1
    assert type(numeric.model_dump()["positive_label"]) is int
    assert textual.model_dump()["positive_label"] == "1"
    assert type(textual.model_dump()["positive_label"]) is str


def test_binary_positive_label_is_checked_after_label_customization(tmp_path: Path) -> None:
    path = _dataset(tmp_path)
    valid = _request(path, metric_average="binary", positive_label="basalt")
    ClassificationPlanCompiler().compile(valid, cli_executable=Path(sys.executable))

    invalid = _request(path, metric_average="binary", positive_label=1)
    with pytest.raises(PlanCompilationError, match="does not exactly match"):
        ClassificationPlanCompiler().compile(invalid, cli_executable=Path(sys.executable))


def test_csv_numeric_positive_label_matches_cli_dtype_inference(tmp_path: Path) -> None:
    path = tmp_path / "numeric-labels.csv"
    path.write_text(
        "SampleID,Label,SIO2,TIO2\n" "S-1,1,49,0.6\nS-2,0,50,0.7\nS-3,1,51,0.8\nS-4,0,52,0.9\n" "S-5,1,53,1.0\nS-6,0,54,1.1\nS-7,1,55,1.2\nS-8,0,56,1.3\n",
        encoding="utf-8",
    )

    numeric = _request(path, metric_average="binary", positive_label=1)
    ClassificationPlanCompiler().compile(numeric, cli_executable=Path(sys.executable))

    textual = _request(path, metric_average="binary", positive_label="1")
    with pytest.raises(PlanCompilationError, match="does not exactly match"):
        ClassificationPlanCompiler().compile(textual, cli_executable=Path(sys.executable))


def test_binary_metric_semantics_bind_to_execution_and_canonical_contract(tmp_path: Path) -> None:
    request = _request(
        _dataset(tmp_path),
        metric_average="binary",
        positive_label="basalt",
        model={"type": "xgboost", "objective": "binary:logistic", "importance_type": "gain"},
    )

    plan = AnalysisPlanCompiler().compile(request, cli_executable=Path(sys.executable))
    execution = json.loads(plan.scientific_execution_contract_json)
    canonical = canonical_scientific_contract(request, plan)

    assert execution["schema_version"] == 4
    assert execution["classification_metric_average"] == "binary"
    assert execution["classification_positive_label"] == {"type": "string", "value": "basalt"}
    assert canonical["parameters"]["metric_average"] == "binary"
    assert canonical["parameters"]["positive_label"] == {"type": "string", "value": "basalt"}


def test_explicit_positive_label_uses_a_metric_only_contract_for_manual_models(tmp_path: Path) -> None:
    request = _request(
        _dataset(tmp_path),
        metric_average="binary",
        positive_label="basalt",
        model={"type": "logistic_regression"},
    )

    plan = AnalysisPlanCompiler().compile(request, cli_executable=Path(sys.executable))
    requirements = planned_artifact_requirements(request, plan)
    assessment = assess_scientific_compatibility(
        request,
        plan,
        requirements,
    )

    execution = json.loads(plan.scientific_execution_contract_json)
    assert execution["classification_metric_average"] == "binary"
    assert execution["classification_positive_label"] == {
        "type": "string",
        "value": "basalt",
    }
    assert execution["model_parameters"] == {}
    attestation_requirement = next(requirement for requirement in requirements if requirement.scientific_type == "parameter_attestation")
    assert attestation_requirement.required_json_keys == (
        "schema_version",
        "contract.source_sha256",
        "effective_model_parameters",
        "verified_parameter_names",
        "estimator_identity.expected.module_root",
        "estimator_identity.expected.class_name",
        "estimator_identity.observed.module",
        "estimator_identity.observed.qualname",
        "verification_status",
        "attestation_sha256",
    )
    assert assessment.execution_ready is True
    assert not any("cannot bind" in issue for issue in assessment.blocking_issues)


def test_native_target_label_mapping_is_required_and_semantically_bound(
    tmp_path: Path,
) -> None:
    request = _request(_dataset(tmp_path), model={"type": "logistic_regression"})
    plan = AnalysisPlanCompiler().compile(request, cli_executable=Path(sys.executable))
    normalized_paths = {path.replace("\\", "/") for path in plan.expected_output_relative_paths}
    target_path = next(path for path in normalized_paths if path.endswith("/artifacts/data/Target Label Mapping.xlsx"))
    mapping = next(
        item for item in plan.artifact_mappings if item.relative_path == target_path and item.scientific_type == "target_label_mapping" and item.output_role == "classification.target_label_mapping"
    )
    requirement = next(item for item in planned_artifact_requirements(request, plan) if item.expected_relative_path == target_path)
    fallback = describe_scientific_output(target_path, plan.workflow_family)

    assert mapping.availability == "available"
    assert requirement.scientific_type == "target_label_mapping"
    assert requirement.output_role == "classification.target_label_mapping"
    assert fallback["scientific_type"] == "target_label_mapping"
    assert fallback["output_role"] == "classification.target_label_mapping"
    native_source = (REPOSITORY_ROOT / "geochemistrypi" / "data_mining" / "model" / "classification.py").read_text(encoding="utf-8")
    assert 'save_data_without_data_identifier(mapping_df, "Target Label Mapping"' in native_source

    caller_requirement = ArtifactRequirement(
        requirement_id="classification.target-label-mapping",
        scientific_type="target_label_mapping",
        output_role="classification.target_label_mapping",
        category="artifacts",
        media_types=("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",),
        expected_relative_path=target_path,
    )
    explicit = request.model_copy(update={"artifact_requirements": (caller_requirement,)})
    explicit_plan = AnalysisPlanCompiler().compile(
        explicit,
        cli_executable=Path(sys.executable),
    )
    explicit_assessment = assess_scientific_compatibility(
        explicit,
        explicit_plan,
        planned_artifact_requirements(explicit, explicit_plan),
    )
    assert explicit_assessment.execution_ready is True

    wrong_role = caller_requirement.model_copy(update={"output_role": "evaluation.predictions"})
    incompatible = request.model_copy(update={"artifact_requirements": (wrong_role,)})
    incompatible_plan = AnalysisPlanCompiler().compile(
        incompatible,
        cli_executable=Path(sys.executable),
    )
    incompatible_assessment = assess_scientific_compatibility(
        incompatible,
        incompatible_plan,
        planned_artifact_requirements(incompatible, incompatible_plan),
    )
    assert incompatible_assessment.execution_ready is False
    assert any("required artifact" in issue.lower() for issue in incompatible_assessment.blocking_issues)


def test_scientific_attestation_requirement_cannot_be_omitted_or_weakened(
    tmp_path: Path,
) -> None:
    base = _request(
        _dataset(tmp_path),
        model={"type": "xgboost"},
    )
    base_plan = AnalysisPlanCompiler().compile(base, cli_executable=Path(sys.executable))
    baseline = next(requirement for requirement in planned_artifact_requirements(base, base_plan) if requirement.scientific_type == "parameter_attestation")
    caller_requirement = ArtifactRequirement(
        requirement_id="caller.metrics",
        scientific_type="holdout_metrics",
        output_role="evaluation.holdout",
        category="metrics",
        expected_relative_path="metrics/Model Score - XGBoost.txt",
    )
    explicit = base.model_copy(update={"artifact_requirements": (caller_requirement,)})
    explicit_plan = AnalysisPlanCompiler().compile(
        explicit,
        cli_executable=Path(sys.executable),
    )

    merged = planned_artifact_requirements(explicit, explicit_plan)

    assert merged == (caller_requirement, baseline)

    weakened = baseline.model_copy(update={"required": False, "required_json_keys": ()})
    conflicting = base.model_copy(update={"artifact_requirements": (weakened,)})
    conflicting_plan = AnalysisPlanCompiler().compile(
        conflicting,
        cli_executable=Path(sys.executable),
    )
    with pytest.raises(ValueError, match="immutable scientific execution attestation"):
        planned_artifact_requirements(conflicting, conflicting_plan)


@pytest.mark.parametrize("model_name", MODEL_ORDER)
def test_every_manual_classification_contract_emits_the_canonical_cli_sidecar(
    tmp_path: Path,
    model_name: str,
) -> None:
    request = _request(_dataset(tmp_path), model={"type": model_name})
    plan = AnalysisPlanCompiler().compile(request, cli_executable=Path(sys.executable))
    payload = json.loads(plan.scientific_execution_contract_json)
    path = tmp_path / f"{model_name}-scientific-execution.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert payload["schema_version"] == 4
    assert payload["workflow_family"] == "supervised_learning"
    assert payload["workflow_mode"] == "classification"
    assert payload["method"] == model_name
    assert payload["classification_metric_average"] == "auto"
    scaling_wait = next(step for step in plan.steps if step.id == "continue_after_scaling")
    assert scaling_wait.output_anchors[0] == "Scientific execution: scaler fitting is deferred until the model-fitting cohort is defined."

    # This is the exact sidecar shape used by models whose manual constructor
    # inputs remain prompt-bound. CLI-owned tests and installed-wheel parity
    # validate the separate Python 3.9 loader without importing it here.
    payload["model_parameters"] = {}
    assert payload["method"] == model_name
    assert payload["model_parameters"] == {}


def test_extra_trees_plan_binds_structured_constructor_parameters_and_seed(tmp_path: Path) -> None:
    request = _request(
        _dataset(tmp_path),
        model={
            "type": "extra_trees",
            "number_of_estimators": 321,
            "maximum_depth": 9,
            "minimum_samples_split": 3,
            "minimum_samples_leaf": 2,
            "maximum_features": 2,
            "bootstrap": True,
            "maximum_samples": 0.75,
            "out_of_bag_score": True,
        },
        reproducibility={"model_seed": 2025},
    )

    plan = AnalysisPlanCompiler().compile(request, cli_executable=Path(sys.executable))
    execution = json.loads(plan.scientific_execution_contract_json)

    assert execution["method"] == "extra_trees"
    assert execution["model_seed"] == 2025
    assert execution["model_parameters"] == {
        "bootstrap": True,
        "max_depth": 9,
        "max_features": 2,
        "max_samples": 0.75,
        "min_samples_leaf": 2,
        "min_samples_split": 3,
        "n_estimators": 321,
        "oob_score": True,
    }


def test_versioned_capability_matrix_matches_public_cli_constants() -> None:
    fixture = json.loads(CAPABILITY_FIXTURE.read_text(encoding="utf-8"))
    cli_models = _assignment(REPOSITORY_ROOT / "geochemistrypi" / "data_mining" / "constants.py", "CLASSIFICATION_MODELS")

    assert tuple(item["id"] for item in fixture["models"]) == MODEL_ORDER
    assert [item["cli_name"] for item in fixture["models"]] == cli_models
    assert [MODEL_DISPLAY_NAMES[model] for model in MODEL_ORDER] == cli_models


@pytest.mark.parametrize("model_name", MODEL_ORDER)
def test_every_public_cli_model_family_compiles_a_manual_plan(tmp_path: Path, model_name: str) -> None:
    request = _request(_dataset(tmp_path), model={"type": model_name})
    plan = ClassificationPlanCompiler().compile(request, cli_executable=Path(sys.executable))
    responses = {step.id: step.response for step in plan.steps}

    assert responses[model_name] == str(MODEL_ORDER.index(model_name) + 1)
    assert any(MODEL_DISPLAY_NAMES[model_name] + " - Hyper-parameters Specification" in step.output_anchors for step in plan.steps)
    assert Path(plan.expected_output_relative_paths[0]).parts[-3:] == (
        "artifacts",
        "model",
        f"{MODEL_DISPLAY_NAMES[model_name]}.joblib",
    )
    assert Path(plan.expected_output_relative_paths[2]).name == (f"Precision-Recall vs. Threshold Diagram - {MODEL_DISPLAY_NAMES[model_name]}.png")
    assert any(Path(path).name == "Target Label Mapping.xlsx" for path in plan.expected_output_relative_paths)


def test_existing_experiment_id_skips_ambiguous_name_prompts(tmp_path: Path) -> None:
    request = _request(_dataset(tmp_path), existing_experiment_id="stable-7")
    plan = ClassificationPlanCompiler().compile(request, cli_executable=Path(sys.executable))

    assert plan.steps[0].id == "run_name"
    assert "use_previous_experiment" not in {step.id for step in plan.steps}
    assert "experiment_name" not in {step.id for step in plan.steps}
    assert plan.public_command[-2:] == ("--existing-experiment-id", "stable-7")


@pytest.mark.parametrize("model_name", MODEL_ORDER)
def test_every_public_cli_model_family_compiles_an_automl_plan_without_manual_prompts(tmp_path: Path, model_name: str) -> None:
    request = _request(_dataset(tmp_path), tuning="automl", model={"type": model_name})
    plan = ClassificationPlanCompiler().compile(request, cli_executable=Path(sys.executable))

    step_ids = {step.id for step in plan.steps}
    assert {step.id: step.response for step in plan.steps}["enable_automl"] == "1"
    assert "continue_after_hyperparameters" not in step_ids
    assert not any("Hyper-parameters Specification" in anchor for step in plan.steps for anchor in step.output_anchors)


def test_label_preprocessing_feature_engineering_and_application_branches_compile(tmp_path: Path) -> None:
    numeric = _dataset(tmp_path, numeric_target=True)
    interval = _request(
        numeric,
        label_customization={"strategy": "interval", "cut_points": (30, 60), "labels": ("low", "middle", "high")},
        metric_average="macro",
        scaling="mean_normalization",
        feature_selection={"method": "select_k_best", "retain_count": 1},
        engineered_features=({"name": "SiTi", "formula": "{SIO2} / {TIO2}"},),
        test_ratio=0.4,
    )
    plan = ClassificationPlanCompiler().compile(interval, cli_executable=Path(sys.executable))
    responses = {step.id: step.response for step in plan.steps}
    assert responses["interval_labels"] == "3"
    assert responses["metric_average"] == "2"
    assert responses["mean_normalization"] == "3"
    assert responses["feature_selection_method"] == "2"
    assert responses["engineered_feature_1_formula"] == "b / c"
    assert responses["feature_columns"] == "[2,4]"

    application = tmp_path / "application.csv"
    application.write_text("SampleID,SIO2,TIO2\nA-1,55,1.1\n", encoding="utf-8")
    inference_plan = ClassificationPlanCompiler().compile(
        _request(_dataset(tmp_path), application_dataset_path=application),
        cli_executable=Path(sys.executable),
    )
    assert inference_plan.public_command[:6] == (
        str(Path(sys.executable).resolve()),
        "data-mining",
        "--training",
        str((tmp_path / "classification.csv").resolve()),
        "--application",
        str(application.resolve()),
    )
    assert inference_plan.public_command[6] == "--world-map-config"
    assert json.loads(inference_plan.public_command[7])["enabled"] is False
    assert any(step.id == "continue_after_inference" for step in inference_plan.steps)


def test_missing_value_contract_rejects_silent_or_unsupported_execution(tmp_path: Path) -> None:
    missing_path = _dataset(tmp_path, missing=True)
    with pytest.raises(PlanCompilationError, match="choose keep, drop_rows, or impute explicitly"):
        ClassificationPlanCompiler().compile(_request(missing_path), cli_executable=Path(sys.executable))
    with pytest.raises(PlanCompilationError, match="only offers XGBoost"):
        ClassificationPlanCompiler().compile(
            _request(missing_path, missing_values={"method": "keep"}),
            cli_executable=Path(sys.executable),
        )

    keep_xgboost = ClassificationPlanCompiler().compile(
        _request(missing_path, missing_values={"method": "keep"}, model={"type": "xgboost"}),
        cli_executable=Path(sys.executable),
    )
    assert {step.id: step.response for step in keep_xgboost.steps}["xgboost"] == "1"

    no_missing_path = _dataset(tmp_path)
    with pytest.raises(PlanCompilationError, match="would be silently skipped"):
        ClassificationPlanCompiler().compile(
            _request(no_missing_path, missing_values={"method": "impute", "strategy": "constant", "fill_value": 0}),
            cli_executable=Path(sys.executable),
        )


@pytest.mark.parametrize(
    ("overrides", "expected_responses"),
    [
        ({"scaling": "none"}, {"skip_feature_scaling": "2"}),
        ({"scaling": "min_max"}, {"min_max": "1"}),
        (
            {"feature_selection": {"method": "generic_univariate", "retain_count": 1}},
            {"feature_selection_method": "1", "feature_selection_retain_count": "1"},
        ),
        (
            {"label_customization": {"strategy": "map", "mapping": {"basalt": "mafic", "granite": "felsic"}}},
            {"map_labels": "2", "label_mapping": "basalt:mafic; granite:felsic"},
        ),
    ],
)
def test_material_preprocessing_and_label_prompt_branches(
    tmp_path: Path,
    overrides: dict,
    expected_responses: dict[str, str],
) -> None:
    plan = ClassificationPlanCompiler().compile(
        _request(_dataset(tmp_path), **overrides),
        cli_executable=Path(sys.executable),
    )
    responses = {step.id: step.response for step in plan.steps}
    assert responses | expected_responses == responses


def test_quantile_and_all_missing_value_processing_prompt_branches(tmp_path: Path) -> None:
    quantile_plan = ClassificationPlanCompiler().compile(
        _request(
            _dataset(tmp_path, numeric_target=True),
            label_customization={"strategy": "quantile", "number_of_classes": 2, "labels": ("low", "high")},
        ),
        cli_executable=Path(sys.executable),
    )
    quantile_responses = {step.id: step.response for step in quantile_plan.steps}
    assert quantile_responses["quantile_labels"] == "4"
    assert quantile_responses["number_of_classes"] == "2"
    assert quantile_responses["class_labels"] == "low; high"

    missing_path = _dataset(tmp_path, missing=True)
    drop_all = ClassificationPlanCompiler().compile(
        _request(missing_path, missing_values={"method": "drop_rows"}),
        cli_executable=Path(sys.executable),
    )
    assert {step.id: step.response for step in drop_all.steps}["drop_all_missing_rows"] == "1"

    drop_specific = ClassificationPlanCompiler().compile(
        _request(missing_path, missing_values={"method": "drop_rows", "columns": ("SIO2",)}),
        cli_executable=Path(sys.executable),
    )
    specific_responses = {step.id: step.response for step in drop_specific.steps}
    assert specific_responses["drop_missing_by_columns"] == "2"
    assert specific_responses["drop_columns"] == "2"

    impute = ClassificationPlanCompiler().compile(
        _request(
            missing_path,
            missing_values={"method": "impute", "strategy": "constant", "fill_value": -999},
        ),
        cli_executable=Path(sys.executable),
    )
    impute_responses = {step.id: step.response for step in impute.steps}
    assert impute_responses["imputation_method"] == "4"
    assert impute_responses["imputation_fill_value"] == "-999"


@pytest.mark.parametrize(
    ("model", "required_steps", "absent_steps"),
    [
        (
            {"type": "logistic_regression", "penalty": "l1", "solver": "saga"},
            {"l1_penalty", "solver"},
            {"l1_ratio"},
        ),
        (
            {"type": "logistic_regression", "penalty": "elasticnet", "solver": "saga", "l1_ratio": 0.25},
            {"elasticnet_penalty", "l1_ratio"},
            {"solver"},
        ),
        (
            {"type": "support_vector_machine", "kernel": "poly"},
            {"kernel", "degree", "gamma"},
            set(),
        ),
        (
            {"type": "random_forest", "bootstrap": False, "maximum_samples": None, "out_of_bag_score": False},
            {"bootstrap", "out_of_bag_score"},
            {"maximum_samples"},
        ),
        (
            {"type": "k_nearest_neighbors", "algorithm": "kd_tree", "metric": "euclidean"},
            {"algorithm", "leaf_size", "metric"},
            {"power"},
        ),
        (
            {"type": "stochastic_gradient_descent", "penalty": "elasticnet"},
            {"penalty", "l1_ratio"},
            set(),
        ),
    ],
)
def test_conditional_model_prompt_branches(
    tmp_path: Path,
    model: dict,
    required_steps: set[str],
    absent_steps: set[str],
) -> None:
    plan = ClassificationPlanCompiler().compile(
        _request(_dataset(tmp_path), model=model),
        cli_executable=Path(sys.executable),
    )
    step_ids = {step.id for step in plan.steps}
    assert required_steps <= step_ids
    assert not absent_steps & step_ids


def test_application_feature_engineering_uses_the_cli_name_based_replay_branch(tmp_path: Path) -> None:
    training = _dataset(tmp_path)
    application = tmp_path / "application.csv"
    application.write_text("SampleID,SIO2,TIO2\nA-1,55,1.1\n", encoding="utf-8")
    request = _request(
        training,
        application_dataset_path=application,
        engineered_features=(
            {"name": "SiTi", "formula": "{SIO2} / {TIO2}"},
            {"name": "SiTiSum", "formula": "{SIO2} + {TIO2}"},
        ),
    )

    plan = ClassificationPlanCompiler().compile(request, cli_executable=Path(sys.executable))
    responses = {step.id: step.response for step in plan.steps}
    assert responses["engineered_feature_1_formula"] == "b / c"
    assert responses["engineered_feature_1_formula_ack"] == ""
    assert responses["engineered_feature_1_constructed"] == ""
    assert responses["engineered_feature_1_statistics"] == ""
    assert responses["engineered_feature_1_continue"] == "1"
    assert responses["engineered_feature_1_continue_to_next"] == ""
    assert responses["engineered_feature_2_formula"] == "b + c"
    assert responses["engineered_feature_2_continue"] == "2"
    preparation = next(step for step in plan.steps if step.id == "continue_after_application_preparation")
    assert preparation.output_anchors[0] == "Application Data Feature-Engineering Selected.xlsx"
    assert any(step.id == "continue_after_inference" for step in plan.steps)


def test_identifier_values_are_not_system_keys_but_scientific_inputs_stay_fail_closed(tmp_path: Path) -> None:
    training = _dataset(tmp_path)
    duplicate_training = tmp_path / "duplicate.csv"
    duplicate_training.write_text(
        training.read_text(encoding="utf-8").replace("S-2,granite", "S-1,granite").replace("S-3,basalt", ",basalt"),
        encoding="utf-8",
    )
    plan = ClassificationPlanCompiler().compile(
        _request(duplicate_training),
        cli_executable=Path(sys.executable),
    )
    assert plan.requires_source_row_pairing is True

    application = tmp_path / "invalid-application.csv"
    application.write_text("SampleID,SIO2,TIO2\nA-1,not-a-number,1.1\n", encoding="utf-8")
    with pytest.raises(PlanCompilationError, match="contains a non-numeric value"):
        ClassificationPlanCompiler().compile(
            _request(training, application_dataset_path=application),
            cli_executable=Path(sys.executable),
        )

    invalid_formula = _request(
        training,
        engineered_features=({"name": "Bad", "formula": "pow({SIO2})"},),
    )
    with pytest.raises(PlanCompilationError, match="wrong number of function arguments"):
        ClassificationPlanCompiler().compile(invalid_formula, cli_executable=Path(sys.executable))

    leaking_formula = _request(
        training,
        engineered_features=({"name": "Leak", "formula": "{Label} + {SIO2}"},),
    )
    with pytest.raises(PlanCompilationError, match="must not use the target column"):
        ClassificationPlanCompiler().compile(leaking_formula, cli_executable=Path(sys.executable))


def test_dataset_provenance_controls_duplicate_header_compatibility_per_role(tmp_path: Path) -> None:
    training = _dataset(tmp_path)
    training_rows = training.read_text(encoding="utf-8").splitlines()
    duplicate_training = tmp_path / "builtin-training.csv"
    duplicate_training.write_text(
        "\n".join([f"{training_rows[0]},FEOT,FEOT", *(f"{row},8.1,8.2" for row in training_rows[1:])]) + "\n",
        encoding="utf-8",
    )
    compiler = ClassificationPlanCompiler()

    with pytest.raises(PlanCompilationError, match="duplicate or colliding column names"):
        compiler.compile(_request(duplicate_training), cli_executable=Path(sys.executable))

    trusted_training = compiler.compile(
        _request(duplicate_training),
        cli_executable=Path(sys.executable),
        dataset_context=DatasetCompilationContext(training_source="builtin"),
    )
    assert trusted_training.name == "classification-logistic_regression-v1"

    duplicate_application = tmp_path / "application.csv"
    duplicate_application.write_text(
        "SampleID,SIO2,TIO2,FEOT,FEOT\nA-1,55,1.1,8.1,8.2\n",
        encoding="utf-8",
    )
    application_request = _request(training, application_dataset_path=duplicate_application)
    with pytest.raises(PlanCompilationError, match="duplicate or colliding column names"):
        compiler.compile(
            application_request,
            cli_executable=Path(sys.executable),
            dataset_context=DatasetCompilationContext(training_source="builtin"),
        )

    trusted_application = compiler.compile(
        application_request,
        cli_executable=Path(sys.executable),
        dataset_context=DatasetCompilationContext(application_source="builtin"),
    )
    assert "--application" in trusted_application.public_command
