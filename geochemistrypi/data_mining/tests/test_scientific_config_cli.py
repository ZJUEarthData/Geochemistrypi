import copy
import importlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import pytest
from jsonschema import Draft202012Validator
from typer.testing import CliRunner

import geochemistrypi.cli as cli_module
import geochemistrypi.scientific_execution as execution_module
from geochemistrypi.scientific_config import build_scientific_execution_template, registered_scientific_workflows, scientific_execution_example, scientific_execution_json_schema
from geochemistrypi.scientific_execution import ScientificExecutionContract, ScientificExecutionContractError, scientific_execution_context


def _write_isolation_forest_example(path: Path) -> Path:
    path.write_text(
        json.dumps(scientific_execution_example("isolation_forest")),
        encoding="utf-8",
    )
    return path.resolve()


def test_data_mining_accepts_scientific_config_without_automation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _write_isolation_forest_example(tmp_path / "scientific.json")
    captured = {}
    monkeypatch.setattr(
        cli_module,
        "_run_cli_pipeline",
        lambda **values: captured.update(values),
    )

    result = CliRunner().invoke(
        cli_module.app,
        [
            "data-mining",
            "--data",
            str(tmp_path / "rocks.xlsx"),
            "--scientific-config",
            str(config),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["scientific_config"] == str(config)
    assert captured["automation_plan"] == ""
    assert captured["automation_events"] == ""


def test_run_cli_pipeline_activates_scientific_contract_for_interactive_cli(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _write_isolation_forest_example(tmp_path / "scientific.json")
    pipeline_module = importlib.import_module("geochemistrypi.data_mining.cli_pipeline")
    observed = {}

    def fake_pipeline(**values) -> None:
        contract = execution_module.active_scientific_execution()
        assert contract is not None
        observed["identity"] = (
            contract.workflow_family,
            contract.workflow_mode,
            contract.method,
        )
        observed["values"] = values

    monkeypatch.setattr(pipeline_module, "cli_pipeline", fake_pipeline)

    with pytest.raises(
        ScientificExecutionContractError,
        match="exactly one Scientific Execution Attestation",
    ):
        cli_module._run_cli_pipeline(
            training_data_path="rocks.xlsx",
            application_data_path="",
            data_source_name="ANY_PATH",
            scientific_config=str(config),
        )

    assert observed["identity"] == (
        "anomaly_detection",
        "outlier_detection",
        "isolation_forest",
    )
    assert execution_module.active_scientific_execution() is None


def test_run_cli_pipeline_nests_automation_inside_scientific_contract(
    monkeypatch,
) -> None:
    pipeline_module = importlib.import_module("geochemistrypi.data_mining.cli_pipeline")
    automation_module = importlib.import_module("geochemistrypi.automation")
    events = []

    @contextmanager
    def fake_scientific_context(path):
        events.append(("scientific-enter", path))
        try:
            yield object()
        finally:
            events.append(("scientific-exit", path))

    @contextmanager
    def fake_automation_adapter(plan, event_path):
        events.append(("automation-enter", plan, event_path))
        try:
            yield object()
        finally:
            events.append(("automation-exit", plan, event_path))

    monkeypatch.setattr(
        execution_module,
        "scientific_execution_context",
        fake_scientific_context,
    )
    monkeypatch.setattr(
        automation_module,
        "automation_input_adapter",
        fake_automation_adapter,
    )
    monkeypatch.setattr(
        pipeline_module,
        "cli_pipeline",
        lambda **_: events.append(("execute",)),
    )

    cli_module._run_cli_pipeline(
        training_data_path="rocks.xlsx",
        application_data_path="",
        data_source_name="ANY_PATH",
        automation_plan="plan.json",
        automation_events="events.json",
        scientific_config="scientific.json",
    )

    assert [event[0] for event in events] == [
        "scientific-enter",
        "automation-enter",
        "execute",
        "automation-exit",
        "scientific-exit",
    ]


def test_every_registered_method_has_a_complete_runtime_valid_template(
    tmp_path: Path,
) -> None:
    workflows = registered_scientific_workflows()
    schema = scientific_execution_json_schema()
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)

    for index, workflow in enumerate(workflows):
        document = build_scientific_execution_template(
            workflow["workflow_family"],
            workflow["workflow_mode"],
            workflow["method"],
        )
        assert set(document) == {
            "schema_version",
            "workflow_family",
            "workflow_mode",
            "method",
            "split_seed",
            "split_strategy",
            "model_seed",
            "cross_validation_folds",
            "evaluation_mode",
            "confusion_matrix_normalization",
            "external_evaluation_identifier_column",
            "external_evaluation_target_columns",
            "target_transformations",
            "classification_metric_average",
            "classification_positive_label",
            "model_parameters",
        }
        path = tmp_path / f"contract-{index}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        assert list(validator.iter_errors(document)) == []
        loaded = ScientificExecutionContract.load(path.resolve())
        assert (loaded.workflow_family, loaded.workflow_mode, loaded.method,) == (
            workflow["workflow_family"],
            workflow["workflow_mode"],
            workflow["method"],
        )


def test_schema_covers_registry_and_retains_descriptions_defaults_and_enums() -> None:
    schema = scientific_execution_json_schema()
    workflows = registered_scientific_workflows()
    branches = schema["allOf"][0]["oneOf"]
    branch_identities = {
        (
            branch["properties"]["workflow_family"]["const"],
            branch["properties"]["workflow_mode"]["const"],
            branch["properties"]["method"]["const"],
        )
        for branch in branches
    }
    registry_identities = {
        (
            item["workflow_family"],
            item["workflow_mode"],
            item["method"],
        )
        for item in workflows
    }

    assert branch_identities == registry_identities
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert all(value.get("description") for value in schema["properties"].values())
    assert schema["properties"]["cross_validation_folds"]["default"] == 10
    assert schema["properties"]["evaluation_mode"]["enum"]
    assert schema["properties"]["classification_metric_average"]["enum"]
    assert (
        len(
            json.dumps(
                schema,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        <= 64 * 1024
    )
    for branch, workflow in zip(branches, workflows):
        assert set(branch["properties"]["model_parameters"]["properties"]) == set(workflow["allowed_model_parameters"])


def test_json_schema_and_runtime_enforce_the_same_cross_field_contracts(
    tmp_path: Path,
) -> None:
    validator = Draft202012Validator(scientific_execution_json_schema())
    classification = build_scientific_execution_template(
        "supervised_learning",
        "classification",
        "xgboost",
    )
    deterministic_knn = build_scientific_execution_template(
        "supervised_learning",
        "classification",
        "k_nearest_neighbors",
    )
    regression = build_scientific_execution_template(
        "supervised_learning",
        "regression",
        "xgboost",
    )
    embedding = build_scientific_execution_template(
        "dimension_reduction",
        "embedding",
        "pca",
    )

    cases = []

    value = copy.deepcopy(deterministic_knn)
    value["model_seed"] = 42
    cases.append(("deterministic-knn-seed", value, False))

    value = copy.deepcopy(classification)
    value["classification_metric_average"] = None
    cases.append(("classification-null-metric", value, False))

    value = copy.deepcopy(regression)
    value["classification_metric_average"] = "macro"
    cases.append(("regression-classification-metric", value, False))

    value = copy.deepcopy(regression)
    value["confusion_matrix_normalization"] = "true"
    cases.append(("regression-confusion-normalization", value, False))

    value = copy.deepcopy(classification)
    value["target_transformations"] = {"Label": {"scale": 1.0, "offset": 0.0}}
    cases.append(("classification-target-transform", value, False))

    value = copy.deepcopy(embedding)
    value["split_seed"] = 2025
    cases.append(("embedding-split-seed", value, False))

    value = copy.deepcopy(regression)
    value.update(
        {
            "evaluation_mode": "external_labeled",
            "split_seed": None,
            "split_strategy": None,
        }
    )
    cases.append(("external-labeled-empty-targets", value, False))

    value = copy.deepcopy(value)
    value["external_evaluation_identifier_column"] = "SAMPLE NAME"
    value["external_evaluation_target_columns"] = ["Target A", "Target B"]
    cases.append(("external-labeled-targets", value, True))

    value = copy.deepcopy(classification)
    value.update(
        {
            "split_seed": 42.0,
            "model_seed": 42.0,
            "cross_validation_folds": 10.0,
            "classification_metric_average": "binary",
            "classification_positive_label": {
                "type": "integer",
                "value": 1.0,
            },
        }
    )
    cases.append(("json-schema-integral-number-semantics", value, True))

    value = copy.deepcopy(value)
    value["classification_positive_label"] = {
        "type": "number",
        "value": 1,
    }
    cases.append(("json-schema-number-semantics", value, True))

    for name, document, expected in cases:
        schema_errors = list(validator.iter_errors(document))
        schema_valid = not schema_errors
        path = (tmp_path / f"{name}.json").resolve()
        path.write_text(json.dumps(document), encoding="utf-8")
        try:
            ScientificExecutionContract.load(path)
        except ScientificExecutionContractError:
            runtime_valid = False
        else:
            runtime_valid = True
        assert schema_valid is expected, (
            name,
            [error.message for error in schema_errors],
        )
        assert runtime_valid is expected, name


def test_json_schema_and_runtime_seed_and_workflow_parity_for_all_registered_methods(
    tmp_path: Path,
) -> None:
    validator = Draft202012Validator(scientific_execution_json_schema())

    def runtime_valid(document, name: str) -> bool:
        path = (tmp_path / f"{name}.json").resolve()
        path.write_text(json.dumps(document), encoding="utf-8")
        try:
            ScientificExecutionContract.load(path)
        except ScientificExecutionContractError:
            return False
        return True

    for index, workflow in enumerate(registered_scientific_workflows()):
        document = build_scientific_execution_template(
            workflow["workflow_family"],
            workflow["workflow_mode"],
            workflow["method"],
        )
        candidates = [("baseline", document)]

        seed_mutation = copy.deepcopy(document)
        if workflow["model_seed_policy"] == "required":
            seed_mutation["model_seed"] = None
        elif workflow["model_seed_policy"] == "conditional":
            seed_mutation["model_seed"] = 2025
        else:
            seed_mutation["model_seed"] = 2025
        candidates.append(("seed-policy", seed_mutation))

        metric_mutation = copy.deepcopy(document)
        metric_mutation["classification_metric_average"] = None if workflow["workflow_mode"] == "classification" else "macro"
        candidates.append(("metric-scope", metric_mutation))

        if workflow["workflow_family"] != "supervised_learning":
            split_mutation = copy.deepcopy(document)
            split_mutation["split_seed"] = 2025
            candidates.append(("split-scope", split_mutation))

        if workflow["workflow_mode"] != "regression":
            transform_mutation = copy.deepcopy(document)
            transform_mutation["target_transformations"] = {"Target": {"scale": 1.0, "offset": 0.0}}
            candidates.append(("transform-scope", transform_mutation))

        if document["evaluation_mode"] != "external_labeled":
            target_mutation = copy.deepcopy(document)
            target_mutation["external_evaluation_target_columns"] = ["Target"]
            candidates.append(("external-target-scope", target_mutation))

        for mutation_name, candidate in candidates:
            schema_valid = not list(validator.iter_errors(candidate))
            observed_runtime_valid = runtime_valid(
                candidate,
                f"{index}-{mutation_name}",
            )
            assert schema_valid is observed_runtime_valid, (
                workflow,
                mutation_name,
            )


def test_active_public_cli_selection_is_bound_before_execution(
    tmp_path: Path,
) -> None:
    pipeline_module = importlib.import_module("geochemistrypi.data_mining.cli_pipeline")
    isolation_path = _write_isolation_forest_example(tmp_path / "isolation.json")

    for mode_num, model_name, is_automl, message in (
        (6, None, False, "Time Series branch"),
        (2, None, False, "cannot configure selected CLI workflow"),
        (5, "all_models", False, "all-models"),
        (5, "Local Outlier Factor", False, "cannot configure selected CLI method"),
    ):
        with pytest.raises(ScientificExecutionContractError, match=message):
            with scientific_execution_context(isolation_path):
                pipeline_module._validate_scientific_cli_selection(
                    mode_num,
                    model_name,
                    is_automl=is_automl,
                )

    classification = build_scientific_execution_template(
        "supervised_learning",
        "classification",
        "xgboost",
    )
    classification_path = (tmp_path / "classification.json").resolve()
    classification_path.write_text(
        json.dumps(classification),
        encoding="utf-8",
    )
    with pytest.raises(ScientificExecutionContractError, match="AutoML"):
        with scientific_execution_context(classification_path):
            pipeline_module._validate_scientific_cli_selection(
                2,
                "XGBoost",
                is_automl=True,
            )

    class _StopAfterSelection(Exception):
        pass

    with pytest.raises(_StopAfterSelection):
        with scientific_execution_context(isolation_path):
            pipeline_module._validate_scientific_cli_selection(
                5,
                "Isolation Forest",
            )
            raise _StopAfterSelection


@pytest.mark.parametrize(
    ("mode_num", "model_name", "message"),
    (
        (2, None, "cannot configure selected CLI workflow"),
        (5, "Local Outlier Factor", "cannot configure selected CLI method"),
    ),
)
def test_mismatched_actual_selection_fails_automation_before_model_execution_or_attestation(
    tmp_path: Path,
    monkeypatch,
    mode_num: int,
    model_name: Optional[str],
    message: str,
) -> None:
    pipeline_module = importlib.import_module("geochemistrypi.data_mining.cli_pipeline")
    contract_path = _write_isolation_forest_example(tmp_path / "isolation.json")
    plan_path = (tmp_path / "automation-plan.json").resolve()
    events_path = (tmp_path / "automation-events.json").resolve()
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "plan_name": "selection-gate",
                "inputs": [],
            }
        ),
        encoding="utf-8",
    )
    fit_calls = []

    def fake_pipeline(**_values) -> None:
        pipeline_module._validate_scientific_cli_selection(
            mode_num,
            model_name,
        )
        fit_calls.append("model-fitted")

    monkeypatch.setattr(pipeline_module, "cli_pipeline", fake_pipeline)

    with pytest.raises(ScientificExecutionContractError, match=message):
        cli_module._run_cli_pipeline(
            training_data_path="rocks.xlsx",
            application_data_path="",
            data_source_name="ANY_PATH",
            automation_plan=str(plan_path),
            automation_events=str(events_path),
            scientific_config=str(contract_path),
        )

    events = json.loads(events_path.read_text(encoding="utf-8"))
    assert events["status"] == "failed"
    assert events["error"]["type"] == "ScientificExecutionContractError"
    assert fit_calls == []
    assert not list(tmp_path.rglob("Scientific Execution Attestation.json"))
    assert execution_module.active_scientific_execution() is None


def test_cli_generates_directly_usable_isolation_forest_example(
    tmp_path: Path,
) -> None:
    output = (tmp_path / "isolation-forest.json").resolve()

    result = CliRunner().invoke(
        cli_module.app,
        [
            "scientific-config",
            "--kind",
            "example",
            "--example",
            "isolation_forest",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    contract = ScientificExecutionContract.load(output)
    assert contract.method == "isolation_forest"
    assert contract.model_seed == 42

    duplicate = CliRunner().invoke(
        cli_module.app,
        [
            "scientific-config",
            "--kind",
            "example",
            "--example",
            "isolation_forest",
            "--output",
            str(output),
        ],
    )
    assert duplicate.exit_code == 2
    assert "already exists" in duplicate.output


@pytest.mark.parametrize(
    "arguments, message",
    [
        (
            ["scientific-config", "--kind", "template"],
            "requires --workflow-family",
        ),
        (
            [
                "scientific-config",
                "--kind",
                "template",
                "--workflow-family",
                "clustering",
                "--workflow-mode",
                "clustering",
                "--method",
                "isolation_forest",
            ],
            "is not registered",
        ),
        (
            ["scientific-config", "--kind", "schema", "--method", "pca"],
            "does not accept workflow",
        ),
        (
            [
                "scientific-config",
                "--kind",
                "example",
                "--example",
                "unknown",
            ],
            "Unknown scientific-config example",
        ),
        (
            [
                "data-mining",
                "--mlflow",
                "--scientific-config",
                "scientific.json",
            ],
            "cannot be combined",
        ),
    ],
)
def test_scientific_config_cli_rejects_invalid_combinations(
    arguments,
    message,
) -> None:
    result = CliRunner().invoke(cli_module.app, arguments)

    assert result.exit_code == 2
    assert message in result.output


def test_main_help_discovers_scientific_config_generator() -> None:
    result = CliRunner().invoke(cli_module.app, ["--help"])

    assert result.exit_code == 0, result.output
    assert "scientific-config" in result.output
    detail = CliRunner().invoke(
        cli_module.app,
        ["scientific-config", "--help"],
    )
    assert detail.exit_code == 0, detail.output
    for option in (
        "--kind",
        "--workflow-family",
        "--workflow-mode",
        "--method",
        "--example",
        "--output",
    ):
        assert option in detail.output
