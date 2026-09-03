# -*- coding: utf-8 -*-
import inspect
import json
import random
from contextlib import ExitStack
from unittest.mock import patch

import matplotlib
import mlflow
import numpy as np
import pandas as pd
import pytest
import typer
from click import unstyle
from click.testing import CliRunner
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeClassifier

from geochemistrypi.cli import app
from geochemistrypi.data_mining.data.data_readiness import create_sub_data_set, data_split
from geochemistrypi.data_mining.model._base import TreeWorkflowMixin
from geochemistrypi.data_mining.model.classification import (
    ClassificationWorkflowBase,
    ExtraTreesClassification,
    LogisticRegressionClassification,
    MLPClassification,
    SGDClassification,
    XGBoostClassification,
)
from geochemistrypi.data_mining.model.func._common_supervised import plot_decision_tree
from geochemistrypi.data_mining.model.func.algo_classification._common import cross_validation, plot_precision_recall, plot_precision_recall_threshold, plot_ROC, score
from geochemistrypi.data_mining.model.func.algo_classification._logistic_regression import plot_logistic_importance
from geochemistrypi.data_mining.model.func.algo_regression._common import display_cross_validation_scores
from geochemistrypi.data_mining.model.regression import MLPRegression, RidgeRegression
from geochemistrypi.data_mining.process import classify as classification_process
from geochemistrypi.data_mining.process.classify import ClassificationModelSelection
from geochemistrypi.scientific_execution import resolve_classification_metric_configuration

matplotlib.use("Agg")


def test_typed_positive_label_resolves_numeric_one_separately_from_string_one() -> None:
    label_config = {
        "typed_label_records": [
            {"semantic_label": {"type": "integer", "value": 1}, "encoded_label": 0},
            {"semantic_label": {"type": "string", "value": "1"}, "encoded_label": 1},
        ]
    }

    numeric = resolve_classification_metric_configuration(
        label_config,
        "binary",
        {"type": "integer", "value": 1},
    )
    textual = resolve_classification_metric_configuration(
        label_config,
        "binary",
        {"type": "string", "value": "1"},
    )

    assert numeric["aggregate_encoded_positive_label"] == 0
    assert numeric["curve_encoded_positive_label"] == 0
    assert textual["aggregate_encoded_positive_label"] == 1
    assert textual["curve_encoded_positive_label"] == 1


def test_binary_metric_consumers_use_resolved_encoded_zero_and_probability_column_zero() -> None:
    configuration = {
        "schema_version": 2,
        "requested_average": "binary",
        "effective_average": "binary",
        "requested_positive_label": {"type": "integer", "value": 10},
        "aggregate_semantic_positive_label": {"type": "integer", "value": 10},
        "aggregate_encoded_positive_label": 0,
        "curve_semantic_positive_label": {"type": "integer", "value": 10},
        "curve_encoded_positive_label": 0,
        "curve_probability_column_index": None,
        "consumers": {},
    }
    y_true = pd.DataFrame({"Label": [0, 0, 1, 1]})
    y_predict = pd.DataFrame({"Label": [0, 1, 1, 1]})

    _, observed = score(y_true, y_predict, average="binary", interactive=False, metric_configuration=configuration)

    assert observed["Precision"] == pytest.approx(1.0)
    assert observed["Recall"] == pytest.approx(0.5)
    assert configuration["consumers"]["holdout_score"]["aggregate_encoded_positive_label"] == 0

    class _ProbabilityModel:
        classes_ = np.array([0, 1])

        @staticmethod
        def predict_proba(X):
            return np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9]])

    probabilities, *_ = plot_ROC(
        pd.DataFrame({"x": [1, 2, 3, 4]}),
        y_true,
        _ProbabilityModel(),
        "ROC",
        "Test",
        metric_configuration=configuration,
    )

    assert probabilities.tolist() == pytest.approx([0.8, 0.3, 0.6, 0.1])
    assert configuration["curve_probability_column_index"] == 0
    assert configuration["consumers"]["roc"]["curve_encoded_positive_label"] == 0


def test_cross_validation_uses_resolved_binary_positive_label() -> None:
    assert mlflow.active_run() is None
    configuration = {
        "schema_version": 2,
        "requested_average": "binary",
        "effective_average": "binary",
        "requested_positive_label": {"type": "integer", "value": 10},
        "aggregate_semantic_positive_label": {"type": "integer", "value": 10},
        "aggregate_encoded_positive_label": 0,
        "curve_semantic_positive_label": {"type": "integer", "value": 10},
        "curve_encoded_positive_label": 0,
        "curve_probability_column_index": None,
        "consumers": {},
    }
    X = pd.DataFrame({"x": [-4, -3, -2, -1, 1, 2, 3, 4], "z": [0, 1, 0, 1, 0, 1, 0, 1]})
    y = pd.DataFrame({"Label": [0, 0, 0, 0, 1, 1, 1, 1]})

    observed = cross_validation(
        LogisticRegression(random_state=0),
        X,
        y,
        average="binary",
        cv_num=2,
        metric_configuration=configuration,
    )

    assert observed["Positive Label"] == 0
    assert configuration["consumers"]["cross_validation"]["aggregate_encoded_positive_label"] == 0
    assert mlflow.active_run() is None


def test_extra_trees_selection_routes_manual_values_through_constructor_contract(
    tmp_path,
    monkeypatch,
) -> None:
    class _StopBeforeFit(RuntimeError):
        pass

    constructor_call = {}
    estimator_arguments = {}

    class _ScientificExecution:
        classification_metric_average = "auto"
        classification_positive_label = None

        @staticmethod
        def constructor_parameters(
            method,
            legacy,
            *,
            workflow_family,
            workflow_mode,
            class_count=None,
        ):
            constructor_call.update(
                {
                    "method": method,
                    "legacy": dict(legacy),
                    "workflow_family": workflow_family,
                    "workflow_mode": workflow_mode,
                    "class_count": class_count,
                }
            )
            return {**legacy, "n_estimators": 321, "max_depth": 9, "random_state": 2025}

    class _ExtraTreesWorkflow:
        @staticmethod
        def manual_hyper_parameters():
            return {
                "n_estimators": 100,
                "max_depth": 4,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": 2,
                "bootstrap": True,
                "oob_score": True,
                "max_samples": 0.8,
            }

        def __init__(self, **parameters):
            estimator_arguments.update(parameters)

        @staticmethod
        def show_info():
            raise _StopBeforeFit

    monkeypatch.setattr(classification_process, "ExtraTreesClassification", _ExtraTreesWorkflow)
    monkeypatch.setenv("GEOPI_OUTPUT_ARTIFACTS_DATA_PATH", str(tmp_path / "data"))
    selection = ClassificationModelSelection(
        "Extra-Trees",
        label_config={
            "typed_label_records": [
                {"semantic_label": {"type": "string", "value": "a"}, "encoded_label": 0},
                {"semantic_label": {"type": "string", "value": "b"}, "encoded_label": 1},
            ]
        },
        labels_already_customized=True,
    )
    selection.scientific_execution = _ScientificExecution()
    X = pd.DataFrame({"x": [1.0, 2.0], "z": [3.0, 4.0]})
    y = pd.DataFrame({"Label": [0, 1]})
    names = pd.Series(["S-1", "S-2"])

    with pytest.raises(_StopBeforeFit):
        selection.activate(X, y, X, X, y, y, names, names, names)

    assert constructor_call == {
        "method": "extra_trees",
        "legacy": {
            "n_estimators": 100,
            "max_depth": 4,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": 2,
            "bootstrap": True,
            "oob_score": True,
            "max_samples": 0.8,
        },
        "workflow_family": "supervised_learning",
        "workflow_mode": "classification",
        "class_count": None,
    }
    assert estimator_arguments["n_estimators"] == 321
    assert estimator_arguments["max_depth"] == 9
    assert estimator_arguments["random_state"] == 2025


def test_extra_trees_preserves_zero_as_an_explicit_model_seed() -> None:
    workflow = ExtraTreesClassification(random_state=0)

    assert workflow.model.get_params(deep=False)["random_state"] == 0


@pytest.mark.parametrize("average", ["micro", "macro", "weighted"])
def test_non_binary_aggregate_metrics_keep_binary_curve_positive_class_separate(average: str) -> None:
    configuration = resolve_classification_metric_configuration(
        {
            "typed_label_records": [
                {"semantic_label": {"type": "string", "value": "first"}, "encoded_label": 0},
                {"semantic_label": {"type": "string", "value": "second"}, "encoded_label": 1},
            ]
        },
        requested_average=average,
    )
    y_true = pd.DataFrame({"Label": [0, 0, 1, 1]})
    y_predict = pd.DataFrame({"Label": [0, 1, 1, 1]})

    class _ProbabilityModel:
        classes_ = np.array([0, 1])

        @staticmethod
        def predict_proba(X):
            return np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9]])

    _, observed = score(
        y_true,
        y_predict,
        average=average,
        interactive=False,
        metric_configuration=configuration,
    )
    for curve in (plot_precision_recall, plot_precision_recall_threshold, plot_ROC):
        probabilities, *_ = curve(
            pd.DataFrame({"x": [1, 2, 3, 4]}),
            y_true,
            _ProbabilityModel(),
            "Curve",
            "Test",
            metric_configuration=configuration,
        )
        assert probabilities.tolist() == pytest.approx([0.2, 0.7, 0.4, 0.9])

    assert observed["Average Method"] == average
    assert observed["Positive Label"] is None
    assert configuration["aggregate_encoded_positive_label"] is None
    assert configuration["curve_encoded_positive_label"] == 1
    assert configuration["curve_probability_column_index"] == 1
    assert configuration["consumers"]["holdout_score"] == {
        "consumer_kind": "aggregate_metric",
        "effective_average": average,
        "aggregate_encoded_positive_label": None,
    }
    assert {configuration["consumers"][name]["consumer_kind"] for name in ("precision_recall", "precision_recall_threshold", "roc")} == {"binary_curve"}


def test_typer_app_can_build_command() -> None:
    command = typer.main.get_command(app)
    assert command.name == "main"


def test_typer_help_renders_with_installed_click() -> None:
    command = typer.main.get_command(app)
    runner = CliRunner()
    result = runner.invoke(command, ["--help"])

    assert result.exit_code == 0, result.output
    output = unstyle(result.output)
    assert "data-mining" in output
    assert "datasets" in output

    subcommand_result = runner.invoke(command, ["data-mining", "--help"])
    assert subcommand_result.exit_code == 0, subcommand_result.output
    subcommand_output = unstyle(subcommand_result.output)
    assert "--data" in subcommand_output
    assert "--automation-plan" in subcommand_output


@pytest.mark.parametrize(
    "arguments",
    [
        ["data-mining", "--data", "a.csv", "--training", "b.csv"],
        ["data-mining", "--application", "application.csv"],
        ["data-mining", "--automation-plan", "plan.json"],
        ["data-mining", "--mlflow", "--data", "a.csv"],
    ],
)
def test_cli_rejects_ambiguous_sources_and_incomplete_automation(arguments) -> None:
    result = CliRunner().invoke(typer.main.get_command(app), arguments)

    assert result.exit_code != 0
    assert "Invalid value" in result.output


def test_cli_version_does_not_hijack_data_mining_options() -> None:
    command = typer.main.get_command(app)
    runner = CliRunner()

    version_result = runner.invoke(command, ["--version"])
    assert version_result.exit_code == 0, version_result.output
    assert "Geochemistry" in version_result.output

    with patch("geochemistrypi.cli._run_cli_pipeline") as pipeline:
        result = runner.invoke(command, ["data-mining", "--data", "train.csv"])

    assert result.exit_code == 0, result.output
    pipeline.assert_called_once()
    assert pipeline.call_args.kwargs["training_data_path"] == "train.csv"


def test_cli_help_and_version_do_not_import_pipeline_output() -> None:
    command = typer.main.get_command(app)
    runner = CliRunner()

    for args in (["--help"], ["--version"]):
        result = runner.invoke(command, args)
        assert result.exit_code == 0, result.output
        assert ">>>" not in result.output
        assert "pkg_resources" not in result.output


def test_cli_startup_text_is_ascii_safe_for_windows_console() -> None:
    import geochemistrypi.cli as cli_module
    from geochemistrypi.data_mining import cli_pipeline

    for source in (inspect.getsource(cli_module), inspect.getsource(cli_pipeline.cli_pipeline)):
        assert "\u2728" not in source
        assert "\u03c0" not in source


def test_classification_module_has_no_import_debug_print() -> None:
    import geochemistrypi.data_mining.model.classification as classification

    source = inspect.getsource(classification)
    assert 'print(">>>' not in source


def test_create_sub_data_set_can_select_string_target_when_numeric_not_required() -> None:
    data = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0],
            "rock_type": ["basalt", "andesite", "dacite"],
        }
    )
    with patch("builtins.input", return_value="2"):
        selected = create_sub_data_set(data, allow_empty_columns=False, require_numeric=False)

    assert selected["rock_type"].tolist() == ["basalt", "andesite", "dacite"]


def test_customize_label_returns_encoded_train_test_without_mapping() -> None:
    X = pd.DataFrame({"feature": range(6)})
    y = pd.DataFrame({"Target": [10, 20, 30, 10, 20, 30]})
    names = pd.Series([f"sample-{idx}" for idx in range(6)], name="Name")
    split = data_split(X, y, names, test_size=0.5, stratify=y["Target"])

    result = ClassificationWorkflowBase.customize_label(
        y,
        split["Y Train"],
        split["Y Test"],
        interactive=False,
        return_config=True,
    )

    y_encoded, y_train_encoded, y_test_encoded, config = result
    assert sorted(y_encoded["Target"].unique().tolist()) == [0, 1, 2]
    assert sorted(y_train_encoded["Target"].unique().tolist()) == [0, 1, 2]
    assert sorted(y_test_encoded["Target"].unique().tolist()) == [0, 1, 2]
    assert config["code_to_custom_label"] == {"0": "10", "1": "20", "2": "30"}


def test_customize_label_preserves_typed_numeric_and_string_label_identities() -> None:
    y = pd.DataFrame({"Target": pd.Series([1, "1", 1, "1"], dtype=object)})

    y_encoded, config = ClassificationWorkflowBase.customize_label(
        y,
        interactive=False,
        return_config=True,
    )

    assert y_encoded["Target"].tolist() == [0, 1, 0, 1]
    assert config["typed_label_records"] == [
        {"semantic_label": {"type": "integer", "value": 1}, "encoded_label": 0, "count": 2},
        {"semantic_label": {"type": "string", "value": "1"}, "encoded_label": 1, "count": 2},
    ]


def test_interval_label_encoding_follows_user_label_order() -> None:
    y = pd.DataFrame({"Target": [52, 40, 60]})
    y_encoded, config = ClassificationWorkflowBase.customize_label(
        y,
        label_mapping={
            "type": "interval",
            "bins": [-float("inf"), 50, 55, float("inf")],
            "labels": ["low", "middle", "high"],
        },
        interactive=False,
        return_config=True,
    )

    assert y_encoded["Target"].tolist() == [1, 0, 2]
    assert config["custom_label_to_code"] == {"low": 0, "middle": 1, "high": 2}


def test_string_label_mapping_follows_user_label_order() -> None:
    y = pd.DataFrame({"rock_type": ["basalt", "dacite", "andesite", "basalt"]})
    y_encoded, config = ClassificationWorkflowBase.customize_label(
        y,
        label_mapping={
            "type": "dict",
            "mapping": {
                "basalt": "mafic",
                "andesite": "intermediate",
                "dacite": "felsic",
            },
        },
        interactive=False,
        return_config=True,
    )

    assert y_encoded["rock_type"].tolist() == [0, 2, 1, 0]
    assert config["custom_label_to_code"] == {"mafic": 0, "intermediate": 1, "felsic": 2}


@pytest.mark.parametrize("num_classes", [4, 5, 6])
def test_customize_label_supports_user_defined_many_class_dict_mapping(num_classes: int) -> None:
    raw_labels = [f"raw-{idx}" for idx in range(num_classes)]
    custom_labels = [f"class-{idx}" for idx in range(num_classes)]
    y = pd.DataFrame({"Target": raw_labels * 2})

    y_encoded, config = ClassificationWorkflowBase.customize_label(
        y,
        label_mapping={
            "type": "dict",
            "mapping": dict(zip(raw_labels, custom_labels)),
        },
        interactive=False,
        return_config=True,
    )

    assert sorted(y_encoded["Target"].unique().tolist()) == list(range(num_classes))
    assert config["num_classes"] == num_classes
    assert config["custom_label_to_code"] == {label: idx for idx, label in enumerate(custom_labels)}


@pytest.mark.parametrize("num_classes", [4, 5, 6])
def test_customize_label_supports_user_defined_many_class_interval_mapping(num_classes: int) -> None:
    y = pd.DataFrame({"Target": np.arange(num_classes * 3)})
    labels = [f"class-{idx}" for idx in range(num_classes)]
    bins = [-float("inf")] + [float(idx * 3) for idx in range(1, num_classes)] + [float("inf")]

    y_encoded, config = ClassificationWorkflowBase.customize_label(
        y,
        label_mapping={"type": "interval", "bins": bins, "labels": labels},
        interactive=False,
        return_config=True,
    )

    assert sorted(y_encoded["Target"].unique().tolist()) == list(range(num_classes))
    assert config["num_classes"] == num_classes
    assert config["custom_label_to_code"] == {label: idx for idx, label in enumerate(labels)}


@pytest.mark.parametrize("num_classes", [4, 5, 6])
def test_customize_label_supports_user_defined_many_class_quantile_mapping(num_classes: int) -> None:
    y = pd.DataFrame({"Target": np.arange(num_classes * 4)})
    labels = [f"class-{idx}" for idx in range(num_classes)]

    y_encoded, config = ClassificationWorkflowBase.customize_label(
        y,
        label_mapping={"type": "quantile", "num_classes": num_classes, "labels": labels},
        interactive=False,
        return_config=True,
    )

    assert sorted(y_encoded["Target"].unique().tolist()) == list(range(num_classes))
    assert config["num_classes"] == num_classes
    assert config["custom_label_to_code"] == {label: idx for idx, label in enumerate(labels)}


@pytest.mark.parametrize("num_classes", [4, 5, 6])
def test_multiclass_score_defaults_to_weighted_and_rejects_binary_average(num_classes: int) -> None:
    y_true = pd.DataFrame({"Target": list(range(num_classes)) * 2})
    y_predict = y_true.copy()

    average, scores = score(y_true, y_predict, average=None, interactive=False)

    assert average == "weighted"
    assert scores["Average Method"] == "weighted"
    with pytest.raises(ValueError, match="Binary average cannot be used"):
        score(y_true, y_predict, average="binary", interactive=False)


def test_logistic_importance_returns_one_row_per_class_and_feature_for_multiclass() -> None:
    X_array, y_array = make_classification(
        n_samples=80,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_classes=4,
        random_state=42,
    )
    X = pd.DataFrame(X_array, columns=[f"feature_{idx}" for idx in range(4)])
    model = LogisticRegression(max_iter=300, multi_class="ovr").fit(X, y_array)

    coefficients = plot_logistic_importance(X.columns, model)

    assert len(coefficients) == len(model.classes_) * X.shape[1]
    assert {"class_label", "var", "coef", "abs_coef"}.issubset(coefficients.columns)
    assert set(coefficients["class_label"]) == set(model.classes_)
    assert set(coefficients["var"]) == set(X.columns)


def test_binary_plot_gate_uses_global_class_count_not_test_subset_only() -> None:
    ClassificationWorkflowBase.y = pd.DataFrame({"Target": [0, 1, 2, 3]})
    ClassificationWorkflowBase.y_train = pd.DataFrame({"Target": [0, 1, 2, 3]})
    ClassificationWorkflowBase.y_test = pd.DataFrame({"Target": [0, 3]})

    assert ClassificationWorkflowBase._get_total_class_count({"num_classes": 4}) == 4
    assert ClassificationWorkflowBase._get_total_class_count({}) == 4


def test_confusion_matrix_output_handles_multiclass_predictions_when_test_subset_is_missing_classes() -> None:
    class ModelWithFourClasses:
        classes_ = np.array([0, 1, 2, 3])

    captured = {}

    def capture_data(df, *args, **kwargs):
        captured["shape"] = df.shape
        captured["columns"] = list(df.columns)

    with patch("geochemistrypi.data_mining.model.classification.save_fig"), patch("geochemistrypi.data_mining.model.classification.save_data", side_effect=capture_data):
        ClassificationWorkflowBase._plot_confusion_matrix(
            y_test=pd.DataFrame({"Target": [0, 3]}),
            y_test_predict=pd.DataFrame({"Target": [0, 1]}),
            name_column="Sample ID",
            trained_model=ModelWithFourClasses(),
            graph_name="Confusion Matrix",
            algorithm_name="Test Model",
            local_path=".",
            mlflow_path=None,
        )

    assert captured["shape"] == (4, 4)
    assert captured["columns"] == ["pred_0", "pred_1", "pred_2", "pred_3"]


def test_sgd_manual_special_components_accepts_multiclass_coefficients() -> None:
    X = pd.DataFrame(
        {
            "feature_0": [0.0, 0.2, 0.4, 1.0, 1.2, 1.4, 2.0, 2.2, 2.4, 3.0, 3.2, 3.4],
            "feature_1": [0.0, 0.1, 0.3, 1.1, 1.2, 1.5, 2.1, 2.3, 2.5, 3.1, 3.3, 3.5],
        }
    )
    y = pd.DataFrame({"Target": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]})
    workflow = SGDClassification(loss="log_loss", max_iter=1000, tol=1e-3, random_state=42)
    workflow.model = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=42).fit(X, y["Target"])
    workflow.data_upload(X_train=X, y=y)

    with patch("geochemistrypi.data_mining.model._base.save_text"):
        workflow.special_components()


def test_xgboost_classification_automl_fit_supports_the_three_argument_dispatch() -> None:
    captured = {}

    class AutoMLDouble:
        def fit(self, **kwargs) -> None:
            captured.update(kwargs)

    X = pd.DataFrame({"feature_0": [0.0, 1.0, 2.0, 3.0], "feature_1": [3.0, 2.0, 1.0, 0.0]})
    y = pd.DataFrame({"Target": [0, 0, 1, 1]})
    workflow = XGBoostClassification(n_estimators=2, max_depth=1)

    with patch("geochemistrypi.data_mining.model.classification.AutoML", return_value=AutoMLDouble()):
        workflow.fit(X, y, True)

    assert captured["X_train"].equals(X)
    assert captured["y_train"].equals(y["Target"])
    assert captured["estimator_list"] == ["xgboost"]
    assert captured["task"] == "classification"
    assert captured["max_iter"] == workflow.automl_max_iterations
    assert captured["seed"] == workflow.random_state
    assert "time_budget" not in captured


def test_automl_settings_use_a_repeatable_trial_budget_and_random_state() -> None:
    workflow = XGBoostClassification(n_estimators=2, max_depth=1)
    workflow.random_state = (42,)
    original = {"time_budget": 10, "metric": "accuracy"}

    first_settings = workflow._prepare_automl_settings(original)
    first_random = random.random()
    first_numpy_random = np.random.random()
    second_settings = workflow._prepare_automl_settings(original)

    assert original == {"time_budget": 10, "metric": "accuracy"}
    assert (
        first_settings
        == second_settings
        == {
            "metric": "accuracy",
            "max_iter": workflow.automl_max_iterations,
            "seed": 42,
        }
    )
    assert random.random() == first_random
    assert np.random.random() == first_numpy_random

    workflow.random_state = None
    assert workflow._prepare_automl_settings(original)["seed"] == workflow.default_random_state


def test_logistic_automl_uses_a_unix_safe_compatibility_budget() -> None:
    workflow = LogisticRegressionClassification()

    prepared = workflow._prepare_automl_settings(workflow.settings)

    assert prepared["estimator_list"] == ["lrl2"]
    assert prepared["time_budget"] == workflow.automl_compatibility_time_budget_seconds
    assert 0 < prepared["time_budget"] <= 2_147_483_647


def test_sgd_automl_validation_fraction_search_space_is_always_valid() -> None:
    workflow = SGDClassification()

    search_space = workflow.customization().search_space((100, 4), "classification")
    validation_fraction = search_space["validation_fraction"]

    assert validation_fraction["init_value"] == 0.1
    assert 0 < validation_fraction["domain"].lower
    assert validation_fraction["domain"].upper < 1


def test_regression_cross_validation_scores_remain_numeric_in_json_output() -> None:
    scores = np.array([1.0, 2.0, 3.0])

    with patch("geochemistrypi.data_mining.model.func.algo_regression._common.mlflow.log_metric"):
        result = display_cross_validation_scores(scores, "Example")

    assert result["Fold Scores"] == [1.0, 2.0, 3.0]
    assert all(isinstance(value, float) for value in result["Fold Scores"])


@pytest.mark.parametrize("workflow", [MLPClassification(), MLPRegression()])
def test_mlp_automl_uses_seeded_fixed_in_process_trials(workflow) -> None:
    first = workflow._automl_mlp_configurations()
    second = workflow._automl_mlp_configurations()

    assert first == second
    assert len(first) == workflow.automl_tuning_trials
    assert all(1 <= config["l1"] < 20 for config in first)
    assert all(1 <= config["l2"] < 30 for config in first)
    assert all(1 <= config["l3"] < 20 for config in first)
    assert all(20 <= config["batch"] < 100 for config in first)
    assert "from ray" not in inspect.getsource(workflow.ray_tune)


@pytest.mark.parametrize(
    ("model", "target"),
    [
        (GradientBoostingClassifier(n_estimators=3, random_state=42), pd.Series([0, 0, 0, 1, 1, 1])),
        (GradientBoostingRegressor(n_estimators=3, random_state=42), pd.Series([0.0, 0.2, 0.4, 1.0, 1.2, 1.4])),
    ],
)
def test_tree_feature_importance_prefers_an_ensemble_top_level_vector(model, target: pd.Series) -> None:
    X = pd.DataFrame(
        {
            "feature_0": [0.0, 0.2, 0.4, 1.0, 1.2, 1.4],
            "feature_1": [1.4, 1.2, 1.0, 0.4, 0.2, 0.0],
        }
    )
    trained_model = model.fit(X, target)
    captured = {}

    def capture_feature_importance(columns, feature_importances, image_config):
        captured["columns"] = tuple(columns)
        captured["feature_importances"] = np.asarray(feature_importances)
        return pd.DataFrame({"feature": columns, "importance": feature_importances})

    with patch("geochemistrypi.data_mining.model._base.plot_feature_importance", side_effect=capture_feature_importance), patch("geochemistrypi.data_mining.model._base.save_fig"), patch(
        "geochemistrypi.data_mining.model._base.save_data"
    ):
        TreeWorkflowMixin._plot_feature_importance(X, "SampleID", trained_model, {}, "Gradient Boosting", "Feature Importance", ".", None)

    assert captured["columns"] == tuple(X.columns)
    np.testing.assert_allclose(captured["feature_importances"], trained_model.feature_importances_)


def test_tree_feature_importance_preserves_multioutput_regressor_support() -> None:
    X = pd.DataFrame(
        {
            "feature_0": [0.0, 0.2, 0.4, 1.0, 1.2, 1.4],
            "feature_1": [1.4, 1.2, 1.0, 0.4, 0.2, 0.0],
        }
    )
    y = pd.DataFrame({"target_0": [0.0, 0.2, 0.4, 1.0, 1.2, 1.4], "target_1": [1.4, 1.2, 1.0, 0.4, 0.2, 0.0]})
    trained_model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=3, random_state=42)).fit(X, y)
    captured = {}

    def capture_feature_importance(columns, feature_importances, image_config):
        captured["feature_importances"] = np.asarray(feature_importances)
        return pd.DataFrame({"feature": columns, "importance": feature_importances})

    with patch("geochemistrypi.data_mining.model._base.plot_feature_importance", side_effect=capture_feature_importance), patch("geochemistrypi.data_mining.model._base.save_fig"), patch(
        "geochemistrypi.data_mining.model._base.save_data"
    ):
        TreeWorkflowMixin._plot_feature_importance(X, "SampleID", trained_model, {}, "Gradient Boosting", "Feature Importance", ".", None)

    expected = np.mean([estimator.feature_importances_ for estimator in trained_model.estimators_], axis=0)
    np.testing.assert_allclose(captured["feature_importances"], expected)


def test_ridge_manual_formula_accepts_single_target_dataframe_coefficients() -> None:
    X = pd.DataFrame({"feature": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]})
    y = pd.DataFrame({"Target": [1.0, 1.4, 2.1, 2.6, 3.2, 3.7]})
    names = pd.Series([f"sample-{index}" for index in range(len(X))], name="SampleID")
    workflow = RidgeRegression()
    workflow.model.fit(X, y)
    assert workflow.model.coef_.ndim == 2
    workflow.data_upload(
        X=X,
        y=y,
        X_train=X,
        X_test=X,
        y_train=y,
        y_test=y,
        y_test_predict=y,
        name_train=names,
        name_test=names,
        name_all=names,
    )
    captured = {}

    def capture_text(value, name, *args, **kwargs) -> None:
        captured[name] = json.loads(value)

    with patch("geochemistrypi.data_mining.model._base.save_text", side_effect=capture_text), patch.object(workflow, "_plot_2d_scatter_diagram"), patch.object(workflow, "_plot_2d_line_diagram"):
        workflow.special_components()

    assert "feature" in captured["Ridge Regression Formula"]["y:"]


def test_process_classify_imports_on_python_38_compatible_annotations() -> None:
    from geochemistrypi.data_mining.process.classify import ClassificationModelSelection

    selector = ClassificationModelSelection("Decision Tree", metric_average="macro")

    assert selector.metric_average == "macro"


def test_classification_request_accepts_metric_average() -> None:
    from geochemistrypi.data_mining.schemas import ClassificationRunRequest

    request = ClassificationRunRequest(dataset_id=1, target_column="Target", model_name="Decision Tree", metric_average="macro")

    assert request.metric_average == "macro"


def test_api_split_stratifies_when_each_class_has_enough_samples() -> None:
    y = pd.DataFrame({"Target": list(range(4)) * 5})
    X = pd.DataFrame({"feature": range(len(y))})
    stratify_target = y["Target"] if y["Target"].value_counts().min() >= 2 else None

    _, _, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_target)

    assert sorted(y_train["Target"].unique().tolist()) == [0, 1, 2, 3]
    assert sorted(y_test["Target"].unique().tolist()) == [0, 1, 2, 3]


def test_data_split_expands_stratified_test_size_to_cover_all_classes() -> None:
    y = pd.DataFrame({"Target": list(range(6)) * 2})
    X = pd.DataFrame({"feature": range(len(y))})
    names = pd.Series([f"sample-{idx}" for idx in range(len(y))], name="Sample ID")

    split = data_split(X, y, names, test_size=0.2, stratify=y["Target"])

    assert sorted(split["Y Train"]["Target"].unique().tolist()) == [0, 1, 2, 3, 4, 5]
    assert sorted(split["Y Test"]["Target"].unique().tolist()) == [0, 1, 2, 3, 4, 5]


def test_api_split_stratifies_on_final_user_defined_quantile_classes() -> None:
    from geochemistrypi.data_mining.router import _build_classification_split_parameters

    y = pd.DataFrame({"Target": np.arange(12)})
    label_mapping = {"type": "quantile", "num_classes": 6, "labels": [f"class-{idx}" for idx in range(6)]}

    split_params = _build_classification_split_parameters(y, label_mapping=label_mapping, default_test_ratio=0.2)

    assert split_params["test_size"] == 6
    assert split_params["class_count"] == 6
    assert sorted(split_params["stratify_target"].unique().tolist()) == [0, 1, 2, 3, 4, 5]


def test_api_metric_average_defaults_to_weighted_for_multiclass_automatic_runs() -> None:
    from geochemistrypi.data_mining.router import _resolve_api_metric_average

    assert _resolve_api_metric_average(None, class_count=4) == "weighted"
    assert _resolve_api_metric_average("macro", class_count=4) == "macro"
    assert _resolve_api_metric_average(None, class_count=2) is None


def test_dash_payload_uses_user_defined_multiclass_count() -> None:
    from geochemistrypi.data_mining.dash_pipeline import _build_classification_payload

    payload = _build_classification_payload(target_col="Target", model_name="Random Forest", num_classes=6)

    assert payload["label_mapping"]["num_classes"] == 6
    assert payload["label_mapping"]["labels"] == ["Level_1", "Level_2", "Level_3", "Level_4", "Level_5", "Level_6"]
    assert payload["metric_average"] == "weighted"
    with pytest.raises(ValueError, match="at least 2"):
        _build_classification_payload(target_col="Target", model_name="Random Forest", num_classes=1)


def test_cli_dependency_check_does_not_install_map_packages_at_runtime() -> None:
    from geochemistrypi.data_mining import cli_pipeline as cli_module
    from geochemistrypi.data_mining.enum_ import DataSource

    class StopAfterDependencyCheck(Exception):
        pass

    def stop_after_dependency_check(*args, **kwargs):
        raise StopAfterDependencyCheck

    with ExitStack() as stack:
        stack.enter_context(patch.object(cli_module, "sleep"))
        stack.enter_context(patch.object(cli_module, "get_os", return_value="Windows"))
        stack.enter_context(patch.object(cli_module, "check_package", return_value=False, create=True))
        stack.enter_context(patch.object(cli_module, "install_package", side_effect=AssertionError("runtime pip install should not be called"), create=True))
        stack.enter_context(patch.object(cli_module.Confirm, "ask", side_effect=stop_after_dependency_check))
        with pytest.raises(StopAfterDependencyCheck):
            cli_module.cli_pipeline("", "", DataSource.ANY_PATH)
    assert cli_module.mlflow.active_run() is None


def test_plot_decision_tree_accepts_default_none_node_ids() -> None:
    X = pd.DataFrame({"feature": [0, 1, 2, 3]})
    y = pd.Series([0, 0, 1, 1])
    model = DecisionTreeClassifier(random_state=42).fit(X, y)
    image_config = {
        "width": 4,
        "height": 3,
        "dpi": 80,
        "max_depth": None,
        "feature_names": ["feature"],
        "class_names": ["class0", "class1"],
        "label": "all",
        "filled": True,
        "impurity": True,
        "node_ids": None,
        "proportion": False,
        "rounded": True,
        "precision": 3,
        "ax": None,
        "fontsize": None,
        "axislabelfont": "Arial",
        "title_label": "Decision Tree",
        "title_size": 10,
        "title_color": "k",
        "title_location": "center",
        "title_font": "Arial",
        "title_pad": 2,
    }

    plot_decision_tree(model, image_config)
