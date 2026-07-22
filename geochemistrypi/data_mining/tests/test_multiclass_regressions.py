# -*- coding: utf-8 -*-
import inspect
from contextlib import ExitStack
from unittest.mock import patch

import matplotlib
import numpy as np
import pandas as pd
import pytest
import typer
from click.testing import CliRunner
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from geochemistrypi.cli import app
from geochemistrypi.data_mining.data.data_readiness import create_sub_data_set, data_split
from geochemistrypi.data_mining.model.classification import ClassificationWorkflowBase, SGDClassification
from geochemistrypi.data_mining.model.func._common_supervised import plot_decision_tree
from geochemistrypi.data_mining.model.func.algo_classification._common import score
from geochemistrypi.data_mining.model.func.algo_classification._logistic_regression import plot_logistic_importance

matplotlib.use("Agg")


def test_typer_app_can_build_command() -> None:
    command = typer.main.get_command(app)
    assert command.name == "main"


def test_typer_help_renders_with_installed_click() -> None:
    command = typer.main.get_command(app)
    runner = CliRunner()
    result = runner.invoke(command, ["--help"])

    assert result.exit_code == 0, result.output
    assert "data-mining" in result.output

    subcommand_result = runner.invoke(command, ["data-mining", "--help"])
    assert subcommand_result.exit_code == 0, subcommand_result.output
    assert "--data" in subcommand_result.output


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
