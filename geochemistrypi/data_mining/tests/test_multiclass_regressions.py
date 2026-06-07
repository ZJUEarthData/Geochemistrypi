# -*- coding: utf-8 -*-
import inspect
from unittest.mock import patch

import matplotlib
import pandas as pd
import typer
from click.testing import CliRunner
from sklearn.tree import DecisionTreeClassifier

from geochemistrypi.cli import app
from geochemistrypi.data_mining.data.data_readiness import create_sub_data_set, data_split
from geochemistrypi.data_mining.model.func._common_supervised import plot_decision_tree


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
    from geochemistrypi.data_mining.model.classification import ClassificationWorkflowBase

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
    from geochemistrypi.data_mining.model.classification import ClassificationWorkflowBase

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
    from geochemistrypi.data_mining.model.classification import ClassificationWorkflowBase

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
