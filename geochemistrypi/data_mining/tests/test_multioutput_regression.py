from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor

from geochemistrypi.data_mining.data.inference import model_inference
from geochemistrypi.data_mining.model._base import LinearWorkflowMixin, TreeWorkflowMixin
from geochemistrypi.data_mining.model.func.algo_regression._common import plot_residuals, score
from geochemistrypi.data_mining.model.regression import (
    BayesianRidgeRegression,
    ClassicalLinearRegression,
    DecisionTreeRegression,
    ElasticNetRegression,
    ExtraTreesRegression,
    GradientBoostingRegression,
    KNNRegression,
    LassoRegression,
    MLPRegression,
    PolynomialRegression,
    RandomForestRegression,
    RidgeRegression,
    SGDRegression,
    SVMRegression,
    XGBoostRegression,
)


def _multi_target_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    feature = pd.DataFrame(
        {
            "SIO2": np.linspace(48.0, 68.0, 20),
            "TIO2": np.linspace(0.4, 2.3, 20),
        }
    )
    target = pd.DataFrame(
        {
            "Target": feature["SIO2"] * 0.7 + feature["TIO2"] * 1.5,
            "TargetB": feature["SIO2"] * -0.2 + feature["TIO2"] * 4.0,
        }
    )
    return feature, target


@pytest.mark.parametrize(
    "model_factory",
    [
        PolynomialRegression,
        lambda: XGBoostRegression(n_estimators=2, max_depth=2, n_jobs=1, verbosity=0),
        lambda: DecisionTreeRegression(max_depth=2),
        lambda: ExtraTreesRegression(n_estimators=2, max_depth=2, n_jobs=1),
        lambda: RandomForestRegression(n_estimators=2, max_depth=2, n_jobs=1),
        SVMRegression,
        lambda: MLPRegression(hidden_layer_sizes=(4,), max_iter=20),
        ClassicalLinearRegression,
        lambda: KNNRegression(n_neighbors=3),
        lambda: GradientBoostingRegression(n_estimators=2, max_depth=2),
        LassoRegression,
        ElasticNetRegression,
        lambda: SGDRegression(max_iter=20),
        BayesianRidgeRegression,
        RidgeRegression,
    ],
    ids=[
        "polynomial",
        "xgboost",
        "decision-tree",
        "extra-trees",
        "random-forest",
        "svm",
        "mlp",
        "linear",
        "knn",
        "gradient-boosting",
        "lasso",
        "elastic-net",
        "sgd",
        "bayesian-ridge",
        "ridge",
    ],
)
def test_every_public_manual_regression_family_fits_multiple_targets(model_factory) -> None:
    feature, target = _multi_target_data()
    workflow = model_factory()

    workflow.fit(feature, target)
    predicted = workflow.predict(feature.iloc[:3])

    assert isinstance(workflow.model, MultiOutputRegressor)
    assert predicted.shape == (3, 2)


def test_multi_target_scores_keep_legacy_aggregate_and_add_named_per_target_metrics() -> None:
    _, actual = _multi_target_data()
    predicted = actual.copy()
    predicted["Target"] += 1.0
    predicted["TargetB"] -= 2.0

    metrics = score(actual, predicted)

    assert set(metrics) == {
        "Root Mean Square Error",
        "Mean Absolute Error",
        "R2 Score",
        "Explained Variance Score",
        "Per Target",
    }
    assert set(metrics["Per Target"]) == {"Target", "TargetB"}
    assert metrics["Per Target"]["Target"]["Root Mean Square Error"] == pytest.approx(1.0)
    assert metrics["Per Target"]["TargetB"]["Root Mean Square Error"] == pytest.approx(2.0)


def test_multi_target_residuals_are_actual_minus_predicted_and_named() -> None:
    _, actual = _multi_target_data()
    predicted = actual.copy()
    predicted["Target"] += 1.0
    predicted["TargetB"] -= 2.0

    residuals = plot_residuals(predicted, actual, "Multi-output")

    assert tuple(residuals.columns) == ("Residuals_Target", "Residuals_TargetB")
    np.testing.assert_allclose(residuals["Residuals_Target"], -1.0)
    np.testing.assert_allclose(residuals["Residuals_TargetB"], 2.0)


@pytest.mark.parametrize(
    "base_estimator",
    [
        DecisionTreeRegressor(max_depth=2, random_state=42),
        RandomForestRegressor(n_estimators=2, max_depth=2, random_state=42),
        ExtraTreesRegressor(n_estimators=2, max_depth=2, random_state=42),
        GradientBoostingRegressor(n_estimators=2, max_depth=2, random_state=42),
    ],
)
def test_multi_target_tree_output_uses_one_real_representative_tree_per_target(base_estimator) -> None:
    feature, target = _multi_target_data()
    model = MultiOutputRegressor(base_estimator).fit(feature, target)

    with patch("geochemistrypi.data_mining.model._base.plot_decision_tree") as plot_tree, patch("geochemistrypi.data_mining.model._base.save_fig") as save_figure:
        TreeWorkflowMixin._plot_tree(model, {}, "Tree Model", "Single Tree Diagram", ".", None)

    assert plot_tree.call_count == 2
    assert all(hasattr(call.args[0], "tree_") for call in plot_tree.call_args_list)
    assert [call.args[0] for call in save_figure.call_args_list] == [
        "Single Tree Diagram - Output 1 - Tree Model",
        "Single Tree Diagram - Output 2 - Tree Model",
    ]


def test_linear_plot_data_names_each_multi_target_prediction_column() -> None:
    feature, target = _multi_target_data()
    feature = feature[["SIO2"]]
    predicted = target.copy()

    with patch("geochemistrypi.data_mining.model._base.plot_2d_line_diagram"), patch("geochemistrypi.data_mining.model._base.save_fig"), patch(
        "geochemistrypi.data_mining.model._base.save_data"
    ) as save_data:
        LinearWorkflowMixin._plot_2d_line_diagram(
            feature,
            target,
            predicted,
            "SampleID",
            "Linear Regression",
            "2D Line Diagram",
            ".",
            None,
        )

    saved = save_data.call_args.args[0]
    assert tuple(saved.columns) == (
        "SIO2",
        "Target",
        "TargetB",
        "Predicted_Target",
        "Predicted_TargetB",
    )


def test_model_inference_preserves_scientist_target_names_and_rejects_width_mismatch() -> None:
    feature, _ = _multi_target_data()
    application = feature.iloc[:3].copy()
    loaded_model = SimpleNamespace(
        predict=lambda _: np.asarray(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
            ]
        )
    )
    run = SimpleNamespace(model_name="Linear Regression")
    active_run = SimpleNamespace(info=SimpleNamespace(run_id="run-123"))

    with patch("geochemistrypi.data_mining.data.inference.mlflow.active_run", return_value=active_run), patch(
        "geochemistrypi.data_mining.data.inference.mlflow.sklearn.load_model",
        return_value=loaded_model,
    ), patch("geochemistrypi.data_mining.data.inference.save_data") as save_data:
        model_inference(
            application,
            "SampleID",
            True,
            run,
            {},
            y_columns=["Target", "TargetB"],
        )

        saved = save_data.call_args.args[0]
        assert tuple(saved.columns) == ("Predicted_Target", "Predicted_TargetB")

        with pytest.raises(ValueError, match="1 names for 2 outputs"):
            model_inference(
                application,
                "SampleID",
                True,
                run,
                {},
                y_columns=["Target"],
            )
