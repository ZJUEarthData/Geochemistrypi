from unittest.mock import patch

import matplotlib
import pandas as pd

from geochemistrypi.data_mining.model.func.algo_regression._common import plot_predicted_actual_density
from geochemistrypi.data_mining.model.regression import RegressionWorkflowBase

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def _density_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_train = pd.DataFrame({"P": [0.0, 0.8, 1.7, 2.6, 3.8]})
    y_train_predict = pd.DataFrame({"P": [0.1, 0.7, 1.9, 2.4, 3.7]})
    y_test = pd.DataFrame({"P": [0.2, 1.1, 2.0, 3.1, 4.2]})
    y_test_predict = pd.DataFrame({"P": [0.3, 0.9, 2.2, 2.8, 4.0]})
    return y_train_predict, y_train, y_test_predict, y_test


def test_density_colorbars_are_outside_the_training_and_testing_panels() -> None:
    plot_predicted_actual_density(*_density_frames(), algorithm_name="XGBoost")

    figure = plt.gcf()
    figure.canvas.draw()
    main_axes = {axis.get_title(): axis for axis in figure.axes if axis.get_title()}
    colorbar_axes = sorted(
        (axis for axis in figure.axes if not axis.get_title()),
        key=lambda axis: axis.get_position().x0,
    )

    assert set(main_axes) == {"Training", "Testing"}
    assert len(colorbar_axes) == 2
    training_bounds = main_axes["Training"].get_position()
    testing_bounds = main_axes["Testing"].get_position()
    red_colorbar_bounds = colorbar_axes[0].get_position()
    blue_colorbar_bounds = colorbar_axes[1].get_position()
    assert training_bounds.x1 < red_colorbar_bounds.x0
    assert red_colorbar_bounds.x1 < testing_bounds.x0
    assert testing_bounds.x1 < blue_colorbar_bounds.x0

    plt.close(figure)


def test_density_workflow_preserves_the_figures_constrained_layout() -> None:
    with patch("geochemistrypi.data_mining.model.regression.plot_predicted_actual_density"):
        with patch("geochemistrypi.data_mining.model.regression.save_fig") as save_figure:
            RegressionWorkflowBase._plot_predicted_actual_density(
                *_density_frames(),
                algorithm_name="XGBoost",
                local_path="local",
                mlflow_path="mlflow",
                graph_name="Density",
            )

    save_figure.assert_called_once_with(
        "Density - XGBoost",
        "local",
        "mlflow",
        tight_layout=False,
    )
