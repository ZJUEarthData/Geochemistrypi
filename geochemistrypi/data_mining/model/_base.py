# -*- coding: utf-8 -*-
import json
import os
import random
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
from multipledispatch import dispatch
from rich import print

from ...scientific_execution import _json_safe
from ..constants import SECTION
from ..data.data_readiness import limit_num_input, num2option, num_input, show_data_columns
from ..utils.base import save_data, save_fig, save_model, save_text
from .func._common_supervised import plot_decision_tree, plot_feature_importance, plot_permutation_importance, show_formula
from .func.algo_regression._linear_regression import plot_2d_line_diagram, plot_2d_scatter_diagram, plot_3d_scatter_diagram, plot_3d_surface_diagram


class WorkflowBase(metaclass=ABCMeta):
    """Base class for all workflow classes in geochemistry π."""

    # Default for child class. They need to be overwritten in child classes.
    name = None
    common_function = []
    special_function = []
    X, y = None, None
    X_train, X_test, y_train, y_test = None, None, None, None
    y_test_predict = None
    default_random_state = 42
    automl_max_iterations = 20
    automl_tuning_trials = 8
    # FLAML 1.0.14 passes an unbounded L2-logistic budget to signal.alarm on
    # Unix, where it overflows the C integer accepted by the system call. The
    # fixed max_iter above remains the real search bound; this finite value is
    # deliberately high enough not to become a wall-clock stopping condition.
    automl_compatibility_time_budget_seconds = 86_400

    @classmethod
    def show_info(cls) -> None:
        """Display what application functions the algorithm will provide."""
        print(f"[bold green]-*-*- {cls.name} Training Process -*-*-[/bold green]")
        print("[bold green]Expected Functionality:[/bold green]")
        function = cls.common_function + cls.special_function
        for i in range(len(function)):
            print(f"[bold green]+ {function[i]}[/bold green]")

    def __init__(self) -> None:
        # Default for child class. They need to be overwritten in child classes.
        self.model = None
        self.naming = None
        self.automl = None
        self.ray_best_model = None
        # Set the random state fixed value for reproducibility of the results.
        self.random_state = self.default_random_state

    def _prepare_automl_settings(self, settings: Dict) -> Dict:
        """Make AutoML searches repeatable across independent CLI processes."""
        random_seed = self._automl_random_seed()
        random.seed(random_seed)
        np.random.seed(random_seed)
        prepared = dict(settings)
        # A wall-clock cutoff can end otherwise identical searches on different
        # machines. Keep it disabled for reproducible trial-based searches. The
        # FLAML 1.0.14 L2-logistic learner is the sole exception: on Unix it
        # passes an unbounded value to signal.alarm(), which overflows a C int.
        if tuple(prepared.get("estimator_list", ())) == ("lrl2",):
            prepared["time_budget"] = self.automl_compatibility_time_budget_seconds
        else:
            prepared.pop("time_budget", None)
        prepared.setdefault("max_iter", self.automl_max_iterations)
        prepared.setdefault("seed", random_seed)
        return prepared

    def _automl_random_seed(self) -> int:
        """Normalize legacy scalar and single-item sequence seed storage."""
        value = self.random_state
        if value is None:
            value = self.default_random_state
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) != 1:
                raise ValueError("AutoML random_state must contain exactly one seed value.")
            value = value[0]
        return int(value)

    def _automl_mlp_configurations(self) -> List[Dict[str, int]]:
        """Generate a fixed-size, serial MLP search without worker processes."""
        generator = np.random.RandomState(self._automl_random_seed())
        return [
            {
                "l1": int(generator.randint(1, 20)),
                "l2": int(generator.randint(1, 30)),
                "l3": int(generator.randint(1, 20)),
                "batch": int(generator.randint(20, 100)),
            }
            for _ in range(self.automl_tuning_trials)
        ]

    @property
    def image_config(self):
        return {
            # Picture layout
            "width": 16,  # number of subgraph rows
            "height": 9,  # number of subgraph columns
            "dpi": 360,  # resolution
            # Main content
            "cmap": "coolwarm_r",  # color setting
            "cmap2": "Wistia",
            "marker_angle": "^",  # point type
            "marker_circle": "o",  # point type
            "edgecolor": "w",  # point edge color
            "markersize1": 18,  # point size
            "markersize2": 6,
            "alpha1": 0.4,  # point transparency
            "alpha2": 0.95,
            "linestyle": "-",
            # bar
            "bar_color": "blue",
            "bar_align": "center",
            "bar_x": range(len(self.X.columns)),  # the sequence of horizontal coordinates of the bar
            "bar_height": None,  # the height(s) of the bars
            "bar_label": self.X.columns,  # The label on the X-axis
            "bar_width": 0.3,  # the width(s) of the bars
            "bottom": 0,  # the y coordinate(s) of the bars bases
            # Convert the font of the axes
            "labelsize": 5,  # the font size of the axis label
            "xrotation": 0,  # x axis label rotation Angle
            "xha": "center",  # x axis 'ha'
            "rot": 90,  # y axis label rotation Angle
            "yha": "center",  # y axis 'ha'
            "axislabelfont": "Times New Roman",  # axis label font
            # Picture title adjustment
            "title_label": self.naming,  # picture name
            "title_size": 15,  # title font size
            "title_color": "k",
            "title_location": "center",
            "title_font": "Times New Roman",
            "title_pad": 2,
            # Tree parameter
            "max_depth": None,  # The maximum depth of the representation
            "feature_names": None,  # Names of each of the features
            "class_names": ["class" + str(i) for i in range(1, 1000)],  # Names of each of the target classes in ascending numerical order
            "label": "all",  # Whether to show informative labels for impurity, etc
            "filled": True,  # color filling
            "impurity": True,  # When set to True, show the impurity at each node
            "node_ids": None,  # When set to True, show the ID number on each node
            "proportion": False,  # When set to True, change the display of ‘values’ and/or ‘samples’ to be proportions and percentages respectively
            "rounded": True,  # When set to True, draw node boxes with rounded corners and use Helvetica fonts instead of Times-Roman
            "precision": 3,  # Number of digits of precision for floating point in the values of impurity, threshold and value attributes of each node
            "ax": None,  # axes to plot to.
            "fontsize": None,  # size of text font
        }

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> None:
        """Placeholder for fit. child classes should implement this method!

        Parameters
        ----------
        X : pd.DataFrame (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and n_features` is the number of features.

        y : pd.DataFrame, default=None (n_samples,) or (n_samples, n_targets)
            Target values。
        """
        return None

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """The interface for the child classes."""
        return pd.DataFrame()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """The interface for the child classes."""
        return pd.DataFrame()

    @abstractmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Placeholder for manual_hyper_parameters. child classes should implement this method!

        Parameters
        ----------
        kwargs : dict
            The hyper parameters of the model.
        """
        return dict()

    @staticmethod
    def score(y_true: Union[pd.DataFrame, np.ndarray], y_predict: Union[pd.DataFrame, np.ndarray]) -> Union[int, float]:
        """The interface for the child classes."""
        return float()

    @staticmethod
    def np2pd(array: np.ndarray, columns_name: Union[List[str], pd.Index]) -> pd.DataFrame:
        """The type of the data set is transformed from numpy.ndarray to pandas.DataFrame.

        Parameters
        ----------
        array : np.ndarray (n_samples, n_features)
            the data set.

        columns_name : list[str] or pd.Index
            the name of the columns of the data set.

        Returns
        -------
        pd.DataFrame (n_samples, n_features)
            the data set.
        """
        return pd.DataFrame(array, columns=columns_name)

    @staticmethod
    def choose_dimension_data(data: pd.DataFrame, dimensions: int) -> Tuple[List[int], pd.DataFrame]:
        """Choose a subgroup data from the whole data set to draw 2d or 3d graph.

        Parameters
        ----------
        data : pd.DataFrame (n_samples, n_features)
            the whole data.
        dimensions : int
            how much dimensions data to keep.

        Returns
        -------
        selected_axis_index : list[int]
            the index of the that dimension, which is shown as the index of the column of the data set.

        selected_axis_data : pd.DataFrame (n_samples, n_features)
            the selected data from the whole data set.
        """
        print(f"-----* {dimensions} Dimensions Data Selection *-----")
        print(f"The software is going to draw related {dimensions}d graphs.")
        print(f"Currently, the data dimension is beyond {dimensions} dimensions.")
        print(f"Please choose {dimensions} dimensions of the data below.")
        data = pd.DataFrame(data)
        selected_axis_index = []
        selected_axis_name = []
        for i in range(1, dimensions + 1):
            num2option(data.columns)
            print(f"Choose dimension - {i} data:")
            index_axis = limit_num_input(data.columns, SECTION[3], num_input)
            selected_axis_index.append(index_axis - 1)
            selected_axis_name.append(data.columns[index_axis - 1])
        selected_axis_data = data.loc[:, selected_axis_name]
        print("The Selected Data Dimension:")
        show_data_columns(selected_axis_name)
        return selected_axis_index, selected_axis_data

    @staticmethod
    def data_upload(
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.DataFrame] = None,
        X_train: Optional[pd.DataFrame] = None,
        X_test: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.DataFrame] = None,
        name_train: Optional[pd.Series] = None,
        name_test: Optional[pd.Series] = None,
        name_all: Optional[pd.Series] = None,
        y_train_predict: Optional[pd.DataFrame] = None,
        y_test_predict: Optional[pd.DataFrame] = None,
    ) -> None:
        """This method loads the required data into the base class's attributes."""
        if X is not None:
            WorkflowBase.X = X
        if y is not None:
            WorkflowBase.y = y
        if X_train is not None:
            WorkflowBase.X_train = X_train
        if X_test is not None:
            WorkflowBase.X_test = X_test
        if y_train is not None:
            WorkflowBase.y_train = y_train
        if y_test is not None:
            WorkflowBase.y_test = y_test
        if name_train is not None:
            WorkflowBase.name_train = name_train
        if name_test is not None:
            WorkflowBase.name_test = name_test
        if name_all is not None:
            WorkflowBase.name_all = name_all
        if y_test_predict is not None:
            WorkflowBase.y_test_predict = y_test_predict
        if y_train_predict is not None:
            WorkflowBase.y_train_predict = y_train_predict

    @staticmethod
    def data_save(df: pd.DataFrame, name: str, df_name: str, local_path: str, mlflow_path: str, slogan: str) -> None:
        """This method saves the data into the local path and the mlflow path.

        Parameters
        ----------
        df : pd.DataFrame
            The data to be saved.

        name: str
            The name.
        df_name : str
            The name of the data.

        local_path : str
            The local path to save the data.

        mlflow_path : str
            The mlflow path to save the data.

        slogan : str
            The title of the output section.
        """
        print(f"-----* {slogan} *-----")
        print(df)
        save_data(df, name, df_name, local_path, mlflow_path)

    @staticmethod
    def save_hyper_parameters(hyper_parameters_dict: Dict, model_name: str, local_path: str) -> None:
        """This method saves the hyper parameters into the local path.

        Parameters
        ----------
        hyper_parameters_dict : dict
            The hyper parameters of the model.

        model_name : str
            The name of the model.

        local_path : str
            The local path to save the hyper parameters.
        """
        # 1. Always save the full dictionary to local file (no length limit)
        hyper_parameters_str = json.dumps(
            _json_safe(hyper_parameters_dict),
            indent=4,
            allow_nan=False,
        )
        save_text(hyper_parameters_str, f"Hyper Parameters - {model_name}", local_path)

        # 2. Log to MLflow with length limit handling (500 characters per value)
        for key, value in hyper_parameters_dict.items():
            # Convert value to string
            value_str = str(value)

            # If the value is within the 500-character limit, log normally
            if len(value_str) <= 500:
                mlflow.log_param(key, value_str)
            else:
                # If the value is too long, try to split it into smaller chunks
                if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                    # For best_config_per_output style lists, log each output's config separately
                    for idx, item in enumerate(value):
                        item_str = json.dumps(item)
                        if len(item_str) <= 500:
                            mlflow.log_param(f"{key}_output_{idx}", item_str)
                        else:
                            # If individual item is still too long, truncate
                            mlflow.log_param(f"{key}_output_{idx}", item_str[:497] + "...")
                else:
                    # For other long values, log a reference to the local file
                    mlflow.log_param(key, f"{key} (saved to local file - length: {len(value_str)})")

    @dispatch()
    def model_save(self) -> None:
        """Persist the model for future use after training the model with Scikit-learn framework."""
        print("-----* Model Persistence *-----")
        GEOPI_OUTPUT_ARTIFACTS_MODEL_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_MODEL_PATH")
        save_model(self.model, self.naming, self.X_train.iloc[[0]], GEOPI_OUTPUT_ARTIFACTS_MODEL_PATH)

    @dispatch(bool)
    def model_save(self, is_automl: bool) -> None:
        """Persist the model for future use after training the model with FLAML framework."""
        print("-----* Model Persistence *-----")
        GEOPI_OUTPUT_ARTIFACTS_MODEL_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_MODEL_PATH")
        save_model(self.auto_model, self.naming, self.X_train.iloc[[0]], GEOPI_OUTPUT_ARTIFACTS_MODEL_PATH)

    @staticmethod
    def _plot_permutation_importance(
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        name_column: str,
        trained_model: object,
        image_config: dict,
        algorithm_name: str,
        graph_name: str,
        local_path: str,
        mlflow_path: str,
    ) -> None:
        """Permutation importance plot."""
        print(f"-----* {graph_name} *-----")  # Permutation Importance
        importances_mean, importances_std, importances = plot_permutation_importance(X_test, y_test, trained_model, image_config)
        save_fig(f"{graph_name} - {algorithm_name}", local_path, mlflow_path)
        save_data(X_test, name_column, f"{graph_name} - X Test", local_path, mlflow_path)
        save_data(y_test, name_column, f"{graph_name} - Y Test", local_path, mlflow_path)
        data_dict = {"importances_mean": importances_mean.tolist(), "importances_std": importances_std.tolist(), "importances": importances.tolist()}
        data_str = json.dumps(data_dict, indent=4)
        save_text(data_str, f"{graph_name} - {algorithm_name}", local_path, mlflow_path)


class TreeWorkflowMixin:
    """Mixin class for tree-based models."""

    @staticmethod
    def _plot_feature_importance(X_train: pd.DataFrame, name_column: str, trained_model: object, image_config: dict, algorithm_name: str, func_name: str, local_path: str, mlflow_path: str) -> None:
        """Draw the feature importance bar diagram."""
        print(f"-----* {func_name} *-----")  # Feature Importance Diagram
        columns_name = X_train.columns

        # Ensemble models such as GradientBoosting expose both a top-level
        # feature_importances_ vector and an estimators_ array. Prefer the public
        # top-level vector; only average child estimators for a real
        # MultiOutputRegressor, which does not expose feature_importances_.
        if hasattr(trained_model, "feature_importances_"):
            feature_importances = trained_model.feature_importances_
        else:
            from sklearn.multioutput import MultiOutputRegressor

            if not isinstance(trained_model, MultiOutputRegressor):
                raise AttributeError(f"{type(trained_model).__name__} does not expose feature_importances_")
            feature_importances = np.mean([est.feature_importances_ for est in trained_model.estimators_], axis=0)

        data = plot_feature_importance(columns_name, feature_importances, image_config)
        save_fig(f"{func_name} - {algorithm_name}", local_path, mlflow_path)
        save_data(data, name_column, f"{func_name} - {algorithm_name}", local_path, mlflow_path, True)

    @staticmethod
    def _plot_tree(trained_model: object, image_config: dict, algorithm_name: str, func_name: str, local_path: str, mlflow_path: str) -> None:
        """Drawing decision tree diagrams."""
        print(f"-----* {func_name} *-----")  # Single Tree Diagram

        from sklearn.multioutput import MultiOutputRegressor

        def representative_tree(estimator: object) -> object:
            """Unwrap a tree ensemble without confusing it with multi-output."""
            children = getattr(estimator, "estimators_", None)
            if children is None:
                return estimator
            if isinstance(children, np.ndarray):
                if children.size == 0:
                    raise ValueError("The fitted tree ensemble does not contain an estimator.")
                child = children.flat[0]
            else:
                if not children:
                    raise ValueError("The fitted tree ensemble does not contain an estimator.")
                child = children[0]
            return representative_tree(child)

        if isinstance(trained_model, MultiOutputRegressor):
            for index, output_estimator in enumerate(trained_model.estimators_, start=1):
                output_func_name = f"{func_name} - Output {index}"
                print(f"-----* {output_func_name} *-----")
                plot_decision_tree(representative_tree(output_estimator), image_config)
                save_fig(f"{output_func_name} - {algorithm_name}", local_path, mlflow_path)
            return

        plot_decision_tree(representative_tree(trained_model), image_config)
        save_fig(f"{func_name} - {algorithm_name}", local_path, mlflow_path)


class LinearWorkflowMixin:
    """Mixin class for linear models."""

    @staticmethod
    def _show_formula(
        coef: np.ndarray,
        intercept: np.ndarray,
        features_name: np.ndarray,
        algorithm_name: str,
        func_name: str,
        regression_classification: str,
        y_train: pd.DataFrame,
        local_path: str,
        mlflow_path: str,
    ) -> None:
        """Show the formula."""
        print(f"-----* {func_name} *-----")
        formula = show_formula(coef, intercept, features_name, regression_classification, y_train)
        formula_str = json.dumps(formula, indent=4)
        save_text(formula_str, f"{func_name}", local_path, mlflow_path)

    @staticmethod
    def _plot_2d_scatter_diagram(feature_data: pd.DataFrame, target_data: pd.DataFrame, data_name: str, algorithm_name: str, func_name: str, local_path: str, mlflow_path: str) -> None:
        """Plot the 2D graph of the linear regression model."""
        print(f"-----* {func_name} *-----")  # 2D Scatter Diagram
        plot_2d_scatter_diagram(feature_data, target_data)
        save_fig(f"{func_name} - {algorithm_name}", local_path, mlflow_path)
        data = pd.concat([feature_data, target_data], axis=1)
        save_data(data, data_name, f"{func_name} - {algorithm_name}", local_path, mlflow_path)

    @staticmethod
    def _plot_2d_line_diagram(
        feature_data: pd.DataFrame, target_data: pd.DataFrame, y_test_predict: pd.DataFrame, data_name: str, algorithm_name: str, func_name: str, local_path: str, mlflow_path: str
    ) -> None:
        """Plot the 2D graph of the linear regression model."""
        print(f"-----* {func_name} *-----")  # 2D Line Diagram
        plot_2d_line_diagram(feature_data, target_data, y_test_predict)
        save_fig(f"{func_name} - {algorithm_name}", local_path, mlflow_path)
        prediction_data = y_test_predict.rename(columns=lambda column: f"Predicted_{column}")
        data = pd.concat([feature_data, target_data, prediction_data], axis=1)
        save_data(data, data_name, f"{func_name} - {algorithm_name}", local_path, mlflow_path)

    @staticmethod
    def _plot_3d_scatter_diagram(feature_data: pd.DataFrame, target_data: pd.DataFrame, data_name: str, algorithm_name: str, func_name: str, local_path: str, mlflow_path: str) -> None:
        """Plot the 3D graph of the linear regression model."""
        print(f"-----*  {func_name} *-----")  # 3D Scatter Diagram
        plot_3d_scatter_diagram(feature_data, target_data)
        save_fig(f"{func_name} - {algorithm_name}", local_path, mlflow_path)
        data = pd.concat([feature_data, target_data], axis=1)
        save_data(data, data_name, f"{func_name} - {algorithm_name}", local_path, mlflow_path)

    @staticmethod
    def _plot_3d_surface_diagram(
        feature_data: pd.DataFrame, target_data: pd.DataFrame, y_test_predict: pd.DataFrame, data_name: str, algorithm_name: str, func_name: str, local_path: str, mlflow_path: str
    ) -> None:
        """Plot the 3D graph of the linear regression model."""
        print(f"-----* {func_name} *-----")  # 3D Surface Diagram
        plot_3d_surface_diagram(feature_data, target_data, y_test_predict)
        save_fig(f"{func_name} - {algorithm_name}", local_path, mlflow_path)
        prediction_data = y_test_predict.rename(columns=lambda column: f"Predicted_{column}")
        data = pd.concat([feature_data, target_data, prediction_data], axis=1)
        save_data(data, data_name, f"{func_name} - {algorithm_name}", local_path, mlflow_path)


class ClusteringMetricsMixin:
    """Mixin class for clustering metrics."""

    @staticmethod
    def _get_num_clusters(labels: pd.Series, func_name: str, algorithm_name: str, store_path: str) -> None:
        """Get and log the number of clusters. It is only used in those algorithms which don't allow to set the number of cluster in advance."""
        print(f"-----* {func_name} *-----")
        num_clusters = len(np.unique(labels.to_numpy()))
        print(f"{func_name}: {num_clusters}")
        num_clusters_dict = {f"{func_name}": num_clusters}
        mlflow.log_metrics(num_clusters_dict)
        num_clusters_str = json.dumps(num_clusters_dict, indent=4)
        save_text(num_clusters_str, f"{func_name} - {algorithm_name}", store_path)
