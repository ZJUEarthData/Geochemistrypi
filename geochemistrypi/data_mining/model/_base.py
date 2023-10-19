# -*- coding: utf-8 -*-
import json
import os
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
from multipledispatch import dispatch
from rich import print

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

    @classmethod
    def show_info(cls) -> None:
        """Display how many functions the algorithm will provide."""
        print("*-*" * 2, cls.name, "is running ...", "*-*" * 2)
        print("Expected Functionality:")
        function = cls.common_function + cls.special_function
        for i in range(len(function)):
            print("+ ", function[i])

    def __init__(self) -> None:
        # Default for child class. They need to be overwritten in child classes.
        self.model = None
        self.naming = None
        self.automl = None
        self.ray_best_model = None
        self.random_state = 42

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
        if y_test_predict is not None:
            WorkflowBase.y_test_predict = y_test_predict

    @staticmethod
    def data_save(df: pd.DataFrame, df_name: str, local_path: str, mlflow_path: str, slogan: str) -> None:
        """This method saves the data into the local path and the mlflow path.

        Parameters
        ----------
        df : pd.DataFrame
            The data to be saved.

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
        save_data(df, df_name, local_path, mlflow_path)

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
        hyper_parameters_str = json.dumps(hyper_parameters_dict, indent=4)
        save_text(hyper_parameters_str, f"Hyper Parameters - {model_name}", local_path)
        mlflow.log_params(hyper_parameters_dict)

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
        trained_model: object,
        image_config: dict,
        algorithm_name: str,
        local_path: str,
        mlflow_path: str,
    ) -> None:
        """Permutation importance plot."""
        print("-----* Permutation Importance Diagram *-----")
        importances_mean, importances_std, importances = plot_permutation_importance(X_test, y_test, trained_model, image_config)
        save_fig(f"Permutation Importance - {algorithm_name}", local_path, mlflow_path)
        save_data(X_test, "Permutation Importance - X Test", local_path, mlflow_path)
        save_data(y_test, "Permutation Importance - Y Test", local_path, mlflow_path)
        data_dict = {"importances_mean": importances_mean.tolist(), "importances_std": importances_std.tolist(), "importances": importances.tolist()}
        data_str = json.dumps(data_dict, indent=4)
        save_text(data_str, f"Permutation Importance - {algorithm_name}", local_path, mlflow_path)


class TreeWorkflowMixin:
    """Mixin class for tree models."""

    @staticmethod
    def _plot_feature_importance(X_train: pd.DataFrame, trained_model: object, image_config: dict, algorithm_name: str, local_path: str, mlflow_path: str) -> None:
        """Draw the feature importance bar diagram."""
        print("-----* Feature Importance Diagram *-----")
        columns_name = X_train.columns
        feature_importances = trained_model.feature_importances_
        data = plot_feature_importance(columns_name, feature_importances, image_config)
        save_fig(f"Feature Importance - {algorithm_name}", local_path, mlflow_path)
        save_data(data, f"Feature Importance - {algorithm_name}", local_path, mlflow_path, True)

    @staticmethod
    def _plot_tree(trained_model: object, image_config: dict, algorithm_name: str, local_path: str, mlflow_path: str) -> None:
        """Drawing decision tree diagrams."""
        print("-----* Single Tree Diagram *-----")
        plot_decision_tree(trained_model, image_config)
        save_fig(f"Tree Diagram - {algorithm_name}", local_path, mlflow_path)


class LinearWorkflowMixin:
    """Mixin class for linear models."""

    @staticmethod
    def _show_formula(coef: np.ndarray, intercept: np.ndarray, features_name: np.ndarray, algorithm_name: str, local_path: str, mlflow_path: str) -> None:
        """Show the formula."""
        print(f"-----* {algorithm_name} Formula *-----")
        formula = show_formula(coef, intercept, features_name)
        formula_str = json.dumps(formula, indent=4)
        save_text(formula_str, f"{algorithm_name} Formula", local_path, mlflow_path)

    @staticmethod
    def _plot_2d_scatter_diagram(feature_data: pd.DataFrame, target_data: pd.DataFrame, algorithm_name: str, local_path: str, mlflow_path: str) -> None:
        """Plot the 2D graph of the linear regression model."""
        print("-----* 2D Scatter Diagram *-----")
        plot_2d_scatter_diagram(feature_data, target_data)
        save_fig(f"2D Scatter Diagram - {algorithm_name}", local_path, mlflow_path)
        data = pd.concat([feature_data, target_data], axis=1)
        save_data(data, f"2D Scatter Diagram - {algorithm_name}", local_path, mlflow_path)

    @staticmethod
    def _plot_2d_line_diagram(feature_data: pd.DataFrame, target_data: pd.DataFrame, y_test_predict: pd.DataFrame, algorithm_name: str, local_path: str, mlflow_path: str) -> None:
        """Plot the 2D graph of the linear regression model."""
        print("-----* 2D Line Diagram *-----")
        plot_2d_line_diagram(feature_data, target_data, y_test_predict)
        save_fig(f"2D Line Diagram - {algorithm_name}", local_path, mlflow_path)
        data = pd.concat([feature_data, target_data, y_test_predict], axis=1)
        save_data(data, f"2D Line Diagram - {algorithm_name}", local_path, mlflow_path)

    @staticmethod
    def _plot_3d_scatter_diagram(feature_data: pd.DataFrame, target_data: pd.DataFrame, algorithm_name: str, local_path: str, mlflow_path: str) -> None:
        """Plot the 3D graph of the linear regression model."""
        print("-----*  3D Scatter Diagram *-----")
        plot_3d_scatter_diagram(feature_data, target_data)
        save_fig(f"3D Scatter Diagram - {algorithm_name}", local_path, mlflow_path)
        data = pd.concat([feature_data, target_data], axis=1)
        save_data(data, f"3D Scatter Diagram - {algorithm_name}", local_path, mlflow_path)

    @staticmethod
    def _plot_3d_surface_diagram(feature_data: pd.DataFrame, target_data: pd.DataFrame, y_test_predict: pd.DataFrame, algorithm_name: str, local_path: str, mlflow_path: str) -> None:
        """Plot the 3D graph of the linear regression model."""
        print("-----* 3D Surface Diagram *-----")
        plot_3d_surface_diagram(feature_data, target_data, y_test_predict)
        save_fig(f"3D Surface Diagram - {algorithm_name}", local_path, mlflow_path)
        data = pd.concat([feature_data, target_data, y_test_predict], axis=1)
        save_data(data, f"3D Surface Diagram - {algorithm_name}", local_path, mlflow_path)
