# -*- coding: utf-8 -*-
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from rich import print

from ....constants import SECTION
from ....data.data_readiness import bool_input


def linear_regression_manual_hyper_parameters() -> Dict:
    """Manually set hyperparameters.

    Returns
    -------
    hyper_parameters : dict
    """
    print("Fit Intercept: This hyperparameter specifies whether to calculate the intercept (also called the bias term) for this model.")
    print("Please specify whether to calculate the intercept for this model. It is generally recommended to leave it as True.")
    fit_intercept = bool_input(SECTION[2])
    hyper_parameters = {"fit_intercept": fit_intercept}
    return hyper_parameters


def plot_2d_scatter_diagram(feature_data: pd.DataFrame, target_data: pd.DataFrame) -> None:
    """Plot a 2D scatter diagram with the data set below.

    Parameters
    ----------
    feature_data : pd.DataFrame (n_samples, n_components)
        Data selected by user.

    target_data : pd.DataFrame (n_samples, n_components)
        The target values.
    """
    plt.figure(figsize=(14, 10))

    # Support multiple Y columns: plot scatter plots for each target variable
    if target_data.shape[1] == 1:
        # Single Y column case
        plt.scatter(feature_data.values, target_data.values)
        plt.xlabel(feature_data.columns[0])
        plt.ylabel(target_data.columns[0])
    else:
        # Multiple Y columns: plot subplots for each target variable
        n_targets = target_data.shape[1]
        fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 5))
        if n_targets == 1:
            axes = [axes]

        for i, target_col in enumerate(target_data.columns):
            axes[i].scatter(feature_data.values, target_data[target_col].values)
            axes[i].set_xlabel(feature_data.columns[0])
            axes[i].set_ylabel(target_col)
            axes[i].grid()

        plt.tight_layout()

    plt.title("2D Scatter Diagram")
    plt.grid()


def plot_2d_line_diagram(feature_data: pd.DataFrame, target_data: pd.DataFrame, y_test_predict: pd.DataFrame) -> None:
    """Plot a 2D line diagram with the data set below.

    Parameters
    ----------
    feature_data : pd.DataFrame (n_samples, n_components)
        Data selected by user.

    target_data : pd.DataFrame (n_samples, n_components)
        The target values.

    y_test_predict : pd.DataFrame (n_samples, n_components)
        The predicted target values.
    """
    plt.figure(figsize=(14, 10))

    # Support multiple Y columns: plot line graphs for each target variable
    if target_data.shape[1] == 1:
        # Single Y column case
        plt.scatter(feature_data.values, target_data.values, label="Data Points")
        plt.plot(feature_data.values, y_test_predict.values, label="Regression Line")
        plt.xlabel(feature_data.columns[0])
        plt.ylabel(target_data.columns[0])
    else:
        # Multiple Y columns: plot subplots for each target variable
        n_targets = target_data.shape[1]
        fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 5))
        if n_targets == 1:
            axes = [axes]

        for i, target_col in enumerate(target_data.columns):
            axes[i].scatter(feature_data.values, target_data[target_col].values, label="Data Points")
            axes[i].plot(feature_data.values, y_test_predict[target_col].values, label="Regression Line")
            axes[i].set_xlabel(feature_data.columns[0])
            axes[i].set_ylabel(target_col)
            axes[i].grid()
            axes[i].legend()

        plt.tight_layout()

    plt.title("2D Line Diagram")
    plt.legend()
    plt.grid()


def plot_3d_scatter_diagram(feature_data: pd.DataFrame, target_data: pd.DataFrame) -> None:
    """Plot a 3D scatter diagram with the data set below.

    Parameters
    ----------
    feature_data : pd.DataFrame (n_samples, n_components)
        Data selected by user.

    target_data : pd.DataFrame (n_samples, n_components)
        The target values.
    """
    x = feature_data.iloc[:, 0]
    y = feature_data.iloc[:, 1]
    name_list = feature_data.columns.values.tolist()
    fig = plt.figure(figsize=(14, 10))
    ax = Axes3D(fig)

    # Support multiple Y columns: plot 3D scatter plots for each target variable
    if target_data.shape[1] == 1:
        # Single Y column case
        z = target_data.iloc[:, 0]
        ax.scatter(x, y, z)
        ax.set_zlabel(target_data.columns[0], fontdict={"size": 10, "color": "blue"})
    else:
        # Multiple Y columns: plot different colors for each target variable
        colors = ["red", "blue", "green", "yellow", "purple", "orange"]
        for i, target_col in enumerate(target_data.columns):
            z = target_data[target_col]
            ax.scatter(x, y, z, c=colors[i % len(colors)], label=target_col, alpha=0.7)
        ax.legend()
        ax.set_zlabel("Target Variables", fontdict={"size": 10, "color": "blue"})

    ax.set_xlabel(name_list[0], fontdict={"size": 10, "color": "black"})
    ax.set_ylabel(name_list[1], fontdict={"size": 10, "color": "r"})
    plt.title("3D Scatter Diagram")
    plt.grid()


def plot_3d_surface_diagram(feature_data: pd.DataFrame, target_data: pd.DataFrame, y_test_predict: pd.DataFrame) -> None:
    """Plot a 3D surface diagram with the data set below.

    Parameters
    ----------
    feature_data : pd.DataFrame (n_samples, n_components)
        Data selected by user.

    target_data : pd.DataFrame (n_samples, n_components)
        The target values.

    y_test_predict : pd.DataFrame (n_samples, n_components)
        The predicted target values.
    """
    x = feature_data.iloc[:, 0]
    y = feature_data.iloc[:, 1]
    name_list = feature_data.columns.values.tolist()
    fig = plt.figure(figsize=(14, 10))
    ax = Axes3D(fig)

    # Support multiple Y columns: plot 3D surface plots for each target variable
    if target_data.shape[1] == 1:
        # Single Y column case
        z = target_data.iloc[:, 0]
        ax.scatter(x, y, z)
        # Create meshgrid for x and y
        x_grid, y_grid = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))
        # Interpolate y_test_predict values on the grid
        z_grid = np.interp(np.ravel(x_grid), x, y_test_predict.iloc[:, 0])
        z_grid = z_grid.reshape(x_grid.shape)
        # Plot the 3D surface using the grid
        ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.3, color="red")
        ax.set_zlabel(target_data.columns[0], fontdict={"size": 10, "color": "blue"})
    else:
        # Multiple Y columns: plot different colors for each target variable
        colors = ["red", "blue", "green", "yellow", "purple", "orange"]
        for i, target_col in enumerate(target_data.columns):
            z = target_data[target_col]
            ax.scatter(x, y, z, c=colors[i % len(colors)], label=target_col, alpha=0.7)

            # Create surface for each target variable
            x_grid, y_grid = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))
            z_grid = np.interp(np.ravel(x_grid), x, y_test_predict[target_col])
            z_grid = z_grid.reshape(x_grid.shape)
            ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.2, color=colors[i % len(colors)])

        ax.legend()
        ax.set_zlabel("Target Variables", fontdict={"size": 10, "color": "blue"})

    ax.set_xlabel(name_list[0], fontdict={"size": 10, "color": "black"})
    ax.set_ylabel(name_list[1], fontdict={"size": 10, "color": "r"})
    plt.title("3D Surface Diagram")
    plt.grid()
    plt.legend()
