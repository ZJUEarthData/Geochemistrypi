# -*- coding: utf-8 -*-
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kde


def plot_2d_scatter_diagram(data: pd.DataFrame, algorithm_name: str) -> None:
    """
    Plot a 2D scatter diagram for dimensionality reduction results.

    Parameters
    ----------
    data : pd.DataFrame (n_samples, n_components)
        Data after dimensionality reduction.

    pc : pd.DataFrame (n_features, n_components)
        principal components.

    algorithm_name : str
        The name of the dimensionality reduction algorithm.

        labels : List[str]
        The type of tag of the samples in the data set.
    """
    markers = ["+", "v", ".", "d", "o", "s", "1", "D", "X", "^", "p", "<", "*", "H", "3", "P"]
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#33a02c",
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
    ]

    marker_cycle = cycle(markers)
    color_cycle = cycle(colors)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    # Plot the data
    for i, label in enumerate(data.index):
        colors = next(color_cycle)
        markers = next(marker_cycle)
        ax.scatter(data.iloc[:, 0], data.iloc[:, 1], c=colors, marker=markers, label=label)

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(f"{algorithm_name} Dimensionality Reduction Results")
    ax.legend(loc="upper right")

    plt.grid(True)


def plot_heatmap(data: pd.DataFrame, algorithm_name: str) -> None:
    """
    Plot a heatmap for dimensionality reduction results.

    Parameters
    ----------
    data : pd.DataFrame (n_samples, n_components)
        Data after dimensionality reduction.

    algorithm_name : str
        The name of the dimensionality reduction algorithm.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, cmap="viridis")
    plt.title(f"{algorithm_name} Dimensionality Reduction Heatmap")
    plt.xlabel("Component")
    plt.ylabel("Sample")


def plot_contour(data: pd.DataFrame, algorithm_name: str) -> None:
    """
    Plot a contour plot for dimensionality reduction results.

    Parameters
    ----------
    data : pd.DataFrame (n_samples, n_components)
        Data after dimensionality reduction.

    algorithm_name : str
        The name of the dimensionality reduction algorithm.
    """
    # Calculate the density
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    buffer = max(x.max() - x.min(), y.max() - y.min()) * 0.05
    xmin, xmax = x.min() - buffer, x.max() + buffer
    ymin, ymax = y.min() - buffer, y.max() + buffer

    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = kde.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    # Plot the contour
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, f, cmap="viridis", alpha=0.5)
    plt.colorbar(label="Density")
    plt.scatter(x, y, marker="o", color="black", alpha=0.5)
    plt.xlabel(f"{data.columns[0]}")
    plt.ylabel(f"{data.columns[1]}")
    plt.title(f"{algorithm_name} Dimensionality Reduction Contour Plot")
    plt.grid(True)
