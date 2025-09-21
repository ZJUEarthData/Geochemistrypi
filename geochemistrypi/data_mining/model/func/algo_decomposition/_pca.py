# -*- coding: utf-8 -*-
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print

from ....constants import SECTION
from ....data.data_readiness import num_input, str_input


def pca_manual_hyper_parameters() -> Dict:
    """Manually set hyperparameters.

    Returns
    -------
    hyper_parameters : dict
    """
    print("N Components: This parameter specifies the number of components to retain after dimensionality reduction.")
    print("Please specify the number of components to retain. A good starting range could be between 2 and 10, such as 4.")
    n_components = num_input(SECTION[2], "N Components: ")
    print("SVD Solver: This parameter specifies the algorithm used to perform the singular value decomposition.")
    print("Please specify the algorithm. It is generally recommended to leave it as 'auto'.")
    svd_solvers = ["auto", "full", "arpack", "randomized"]
    svd_solver = str_input(svd_solvers, SECTION[2])
    hyper_parameters = {"n_components": n_components, "svd_solver": svd_solver}
    return hyper_parameters


def biplot(reduced_data: pd.DataFrame, pc: pd.DataFrame, algorithm_name: str, labels: Optional[List[str]] = None) -> None:
    """Plot a compositional bi-plot for two principal components.

    Parameters
    ----------
    reduced_data : pd.DataFrame (n_samples, n_components)
        Data processed by PCA.

    pc : pd.DataFrame (n_features, n_components)
        principal components.

    algorithm_name : str
        the name of the algorithm

    labels : List[str]
        The type of tag of the samples in the data set.
    """
    plt.figure(figsize=(14, 10))

    x = reduced_data.iloc[:, 0]  # features' contributions for PC1
    y = reduced_data.iloc[:, 1]  # features' contributions for PC2
    scalex = 1.0 / (x.max() - x.min())
    scaley = 1.0 / (y.max() - y.min())

    # Draw a data point projection plot that is projected to
    # a two-dimensional plane using normal PCA
    if labels:
        legend = []
        classes = np.unique(labels)
        for i, label in enumerate(classes):
            plt.scatter(x[labels == label] * scalex, y[labels == label] * scaley, linewidth=0.01)
            legend.append("Label: {}".format(label))
        plt.legend(legend)
    else:
        plt.scatter(x * scalex, y * scaley, linewidths=0.01)

    # principal component's weight coefficient
    # it can obtain the expression of principal component in feature space
    n = pc.shape[0]

    # plot arrows as the variable contribution,
    # each variable has a score for PC1 and for PC2 respectively
    for i in range(n):
        plt.arrow(
            0,
            0,
            pc.iloc[i, 0],
            pc.iloc[i, 1],
            color="k",
            alpha=0.7,
            linewidth=1,
        )
        plt.text(pc.iloc[i, 0] * 1.01, pc.iloc[i, 1] * 1.01, pc.index[i], ha="center", va="center", color="k", fontsize=12)

    plt.xlabel(f"{pc.columns[0]}")
    plt.ylabel(f"{pc.columns[1]}")
    plt.title(f"Compositional Bi-plot - {algorithm_name}")
    plt.grid()


def triplot(reduced_data: pd.DataFrame, pc: pd.DataFrame, algorithm_name: str, labels: Optional[List[str]] = None) -> None:
    """Plot a compositional tri-plot in 3d for three principal components.

    Parameters
    ----------
    reduced_data : pd.DataFrame (n_samples, n_components)
        Data processed by PCA.

    pc : pd.DataFrame (n_features, n_components)
        principal components.

    algorithm_name : str
        the name of the algorithm

    store_path : str
        the local path to store the graph produced

    labels : List[str]
        The type of tag of the samples in the data set.
    """

    plt.figure(figsize=(14, 12))
    ax = plt.axes(projection="3d")

    x = reduced_data.iloc[:, 0]  # features' contributions for PC1
    y = reduced_data.iloc[:, 1]  # features' contributions for PC2
    z = reduced_data.iloc[:, 2]  # features' contributions for PC3
    scalex = 1.0 / (x.max() - x.min())
    scaley = 1.0 / (y.max() - y.min())
    scalez = 1.0 / (z.max() - z.min())

    # Draw a data point projection plot that is projected to
    # a three-dimensional space using normal PCA
    if labels:
        legend = []
        classes = np.unique(labels)  # label type
        for i, label in enumerate(classes):
            ax.scatter3D(x[labels == label] * scalex, y[labels == label] * scaley, z[labels == label] * scalez, linewidth=0.01)
            # hyperparameter in plt.scatter(): c=colors[i], marker=markers[i]
            legend.append("Label: {}".format(label))
        ax.legend(legend)
    else:
        ax.scatter3D(x * scalex, y * scaley, z * scalez, linewidths=0.01)

    # the initial angle to draw the 3d plot
    azim = -60  # azimuth
    elev = 30  # elevation
    ax.view_init(elev, azim)  # set the angles

    # principal component's weight coefficient
    # it can obtain the expression of principal component in feature space
    n = pc.shape[0]

    # plot arrows as the column_name contribution,
    # each column_name has a score for PC1, for PC2 and for PC3 respectively
    for i in range(n):
        ax.quiver(0, 0, 0, pc.iloc[i, 0], pc.iloc[i, 1], pc.iloc[i, 2], color="k", alpha=0.7, linewidth=1, arrow_length_ratio=0.05)
        ax.text(
            pc.iloc[i, 0] * 1.1,
            pc.iloc[i, 1] * 1.1,
            pc.iloc[i, 2] * 1.1,
            pc.index[i],
            ha="center",
            va="center",
            color="k",
            fontsize=12,
        )

    ax.set_xlabel(f"{pc.columns[0]}")
    ax.set_ylabel(f"{pc.columns[1]}")
    ax.set_zlabel(f"{pc.columns[2]}")
    plt.title(f"Compositional Tri-plot - {algorithm_name}")
    plt.grid()
