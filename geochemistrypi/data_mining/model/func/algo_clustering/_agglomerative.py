from itertools import cycle
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patheffects import withStroke
from rich import print

from ....constants import SECTION
from ....data.data_readiness import num_input, str_input


def agglomerative_manual_hyper_parameters() -> Dict:
    """Manually set hyperparameters.

    Returns
    -------
    hyper_parameters : dict
    """
    print("N Clusters: The number of clusters to form as well as the number of centroids to generate.")
    print("Please specify the number of clusters for Agglomerative. A good starting range could be between 2 and 10, such as 4.")
    n_clusters = num_input(SECTION[2], "N Clusters: ")
    print("linkage: The linkage criterion determines which distance to use between sets of observation. ")
    print("Please specify the linkage criterion. It is generally recommended to leave it set to ward.")
    linkages = ["ward", "complete", "average", "single"]
    linkage = str_input(linkages, SECTION[2])
    hyper_parameters = {
        "n_clusters": n_clusters,
        "linkage": linkage,
    }
    return hyper_parameters


def scatter2d(data: pd.DataFrame, cluster_labels: pd.DataFrame, algorithm_name: str) -> None:
    """make 2d scatter plot for clustering results"""
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
    fig, ax = plt.subplots(figsize=(10, 10))
    cluster_labels = cluster_labels + 1

    categorys = np.unique(cluster_labels)
    for category in categorys:
        # for category in categorys:
        markers = next(marker_cycle)
        colors = next(color_cycle)
        ax.scatter(
            data.iloc[:, 0][cluster_labels == category],
            data.iloc[:, 1][cluster_labels == category],
            label=str(category),
            color=colors,
            marker=markers,
            s=30,
            linewidths=0.01,
            path_effects=[withStroke(linewidth=3, foreground="black")],
        )
    ax.spines["top"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.spines["bottom"].set_linewidth(1)
    ax.spines["left"].set_linewidth(1)

    plt.xlabel(
        f"{data.columns[0]}",
        font={
            "family": "Time New Roam",
            "size": 16,
        },
        weight="bold",
    )
    plt.ylabel(f"{data.columns[1]}", font={"family": "Time New Roam", "size": 16}, weight="bold")
    plt.legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0, fontsize=10, markerscale=2.5)
    plt.title(f"Cluster Data Bi-plot - {algorithm_name}")
