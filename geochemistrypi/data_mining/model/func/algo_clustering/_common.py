# -*- coding: utf-8 -*-
from itertools import cycle
from typing import Dict

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich import print
from sklearn.metrics import calinski_harabasz_score, silhouette_samples, silhouette_score


def score(data: pd.DataFrame, labels: pd.Series) -> Dict:
    """Calculate the scores of the clustering model.

    Parameters
    ----------
    data : pd.DataFrame (n_samples, n_components)
        The true values.

    labels : pd.Series (n_samples, )
        Labels of each point.

    Returns
    -------
    scores : dict
        The scores of the clustering model.
    """
    silhouette = silhouette_score(data, labels)
    calinski_harabaz = calinski_harabasz_score(data, labels)
    print("silhouette_score: ", silhouette)
    print("calinski_harabasz_score:", calinski_harabaz)
    scores = {
        "silhouette_score": silhouette,
        "calinski_harabasz_score": calinski_harabaz,
    }
    return scores


def scatter2d(data: pd.DataFrame, labels: pd.Series, cluster_centers_: pd.DataFrame, algorithm_name: str) -> None:
    """
    Draw the result-2D diagram for analysis.

    Parameters
    ----------
    data : pd.DataFrame (n_samples, n_components)
       The features of the data.

    labels : pd.Series (n_samples,)
        Labels of each point.

    cluster_centers_: pd.DataFrame (n_samples,)
        Coordinates of cluster centers. If the algorithm stops before fully converging (see tol and max_iter), these will not be consistent with labels_.

    algorithm_name : str
        the name of the algorithm
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

    fig = plt.figure()
    fig.set_size_inches(18, 10)
    plt.subplot(111)
    # Plot the data
    for i, label in enumerate(set(labels)):
        cluster_data = data[labels == label]
        color = next(color_cycle)
        marker = next(marker_cycle)
        plt.scatter(cluster_data.iloc[:, 0], cluster_data.iloc[:, 1], c=color, marker=marker)

    # Plot the cluster centers
    if cluster_centers_ is not None:
        # Draw white circles at cluster centers
        plt.scatter(cluster_centers_.iloc[:, 0], cluster_centers_.iloc[:, 1], c="white", marker="o", alpha=1, s=200, edgecolor="k")

        # Label the cluster centers
        for i, c in enumerate(cluster_centers_.to_numpy()):
            plt.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    plt.xlabel(f"{data.columns[0]}")
    plt.ylabel(f"{data.columns[1]}")
    plt.title(f"Cluster Data Bi-plot - {algorithm_name}")


def scatter3d(data: pd.DataFrame, labels: pd.Series, algorithm_name: str) -> None:
    """
    Draw the result-3D diagram for analysis.

    Parameters
    ----------
    data : pd.DataFrame (n_samples, n_components)
       The features of the data.

    labels : pd.Series (n_samples,)
        Labels of each point.

    algorithm_name : str
        the name of the algorithm
    """
    plt.figure()
    namelist = data.columns.values.tolist()
    fig = plt.figure(figsize=(12, 6), facecolor="w")
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9)

    # Plot the data without cluster results
    ax = fig.add_subplot(121, projection="3d")
    ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], alpha=0.3, c="#FF0000", marker=".")
    ax.set_xlabel(namelist[0])
    ax.set_ylabel(namelist[1])
    ax.set_zlabel(namelist[2])
    plt.grid(True)

    ax2 = fig.add_subplot(122, projection="3d")
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

    # Plot the data with cluster results
    for i, label in enumerate(set(labels)):
        cluster_data = data[labels == label]
        color = next(color_cycle)
        marker = next(marker_cycle)
        ax2.scatter(cluster_data.iloc[:, 0], cluster_data.iloc[:, 1], cluster_data.iloc[:, 2], c=color, marker=marker, s=6, cmap=plt.cm.Paired, edgecolors="none")

    ax2.set_xlabel(namelist[0])
    ax2.set_ylabel(namelist[1])
    ax2.set_zlabel(namelist[2])
    plt.grid(True)
    ax.set_title(f"Base Data Tri-plot - {algorithm_name}")
    ax2.set_title(f"Cluster Data Tri-plot - {algorithm_name}")


def plot_silhouette_diagram(data: pd.DataFrame, labels: pd.Series, cluster_centers_: pd.DataFrame, model: object, algorithm_name: str) -> None:
    """
    Draw the silhouette diagram for analysis.

    Parameters
    ----------
    data : pd.DataFrame (n_samples, n_components)
       The true values.

    labels : pd.Series (n_samples,)
        Labels of each point.

    cluster_centers_: pd.DataFrame (n_samples,)
        Coordinates of cluster centers. If the algorithm stops before fully converging (see tol and max_iter), these will not be consistent with labels_.

    model : sklearn algorithm model
        The sklearn algorithm model trained with X.

    algorithm_name : str
        the name of the algorithm

    References
    ----------
    Silhouette analysis can be used to study the separation distance between the resulting clusters.
    The silhouette plot displays a measure of how close each point in one cluster is to other points in the
    neighboring clusters and thus provides a way to assess parameters like number of clusters visually.
    This measure has a range of [-1, 1].

    https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    """
    if hasattr(model, "n_clusters"):
        n_clusters = model.n_clusters
    else:
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 10)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(data, labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(data, labels)

    if n_clusters >= 20:
        Fontsize = 5
        y_long = 7
    else:
        Fontsize = None
        y_long = 10

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), fontsize=Fontsize)

        # Compute the new y_lower for next plot
        y_lower = y_upper + y_long  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(labels.astype(float) / n_clusters)
    ax2.scatter(data.iloc[:, 0], data.iloc[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")

    if cluster_centers_ is not None:
        # Draw white circles at cluster centers
        ax2.scatter(
            cluster_centers_.iloc[:, 0],
            cluster_centers_.iloc[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        # Label the cluster centers
        for i, c in enumerate(cluster_centers_.to_numpy()):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")
    plt.suptitle(
        f"Silhouette analysis for clustering on sample data with n_clusters = %d - {algorithm_name}" % n_clusters,
        fontsize=14,
        fontweight="bold",
    )


def plot_silhouette_value_diagram(data: pd.DataFrame, labels: pd.Series, algorithm_name: str) -> None:
    """Calculate the scores of the clustering model.

    Parameters
    ----------
    data : pd.DataFrame (n_samples, n_components)
        The true values.

    labels : pd.Series (n_samples, )
        Labels of each point.

    algorithm_name : str
        The name of the algorithm model.
    """
    silhouette_values = silhouette_samples(data, labels)
    sns.histplot(silhouette_values, bins=30, kde=True)
    plt.title(f"Silhouette value Diagram - {algorithm_name}")
    plt.xlabel("Silhouette Coefficient")
    plt.ylabel("Frequency")
