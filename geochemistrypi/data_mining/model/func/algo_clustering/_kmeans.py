from typing import Dict

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print
from sklearn.metrics import silhouette_samples, silhouette_score

from ....constants import SECTION
from ....data.data_readiness import float_input, num_input, str_input


def kmeans_manual_hyper_parameters() -> Dict:
    """Manually set hyperparameters.

    Returns
    -------
    hyper_parameters : dict
    """
    print("N Clusters: The number of clusters to form as well as the number of centroids to generate.")
    print("Please specify the number of clusters for KMeans. A good starting range could be between 2 and 10, such as 4.")
    n_clusters = num_input(SECTION[2], "N Clusters: ")
    print("Init: Method for initialization of centroids. The centroids represent the center points of the clusters in the dataset.")
    print("Please specify the method for initialization of centroids. It is generally recommended to leave it as 'k-means++'.")
    inits = ["k-means++", "random"]
    init = str_input(inits, SECTION[2])
    print("Max Iter: Maximum number of iterations of the k-means algorithm for a single run.")
    print("Please specify the maximum number of iterations of the k-means algorithm for a single run. A good starting range could be between 100 and 500, such as 300.")
    max_iters = num_input(SECTION[2], "Max Iter: ")
    print("Tolerance: Relative tolerance with regards to inertia to declare convergence.")
    print("Please specify the relative tolerance with regards to inertia to declare convergence. A good starting range could be between 0.0001 and 0.001, such as 0.0005.")
    tol = float_input(0.0005, SECTION[2], "Tolerance: ")
    print("Algorithm: The algorithm to use for the computation.")
    print("Please specify the algorithm to use for the computation. It is generally recommended to leave it as 'auto'.")
    print("Auto: selects 'elkan' for dense data and 'full' for sparse data. 'elkan' is generally faster on data with lower dimensionality, while 'full' is faster on data with higher dimensionality")
    algorithms = ["auto", "full", "elkan"]
    algorithm = str_input(algorithms, SECTION[2])
    hyper_parameters = {"n_clusters": n_clusters, "init": init, "max_iter": max_iters, "tol": tol, "algorithm": algorithm}
    return hyper_parameters


def plot_silhouette_diagram_kmeans(data: pd.DataFrame, cluster_labels: pd.DataFrame, cluster_centers_: np.ndarray, n_clusters: int, algorithm_name: str) -> None:
    """
    Draw the silhouette diagram for analysis.

    Parameters
    ----------
    data: pd.DataFrame (n_samples, n_components)
        Data for silhouette.

    cluster_labels: pd.DataFrame (n_samples,)
        Labels of each point.

    cluster_centers_: np.ndarray (n_samples,)
        Coordinates of cluster centers. If the algorithm stops before fully converging (see tol and max_iter), these will not be consistent with labels_.

    n_clusters: int
        Number of features seen during fit.

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
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

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
    silhouette_avg = silhouette_score(data, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(data, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

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
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(data.iloc[:, 0], data.iloc[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")

    # Labeling the clusters
    centers = cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")
    plt.suptitle(
        f"Silhouette analysis for KMeans clustering on sample data with n_clusters = %d - {algorithm_name}" % n_clusters,
        fontsize=14,
        fontweight="bold",
    )


def scatter2d(data: pd.DataFrame, cluster_labels: pd.DataFrame, algorithm_name: str) -> None:
    plt.figure()
    plt.subplot(111)
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=cluster_labels)

    plt.xlabel(f"{data.columns[0]}")
    plt.ylabel(f"{data.columns[1]}")
    plt.title(f"Cluster Data Bi-plot - {algorithm_name}")


def scatter3d(data: pd.DataFrame, cluster_labels: pd.DataFrame, algorithm_name: str) -> None:
    plt.figure()
    namelist = data.columns.values.tolist()
    fig = plt.figure(figsize=(12, 6), facecolor="w")
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9)

    ax = fig.add_subplot(121, projection="3d")
    ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], alpha=0.3, c="#FF0000", s=6)
    ax.set_xlabel(namelist[0])
    ax.set_ylabel(namelist[1])
    ax.set_zlabel(namelist[2])
    plt.grid(True)

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], c=cluster_labels, s=6, cmap=plt.cm.Paired, edgecolors="none")
    ax2.set_xlabel(namelist[0])
    ax2.set_ylabel(namelist[1])
    ax2.set_zlabel(namelist[2])
    plt.grid(True)
    ax.set_title(f"Base Data Tri-plot - {algorithm_name}")
    ax2.set_title(f"Cluster Data Tri-plot - {algorithm_name}")
