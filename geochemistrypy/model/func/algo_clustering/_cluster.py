from sklearn.metrics import silhouette_samples, silhouette_score
from utils.base import save_fig
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd


def plot_silhouette_diagram(data: pd.DataFrame, cluster_labels: pd.DataFrame,
                            cluster_centers_: np.ndarray, n_clusters: int, algorithm_name: str) -> None:
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
    ax2.scatter(
        data.iloc[:, 0], data.iloc[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

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
        f"Silhouette analysis for KMeans clustering on sample data with n_clusters = %d - {algorithm_name}"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )


def scatter2d(data: pd.DataFrame, cluster_labels: pd.DataFrame, algorithm_name: str) -> None:
    plt.subplot(111)
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c = cluster_labels)

    plt.xlabel(f"{data.columns[0]}")
    plt.ylabel(f"{data.columns[1]}")
    plt.title(f"Compositional Bi-plot - {algorithm_name}")

def scatter3d() -> None:
    pass
"""

def plot_2d_graph(self):
    print("")
    print("-----* 2D Scatter Plot *-----")
    # Get name
    namelist = self.X.columns.values.tolist()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('2D Scatter Plot')
    plt.xlabel(namelist[0])
    plt.ylabel(namelist[1])

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    if len(namelist) == 3:
        h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = self.X[namelist[0]].min() - 1, self.X[namelist[0]].max() + 1
        y_min, y_max = self.X[namelist[1]].min() - 1, self.X[namelist[1]].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        print(np.c_[xx.ravel(), yy.ravel()])
        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.imshow(
            Z,
            interpolation="nearest",
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap=plt.cm.Paired,
            aspect="auto",
            origin="lower",
        )

    plt.plot(self.X[namelist[0]], self.X[namelist[1]], "k.", markersize=2)
    # has wrong
    # c=self.X['clustering result'], marker='o', cmap=plt.cm.Paired)

    # Plot the centroids as a white X
    plt.scatter(
        self.model.cluster_centers_[:, [0]],
        self.model.cluster_centers_[:, [1]],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )
    plt.xlabel(namelist[0])
    plt.ylabel(namelist[1])
    ax1.scatter(self.X.iloc[:, [0]], self.X.iloc[:, [1]], c=self.X['clustering result'],
                marker='o', cmap=plt.cm.Paired)
    ax1.scatter(self.model.cluster_centers_[:, [0]], self.model.cluster_centers_[:, [1]],
                c=list(set(self.X['clustering result'])), marker='o', cmap=plt.cm.Paired, s=60)

    # plt.legend('x1')
    save_fig(f"Scatter Plot - {self.naming}", MODEL_OUTPUT_IMAGE_PATH)


def plot_3d_graph(self):
    print("")
    print("-----* Plot 3d Graph *-----")
    nameList = self.X.columns.values.tolist()
    fig = plt.figure(figsize=(12, 6), facecolor='w')
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9)

    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(self.X.iloc[:, [0]], self.X.iloc[:, [1]], self.X.iloc[:, [2]], alpha=0.3, c="#FF0000", s=6)
    plt.title('3D Scatter Plot')
    ax.set_xlabel(nameList[0])
    ax.set_ylabel(nameList[1])
    ax.set_zlabel(nameList[1])
    plt.grid(True)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(self.X.iloc[:, [0]], self.X.iloc[:, [1]], self.X.iloc[:, [2]],
                c=self.X['clustering result'], s=6, cmap=plt.cm.Paired, edgecolors='none')
    ax2.set_xlabel(nameList[0])
    ax2.set_ylabel(nameList[1])
    ax2.set_zlabel(nameList[1])
    plt.grid(True)
    save_fig(f"Plot 3d Graph - {self.naming}", MODEL_OUTPUT_IMAGE_PATH)
"""