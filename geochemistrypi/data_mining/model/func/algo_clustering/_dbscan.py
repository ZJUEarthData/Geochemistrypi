import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def dbscan_result_plot(data: pd.DataFrame, trained_model: any, algorithm_name: str) -> None:
    """
    Draw the clustering result diagram for analysis.

    Parameters
    ----------
    data: pd.DataFrame (n_samples, n_components)
        Data for silhouette.

    trained_model: any
        The algorithm which to be used

    algorithm_name : str
        the name of the algorithm

    References
    ----------
    The DBSCAN algorithm is deterministic, always generating the same clusters when given the same data in the same order.

    https://scikit-learn.org/stable/modules/clustering.html/dbscan

    """
    db = trained_model.fit(data)
    labels = trained_model.labels_
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print("Estimated number of clusters: %d\n" % n_clusters_)
    unique_labels = set(labels)
    plt.figure()
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]
        class_member_mask = labels == k
        xy = data[class_member_mask & core_samples_mask]
        plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="w", markersize=5.5)
        xy = data[class_member_mask & ~core_samples_mask]
        plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="w", markersize=3,alpha=0.75)
    plt.title("cluster Numbers =" + str(n_clusters_), y=-0.2)
    plt.title(f'Plot - {algorithm_name} - 2D')