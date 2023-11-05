from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print

from ....constants import SECTION
from ....data.data_readiness import float_input, num_input, str_input


def dbscan_manual_hyper_parameters() -> Dict:
    """Manually set hyperparameters.

    Returns
    -------
    hyper_parameters : dict
    """
    print("Eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.")
    print("Please specify the maximum distance. A good starting range could be between 0.1 and 1.0, such as 0.5.")
    eps = float_input(0.5, SECTION[2], "Eps: ")
    print("Min Samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.")
    print("Please specify the number of samples. A good starting range could be between 5 and 20, such as 5.")
    min_samples = num_input(SECTION[2], "Min Samples: ")
    print("Metric: The metric to use when calculating distance between instances in a feature array.")
    print("Please specify the metric to use when calculating distance between instances in a feature array. It is generally recommended to leave it as 'euclidean'.")
    metrics = ["euclidean", "manhattan", "chebyshev", "minkowski", "cosine", "correlation"]
    metric = str_input(metrics, SECTION[2])
    print("Algorithm: The algorithm to be used by the NearestNeighbors module to compute pointwise distances and find nearest neighbors.")
    print("Please specify the algorithm. It is generally recommended to leave it as 'auto'.")
    algorithms = ["auto", "ball_tree", "kd_tree", "brute"]
    algorithm = str_input(algorithms, SECTION[2])
    print("Leaf Size: Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree.")
    print("Please specify the leaf size. A good starting range could be between 10 and 30, such as 30.")
    leaf_size = num_input(SECTION[2], "Leaf Size: ")
    p = None
    if metric == "minkowski":
        print("P: The power of the Minkowski metric to be used to calculate distance between points.")
        print("Please specify the power of the Minkowski metric. A good starting range could be between 1 and 2, such as 2.")
        p = num_input(SECTION[2], "P: ")
    hyper_parameters = {
        "eps": eps,
        "min_samples": min_samples,
        "metric": metric,
        "algorithm": algorithm,
        "leaf_size": leaf_size,
        "p": p,
    }
    return hyper_parameters


def dbscan_result_plot(data: pd.DataFrame, trained_model: any, image_config: dict, algorithm_name: str) -> None:
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
    print("Estimated number of clusters: %d" % n_clusters_)
    unique_labels = set(labels)

    # create drawing canvas
    fig, ax = plt.subplots(figsize=(image_config["width"], image_config["height"]), dpi=image_config["dpi"])

    # draw the main content
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]
        class_member_mask = labels == k
        xy = data[class_member_mask & core_samples_mask]
        ax.plot(
            xy.iloc[:, 0],
            xy.iloc[:, 1],
            image_config["marker_angle"],
            markerfacecolor=tuple(col),
            markeredgecolor=image_config["edgecolor"],
            markersize=image_config["markersize1"],
            alpha=image_config["alpha1"],
        )
        xy = data[class_member_mask & ~core_samples_mask]
        ax.plot(
            xy.iloc[:, 0],
            xy.iloc[:, 1],
            image_config["marker_circle"],
            markerfacecolor=tuple(col),
            markeredgecolor=image_config["edgecolor"],
            markersize=image_config["markersize2"],
            alpha=image_config["alpha2"],
        )

    # automatically optimize picture layout structure
    fig.tight_layout()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x_adjustment = (xmax - xmin) * 0.1
    y_adjustment = (ymax - ymin) * 0.1
    ax.axis([xmin - x_adjustment, xmax + x_adjustment, ymin - y_adjustment, ymax + y_adjustment])

    # convert the font of the axes
    plt.tick_params(labelsize=image_config["labelsize"])  # adjust the font size of the axis label
    # plt.setp(ax.get_xticklabels(), rotation=image_config['xrotation'], ha=image_config['xha'],
    #          rotation_mode="anchor")  # axis label rotation Angle
    # plt.setp(ax.get_yticklabels(), rotation=image_config['rot'], ha=image_config['yha'],
    #          rotation_mode="anchor")  # axis label rotation Angle
    x1_label = ax.get_xticklabels()  # adjust the axis label font
    [x1_label_temp.set_fontname(image_config["axislabelfont"]) for x1_label_temp in x1_label]
    y1_label = ax.get_yticklabels()
    [y1_label_temp.set_fontname(image_config["axislabelfont"]) for y1_label_temp in y1_label]

    ax.set_title(
        label=algorithm_name,
        fontdict={
            "size": image_config["title_size"],
            "color": image_config["title_color"],
            "family": image_config["title_font"],
        },
        loc=image_config["title_location"],
        pad=image_config["title_pad"],
    )
