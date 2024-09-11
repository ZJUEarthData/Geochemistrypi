# -*- coding: utf-8 -*-
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print

from ....constants import SECTION
from ....data.data_readiness import float_input, num_input


def local_outlier_factor_manual_hyper_parameters() -> Dict:
    """Manually set hyperparameters.

    Returns
    -------
    hyper_parameters : dict
    """
    print("N neighbors: The number of neighbors to use.")
    print("Please specify the number of neighbors. A good starting range could be between 10 and 50, such as 20.")
    n_neighbors = num_input(SECTION[2], "@N Neighbors: ")
    print("Leaf size: The leaf size used in the ball tree or KD tree.")
    print("Please specify the leaf size. A good starting range could be between 20 and 50, such as 30.")
    leaf_size = num_input(SECTION[2], "@Leaf Size: ")
    print("P: The power parameter for the Minkowski metric.")
    print("Please specify the power parameter. When p = 1, this is equivalent to using manhattan_distance, and when p = 2 euclidean_distance is applied. For arbitrary p, minkowski_distance is used.")
    p = float_input(2.0, SECTION[2], "@P: ")
    print("Contamination: The amount of contamination of the data set.")
    print("Please specify the contamination of the data set. A good starting range could be between 0.1 and 0.5, such as 0.3.")
    contamination = float_input(0.3, SECTION[2], "@Contamination: ")
    print("N jobs: The number of parallel jobs to run.")
    print("Please specify the number of jobs. Use -1 to use all available CPUs, 1 for no parallelism, or specify the number of CPUs to use. A good starting value is 1.")
    n_jobs = num_input(SECTION[2], "@N Jobs: ")
    hyper_parameters = {
        "n_neighbors": n_neighbors,
        "leaf_size": leaf_size,
        "p": p,
        "contamination": contamination,
        "n_jobs": n_jobs,
    }
    return hyper_parameters


def plot_lof_scores(columns_name: pd.Index, lof_scores: np.ndarray, image_config: dict) -> pd.DataFrame:
    """Draw the LOF scores bar diagram.

    Parametersplot_lof_scores
    ----------
    columns_name : pd.Index
        The name of the columns.

    lof_scores : np.ndarray
        The LOF scores values.

    image_config : dict
        The configuration of the image.

    Returns
    -------
    lof_scores_df : pd.DataFrame
        The LOF scores values.
    """
    # create drawing canvas
    fig, ax = plt.subplots(figsize=(image_config["width"], image_config["height"]), dpi=image_config["dpi"])

    # # print the LOF scores value orderly
    # for feature_name, score in zip(list(columns_name), lof_scores):
    #     print(feature_name, ":", score)

    # draw the main content
    lof_scores_df = pd.DataFrame({"Feature": columns_name, "LOF Score": lof_scores})
    lof_scores_df = lof_scores_df.sort_values(["LOF Score"], ascending=True)
    lof_scores_df["LOF Score"] = lof_scores_df["LOF Score"].astype(float)
    lof_scores_df = lof_scores_df.sort_values(["LOF Score"])
    lof_scores_df.set_index("Feature", inplace=True)
    lof_scores_df.plot.barh(alpha=image_config["alpha2"], rot=0)

    # automatically optimize picture layout structure
    fig.tight_layout()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x_adjustment = (xmax - xmin) * 0.01
    y_adjustment = (ymax - ymin) * 0.01
    ax.axis([xmin - x_adjustment, xmax + x_adjustment, ymin - y_adjustment, ymax + y_adjustment])

    # convert the font of the axes
    x1_label = ax.get_xticklabels()  # adjust the axis label font
    [x1_label_temp.set_fontname(image_config["axislabelfont"]) for x1_label_temp in x1_label]
    y1_label = ax.get_yticklabels()
    [y1_label_temp.set_fontname(image_config["axislabelfont"]) for y1_label_temp in y1_label]

    ax.set_title(
        label=image_config["title_label"],
        fontdict={
            "size": image_config["title_size"],
            "color": image_config["title_color"],
            "family": image_config["title_font"],
        },
        loc=image_config["title_location"],
        pad=image_config["title_pad"],
    )

    return lof_scores_df
