from typing import Dict

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

from ....data.data_readiness import num_input, str_input
from ....global_variable import SECTION


def decision_tree_manual_hyper_parameters() -> Dict:
    """Manually set hyperparameters.

    Returns
    -------
    hype_parameters : dict
    """
    print(
        "Criterion: The function to measure the quality of a split. Supported criteria are “squared_error” for the mean squared error, which is equal to variance reduction"
        " as feature selection criterion and minimizes the L2 loss using the mean of each terminal node, “friedman_mse”, which uses mean squared error"
        " with Friedman’s improvement score for potential splits, “absolute_error” for the mean absolute error, which minimizes the L1 loss using the median of each terminal node, and “poisson”"
        " which uses reduction in Poisson deviance to find splits."
    )
    print("The default value is 'squared_error'. Optional criterions are 'friedman_mse', 'absolute_error' and 'poisson'.")
    criterions = ["squared_error", "friedman_mse", "absolute_error", "poisson"]
    criterion = str_input(criterions, SECTION[2])
    print("Max Depth: The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.")
    print("Please specify the maximum depth of the tree. A good starting range could be between 3 and 15, such as 4.")
    max_depth = num_input(SECTION[2], "@Max Depth: ")
    print("Min Samples Split: The minimum number of samples required to split an internal node.")
    print("Please specify the minimum number of samples required to split an internal node. A good starting range could be between 2 and 10, such as 3.")
    min_samples_split = num_input(SECTION[2], "@Min Samples Split: ")
    print("Min Samples Leaf: The minimum number of samples required to be at a leaf node.")
    print("Please specify the minimum number of samples required to be at a leaf node. A good starting range could be between 1 and 10, such as 2.")
    min_samples_leaf = num_input(SECTION[2], "@Min Samples Leaf: ")
    print("Max Features: The number of features to consider when looking for the best split.")
    print("Please specify the number of features to consider when looking for the best split. A good starting range could be between 1 and the total number of features in the dataset.")
    max_features = num_input(SECTION[2], "@Max Features: ")
    hyper_parameters = {
        "criterion": criterion,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
    }
    return hyper_parameters


def decision_tree_plot(trained_model: object, image_config: dict) -> None:
    """Drawing decision tree diagrams.

    Parameters
    ----------
    trained_model : sklearn algorithm model
        The sklearn algorithm model trained with X_train data.

    image_config : dict
        Image Configuration
    """

    # create drawing canvas
    fig, ax = plt.subplots(figsize=(image_config["width"], image_config["height"]), dpi=image_config["dpi"])

    # draw the main content
    plot_tree(
        trained_model,
        max_depth=image_config["max_depth"],
        feature_names=image_config["feature_names"],
        class_names=image_config["class_names"],
        label=image_config["label"],
        filled=image_config["filled"],
        impurity=image_config["impurity"],
        node_ids=image_config["node_ids"],
        proportion=image_config["proportion"],
        rounded=image_config["rounded"],
        precision=image_config["precision"],
        ax=image_config["ax"],
        fontsize=image_config["fontsize"],
    )

    # automatically optimize picture layout structure
    fig.tight_layout()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x_adjustment = (xmax - xmin) * 0.01
    y_adjustment = (ymax - ymin) * 0.01
    ax.axis([xmin - x_adjustment, xmax + x_adjustment, ymin - y_adjustment, ymax + y_adjustment])

    # convert the font of the axes
    # plt.tick_params(labelsize=image_config['labelsize'])  # adjust the font size of the axis label
    # plt.setp(ax.get_xticklabels(), rotation=image_config['xrotation'], ha=image_config['xha'],
    #          rotation_mode="anchor")  # axis label rotation Angle
    # plt.setp(ax.get_yticklabels(), rotation=image_config['rot'], ha=image_config['yha'],
    #          rotation_mode="anchor")  # axis label rotation Angle
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
