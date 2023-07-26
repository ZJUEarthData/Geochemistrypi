from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import plot_tree

# <------
# Used by decsion tree including classification and regression


def plot_decision_tree(trained_model: object, image_config: Dict) -> None:
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


# Used by decsion tree including classification and regression
# ------>

# <------
# Used by decsion tree, svm including classification and regression


def plot_2d_decision_boundary(X: pd.DataFrame, X_test: pd.DataFrame, trained_model: object, image_config: Dict) -> None:
    """Plot the decision boundary of the trained model with the testing data set below.

    Parameters
    ----------
    X : pd.DataFrame (n_samples, n_components)
        The complete feature data.

    X_test : pd.DataFrame (n_samples, n_components)
        The testing feature data.

    trained_model : sklearn algorithm model
        The sklearn algorithm model trained with X_train data.

    image_config : dict
        The configuration of the image.
    """

    # Prepare the data for contour plot

    x0s = np.arange(X.iloc[:, 0].min(), X.iloc[:, 0].max(), (X.iloc[:, 0].max() - X.iloc[:, 0].min()) / 50)
    x1s = np.arange(X.iloc[:, 1].min(), X.iloc[:, 1].max(), (X.iloc[:, 1].max() - X.iloc[:, 1].min()) / 50)
    x0, x1 = np.meshgrid(x0s, x1s)
    input_array = np.c_[x0.ravel(), x1.ravel()]
    labels = trained_model.predict(input_array).reshape(x0.shape)

    # Use the code below when the dimensions of X are greater than 2

    # create drawing canvas
    fig, ax = plt.subplots(figsize=(image_config["width"], image_config["height"]), dpi=image_config["dpi"])
    ax.patch.set_facecolor("cornsilk")

    # draw the main content
    ax.contourf(x0, x1, labels, cmap=image_config["cmap2"], alpha=image_config["alpha1"])
    ax.scatter(
        X_test.iloc[:, 0],
        X_test.iloc[:, 1],
        alpha=image_config["alpha2"],
        marker=image_config["marker_angle"],
        cmap=image_config["cmap"],
    )
    # automatically optimize picture layout structure
    fig.tight_layout()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x_adjustment = (xmax - xmin) * 0.01
    y_adjustment = (ymax - ymin) * 0.01
    ax.axis([xmin - x_adjustment, xmax + x_adjustment, ymin - y_adjustment, ymax + y_adjustment])

    # convert the font of the axes
    ax.set_xlabel(X.columns[0])
    ax.set_ylabel(X.columns[1])
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
        label=image_config["title_label"],
        fontdict={
            "size": image_config["title_size"],
            "color": image_config["title_color"],
            "family": image_config["title_font"],
        },
        loc=image_config["title_location"],
        pad=image_config["title_pad"],
    )


# Used by decsion tree, svm including classification and regression
# ------>
