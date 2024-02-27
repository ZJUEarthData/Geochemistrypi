from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print
from sklearn.inspection import permutation_importance
from sklearn.tree import plot_tree

# <------
# Used by tree-based models including classification and regression besides XGBoost


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


# Used by tree-based models including classification and regression besides XGBoost
# ------>

# <------
# Used by tree-based models, like, random forest, extra-trees, xgboost including classification and regression


def plot_feature_importance(columns_name: pd.Index, feature_importance: np.ndarray, image_config: dict) -> pd.DataFrame:
    """Draw the feature importance bar diagram.

    Parameters
    ----------
    columns_name : pd.Index
        The name of the columns.

    feature_importance : np.ndarray
        The feature importance values.

    image_config : dict
        The configuration of the image.

    Returns
    -------
    importance : pd.DataFrame
        The feature importance values.
    """
    # create drawing canvas
    fig, ax = plt.subplots(figsize=(image_config["width"], image_config["height"]), dpi=image_config["dpi"])

    # print the feature importance value orderly
    for feature_name, score in zip(list(columns_name), feature_importance):
        print(feature_name, ":", score)

    # draw the main content
    importance = pd.DataFrame({"Feature": columns_name, "Importance": feature_importance})
    importance = importance.sort_values(["Importance"], ascending=True)
    importance["Importance"] = (importance["Importance"]).astype(float)
    importance = importance.sort_values(["Importance"])
    importance.set_index("Feature", inplace=True)
    importance.plot.barh(alpha=image_config["alpha2"], rot=0)

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

    return importance


# Used by tree-based models, like, random forest, extra-trees, xgboost including classification and regression
# ------>

# <------
# Used by linear models including classification and regression


def show_formula(coef: np.ndarray, intercept: np.ndarray, features_name: np.ndarray, regression_classification: str, y_train: pd.DataFrame) -> Dict:
    """Show the formula of linear models.

    Parameters
    ----------
    coef : np.ndarray
        Coefficient of the features in the decision function.

    intercept : np.ndarray or float
        Independent term in decision function.

    features_name : np.ndarray
        Name of the features.

    regression_classification : str
        Indicates whether it's a regression or classification model.

    y_train : pd.DataFrame
        The train label data.

    Returns
    -------
    formula : dict
        The formula of linear models.
    """
    formula = {}

    if regression_classification == "Regression":
        if len(y_train.columns) == 1:  # Single target
            coef = np.around(coef, decimals=3)[0]
            # Check if intercept is a scalar
            if isinstance(intercept, np.ndarray):
                intercept = np.around(intercept, decimals=3)[0]
            else:
                intercept = np.around(intercept, decimals=3)

            terms = [("-" if c < 0 else "+") + " " + str(abs(c)) + f if c != 0 else "" for c, f in zip(coef, features_name)]
            terms_first = (terms[0][2:] if coef[0] > 0 else terms[0]).replace(" ", "")
            formula["y:"] = terms_first + " " + " ".join(terms[1:]) + (" - " if intercept < 0 else " + ") + str(abs(intercept))
            print("y = ", formula["y:"])

        else:  # Multiple targets
            coef = np.around(coef, decimals=3)
            intercept = np.around(intercept, decimals=3)

            for idx, (coef_temp, intercept_temp) in enumerate(zip(coef, intercept)):
                terms_temp = [("-" if c < 0 else "+") + " " + str(abs(c)) + f if c != 0 else "" for c, f in zip(coef_temp, features_name)]
                terms_temp_first = (terms_temp[0][2:] if coef_temp[0] > 0 else terms_temp[0]).replace(" ", "")
                formula["y (" + y_train.columns[idx] + ") = "] = terms_temp_first + " " + " ".join(terms_temp[1:]) + (" - " if intercept_temp < 0 else " + ") + str(abs(intercept_temp))
                print("y (" + y_train.columns[idx] + ") = ", formula["y (" + y_train.columns[idx] + ") = "])

    elif regression_classification == "Classification":
        if coef.shape[0] == 1:  # Binary classification
            coef = np.around(coef, decimals=3)[0]
            # Check if intercept is a scalar
            if isinstance(intercept, np.ndarray):
                intercept = np.around(intercept, decimals=3)[0]
            else:
                intercept = np.around(intercept, decimals=3)

            terms = [("-" if c < 0 else "+") + " " + str(abs(c)) + f if c != 0 else "" for c, f in zip(coef, features_name)]
            terms_first = (terms[0][2:] if coef[0] > 0 else terms[0]).replace(" ", "")
            formula["y:"] = terms_first + " " + " ".join(terms[1:]) + (" - " if intercept < 0 else " + ") + str(abs(intercept))
            print("y = ", formula["y:"])

        else:  # Multiclass classification
            label_min = int(y_train.min())
            coef = np.around(coef, decimals=3)
            intercept = np.around(intercept, decimals=3)

            for idx, (coef_temp, intercept_temp) in enumerate(zip(coef, intercept), label_min):  # The range of idx is between label_min and label_max.
                terms_temp = [("-" if c < 0 else "+") + " " + str(abs(c)) + f if c != 0 else "" for c, f in zip(coef_temp, features_name)]
                terms_temp_first = (terms_temp[0][2:] if coef_temp[0] > 0 else terms_temp[0]).replace(" ", "")
                formula[f"y (label={idx}):"] = terms_temp_first + " " + " ".join(terms_temp[1:]) + (" - " if intercept_temp < 0 else " + ") + str(abs(intercept_temp))
                print(f"y (label={idx}) = ", formula[f"y (label={idx}):"])

    return formula


# Used by linear models including classification and regression
# ------>

# <------
# Used regresssion and classification models


def plot_permutation_importance(X_test: pd.DataFrame, y_test: pd.DataFrame, trained_model: object, image_config: dict) -> tuple:
    """Plot the permutation Importance.

    Parameters
    ----------
    X_test : pd.DataFrame (n_samples, n_components)
        The testing target values.

    y_test : pd.DataFrame (n_samples, n_components)
    The testing target values.

    trained_model : sklearn algorithm model
        The sklearn algorithm model trained with X_train data.

    image_config : dict
        Image Configuration

    Returns
    -------
    result.importances_mean : ndarray
        The mean of feature importance over repetitions.

    result.importances_std : ndarray
        The standard deviation over repetitions.

    result.importances : ndarray
        The matrix of all feature importance values.
    """

    # create drawing canvas
    fig, ax = plt.subplots(figsize=(image_config["width"], image_config["height"]), dpi=image_config["dpi"])

    columns_name = X_test.columns
    result = permutation_importance(trained_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()
    ax.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        labels=np.array(columns_name),
    )

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

    return result.importances_mean, result.importances_std, result.importances


# Used regresssion and classification models
# ------>
