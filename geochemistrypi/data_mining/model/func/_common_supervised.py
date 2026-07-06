from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print
from sklearn.inspection import permutation_importance
from sklearn.tree import plot_tree

# <------
# Used by tree-based models including classification and regression besides XGBoost


def plot_decision_tree(trained_model: object, image_config: dict) -> None:
    """Plot the decision tree.

    Parameters
    ----------
    trained_model : object
        Trained model
    image_config : dict
        Image Configuration
    """
    # create drawing canvas
    fig, ax = plt.subplots(figsize=(image_config["width"], image_config["height"]), dpi=image_config["dpi"])

    # draw the main content
    # 修复：处理node_ids参数为None的情况
    node_ids = image_config["node_ids"]
    if node_ids is None:
        node_ids = False  # 设置默认值为False

    plot_tree(
        trained_model,
        max_depth=image_config["max_depth"],
        feature_names=image_config["feature_names"],
        class_names=image_config["class_names"],
        label=image_config["label"],
        filled=image_config["filled"],
        impurity=image_config["impurity"],
        node_ids=node_ids,  # 使用处理后的值
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
        # 首先确保coef是numpy数组
        if not isinstance(coef, np.ndarray):
            try:
                coef = np.array(coef)
            except (ValueError, TypeError):
                # 如果转换失败，尝试将其展平为标量列表
                coef = np.array([c for c in coef])

        # 同样确保intercept是合适的格式
        if isinstance(intercept, np.ndarray):
            intercept_flat = np.ravel(intercept)
            if len(intercept_flat) > 0:
                intercept = intercept_flat[0]
            else:
                intercept = 0
        else:
            intercept = np.around(intercept, decimals=3)

        # 检查coef是否为二维数组（多输出情况）
        if len(y_train.columns) == 1 and len(coef.shape) > 1 and coef.shape[0] > 1:
            # 这种情况是：虽然只有一个目标变量，但coef是二维数组（可能是MultiOutputRegressor的结果）
            for idx in range(coef.shape[0]):
                # 对每个输出单独处理
                coef_single = coef[idx]
                intercept_single = intercept[idx] if isinstance(intercept, np.ndarray) and intercept.size > 1 else intercept

                # 确保coef_single是标量
                terms = []
                for c, f in zip(coef_single, features_name):
                    if isinstance(c, np.ndarray) and c.size > 0:
                        c_val = c[0]
                    else:
                        c_val = c
                    if c_val != 0:
                        terms.append(("-" if c_val < 0 else "+") + " " + str(abs(c_val)) + f)
                    else:
                        terms.append("")

                # 确保coef_single[0]是标量
                if isinstance(coef_single[0], np.ndarray) and coef_single[0].size > 0:
                    coef_first_val = coef_single[0][0]
                else:
                    coef_first_val = coef_single[0]

                terms_first = (terms[0][2:] if coef_first_val > 0 else terms[0]).replace(" ", "")
                formula[f"y (output {idx+1}):"] = terms_first + " " + " ".join(terms[1:]) + (" - " if intercept_single < 0 else " + ") + str(abs(intercept_single))
                print(f"y (output {idx+1}) = ", formula[f"y (output {idx+1}):"])
        elif len(y_train.columns) == 1:
            # 单输出情况，但确保正确处理可能的数组系数
            coef_flat = np.ravel(coef)
            if len(coef_flat) > 0:
                coef = coef_flat

            # 检查intercept是否为标量
            if isinstance(intercept, np.ndarray):
                intercept_flat = np.ravel(intercept)
                if len(intercept_flat) > 0:
                    intercept = intercept_flat[0]
                else:
                    intercept = 0
            else:
                intercept = np.around(intercept, decimals=3)

            # 处理terms
            terms = []
            for c, f in zip(coef, features_name):
                if isinstance(c, np.ndarray) and c.size > 0:
                    c_val = c[0]
                else:
                    c_val = c
                if c_val != 0:
                    terms.append(("-" if c_val < 0 else "+") + " " + str(abs(c_val)) + f)
                else:
                    terms.append("")

            # 处理terms_first
            if len(coef) > 0:
                if isinstance(coef[0], np.ndarray) and coef[0].size > 0:
                    coef_first_val = coef[0][0]
                else:
                    coef_first_val = coef[0]

                terms_first = (terms[0][2:] if coef_first_val > 0 else terms[0]).replace(" ", "")
                formula["y:"] = terms_first + " " + " ".join(terms[1:]) + (" - " if intercept < 0 else " + ") + str(abs(intercept))
                print("y = ", formula["y:"])
        else:
            # 多目标变量情况
            # 确保coef和intercept是适合迭代的格式
            if len(coef.shape) == 1:
                # 如果coef是一维数组，转换为二维数组以便迭代
                coef = coef.reshape(1, -1)

            if not isinstance(intercept, np.ndarray) or intercept.ndim == 0:
                # 如果intercept是标量，转换为数组以便迭代
                intercept = np.array([intercept] * len(coef))

            for idx, (coef_temp, intercept_temp) in enumerate(zip(coef, intercept)):
                # 确保不超出y_train的列数
                if idx < len(y_train.columns):
                    terms_temp = []
                    for c, f in zip(coef_temp, features_name):
                        if isinstance(c, np.ndarray) and c.size > 0:
                            c_val = c[0]
                        else:
                            c_val = c
                        if c_val != 0:
                            terms_temp.append(("-" if c_val < 0 else "+") + " " + str(abs(c_val)) + f)
                        else:
                            terms_temp.append("")

                    if len(coef_temp) > 0:
                        if isinstance(coef_temp[0], np.ndarray) and coef_temp[0].size > 0:
                            coef_temp_first_val = coef_temp[0][0]
                        else:
                            coef_temp_first_val = coef_temp[0]

                        terms_temp_first = (terms_temp[0][2:] if coef_temp_first_val > 0 else terms_temp[0]).replace(" ", "")
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
