# -*- coding: utf-8 -*-
from typing import Dict

import numpy as np
import pandas as pd
import xgboost
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance

from ....data.data_readiness import float_input, num_input
from ....global_variable import SECTION


def xgboost_manual_hyper_parameters() -> Dict:
    """Manually set hyperparameters.

    Returns
    -------
    hyper_parameters : dict
    """
    print("N Estimators: The number of trees in the forest.")
    print("Please specify the number of trees in the forest. A good starting range could be between 50 and 500, such as 100.")
    n_estimators = num_input(SECTION[2], "@N Estimators: ")
    print("Learning Rate: It controls the step-size in updating the weights. It shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators.")
    print("Please specify the initial learning rate of Xgboost, such as 0.1.")
    learning_rate = float_input(0.01, SECTION[2], "@Learning Rate: ")
    print("Max Depth: The maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.")
    print("Please specify the maximum depth of a tree. A good starting range could be between 3 and 15, such as 4.")
    max_depth = num_input(SECTION[2], "@Max Depth: ")
    print("Subsample: The fraction of samples to be used for fitting the individual base learners.")
    print("Please specify the fraction of samples to be used for fitting the individual base learners. A good starting range could be between 0.5 and 1.0, such as 0.8.")
    subsample = float_input(1, SECTION[2], "@Subsample: ")
    print("Colsample Bytree: The fraction of features to be used for fitting the individual base learners.")
    print("Please specify the fraction of features to be used for fitting the individual base learners. A good starting range could be between 0.5 and 1.0, such as 1.")
    colsample_bytree = float_input(1, SECTION[2], "@Colsample Bytree: ")
    print("Alpha: L1 regularization term on weights.")
    print("Please specify the L1 regularization term on weights. A good starting range could be between 0 and 1.0, such as 0.")
    alpha = float_input(0, SECTION[2], "@Alpha: ")
    print("Lambda: L2 regularization term on weights.")
    print("Please specify the L2 regularization term on weights. A good starting range could be between 0 and 1.0, such as 1.")
    lambd = float_input(1, SECTION[2], "@Lambda: ")
    hyper_parameters = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "alpha": alpha,
        "lambd": lambd,
    }
    return hyper_parameters


def feature_importance(X: pd.DataFrame, trained_model: object, image_config: dict) -> None:
    """Plot the Feature Importance.

    Parameters
    ----------
    X: pd.DataFrame (n_samples, n_components)
        The input data.

    trained_model : sklearn algorithm model
        The sklearn algorithm model trained with X_train data.
    """
    columns_name = X.columns

    # print the feature importance value orderly
    for feature_name, score in zip(list(columns_name), trained_model.feature_importances_):
        print(feature_name, ":", score)
    plt.figure(figsize=(image_config["width"], image_config["height"]), dpi=image_config["dpi"])
    plt.bar(range(len(columns_name)), trained_model.feature_importances_, tick_label=columns_name)


def histograms_feature_weights(X: pd.DataFrame, trained_model: object, image_config: dict) -> None:
    """Plot the Feature Importance, histograms present feature weights for XGBoost predictions.

    Parameters
    ----------
    X: pd.DataFrame (n_samples, n_components)
        The input data.

    trained_model : sklearn algorithm model
        The sklearn algorithm model trained with X_train data.
    """

    # columns_name = X.columns
    plt.rcParams["figure.figsize"] = (image_config["width"], image_config["height"])
    xgboost.plot_importance(trained_model)


def permutation_importance_(X: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, trained_model: object, image_config: dict) -> None:
    """Plot the permutation Importance.

    Parameters
    ----------
    X: pd.DataFrame (n_samples, n_components)
        The input data.

    X_test : pd.DataFrame (n_samples, n_components)
        The testing target values.

    y_test : pd.DataFrame (n_samples, n_components)
    The testing target values.

    trained_model : sklearn algorithm model
        The sklearn algorithm model trained with X_train data.
    """

    columns_name = X.columns
    plt.figure(figsize=(image_config["width"], image_config["height"]), dpi=image_config["dpi"])
    result = permutation_importance(trained_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()
    plt.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        labels=np.array(columns_name),
    )
