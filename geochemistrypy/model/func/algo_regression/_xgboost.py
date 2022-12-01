# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xgboost
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance
from global_variable import MODEL_OUTPUT_IMAGE_PATH
from utils.base import save_fig


def feature_importance(X: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, algorithm_name: str,):
    """Plot the Feature Importance.
    Parameters
    ----------
    X: pd.DataFrame (n_samples, n_components)
        The input data.

    X_test : pd.DataFrame (n_samples, n_components)
        The testing target values.

    y_test : pd.DataFrame (n_samples, n_components)
    The testing target values.

    algorithm_name : str
        The name of the algorithm model.
    """

    print("-----* Feature Importance *-----")
    columns_name = X.columns

    # print the feature importance value orderly
    for feature_name, score in zip(list(columns_name),algorithm_name.feature_importances_):
        print(feature_name, ":", score)

    # histograms present feature weights for XGBoost predictions
    plt.figure(figsize=(40, 6))
    plt.bar(range(len(columns_name)), algorithm_name.feature_importances_, \
            tick_label=columns_name)
    save_fig("xgb_feature_importance", MODEL_OUTPUT_IMAGE_PATH)

    # feature importance map ranked by importance
    plt.rcParams["figure.figsize"] = (14, 8)
    xgboost.plot_importance(algorithm_name)
    save_fig("xgb_feature_importance_score", MODEL_OUTPUT_IMAGE_PATH)

    # Permutation importance
    print("-----* Permutation Importance  *-----")
    plt.figure(figsize=(10, 8))
    result = permutation_importance(algorithm_name, X_test,
                                    y_test, n_repeats=10,
                                    random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()
    plt.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        labels=np.array(columns_name),
    )
    save_fig("xgb_feature_importance_T", MODEL_OUTPUT_IMAGE_PATH)

