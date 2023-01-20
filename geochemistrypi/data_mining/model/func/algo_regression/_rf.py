# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance


def feature_importance__(X: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, trained_model: object) -> None:
    """Plot the Feature Importance.

    Parameters
    ----------
    X : pd.DataFrame (n_samples, n_components)
        The input data.

    X_test : pd.DataFrame (n_samples, n_components)
        The testing target values.

    y_test : pd.DataFrame (n_samples, n_components)
        The testing target values.

    trained_model : sklearn algorithm model
        The sklearn algorithm model trained with X_train data.
    """
    columns_name = X.columns
    feature_importance = trained_model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(pos, feature_importance[sorted_idx], align="center")

    plt.yticks(pos, np.array(columns_name)[sorted_idx])
    plt.title("Feature Importance ")

    result = permutation_importance(
        trained_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    )
    sorted_idx = result.importances_mean.argsort()
    plt.subplot(1, 2, 2)
    plt.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        labels=np.array(columns_name)[sorted_idx],
    )
    plt.title("Permutation Importance ")
