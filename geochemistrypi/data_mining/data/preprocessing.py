# -*- coding: utf-8 -*-
from typing import List

import numpy as np
import pandas as pd
from rich import print
from sklearn.feature_selection import GenericUnivariateSelect, SelectKBest, f_classif, f_regression
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .data_readiness import show_data_columns


def feature_scaler(X: pd.DataFrame, method: List[str], method_idx: int) -> tuple[dict, np.ndarray]:
    """Apply feature scaling methods.

    Parameters
    ----------
    X : pd.DataFrame
        The dataset.

    method : str
        The feature scaling methods.

    method_idx : int
        The index of methods.

    Returns
    -------
    feature_scaling_config : dict
        The feature scaling configuration.

    X_scaled : np.ndarray
        The dataset after imputing.
    """
    if method[method_idx] == "Min-max Scaling":
        scaler = MinMaxScaler()
    elif method[method_idx] == "Standardization":
        scaler = StandardScaler()
    try:
        X_scaled = scaler.fit_transform(X)
    except ValueError:
        print("The selected feature scaling method is not applicable to the dataset!")
        print("Please check the dataset to find the reason.")
    feature_scaling_config = {type(scaler).__name__: scaler.get_params()}
    return feature_scaling_config, X_scaled


def feature_selector(X: pd.DataFrame, y: pd.DataFrame, feature_selection_task: int, method: List[str], method_idx: int) -> tuple[dict, pd.DataFrame]:
    """Apply feature selection methods.

    Parameters
    ----------
    X : pd.DataFrame
        The feature dataset.

    y : pd.DataFrame
        The label dataset.

    feature_selection_task : int
        Feature selection for regression or classification tasks.

    method : str
        The feature selection methods.

    method_idx : int
        The index of methods.

    Returns
    -------
    feature_selection_config : dict
        The feature selection configuration.

    X_selected : pd.DataFrame
        The feature dataset after selecting.
    """
    print("--Original Features-")
    show_data_columns(X.columns)

    features_num = len(X.columns)
    print(f"The original number of features is {features_num}, and your input must be less than {features_num}.")
    features_retain_num = int(input("Please enter the number of features to retain.\n" "@input: "))

    if feature_selection_task == 1:
        score_func = f_regression
    elif feature_selection_task == 2:
        score_func = f_classif

    if method[method_idx] == "GenericUnivariateSelect":
        selector = GenericUnivariateSelect(score_func=score_func, mode="k_best", param=features_retain_num)
    elif method[method_idx] == "SelectKBest":
        selector = SelectKBest(score_func=score_func, k=features_retain_num)

    try:
        selector.fit(X, y)
        features_selected = selector.get_feature_names_out()
        X = X[features_selected]
    except ValueError:
        print("The selected feature selection method is not applicable to the dataset!")
        print("Please check the dataset to find the reason.")

    feature_selection_config = {type(selector).__name__: selector.get_params()}
    return feature_selection_config, X
