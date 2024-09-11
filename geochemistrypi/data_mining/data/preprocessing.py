# -*- coding: utf-8 -*-
from typing import List, Optional

import numpy as np
import pandas as pd
from rich import print
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import GenericUnivariateSelect, SelectKBest, f_classif, f_regression
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .data_readiness import show_data_columns


class MeanNormalScaler(BaseEstimator, TransformerMixin):
    """Custom Scikit-learn transformer for mean normalization.

    MeanNormalization involves subtracting the mean of each feature from the feature values
    and then dividing by the range (maximum value minus minimum value) of that feature.

    The transformation is given by:

        X_scaled = (X - X.mean()) / (X.max() - X.min())

    """

    def __init__(self: object, copy: bool = True):
        self.copy = copy
        self.mean_ = None
        self.scale_ = None

    def fit(self: object, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> object:
        """
        Compute the mean and range (max - min) for each feature.

        Parameters
        ----------
        X : pd.DataFrame
            The input dataframe where each column represents a feature.

        y : pd.DataFrame, optional (default: None)
            Ignored.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        return self

    def transform(self: object, X: pd.DataFrame, y: Optional[pd.DataFrame] = None, copy: bool = None) -> np.ndarray:
        """
        Apply mean normalization to the data.

        Parameters
        ----------
        X : pd.DataFrame
            The input dataframe where each column represents a feature.

        y : pd.DataFrame, optional (default: None)
            Ignored.

        copy : bool, optional (default: None)
            Copy the input X or not.

        Returns
        -------
        X_tr : np.ndarray
            The normalized data.
        """
        copy = copy if copy is not None else self.copy
        X = X if not self.copy else X.copy()
        return (X - self.mean_) / self.scale_

    def inverse_transform(self: object, X: pd.DataFrame) -> np.ndarray:
        """
        Reverse the mean normalization transformation.

        Parameters
        ----------
        X : pd.DataFrame
            The input dataframe where each column represents a feature.

        Returns
        -------
        X_tr : np.ndarray
            The original data.
        """
        X = X if not self.copy else X.copy()
        return X * self.scale_ + self.mean_


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
    elif method[method_idx] == "Mean Normalization":
        scaler = MeanNormalScaler()
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
    print("-- Original Features --")
    show_data_columns(X.columns)

    features_num = len(X.columns)
    print(f"The original number of features is {features_num}, and your input must be less than {features_num}.")
    features_retain_num = int(input("Please enter the number of features to retain.\n" "@input: "))

    if feature_selection_task == 1:
        score_func = f_regression
    elif feature_selection_task == 2:
        score_func = f_classif

    if method[method_idx] == "Generic Univariate Select":
        selector = GenericUnivariateSelect(score_func=score_func, mode="k_best", param=features_retain_num)
    elif method[method_idx] == "Select K Best":
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
