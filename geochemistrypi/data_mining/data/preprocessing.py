# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from rich import print
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import GenericUnivariateSelect, SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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


@dataclass
class FittedSupervisedPreprocessor:
    """A supervised preprocessor fitted exclusively on training data."""

    pipeline: Optional[Pipeline]
    input_features: List[str]
    feature_names: List[str]
    imputation_config: Dict[str, Dict[str, Any]]
    feature_scaling_config: Dict[str, Dict[str, Any]]
    feature_selection_config: Dict[str, Dict[str, Any]]

    @property
    def transformer_config(self) -> Dict[str, Dict[str, Any]]:
        """Return the ordered transformer configuration used by the pipeline."""
        config: Dict[str, Dict[str, Any]] = {}
        config.update(self.imputation_config)
        config.update(self.feature_scaling_config)
        config.update(self.feature_selection_config)
        return config

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform a feature frame while preserving its index and feature names."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if list(X.columns) != self.input_features:
            raise ValueError("Feature columns must match the training columns in the same order. " f"Expected {self.input_features}, received {list(X.columns)}.")
        if self.pipeline is None:
            return X.copy()
        transformed = self.pipeline.transform(X)
        return pd.DataFrame(transformed, index=X.index, columns=self.feature_names)


def _make_imputer(method: str, fill_value: Optional[float] = None) -> SimpleImputer:
    strategy = {
        "Mean Value": "mean",
        "Median Value": "median",
        "Most Frequent Value": "most_frequent",
        "Constant(Specified Value)": "constant",
    }.get(method)
    if strategy is None:
        raise ValueError(f"Unsupported imputation method: {method}")
    if strategy == "constant":
        return SimpleImputer(missing_values=np.nan, strategy=strategy, fill_value=fill_value)
    return SimpleImputer(missing_values=np.nan, strategy=strategy)


def _make_scaler(method: str) -> BaseEstimator:
    scaler_by_method = {
        "Min-max Scaling": MinMaxScaler,
        "Standardization": StandardScaler,
        "Mean Normalization": MeanNormalScaler,
    }
    try:
        return scaler_by_method[method]()
    except KeyError as exc:
        raise ValueError(f"Unsupported feature scaling method: {method}") from exc


def _make_selector(task: str, method: str, features_to_retain: int) -> BaseEstimator:
    if task == "regression":
        score_func = f_regression
    elif task == "classification":
        score_func = f_classif
    else:
        raise ValueError("task must be either 'regression' or 'classification'.")

    if method == "Generic Univariate Select":
        return GenericUnivariateSelect(score_func=score_func, mode="k_best", param=features_to_retain)
    if method == "Select K Best":
        return SelectKBest(score_func=score_func, k=features_to_retain)
    raise ValueError(f"Unsupported feature selection method: {method}")


def fit_supervised_preprocessor(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    *,
    task: str,
    imputation_method: Optional[str] = None,
    imputation_fill_value: Optional[float] = None,
    scaling_method: Optional[str] = None,
    selection_method: Optional[str] = None,
    features_to_retain: Optional[int] = None,
) -> FittedSupervisedPreprocessor:
    """Fit imputation, scaling, and selection using training rows only.

    Call this function only after the train/test split. The returned object can
    then transform the training, test, full, and application feature frames
    with the same fitted statistics.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if X_train.empty or len(X_train.columns) == 0:
        raise ValueError("X_train must contain at least one row and one feature.")
    if len(X_train.columns) != len(set(X_train.columns)):
        raise ValueError("X_train feature names must be unique.")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must contain the same number of rows.")
    if task not in {"regression", "classification"}:
        raise ValueError("task must be either 'regression' or 'classification'.")
    if selection_method is not None:
        if features_to_retain is None:
            raise ValueError("features_to_retain is required when feature selection is enabled.")
        if not 1 <= features_to_retain <= len(X_train.columns):
            raise ValueError(f"features_to_retain must be between 1 and {len(X_train.columns)}.")
        if isinstance(y_train, pd.DataFrame) and y_train.shape[1] != 1:
            raise ValueError("Univariate feature selection requires exactly one target column.")

    steps = []
    imputer = _make_imputer(imputation_method, imputation_fill_value) if imputation_method else None
    scaler = _make_scaler(scaling_method) if scaling_method else None
    selector = _make_selector(task, selection_method, features_to_retain) if selection_method else None
    if imputer is not None:
        steps.append(("imputer", imputer))
    if scaler is not None:
        steps.append(("scaler", scaler))
    if selector is not None:
        steps.append(("selector", selector))

    input_features = list(X_train.columns)
    if not steps:
        return FittedSupervisedPreprocessor(
            pipeline=None,
            input_features=input_features,
            feature_names=input_features.copy(),
            imputation_config={},
            feature_scaling_config={},
            feature_selection_config={},
        )

    target = y_train.iloc[:, 0] if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1 else y_train
    pipeline = Pipeline(steps)
    pipeline.fit(X_train, target)

    feature_names = input_features
    if selector is not None:
        feature_names = list(selector.get_feature_names_out(input_features))

    return FittedSupervisedPreprocessor(
        pipeline=pipeline,
        input_features=input_features,
        feature_names=feature_names,
        imputation_config={type(imputer).__name__: imputer.get_params()} if imputer is not None else {},
        feature_scaling_config={type(scaler).__name__: scaler.get_params()} if scaler is not None else {},
        feature_selection_config={type(selector).__name__: selector.get_params()} if selector is not None else {},
    )


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
    scaler = _make_scaler(method[method_idx])
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
