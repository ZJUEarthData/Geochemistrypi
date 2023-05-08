# -*- coding: utf-8 -*-
from typing import List

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def feature_scaler(X: pd.DataFrame, method: List[str], method_idx: int) -> pd.DataFrame:
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
    X_scaled : np.ndarray
        The dataset after imputing.
    """
    if method[method_idx] == "Min-max Scaling":
        scaler = MinMaxScaler()
        return scaler.fit_transform(X)
    elif method[method_idx] == "Standardization":
        scaler = StandardScaler()
        return scaler.fit_transform(X)
