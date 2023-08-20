# -*- coding: utf-8 -*-
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


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
