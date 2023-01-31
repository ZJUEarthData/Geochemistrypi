# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def imputer(data: pd.DataFrame, method: str) -> np.ndarray:
    """Apply imputation on missing values.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset with missing values.

    method : str
        The imputation method.

    Returns
    -------
    data_imputed : np.ndarray
        The dataset after imputing.
    """
    method2option = {'Mean': 'mean', 'Median': 'median', 'Most Frequent': 'most_frequent'}[method]
    imp = SimpleImputer(missing_values=np.nan, strategy=method2option)
    print(f"Successfully fill the missing values with the {method2option} value "
          f"of each feature column respectively.")
    return imp.fit_transform(data)
