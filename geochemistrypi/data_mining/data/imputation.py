# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from rich import print
from sklearn.impute import SimpleImputer

from ..constants import SECTION
from .data_readiness import float_input


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
    method2option = {
        "Mean Value": "mean",
        "Median Value": "median",
        "Most Frequent Value": "most_frequent",
        "Constant(Specified Value)": "constant",
    }[method]
    if method2option == "constant":
        filled_value = float_input(0, SECTION[2], "@Specified Value: ")
        imp = SimpleImputer(missing_values=np.nan, strategy=method2option, fill_value=filled_value)
    else:
        imp = SimpleImputer(missing_values=np.nan, strategy=method2option)
    print(f"Successfully fill the missing values with the {method2option} value " f"of each feature column respectively.")
    return imp.fit_transform(data)
