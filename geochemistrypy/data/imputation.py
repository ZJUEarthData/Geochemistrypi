# -*- coding: utf-8 -*-
import numpy as np
from sklearn.impute import SimpleImputer


def imputer(data, method):
    imp = SimpleImputer(missing_values=np.nan, strategy=method)
    print(f"Successfully fill the missing values with the {method} value "
          f"of each feature column respectively.")
    return imp.fit_transform(data)