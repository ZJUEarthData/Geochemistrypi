# -*- coding: utf-8 -*-
import numpy as np
from sklearn.impute import SimpleImputer


def imputer(data, method):
    method2option = {'Mean': 'mean', 'Median': 'median', 'Most Frequent': 'most_frequent'}[method]
    imp = SimpleImputer(missing_values=np.nan, strategy=method2option)
    print(f"Successfully fill the missing values with the {method2option} value "
          f"of each feature column respectively.")
    return imp.fit_transform(data)
