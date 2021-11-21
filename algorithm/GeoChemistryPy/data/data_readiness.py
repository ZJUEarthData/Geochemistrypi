import sys
sys.path.append("..")
from global_variable import *
import numpy as np
import pandas as pd
import re
from typing import Optional, List
from sklearn.impute import SimpleImputer


def read_data(file_name):
    data_path = os.path.join(DATASET_PATH, file_name)
    data = pd.read_excel(data_path, engine="openpyxl")
    return data


def show_data_columns(columns_name, columns_index=None):
    if columns_index == None:
        for i, j in enumerate(columns_name):
            print(i+1, "-", j)
    else:
        for i, j in zip(columns_index, columns_name):
            print(i+1, "-", j)


def select_columns(columns_range: Optional[str] = None) -> List[int]:

    columns_selected = []
    temp = columns_range.split(";")
    for i in range(len(temp)):
        if isinstance(eval(temp[i]), int):
            columns_selected.append(eval(temp[i]))
        else:
            min_max = eval(temp[i])
            index1 = min_max[0]
            index2 = min_max[1]
            j = [index2 - j for j in range(index2 - index1 + 1)]
            columns_selected = columns_selected + j

    # delete the repetitive index in the list
    columns_selected = list(set(columns_selected))
    # sort the index list
    columns_selected.sort()
    # reindex by subtracting 1 due to python list traits
    columns_selected = [columns_selected[i] - 1 for i in range(len(columns_selected))]
    return columns_selected


def num2option(items: List = None) -> None:
    for i, j in enumerate(items):
        print(str(i+1) + " - " + j)


def np2pd(array, columns_name):
    return pd.DataFrame(array, columns=columns_name)


def imputer(data, method):
    imp = SimpleImputer(missing_values=np.nan, strategy=method)
    print(f"Successfully fill the missing values with the {method} value "
          f"of each feature column respectively")
    return imp.fit_transform(data)