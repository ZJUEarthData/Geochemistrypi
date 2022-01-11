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

# TODO: is the directory exists? -> build automatically
def save_data(file_name):
    pass

def basic_info(data):
    print(data.info())


def show_data_columns(columns_name, columns_index=None):
    print("Index - Column Name")
    if columns_index is None:
        for i, j in enumerate(columns_name):
            print(i+1, "-", j)
    else:
        # specify the designated column index
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


def create_sub_data_set(data):
    sub_data_set_columns_range = input('Select the data range you want to process.\n'
                                       'Input format:\n'
                                       'Format 1: "[**, **]; **; [**, **]", such as "[1, 3]; 7; [10, 13]" '
                                       '--> you want to deal with the columns 1, 2, 3, 7, 10, 11, 12, 13 \n'
                                       'Format 2: "xx", such as "7" --> you want to deal with the columns 7 \n'
                                       '@input: ')
    # column name
    sub_data_set_columns_selected = select_columns(sub_data_set_columns_range)
    # select designated column
    sub_data_set = data.iloc[:, sub_data_set_columns_selected]
    show_data_columns(sub_data_set.columns, sub_data_set_columns_selected)
    return sub_data_set


def num2option(items: List = None) -> None:
    """pattern show: num - item

    :param items: list, a series of items need to be enumerated
    """
    for i, j in enumerate(items):
        print(str(i+1) + " - " + j)


def np2pd(array, columns_name):
    return pd.DataFrame(array, columns=columns_name)

