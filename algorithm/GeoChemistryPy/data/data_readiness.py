import sys
sys.path.append("..")
from global_variable import *
import pandas as pd
import re
from typing import Optional, List


def read_data(file_name):
    data_path = os.path.join(DATASET_PATH, file_name)
    data = pd.read_excel(data_path, engine="openpyxl")
    return data


def show_data_columns(data):
    feature_name = list(data.columns)
    for i, j in enumerate(feature_name):
        print(i+1, "-", j)


def select_columns(columns_range: Optional[str] = None) -> List[int]:

    columns_selected = []

    # recognize "(xx:xx)" pattern
    pattern1 = r'\(\d{1,2}.*?\d{1,2}\)'
    continues_column = re.findall(pattern1, columns_range)
    # find the integers in a specific range
    for i in range(len(continues_column)):
        min_max = re.findall(r"\d{1,2}", continues_column[i])
        index1 = int(min_max[0])
        index2 = int(min_max[1])
        temp = [index2 - j for j in range(index2-index1+1)]
        columns_selected = columns_selected + temp

    # recognize "xx; | ;xx" pattern
    pattern2 = r'\s*\d{1,2}\s*;|;\s*\d{1,2}\s*'
    discrete_column = re.findall(pattern2, columns_range)
    # extract the integers
    for i in range(len(discrete_column)):
        index = re.sub(r';|\s', '', discrete_column[i])
        columns_selected.append(int(index))

    # delete the repetitive index in the list
    columns_selected = list(set(columns_selected))
    # sort the index list
    columns_selected.sort()
    return columns_selected



def imputing():
    pass