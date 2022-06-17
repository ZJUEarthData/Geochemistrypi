import sys
import os
import re
import openpyxl.utils.exceptions
from global_variable import DATASET_PATH
import pandas as pd
from typing import Optional, List
# from utils.exceptions import InvalidFileError


def read_data(file_name: Optional[str] = None, is_own_data: int = 2, prefix: Optional[str] = None,
              slogan: Optional[str] = "@File: "):
    """process the data set"""
    if is_own_data == 1:
        while True:
            # Capture exception: The path is invalid -> doesn't exist or not xlsx format
            file_name = input(f"({prefix}) ➜ {slogan}").strip()
            if os.path.exists(file_name):
                if '.xlsx' in re.findall(r'.xlsx.*', file_name):
                    print("Successfully loading the data set ...")
                    data_path = file_name
                    break
                else:
                    print("Caution: Please make sure the data is stored in xlsx format!")
            else:
                print("Caution: The path is invalid. Please input the correct path including the"
                      " stored path and suffix!")
    else:
        data_path = os.path.join(DATASET_PATH, file_name)
    try:
        data = pd.read_excel(data_path, engine="openpyxl")
        return data
    except ImportError as err:
        print(err)
        print("Warning: on Mac, input the following command in terminal: pip3 install openpyxl")
        raise
    except FileNotFoundError as err:
        print(err)
        print("Warning: please put your own data in the right place and input the completed data set name including"
              " the stored path and suffix")
        raise
    except openpyxl.utils.exceptions.InvalidFileException as err:
        print(err)
        print("Warning: please put your own data in the right place and input the completed data set name including"
              " the stored path and suffix")
        raise
    except Exception:
        print(f"Unexpected error: {sys.exc_info()[0]} - check the last line of Traceback about the error information")
        raise


def basic_info(data):
    print(data.info())


def show_data_columns(columns_name, columns_index=None):
    print('-' * 20)
    print("Index - Column Name")
    if columns_index is None:
        for i, j in enumerate(columns_name):
            print(i+1, "-", j)
    else:
        # specify the designated column index
        for i, j in zip(columns_index, columns_name):
            print(i+1, "-", j)
    print('-' * 20)


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


def num2option(items: List[str]) -> None:
    """list all the options serially.

    Parameters
    ----------
    items : list
        a series of items need to be enumerated
    """
    for i, j in enumerate(items):
        print(str(i+1) + " - " + j)


def num_input(prefix: Optional[str] = None, slogan: Optional[str] = "@Number: ") -> int:
    """get the number of the desired option"""
    # capture exception: input is not digit
    while True:
        option = input(f"({prefix}) ➜ {slogan}").strip()
        if option.isdigit():
            option = int(option)
            if isinstance(option, int):
                break
        else:
            print("Caution: The input is not a positive integer number. Please input the right number again!")
    return option


def limit_num_input(option_list: List[str], prefix: str, input_func: num_input) -> int:
    """limit the scope of the option"""
    while True:
        # in case that the option number is beyond the maximum
        option = input_func(prefix)
        if option not in range(1, len(option_list)+1):
            print("Caution: The number is invalid. Please enter the correct number inside the scope!")
        else:
            break
    return option


def np2pd(array, columns_name):
    return pd.DataFrame(array, columns=columns_name)
