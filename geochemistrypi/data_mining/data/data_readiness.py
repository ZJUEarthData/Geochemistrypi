import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import openpyxl.utils.exceptions
import pandas as pd
from rich import print
from sklearn.model_selection import train_test_split

from ..constants import BUILT_IN_DATASET_PATH

# from utils.exceptions import InvalidFileError


def read_data(file_path: Optional[str] = None, is_own_data: int = 2, prefix: Optional[str] = None, slogan: Optional[str] = "@File: "):
    """Read the data set.

    Parameters
    ----------
    file_path : str, optional
        The path of the data set, by default None

    is_own_data : int, default=2
        1: own data set; 2: built-in data set

    prefix : str, optional
        The prefix of the data set, by default None

    slogan : str, optional
        The slogan of the data set, by default "@File: "

    Returns
    -------
    pd.DataFrame
        The data set read
    """
    if is_own_data == 1:
        data_path = file_path
    else:
        data_path = os.path.join(BUILT_IN_DATASET_PATH, file_path)
    try:
        if data_path.endswith(".xlsx"):
            data = pd.read_excel(data_path, engine="openpyxl")
        elif data_path.endswith(".csv"):
            data = pd.read_csv(data_path)
        return data
    except ImportError as err:
        print(err)
        print("[red]Warning: on Mac, input the following command in terminal: pip3 install openpyxl[red]")
        raise err
    except FileNotFoundError as err:
        print(err)
        print("[red]Warning: please put your own data in the right place and input the completed data set name including" " the stored path and suffix[red]")
        raise err
    except openpyxl.utils.exceptions.InvalidFileException as err:
        print(err)
        print("[red]Warning: please put your own data in the right place and input the completed data set name including" " the stored path and suffix[red]")
        raise err
    except Exception:
        print(f"[red]Unexpected error: {sys.exc_info()[0]} - check the last line of Traceback about the error information[red]")
        raise Exception


def basic_info(data: pd.DataFrame) -> None:
    """Show the basic information of the data set.

    Parameters
    ----------
    data : pd.DataFrame
        The data set to be shown.
    """
    print(data.info())


def show_data_columns(columns_name: pd.Index, columns_index: Optional[List] = None) -> None:
    """Show the column names of the data set.

    Parameters
    ----------
    columns_name : pd.Index
        The column names of the data set.

    columns_index : list, default=None
        The column index of the data set.
    """
    print("-" * 20)
    print("Index - Column Name")
    if columns_index is None:
        for i, j in enumerate(columns_name):
            print(i + 1, "-", j)
    else:
        # specify the designated column index
        for i, j in zip(columns_index, columns_name):
            print(i + 1, "-", j)
    print("-" * 20)


def select_columns(columns_range: Optional[str] = None) -> List[int]:
    """Select the columns of the data set.

    Parameters
    ----------
    columns_range : str, default=None
        The columns range of the data set.

    Returns
    -------
    list
        The columns selected.
    """
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


def create_sub_data_set(data: pd.DataFrame) -> pd.DataFrame:
    """Create a sub data set.

    Parameters
    ----------
    data : pd.DataFrame
        The data set to be processed.

    Returns
    -------
    pd.DataFrame
        The sub data set.
    """
    sub_data_set_columns_range = str(
        input(
            "Select the data range you want to process.\n"
            "Input format:\n"
            'Format 1: "[**, **]; **; [**, **]", such as "[1, 3]; 7; [10, 13]" '
            "--> you want to deal with the columns 1, 2, 3, 7, 10, 11, 12, 13 \n"
            'Format 2: "xx", such as "7" --> you want to deal with the columns 7 \n'
            "@input: "
        )
    )
    while True:
        if ("【" in sub_data_set_columns_range) or ("】" in sub_data_set_columns_range):
            print("There is a problem with the format of the parentheses entered !")
            time.sleep(0.5)
            sub_data_set_columns_range = str(input("-----* Please enter again *-----\n@input: "))
            judge = True
        else:
            monitor_number = 0
            for i in ["[", "]"]:
                if i in sub_data_set_columns_range:
                    monitor_number = monitor_number + 1
            if monitor_number % 2 != 0:
                print("There is a problem with the format of the parentheses entered !")
                time.sleep(0.5)
                sub_data_set_columns_range = str(input("-----* Please enter again *-----\n@input: "))
                judge = True
            sub_data_set_columns_range = sub_data_set_columns_range.replace(" ", "")
            temp = sub_data_set_columns_range.split(";")
            if len(sub_data_set_columns_range) != 0:
                for i in range(len(temp)):
                    if isinstance(eval(temp[i]), int):
                        if int(temp[i]) > int(data.shape[1]):
                            print("The input {} is incorrect!".format(temp[i]))
                            print("The number you entered is out of the range of options: 1 - {}".format(data.shape[1]))
                            time.sleep(0.5)
                            sub_data_set_columns_range = input("-----* Please enter again *-----\n@input: ")
                            judge = True
                            break
                        else:
                            judge = False
                    else:
                        min_max = eval(temp[i])
                        if int(min_max[0]) >= int(min_max[1]):
                            print("There is a problem with the format of the data you entered!")
                            time.sleep(0.5)
                            sub_data_set_columns_range = input("-----* Please enter again *-----\n@input: ")
                            judge = True
                            break
                        elif int(min_max[1]) > int(data.shape[1]):
                            print("The input {} is incorrect!".format(temp[i]))
                            print("The number you entered is out of the range of options: 1 - {}".format(data.shape[1]))
                            time.sleep(0.5)
                            sub_data_set_columns_range = input("-----* Please enter again *-----\n@input: ")
                            judge = True
                            break
                        else:
                            judge = False
            else:
                print("You have not entered the sequence number of the selected data!")
                print("The number you entered should be in the range of options: 1 - {}".format(data.shape[1]))
                time.sleep(0.5)
                sub_data_set_columns_range = input("-----* Please enter again *-----\n@input: ")
                judge = True

        if judge is False:
            break

    while True:
        try:
            # column name
            sub_data_set_columns_selected = select_columns(sub_data_set_columns_range)
            judge = False
        except SyntaxError:
            print("Warning: Please use English input method editor.")
            judge = True
            sub_data_set_columns_range = str(input("@input: "))
        except NameError:
            print("Warning: Please follow the rules and re-enter.")
            judge = True
            sub_data_set_columns_range = str(input("@input: "))
        except UnicodeDecodeError:
            print("Warning: Please use English input method editor.")
            judge = True
            sub_data_set_columns_range = str(input("@input: "))
        except IndexError:
            print("Warning: Please follow the rules and re-enter.")
            judge = True
            sub_data_set_columns_range = str(input("@input: "))
        except TypeError:
            print("Warning: Please follow the rules and re-enter.")
            judge = True
            sub_data_set_columns_range = str(input("@input: "))
        else:
            data_checking = data.iloc[:, sub_data_set_columns_selected]
            for i in data_checking.columns.values:
                df_test = pd.DataFrame(data_checking[i])
                test_columns = df_test.columns
                v_value = int(df_test.isnull().sum())
                if v_value == len(df_test):
                    print(f"Warning: The selected column {df_test.columns.values} is an empty column!")
                    judge = True
                elif df_test[test_columns[0]].dtype in ["int64", "float64"]:
                    continue
                else:
                    print(f"Warning: The data type of selected column {df_test.columns.values} is not numeric!" " Please make sure that the selected data type is numeric and re-enter.")
                    judge = True
            if judge is True:
                sub_data_set_columns_range = str(input("@input: "))
        if judge is False:
            break

    # select designated column
    sub_data_set = data.iloc[:, sub_data_set_columns_selected]
    show_data_columns(sub_data_set.columns, sub_data_set_columns_selected)
    return sub_data_set


def data_split(X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series], test_size: float = 0.2) -> Dict:
    """Split arrays or matrices into random train and test subsets.

    Parameters
    ----------
    X : pd.DataFrame
        The data to be split.

    y : pd.DataFrame or pd.Series
        The target variable to be split.

    test_size : float, default=0.2
        Represents the proportion of the dataset to include in the test split.

    Returns
    -------
    dict
        A dictionary containing the split data.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return {"X Train": X_train, "X Test": X_test, "Y Train": y_train, "Y Test": y_test}


def num2option(items: List[str]) -> None:
    """List all the options serially.

    Parameters
    ----------
    items : list
        a series of items need to be enumerated
    """
    for i, j in enumerate(items):
        print(str(i + 1) + " - " + j)


def num_input(prefix: Optional[str] = None, slogan: Optional[str] = "@Number: ") -> int:
    """Get the number of the desired option.

    Parameters
    ----------
    prefix : str, default=None
        It indicates which section the user currently is in on the UML, which is shown on the command-line console.

    slogan : str, default="@Number: "
        It acts like the first parameter of input function in Python, which output the hint.

    Returns
    -------
    option: int
        An option number. Be careful that 'option = real index  + 1'
    """
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


def np2pd(array: np.ndarray, columns_name: List[str]) -> pd.DataFrame:
    """Convert numpy array to pandas dataframe.

    Parameters
    ----------
    array : np.ndarray
        The numpy array to be converted.

    columns_name : List[str]
        The column names of the dataframe.

    Returns
    -------
    pd.DataFrame
        The converted dataframe.
    """
    return pd.DataFrame(array, columns=columns_name)


def limit_num_input(option_list: List[str], prefix: str, input_func: num_input) -> int:
    """Limit the scope of the option.

    Parameters
    ----------
    option_list : List[str]
        All the options provided are stored in a list.

    prefix : str
        It indicates which section the user currently is in on the UML, which is shown on the command-line console.

    input_func: function
        The function of input_func.

    Returns
    -------
    option: int
        An option number. Be careful that 'option = real index  + 1'
    """
    while True:
        # in case that the option number is beyond the maximum
        option = input_func(prefix)
        if option not in range(1, len(option_list) + 1):
            print("Caution: The number is invalid. Please enter the correct number inside the scope!")
        else:
            break
    return option


def float_input(default: float, prefix: Optional[str] = None, slogan: Optional[str] = "@Number: ") -> float:
    """Get the number of the desired option.

    Parameters
    ----------
    default: float
         If the user does not enter anything, it is assigned to option.

    prefix : str, default=None
        It indicates which section the user currently is in on the UML, which is shown on the command-line console.

    slogan : str, default="@Number: "
        It acts like the first parameter of input function in Python, which output the hint.

    Returns
    -------
    option: float or int
        An option number.
    """
    while True:
        option = input(f"({prefix}) ➜ {slogan}").strip()
        if option.isdigit() or option.replace(".", "").isdigit():
            option = float(option)
            break
        elif len(option) == 0:
            option = default
            break
        else:
            print("Caution: The input is not a positive number. Please input the right number again!")
    return option


def str_input(option_list: List[str], prefix: Optional[str] = None) -> str:
    """Get the string of the desired option.

    Parameters
    ----------
    option_list : list
        All the options provided are stored in a list.

    prefix : str, default=None
        It indicates which section the user currently is in on the UML, which is shown on the command-line console.

    Returns
    -------
    option: str
        A string of the desired option.
    """
    num2option(option_list)
    option_num = limit_num_input(option_list, prefix, num_input)
    option = option_list[option_num - 1]
    return option


def bool_input(prefix: Optional[str] = None) -> bool:
    """Get the number of the desired option.

    Parameters
    ----------
    prefix : str, default=None
        It indicates which section the user currently is in on the UML, which is shown on the command-line console.

    Returns
    -------
    bool
        A boolean value.
    """
    bool_value = ["True", "False"]
    option = str_input(bool_value, prefix)
    return True if option == "True" else False


def tuple_input(default: Tuple[int], prefix: Optional[str] = None, slogan: Optional[str] = None) -> Tuple[int]:
    """Get the tuple of the desired option.

    Parameters
    ----------
    default: Tuple[int]
         If the user does not enter anything, it is assigned to option.

    prefix : str, default=None
        It indicates which section the user currently is in on the UML, which is shown on the command-line console.

    slogan : str, default=None
        It acts like the first parameter of input function in Python, which output the hint.

    Returns
    -------
    option: tuple
        A numeric tuple.
    """
    while True:
        option = input(
            "Determine the architecture of the multi-layer perceptron.\n"
            "Input format:\n"
            'Format 1: "(**,)", such as "(100,)"\n'
            "--> You want to set one hidden layer with 100 neurons for the multi-layer perceptron.\n"
            'Format 2: "(**, **)", such as "(50, 25)"\n'
            "--> You want to set two hidden layers in order with 50 neurons and 25 neurons respectively"
            " for the multi-layer perceptron.\n"
            'Format 3: "(**, **, **)", such as "(64, 32, 8)"\n'
            "--> You want to set three hidden layers in order 64 neurons, 32 neurons and 8 neurons"
            " respectively for the multi-layer perceptron.\n"
            f"({prefix}) ➜ {slogan}"
        ).strip()
        if len(option) == 0:
            option = default
            break
        else:
            option = eval(option)
            break
    return option
