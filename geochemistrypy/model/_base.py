# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from typing import Optional
from data.data_readiness import num2option, num_input, limit_num_input, create_sub_data_set, show_data_columns
from global_variable import SECTION
from typing import Tuple, List, Union
from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split


class WorkflowBase(metaclass=ABCMeta):
    """Base class for all workflow classes in geochemistry π."""

    # Default for child class. They need to be overwritten in child classes.
    name = None
    common_function = []
    special_function = []
    X, y = None, None
    X_train, X_test, y_train, y_test = None, None, None, None
    y_test_predict = None

    @classmethod
    def show_info(cls) -> None:
        """Display how many functions the algorithm will provide."""
        print("*-*" * 2, cls.name, "is running ...", "*-*" * 2)
        print("Expected Functionality:")
        function = cls.common_function + cls.special_function
        for i in range(len(function)):
            print("+ ", function[i])

    def __init__(self) -> None:
        # Default for child class. They need to be overwritten in child classes.
        self.model = None
        self.naming = None
        self.automl = None
        self.random_state = 42

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> None:
        """Placeholder for fit. child classes should implement this method!

        Parameters
        ----------
        X : pd.DataFrame (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and n_features` is the number of features.

        y : pd.DataFrame, default=None (n_samples,) or (n_samples, n_targets)
            Target values。
        """
        return None

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """The interface for the child classes."""
        return pd.DataFrame()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """The interface for the child classes."""
        return pd.DataFrame()

    @staticmethod
    def score(y_true: Union[pd.DataFrame, np.ndarray], y_predict: Union[pd.DataFrame, np.ndarray])\
            -> Union[int, float]:
        """The interface for the child classes."""
        return float()

    @staticmethod
    def np2pd(array, columns_name):
        """The type of the data set is transformed from numpy.ndarray to pandas.DataFrame."""
        return pd.DataFrame(array, columns=columns_name)

    @staticmethod
    def choose_dimension_data(data: pd.DataFrame, dimensions: int) -> Tuple[List[int], pd.DataFrame]:
        """Choose a subgroup data from the whole data set to draw 2d or 3d graph.

        Parameters
        ----------
        data : pd.DataFrame (n_samples, n_features)
            the whole data.
        dimensions : int
            how much dimensions data to keep.

        Returns
        -------
        selected_axis_index : list[int]
            the index of the that dimension, which is shown as the index of the column of the data set
        selected_axis_data : pd.DataFrame (n_samples, n_features)
            the selected data from the whole data set
        """
        print(f"-----* {dimensions} Dimensions Data Selection *-----")
        print(f"The system is going to draw related {dimensions}d graphs.")
        print(f"Currently, the data dimension is beyond {dimensions} dimensions.")
        print(f"Please choose {dimensions} dimensions of the data below.")
        data = pd.DataFrame(data)
        selected_axis_index = []
        selected_axis_name = []
        for i in range(1, dimensions+1):
            num2option(data.columns)
            print(f'Choose dimension - {i} data:')
            index_axis = limit_num_input(data.columns, SECTION[3], num_input)
            selected_axis_index.append(index_axis-1)
            selected_axis_name.append(data.columns[index_axis-1])
        selected_axis_data = data.loc[:, selected_axis_name]
        print(f"The Selected Data Dimension:")
        show_data_columns(selected_axis_name)
        return selected_axis_index, selected_axis_data

    @staticmethod
    def data_upload(X: pd.DataFrame,
                    y: Optional[pd.DataFrame] = None,
                    X_train: Optional[pd.DataFrame] = None,
                    X_test: Optional[pd.DataFrame] = None,
                    y_train: Optional[pd.DataFrame] = None,
                    y_test: Optional[pd.DataFrame] = None,
                    y_test_predict: Optional[pd.DataFrame] = None) -> None:
        """This method loads the required data into the base class's attributes."""
        WorkflowBase.X = X
        WorkflowBase.y = y
        WorkflowBase.X_train = X_train
        WorkflowBase.X_test = X_test
        WorkflowBase.y_train = y_train
        WorkflowBase.y_test = y_test
        WorkflowBase.y_test_predict = y_test_predict

    def data_split(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series], test_size: float = 0.2)\
            -> Tuple[pd.DataFrame, pd.DataFrame, Union[pd.DataFrame, pd.Series], Union[pd.DataFrame, pd.Series]]:
        """Split arrays or matrices into random train and test subsets."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=self.random_state)
        return X_train, X_test, y_train, y_test

