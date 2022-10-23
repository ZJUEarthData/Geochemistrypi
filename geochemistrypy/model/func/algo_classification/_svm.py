# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Any, Optional, List

# TODO(Sany sanyhew1097618435@163.com): if 3d decision boundary doesn't make sense,
#  delete the parameters 'contour_data' and 'labels'.


def plot_2d_decision_boundary(X: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, trained_model: Any,
                              algorithm_name: str, contour_data: Optional[List[np.ndarray]] = None,
                              labels: Optional[np.ndarray] = None):
    """Plot the decision boundary of the trained model with the testing data set below.
<<<<<<< HEAD
=======

>>>>>>> 8efb9b5fb6c369cd87df691f6477cdb8af4c109c
    Parameters
    ----------
    X : pd.DataFrame (n_samples, n_components)
        The complete feature data.
<<<<<<< HEAD
    X_test : pd.DataFrame (n_samples, n_components)
        The testing feature data.
    y_test : pd.DataFrame (n_samples, n_components)
        The testing target values.
    trained_model : sklearn algorithm model
        The sklearn algorithm model trained with X_train data.
=======

    X_test : pd.DataFrame (n_samples, n_components)
        The testing feature data.

    y_test : pd.DataFrame (n_samples, n_components)
        The testing target values.

    trained_model : sklearn algorithm model
        The sklearn algorithm model trained with X_train data.

>>>>>>> 8efb9b5fb6c369cd87df691f6477cdb8af4c109c
    algorithm_name : str
        The name of the algorithm model.
    """

    # Prepare the data for contour plot
    x0s = np.arange(X.iloc[:, 0].min(), X.iloc[:, 0].max(), (X.iloc[:, 0].max()-X.iloc[:, 0].min())/50)
    x1s = np.arange(X.iloc[:, 1].min(), X.iloc[:, 1].max(), (X.iloc[:, 1].max()-X.iloc[:, 1].min())/50)
    x0, x1 = np.meshgrid(x0s, x1s)
    input_array = np.c_[x0.ravel(), x1.ravel()]
    labels = trained_model.predict(input_array).reshape(x0.shape)

    # Use the code below when the dimensions of X are greater than 2
    # x0, x1 = contour_data

    # Set up the canvas
    plt.figure()

    # Draw the contour plot and the scatter plot
    plt.contourf(x0, x1, labels, alpha=0.1)
    plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=np.array(y_test), alpha=1)

    # Set up the axis name, graph title, storage path
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
<<<<<<< HEAD
    plt.title(f'2d Decision Boundary - {algorithm_name}')
=======
    plt.title(f'2d Decision Boundary - {algorithm_name}')
>>>>>>> 8efb9b5fb6c369cd87df691f6477cdb8af4c109c
