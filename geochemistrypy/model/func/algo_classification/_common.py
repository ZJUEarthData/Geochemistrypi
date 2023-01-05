# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import Tuple, List


def confusion_matrix_plot(y_test: pd.DataFrame, y_test_predict: pd.DataFrame, trained_model: object) -> None:
    cm = confusion_matrix(y_test, y_test_predict)
    print(cm)
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=trained_model.classes_)
    disp.plot()


def contour_data(X: pd.DataFrame, trained_model: object) -> Tuple[List[np.ndarray], np.ndarray]:
    """Build up coordinate matrices as the data of contour plot.

    Parameters
    ----------
    X : pd.DataFrame (n_samples, n_components)
        The complete feature data.

    trained_model : object
        Te algorithm model class from sklearn is trained.

    Returns
    -------
    matrices : List[np.ndarray]
        Coordinate matrices.

    labels : np.ndarray
        Predicted value by the trained model with coordinate data as input data.
    """

    # build up coordinate matrices from coordinate vectors.
    xi = [np.arange(X.iloc[:, i].min(), X.iloc[:, i].max(), (X.iloc[:, i].max()-X.iloc[:, i].min())/50)
          for i in range(X.shape[1])]
    ndim = len(xi)
    s0 = (1,) * ndim
    matrices = [np.asanyarray(x).reshape(s0[:i] + (-1,) + s0[i + 1:]) for i, x in enumerate(xi)]
    matrices[0].shape = (1, -1) + s0[2:]
    matrices[1].shape = (-1, 1) + s0[2:]
    matrices = np.broadcast_arrays(*matrices, subok=True)

    # get the labels of the coordinate matrices through the trained model
    input_array = np.column_stack((i.ravel() for i in matrices))
    labels = trained_model.predict(input_array).reshape(matrices[0].shape)

    return matrices, labels
