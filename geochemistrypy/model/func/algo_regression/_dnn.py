# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Any

def plot_pred(y_test_predict: Any, y_test: pd.DataFrame, algorithm_name: str):
    """Plot the testing predict values of the trained model and the testing target values.
    Parameters
    ----------
    y_test_predict : Any
        The testing predict values.
    y_test : pd.DataFrame (n_samples, n_components)
        The testing target values.
    algorithm_name : str
        The name of the algorithm model.
    """

    y_test = np.array(y_test).reshape(1, len(y_test)).flatten()
    y_test_predict = np.array(y_test_predict).flatten()
    plt.figure(figsize=(8, 6))
    plt.plot([i for i in range(len(y_test))], y_test, color='gold', label='true')
    plt.plot([i for i in range(len(y_test))], y_test_predict, color='red', label='predict')
    plt.legend()
    plt.title(f'Ground Truth v.s. Prediction - {algorithm_name}')