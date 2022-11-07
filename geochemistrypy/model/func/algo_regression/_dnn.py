# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd

def plot_pred(y_test_predict: pd.DataFrame, y_test: pd.DataFrame, algorithm_name: str):
    """Plot the testing predict values of the trained model and the testing target values.
    Parameters
    ----------
    y_test_predict : pd.DataFrame (n_samples, n_components)
        The testing predict values.
    y_test : pd.DataFrame (n_samples, n_components)
        The testing target values.
    algorithm_name : str
        The name of the algorithm model.
    """

    y_test.index = range(len(y_test))
    plt.figure(figsize=(8, 6))
    plt.plot(y_test_predict, color='red', label='predict')
    plt.plot(y_test, color='gold', label='true')
    plt.legend()
    plt.title(f'Ground Truth v.s. Prediction - {algorithm_name}')