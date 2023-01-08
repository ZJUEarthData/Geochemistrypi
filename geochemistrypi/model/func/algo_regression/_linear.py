# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


def show_formula(coef, intercept, columns_name):
    term = []
    coef = np.around(coef, decimals=3).tolist()[0]

    for i in range(len(coef)):
        # the first value stay the same
        if i == 0:
            # not append if zero
            if coef[i] != 0:
                temp = str(coef[i]) + columns_name[i]
                term.append(temp)
        else:
            # add plus symbol if positive, maintain if negative, not append if zero
            if coef[i] > 0:
                temp = '+' + str(coef[i]) + columns_name[i]
                term.append(temp)
            elif coef[i] < 0:
                temp = str(coef[i]) + columns_name[i]
                term.append(temp)
    if intercept[0] >= 0:
        # formula of linear regression
        formula = ''.join(term) + '+' + str(intercept[0])
    else:
        formula = ''.join(term) + str(intercept[0])
    print('y =', formula)


def plot_2d_graph(feature_data: pd.DataFrame, target_data: pd.DataFrame = None) -> None:
    """Plot a 2D graph with the data set below.

        Parameters
        ----------
        feature_data : pd.DataFrame (n_samples, n_components)
            Data selected by user.

        target_data : pd.DataFrame (n_samples, n_components)
            The target values.
        """
    plt.figure(figsize=(14, 10))
    plt.scatter(feature_data.values, target_data.values)
    plt.xlabel(feature_data.columns)
    plt.ylabel(target_data.columns)
    plt.title("2D Scatter Graph")
    plt.grid()


def plot_3d_graph(feature_data: pd.DataFrame, target_data: pd.DataFrame = None) -> None:
    """Plot a 3D graph with the data set below.

        Parameters
        ----------
        feature_data : pd.DataFrame (n_samples, n_components)
            Data selected by user.

        target_data : pd.DataFrame (n_samples, n_components)
            The target values.
        """
    x = feature_data.iloc[:, 0]
    y = feature_data.iloc[:, 1]
    z = target_data
    nameList = feature_data.columns.values.tolist()
    fig = plt.figure(figsize=(14, 10))
    ax = Axes3D(fig)
    ax.scatter(x, y, z)
    ax.set_xlabel(nameList[0], fontdict={'size': 10, 'color': 'black'})
    ax.set_ylabel(nameList[1], fontdict={'size': 10, 'color': 'r'})
    ax.set_zlabel(target_data.columns, fontdict={'size': 10, 'color': 'blue'})
    plt.title("3D Scatter Graph")
    plt.grid()
