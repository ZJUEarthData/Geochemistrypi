import sys
sys.path.append("..")
from core.base import save_fig
from global_variable import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm


def basic_statistic(data):
    print("Some basic statistic information of the designated data set:")
    print(data.describe())


def is_null_value(data):
    print("Check which column has null values:")
    print(data.isnull().any())


def correlation_plot(col, df):
    """A heatmap describing the correlation between the required columns

    :param col: A list of columns that need to plot
    :param df: The dataframe
    """
    plot_df = df[col]
    plot_df_cor = plot_df.corr()
    plt.figure(figsize=(20, 20))
    sns.heatmap(plot_df_cor, cmap = 'coolwarm', annot=True, linewidths=.5)
    print("Successfully calculate the pair-wise correlation coefficient among the selected columns")
    save_fig("correlation_plot", STATISTIC_IMAGE_PATH)


def distribution_plot(col, df):
    """the histogram containing the respective distribution subplots of the required columns

    :param col: A list of columns that need to plot
    :param df: The dataframe
    """
    n = int(np.sqrt(len(col))) + 1
    plt.figure(figsize=(n*2, n*2))
    df.hist()
    print("Successfully plot the distribution plot of the selected columns")
    save_fig("distribution_histogram", STATISTIC_IMAGE_PATH)


def probability_plot(col, df_origin, df_impute):
    """A large graph containing the respective probability plots (origin vs. impute) of the required columns

    :param col: A list of columns that need to plot
    :param df_origin: The original dataframe
    :param df_impute: The dataframe after missing value imputation
    """
    r, c = len(col) // 4 + 1, 4
    fig = plt.figure(figsize=(c*8, r*8))
    for i in range(len(col)):
        feature = col[i]
        pp_origin = sm.ProbPlot(df_origin[feature].dropna(), fit=True)
        pp_impute = sm.ProbPlot(df_impute[feature], fit=True)
        ax = fig.add_subplot(r, c, i+1)
        pp_origin.ppplot(line="45", other=pp_impute, ax=ax)
        plt.title(f"{feature}, origin vs. impute")
    print("Successfully graph the respective probability plot (origin vs. impute) of the selected columns")
    save_fig("probability_plot", STATISTIC_IMAGE_PATH)
