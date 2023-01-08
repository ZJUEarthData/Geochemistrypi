# import sys
from utils.base import save_fig
from global_variable import STATISTIC_IMAGE_PATH
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
# sys.path.append("..")


def basic_statistic(data: pd.DataFrame) -> None:
    print("Some basic statistic information of the designated data set:")
    print(data.describe())


def is_null_value(data: pd.DataFrame) -> None:
    print("Check which column has null values:")
    print("--" * 10)
    print(data.isnull().any())
    print("--" * 10)


def is_imputed(data: pd.DataFrame) -> bool:
    """Check whether the data set has null value or not

    :param data: pd.DataFrame, the data set
    :return: bool, True if it has null value
    """
    flag = data.isnull().any().any()
    if flag:
        print("Tip: you'd better use imputation techniques to deal with the missing values.")
    else:
        print("Tip: you don't need to deal with the missing values, we'll just pass this part!")
    return flag


def ratio_null_vs_filled(data: pd.DataFrame) -> None:
    print('The ratio of the null values in each column:')
    print("--" * 10)
    print(data.isnull().mean().sort_values(ascending=False))
    print("--" * 10)


def correlation_plot(col: pd.Index, df: pd.DataFrame) -> None:
    """A heatmap describing the correlation between the required columns

    :param col: pd.Index, a list of columns that need to plot
    :param df: pd.DataFrame, the dataframe
    """
    plot_df = df[col]
    plot_df_cor = plot_df.corr()
    plt.figure(figsize=(20, 20))
    sns.heatmap(plot_df_cor, cmap='coolwarm', annot=True, linewidths=.5)
    print("Successfully calculate the pair-wise correlation coefficient among the selected columns.")
    save_fig("correlation_plot", STATISTIC_IMAGE_PATH)


def distribution_plot(col: pd.Index, df: pd.DataFrame) -> None:
    """The histogram containing the respective distribution subplots of the required columns

    :param col: pd.Index, a list of columns that need to plot
    :param df: pd.DataFrame, the dataframe
    """
    n = int(np.sqrt(len(col))) + 1
    plt.figure(figsize=(n*2, n*2))
    df.hist()
    print("Successfully plot the distribution plot of the selected columns.")
    save_fig("distribution_histogram", STATISTIC_IMAGE_PATH)


def probability_plot(col: pd.Index, df_origin: pd.DataFrame, df_impute: pd.DataFrame) -> None:
    """A large graph containing the respective probability plots (origin vs. impute) of the required columns

    :param col: pd.Index, a list of columns that need to plot
    :param df_origin: pd.DataFrame, the original dataframe
    :param df_impute: pd.DataFrame, the dataframe after missing value imputation
    """
    r, c = len(col) // 4 + 1, 4
    fig = plt.figure(figsize=(c*8, r*8))
    for i in range(len(col)):
        feature = col[i]
        pp_origin = sm.ProbPlot(df_origin[feature].dropna(), fit=True)
        pp_impute = sm.ProbPlot(df_impute[feature], fit=True)
        ax = fig.add_subplot(r, c, i+1)
        pp_origin.ppplot(line="45", other=pp_impute, ax=ax)
        plt.title(f"{feature}, origin data vs. imputed data")
    print("Successfully graph the respective probability plot (origin vs. impute) of the selected columns")
    save_fig("probability_plot", STATISTIC_IMAGE_PATH)
