import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from rich import print

from ..constants import MLFLOW_ARTIFACT_IMAGE_STATISTIC_PATH
from ..utils.base import save_data, save_fig


def basic_statistic(data: pd.DataFrame) -> None:
    """Some basic statistic information of the designated data set.

    Parameters
    ----------
    data : pd.DataFrame
        The data set.
    """
    print("Some basic statistic information of the designated data set:")
    print(data.describe())


def is_null_value(data: pd.DataFrame) -> None:
    """Check whether the data set has null value or not.

    Parameters
    ----------
    data : pd.DataFrame
        The data set.
    """
    print("Check which column has null values:")
    print("--" * 10)
    print(data.isnull().any())
    print("--" * 10)


def is_imputed(data: pd.DataFrame) -> bool:
    """Check whether the data set has null value or not.

    Parameters
    ----------
    data : pd.DataFrame
        The data set.

    Returns
    -------
    flag : bool
        True if it has null value.
    """
    flag = data.isnull().any().any()
    if flag:
        print("Note: you'd better use imputation techniques to deal with the missing values.")
    else:
        print("Note: you don't need to deal with the missing values, we'll just pass this step!")
    return flag


def ratio_null_vs_filled(data: pd.DataFrame) -> None:
    """The ratio of the null values in each column.

    Parameters
    ----------
    data : pd.DataFrame
        The data set.
    """
    print("The ratio of the null values in each column:")
    print("--" * 10)
    print(data.isnull().mean().sort_values(ascending=False))
    print("--" * 10)


def correlation_plot(col: pd.Index, df: pd.DataFrame) -> None:
    """A heatmap describing the correlation between the required columns.

    Parameters
    ----------
    col : pd.Index
        A list of columns that need to plot.

    df : pd.DataFrame
        The data set.
    """
    plot_df = df[col]
    plot_df_cor = plot_df.corr()
    plt.figure(figsize=(20, 20))
    sns.heatmap(plot_df_cor, cmap="coolwarm", annot=True, linewidths=0.5)
    print("Successfully calculate the pair-wise correlation coefficient among the selected columns.")
    save_fig("Correlation Plot", os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_STATISTIC_PATH"), MLFLOW_ARTIFACT_IMAGE_STATISTIC_PATH)
    save_data(df, "Correlation Plot", os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_STATISTIC_PATH"), MLFLOW_ARTIFACT_IMAGE_STATISTIC_PATH)


def distribution_plot(col: pd.Index, df: pd.DataFrame) -> None:
    """The histogram containing the respective distribution subplots of the required columns.

    Parameters
    ----------
    col : pd.Index
        A list of columns that need to plot.

    df : pd.DataFrame
        The data set.
    """
    n = int(np.sqrt(len(col))) + 1
    plt.figure(figsize=(n * 2, n * 2))
    for i in range(len(col)):
        plt.subplot(n, n, i + 1)
        plt.hist(df[col[i]])
        plt.title(col[i])
    print("Successfully draw the distribution plot of the selected columns.")
    save_fig("Distribution Histogram", os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_STATISTIC_PATH"), MLFLOW_ARTIFACT_IMAGE_STATISTIC_PATH)
    save_data(df, "Distribution Histogram", os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_STATISTIC_PATH"), MLFLOW_ARTIFACT_IMAGE_STATISTIC_PATH)


def log_distribution_plot(col: pd.Index, df: pd.DataFrame) -> None:
    """The histogram containing the respective distribution subplots after log transformation of the required columns.

    Parameters
    ----------
    col : pd.Index
        A list of columns that need to plot.

    df : pd.DataFrame
        The data set.
    """
    # Log transform the required columns
    df_log_transformed = df[col].applymap(lambda x: np.log(x + 1))

    n = int(np.sqrt(len(col))) + 1
    plt.figure(figsize=(n * 2, n * 2))
    for i in range(len(col)):
        plt.subplot(n, n, i + 1)
        plt.hist(df_log_transformed[col[i]])
        plt.title(col[i])

    print("Successfully draw the distribution plot after log transformation of the selected columns.")
    save_fig("Distribution Histogram After Log Transformation", os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_STATISTIC_PATH"), MLFLOW_ARTIFACT_IMAGE_STATISTIC_PATH)
    save_data(df_log_transformed, "Distribution Histogram After Log Transformation", os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_STATISTIC_PATH"), MLFLOW_ARTIFACT_IMAGE_STATISTIC_PATH)


def probability_plot(col: pd.Index, df_origin: pd.DataFrame, df_impute: pd.DataFrame) -> None:
    """A large graph containing the respective probability plots (origin vs. impute) of the required columns.

    Parameters
    ----------
    col : pd.Index
        A list of columns that need to plot.

    df_origin : pd.DataFrame (n_samples, n_components)
        The original dataset with missing value.

    df_impute : pd.DataFrame (n_samples, n_components)
        The dataset after imputation.
    """
    r, c = len(col) // 4 + 1, 4
    fig = plt.figure(figsize=(c * 8, r * 8))
    for i in range(len(col)):
        feature = col[i]
        pp_origin = sm.ProbPlot(df_origin[feature].dropna(), fit=True)
        pp_impute = sm.ProbPlot(df_impute[feature], fit=True)
        ax = fig.add_subplot(r, c, i + 1)
        pp_origin.ppplot(line="45", other=pp_impute, ax=ax)
        plt.title(f"{feature}, origin data vs. imputed data")

    data = pd.concat([df_origin, df_impute], axis=1)
    print("Successfully draw the respective probability plot (origin vs. impute) of the selected columns")
    save_fig("Probability Plot", os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_STATISTIC_PATH"), MLFLOW_ARTIFACT_IMAGE_STATISTIC_PATH)
    save_data(data, "Probability Plot", os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_STATISTIC_PATH"), MLFLOW_ARTIFACT_IMAGE_STATISTIC_PATH)
