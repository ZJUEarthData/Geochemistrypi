# -*- coding: utf-8 -*-
import random

import numpy as np
import pandas as pd
from rich import print
from scipy.stats import kruskal, wilcoxon


def test_once(df_orig: pd.DataFrame, df_impute: pd.DataFrame, test: str) -> np.ndarray:
    """Do hypothesis testing on each pair-wise column once, non-parametric test.
    Null hypothesis: the distributions of the data set before and after imputing remain the same.

    Parameters
    ----------
    df_orig : pd.DataFrame (n_samples, n_components)
        The original dataset with missing value.

    df_impute : pd.DataFrame (n_samples, n_components)
        The dataset after imputation.

    test : str
        The statistics test method used.

    Returns
    -------
    pvals : np.ndarray
        A numpy array containing the p-values of the tests on each column in the column order
    """
    cols = df_orig.columns
    pvals = np.array([])

    if test == "wilcoxon":
        for c in cols:
            try:
                df_new_orig = df_orig[c].dropna()
                stat, pval = wilcoxon(df_new_orig, df_impute[c])
                pvals = np.append(pvals, pval)
            except Exception:
                pvals = np.append(pvals, 0)

    if test == "kruskal":
        for c in cols:
            df_new_orig = df_orig[c].dropna()
            stat, pval = kruskal(df_new_orig, df_impute[c], nan_policy="omit")
            pvals = np.append(pvals, pval)

    return pvals


def monte_carlo_simulator(
    df_orig: pd.DataFrame,
    df_impute: pd.DataFrame,
    sample_size: int,
    iteration: int,
    test: str,
    confidence: float = 0.05,
) -> None:
    """Check which column rejects hypothesis testing, p value < significance level, to find whether
    the imputation change the distribution of the original data set.

    Parameters
    ----------
    df_orig : pd.DataFrame (n_samples, n_components)
        The original dataset with missing value.

    df_impute : pd.DataFrame (n_samples, n_components)
        The dataset after imputation.

    test : str
        The statistics test method used.

    sample_size : int
        The size of the sample for each iteration.

    iteration : int
        The number of iterations of Monte Carlo simulation.

    confidence : float
        Confidence level, default to be 0.05
    """
    random.seed(2)
    simu_pvals = np.array([0] * df_orig.shape[1])
    for i in range(iteration):
        # monte carlo sampling
        sample_idx = random.sample(range(df_orig.shape[0]), sample_size)
        sample_orig = df_orig.iloc[sample_idx]
        sample_impute = df_impute.iloc[sample_idx]

        # hypothesis testing, non-parametric test
        one_pval = test_once(df_orig=sample_orig, df_impute=sample_impute, test=test)
        simu_pvals = simu_pvals + one_pval

    # average p value
    col_res = simu_pvals / iteration
    # check which column rejects hypothesis testing, 0 < p value < significance level
    rejected_col = df_orig.columns[np.where((col_res < confidence) & (col_res > 0))[0]]

    print("Significance Level: ", confidence)
    print("The number of iterations of Monte Carlo simulation: ", iteration)
    print("The size of the sample for each iteration (half of the whole data set): ", sample_size)
    print("Average p-value: ")
    print("\n".join("{} {}".format(x, y) for x, y in zip(df_orig.columns, col_res)))
    print("Note: 'p-value < 0.05' means imputation method doesn't apply to that column.")
    print("The columns which rejects null hypothesis: ", end="")
    print("None") if not rejected_col.size else print(*list(rejected_col))
