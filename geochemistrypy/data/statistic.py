# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, kruskal
import random


def test_once(df_orig: pd.DataFrame, df_impute: pd.DataFrame, test: str = 'wilcoxon') -> np.ndarray:
    """Do hypothesis testing on each pair-wise column once, non-parametric test.
    Null hypothesis: the distributions of the data set before and after imputing remain the same.

    :param df_orig: pd.DataFrame, the original dataset with missing value
    :param df_impute: pd.DataFrame, the dataset after imputation
    :param test:str, the statistics test method used
    :return: a numpy array containing the p-values of the tests on each column in the column order
    """
    cols = df_orig.columns
    pvals = np.array([])

    if test == 'wilcoxon':
        for c in cols:
            try:
                stat, pval = wilcoxon(df_orig[c], df_impute[c])
                pvals = np.append(pvals, pval)
            except Exception:
                pvals = np.append(pvals, 0)

    if test == 'kruskal':
        for c in cols:
            stat, pval = kruskal(df_orig[c], df_impute[c], nan_policy='omit')
            pvals = np.append(pvals, pval)

    return pvals


def monte_carlo_simulator(df_orig: pd.DataFrame, df_impute: pd.DataFrame, sample_size: int, iteration: int,
                          test: str = 'wilcoxon', confidence: float = 0.05) -> pd.DataFrame:
    """Check which column rejects hypothesis testing, p value < significance level, to find whether
    the imputation change the distribution of the original data set.

    :param df_orig: The original dataset with missing value
    :param df_impute: The dataset after imputation
    :param test: The statistics test used
    :param sample_size: The size of the sample for each iteration
    :param iteration: Number of iterations of Monte Carlo Simulation
    :param confidence: Confidence level, default to be 0.05
    :return: The column names that reject the null hypothesis,
    """
    random.seed(2)
    simu_pvals = np.array([0] * df_orig.shape[1])
    for i in range(iteration):
        # TODO(sany hecan@mail2.sysu.edu.cn): which way to perform monte carlo sampling, random.sample?
        # monte carlo sampling
        sample_idx = random.sample(range(df_orig.shape[0]), sample_size)
        sample_orig = df_orig.iloc[sample_idx]
        sample_impute = df_impute.iloc[sample_idx]

        # hypothesis testing, non-parametric test
        one_pval = test_once(df_orig=sample_orig, df_impute=sample_impute, test=test)
        simu_pvals = simu_pvals + one_pval

    # average p value
    col_res = simu_pvals / iteration
    # check which column rejects hypothesis testing, p value < significance level
    return df_orig.columns[np.where(col_res < confidence)[0]]
