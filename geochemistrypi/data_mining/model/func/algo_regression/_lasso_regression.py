# -*- coding: utf-8 -*-
from typing import Dict

from rich import print

from ....constants import SECTION
from ....data.data_readiness import bool_input, float_input, num_input, str_input


def lasso_regression_manual_hyper_parameters() -> Dict:
    """Manually set hyperparameters.

    Returns
    -------
    hyper_parameters : dict
    """
    print("Alpha: This hyperparameter represents the coefficient of the norm, which controls the degree of contraction constraint.")
    print("Please indicate the coefficient of alpha. A good starting range could be between 0.001 and 2, such as 1.")
    alpha = float_input(0.01, SECTION[2], "@Alpha: ")
    print("Fit Intercept: This hyperparameter represents whether the model is evaluated with constant terms.")
    print("Please indicate whether there is a parameter entry. It is generally recommended to leave it set to True.")
    fit_intercept = bool_input(SECTION[2])
    print("Max Iter: This hyperparameter represents the maximum number of iterations for the solver to converge.")
    print("Please indicate the maximum number of iterations. A good starting range could be between 1000 and 10000, such as 1000.")
    max_iter = num_input(SECTION[2], "@Max Iter: ")
    print("Tolerance: This hyperparameter represents the tolerance of the optimization method.")
    print("Please indicate the tolerance. A good starting range could be between 0.0001 and 0.001, such as 0.0001.")
    tol = float_input(0.0001, SECTION[2], "@Tolerance: ")
    print("Selection: This hyperparameter represents the method of selecting the regularization coefficient.")
    print("Please indicate the method of selecting the regularization coefficient. It is generally recommended to leave it set to 'cyclic'.")
    selections = ["cyclic", "random"]
    selection = str_input(selections, SECTION[2])
    hyper_parameters = {
        "alpha": alpha,
        "fit_intercept": fit_intercept,
        "max_iter": max_iter,
        "tol": tol,
        "selection": selection,
    }
    return hyper_parameters
