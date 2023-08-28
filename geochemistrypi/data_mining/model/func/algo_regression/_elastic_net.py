from typing import Dict

from rich import print

from ....constants import SECTION
from ....data.data_readiness import bool_input, float_input, num_input, str_input


def elastic_net_manual_hyper_parameters() -> Dict:
    """Manually set hyperparameters.

    Returns
    -------
    hyper_parameters : dict
        The hyperparameters.
    """
    print("Alpha: This hyperparameter represents the coefficient of the norm, which controls the degree of contraction constraint.")
    print("Please indicate the coefficient of alpha. A good starting value could be between 0.1 and 1, such as 1.")
    alpha = float_input(1, SECTION[2], "@Alpha: ")
    print("l1_ratio: This parameter represents the proportion of the L1 norm in the regularization term.")
    print("Please indicate the l1_ratio. A good starting value could be between 0 and 1, such as 0.5.")
    l1_ratio = float_input(0.5, SECTION[2], "@L1 Ratio: ")
    print("Fit Intercept: This hyperparameter represents whether the model is evaluated with constant terms.")
    print("Please indicate whether there is a parameter entry. It is generally recommended to set it as True.")
    fit_intercept = bool_input(SECTION[2])
    print("Max Iter: This hyperparameter represents the maximum number of iterations for the solver to converge.")
    print("Please indicate the maximum number of iterations. A good starting value could be between 1000 and 10000, such as 1000.")
    max_iter = num_input(SECTION[2], "@Max Iter: ")
    print("Tolerance: This hyperparameter represents the tolerance of the optimization method.")
    print("Please indicate the tolerance. A good starting value could be between 0.0001 and 0.001, such as 0.0001.")
    tol = float_input(0.0001, SECTION[2], "@Tolerance: ")
    print("Selection: This hyperparameter represents the method of selecting the regularization coefficient.")
    print("Please indicate the method of selecting the regularization coefficient. It is generally recommended to set it as 'cyclic'.")
    selections = ["cyclic", "random"]
    selection = str_input(selections, SECTION[2])
    hyper_parameters = {
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "fit_intercept": fit_intercept,
        "max_iter": max_iter,
        "tol": tol,
        "selection": selection,
    }
    return hyper_parameters
