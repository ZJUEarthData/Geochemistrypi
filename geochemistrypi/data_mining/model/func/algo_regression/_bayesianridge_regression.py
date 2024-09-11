from typing import Dict

from rich import print

from ....constants import SECTION
from ....data.data_readiness import bool_input, float_input


def bayesian_ridge_manual_hyper_parameters() -> Dict:
    """Manually set hyperparameters for Bayesian Ridge Regression.

    Returns
    -------
    hyper_parameters : dict
        The hyperparameters.
    """
    # print("Maximum Number of Iterations: The maximum number of iterations for training. The default is 300.")
    # max_iter = num_input(SECTION[2], "@Max Iterations: ")

    print("Tolerance: The tolerance for stopping criteria. A good starting value could be between 1e-6 and 1e-3, such as 1e-4. The default is 0.001.")
    tol = float_input(1e-4, SECTION[2], "@Tol: ")

    print("Alpha 1: Hyperparameter for the Gamma distribution prior over the alpha parameter. A good starting value could be between 1e-6 and 1e-3, such as 1e-6. The default is 0.000001.")
    alpha_1 = float_input(1e-6, SECTION[2], "@Alpha 1: ")

    print("Alpha 2: Hyperparameter for the Gamma distribution prior over the beta parameter. A good starting value could be between 1e-6 and 1e-3, such as 1e-6. The default is 0.000001.")
    alpha_2 = float_input(1e-6, SECTION[2], "@Alpha 2: ")

    print("Lambda 1: Hyperparameter for the Gaussian distribution prior over the weights. A good starting value could be between 1e-6 and 1e-3, such as 1e-6. The default is 0.000001.")
    lambda_1 = float_input(1e-6, SECTION[2], "@Lambda 1: ")

    print("Lambda 2: Hyperparameter for the Gamma distribution prior over the noise. A good starting value could be between 1e-6 and 1e-3, such as 1e-6. The default is 0.000001.")
    lambda_2 = float_input(1e-6, SECTION[2], "@Lambda 2: ")

    print("Alpha Init: Initial guess for alpha. The default is 0.000001.")
    alpha_init = float_input(1.0, SECTION[2], "@Alpha Init: ")

    print("Lambda Init: Initial guess for lambda. The default is 0.000001.")
    lambda_init = float_input(1.0, SECTION[2], "@Lambda Init: ")

    print("Compute Score: Whether to compute the objective function at each step. The default is False.")
    compute_score = bool_input(SECTION[2])

    print("Fit Intercept: Whether to fit an intercept term. It is generally recommended to set it as True.")
    fit_intercept = bool_input(SECTION[2])

    print("Copy X: Whether to copy X in fit. The default is True.")
    copy_X = bool_input(SECTION[2])

    print("Verbose: Verbosity of the solution. The default is False.")
    verbose = bool_input(SECTION[2])

    hyper_parameters = {
        "tol": tol,
        "alpha_1": alpha_1,
        "alpha_2": alpha_2,
        "lambda_1": lambda_1,
        "lambda_2": lambda_2,
        "alpha_init": alpha_init,
        "lambda_init": lambda_init,
        "compute_score": compute_score,
        "fit_intercept": fit_intercept,
        "copy_X": copy_X,
        "verbose": verbose,
    }
    return hyper_parameters
