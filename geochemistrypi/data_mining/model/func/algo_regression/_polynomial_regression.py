# -*- coding: utf-8 -*-
from typing import Dict

from rich import print

from ....constants import SECTION
from ....data.data_readiness import bool_input, num_input


def polynomial_regression_manual_hyper_parameters() -> Dict:
    """Manually set hyperparameters.

    Returns
    -------
    hyper_parameters : dict
    """
    print("Degree: This hyperparameter specifies the degree of the polynomial features.")
    print("Please specify the degree of the polynomial features. A good starting range could be between 1 and 5, such as 2.")
    degree = num_input(SECTION[2], "@Degree: ")
    print("Interaction Only: This hyperparameter specifies whether to only include interaction features.")
    print("Please specify whether to only include interaction features. It is generally recommended to leave it set to False.")
    interaction_only = bool_input(SECTION[2])
    print("Include Bias: This hyperparameter specifies whether to include a bias (also called the intercept) term in the model.")
    print("Please specify whether to include a bias term in the model. It is generally recommended to leave it set to True.")
    include_bias = bool_input(SECTION[2])
    hyper_parameters = {"degree": degree, "interaction_only": interaction_only, "include_bias": include_bias}
    return hyper_parameters
