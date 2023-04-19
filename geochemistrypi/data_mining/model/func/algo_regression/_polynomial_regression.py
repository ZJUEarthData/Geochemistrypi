# -*- coding: utf-8 -*-
import numpy as np
from typing import Dict, List
from ....data.data_readiness import float_input, num_input, str_input
from ....global_variable import SECTION


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
    interaction_onlys = ["True", "False"]
    interaction_only = bool(str_input(interaction_onlys, SECTION[2]))
    print("Include Bias: This hyperparameter specifies whether to include a bias (also called the intercept) term in the model.")
    print("Please specify whether to include a bias term in the model. It is generally recommended to leave it set to True.")
    include_biases = ["True", "False"]
    include_bias = bool(str_input(include_biases, SECTION[2]))
    hyper_parameters = {'degree': degree, 'interaction_only': interaction_only, 'include_bias': include_bias}
    return hyper_parameters


def show_formula(coef: np.ndarray, intercept: np.ndarray, features_name: List) -> None:
    """Show the formula of polynomial regression.
    
    Parameters
    ----------
    coef : array
        Coefficient of the features in the decision function.

    intercept : array
        Independent term in decision function.
    
    features_name : list
        Name of the features.
    """
    term = []
    coef = np.around(coef, decimals=3).tolist()[0]

    for i in range(len(coef)):
        # the first value stay the same
        if i == 0:
            # not append if zero
            if coef[i] != 0:
                temp = str(coef[i]) + features_name[i]
                term.append(temp)
        else:
            # add plus symbol if positive, maintain if negative, not append if zero
            if coef[i] > 0:
                temp = '+' + str(coef[i]) + features_name[i]
                term.append(temp)
            elif coef[i] < 0:
                temp = str(coef[i]) + features_name[i]
                term.append(temp)
    if intercept[0] >= 0:
        # formula of polynomial regression
        formula = ''.join(term) + '+' + str(intercept[0])
    else:
        formula = ''.join(term) + str(intercept[0])
    print('y =', formula)
