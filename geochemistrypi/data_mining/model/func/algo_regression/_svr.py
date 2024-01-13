# -*- coding: utf-8 -*-
from typing import Dict

from rich import print

from ....constants import SECTION
from ....data.data_readiness import bool_input, float_input, num_input, str_input


def svr_manual_hyper_parameters() -> Dict:
    """Manually set hyperparameters.

    Returns
    -------
    hyper_parameters : dict
    """
    print("Kernel: This hyperparameter specifies the kernel function to be used for mapping the input data to a higher-dimensional feature space.")
    print("Please specify the kernel type to be used in the algorithm. It is generally recommended to leave it set to Radial basis function (RBF) kernel.")
    kernels = ["linear", "poly", "rbf", "sigmoid"]
    kernel = str_input(kernels, SECTION[2])
    degree = None
    if kernel == "poly":
        print("Degree: This hyperparameter is only used when the kernel is set to polynomial. It specifies the degree of the polynomial kernel function.")
        print("Please specify the degree of the polynomial kernel function. A good starting range could be between 2 and 5, such as 3.")
        degree = num_input(SECTION[2], "@Degree: ")
    gamma = None
    if kernel in ["poly", "rbf", "sigmoid"]:
        print("Gamma: This hyperparameter is only used when the kernel is set to Polynomial, RBF, or Sigmoid. It specifies the kernel coefficient for rbf, poly and sigmoid.")
        print("Please specify the kernel coefficient for rbf, poly and sigmoid. A good starting range could be between 0.0001 and 10, such as 0.1.")
        gamma = float_input(0.1, SECTION[2], "@Gamma: ")
    # coef0 = None
    # if kernel in ["poly", "sigmoid"]:
    #     print("Coef0: This hyperparameter is only used when the kernel is set to Polynomial or Sigmoid. It specifies the independent term in kernel function.")
    #     print("The coef0 parameter controls the influence of higher-order terms in the polynomial and sigmoid kernels, and it can help to adjust the balance between the linear"
    #           " and nonlinear parts of the decision boundary.")
    #     print("Please specify the independent term in kernel function. A good starting range could be between 0.0 and 1.0, such as 0.0.")
    #     coef0 = float_input(SECTION[2], "@Coef0: ")
    print("C: This hyperparameter specifies the penalty parameter C of the error term.")
    print("The C parameter controls the trade off between smooth decision boundary and classifying the training points correctly.")
    print("Please specify the penalty parameter C of the error term. A good starting range could be between 0.001 and 1000, such as 1.0.")
    C = float_input(1, SECTION[2], "@C: ")
    print("Shrinking: This hyperparameter specifies whether to use the shrinking heuristic.")
    print("The shrinking heuristic is a technique that speeds up the training process by only considering the support vectors in the decision function.")
    print("Please specify whether to use the shrinking heuristic. It is generally recommended to leave it set to True.")
    shrinking = bool_input(SECTION[2])
    hyper_parameters = {"kernel": kernel, "C": C, "shrinking": shrinking}
    if not degree:
        # Use the default value provided by sklearn.svm.SVR
        hyper_parameters["degree"] = 3
    else:
        hyper_parameters["degree"] = degree
    if not gamma:
        # Use the default value provided by sklearn.svm.SVR
        hyper_parameters["gamma"] = "scale"
    else:
        hyper_parameters["gamma"] = gamma
    # if coef0:
    #     # Use the default value provided by sklearn.svm.SVR
    #     hyper_parameters["coef0"] = 0.0
    return hyper_parameters
