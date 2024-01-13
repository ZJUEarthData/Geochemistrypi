# -*- coding: utf-8 -*-
from typing import Dict

from rich import print

from ....constants import SECTION
from ....data.data_readiness import float_input, num_input, str_input, tuple_input


def multi_layer_perceptron_manual_hyper_parameters() -> Dict:
    """Manually set hyperparameters.

    Returns
    -------
    hyper_parameters : dict
    """
    print("Hidden Layer Sizes: The ith element represents the number of neurons in the ith hidden layer.")
    print("Please specify the size of hidden layer and the number of neurons in the each hidden layer.")
    hidden_layer = tuple_input((50, 25, 5), SECTION[2], "@Hidden Layer Sizes: ")
    print("Activation: Activation function for the hidden layer.")
    print("Please specify the activation function for the hidden layer. It is generally recommended to leave it set to ReLU.")
    activations = ["identity", "logistic", "tanh", "relu"]
    activation = str_input(activations, SECTION[2])
    print("Solver: The solver for weight optimization.")
    print("Please specify the solver for weight optimization. It is generally recommended to leave it set to Adam.")
    solvers = ["lbfgs", "sgd", "adam"]
    solver = str_input(solvers, SECTION[2])
    print("Alpha: L2 penalty (regularization term) parameter.")
    print("Please specify the L2 penalty (regularization term) parameter. A good starting range could be between 0.0001 and 10, such as 0.0001.")
    alpha = float_input(0.0001, SECTION[2], "@Alpha: ")
    print("Learning Rate: It controls the step-size in updating the weights.")
    print("Please specify the learning rate. It is generally recommended to leave it set to Adaptive.")
    learning_rates = ["constant", "invscaling", "adaptive"]
    learning_rate = str_input(learning_rates, SECTION[2])
    print("Max Iterations: Maximum number of iterations. The solver iterates until convergence (determined by 'tol') or this number of iterations.")
    print("Please specify the maximum number of iterations. A good starting range could be between 100 and 1000, such as 200.")
    max_iter = num_input(SECTION[2], "@Max Iterations: ")
    hyper_parameters = {
        "hidden_layer_sizes": hidden_layer,
        "activation": activation,
        "solver": solver,
        "alpha": alpha,
        "learning_rate": learning_rate,
        "max_iter": max_iter,
    }
    return hyper_parameters
