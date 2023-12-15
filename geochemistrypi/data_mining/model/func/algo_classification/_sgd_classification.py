from typing import Dict

from rich import print

from ....constants import SECTION
from ....data.data_readiness import bool_input, float_input, num_input, str_input


def sgd_classificaiton_manual_hyper_parameters() -> Dict:
    """Manually set hyperparameters for SGD Classification.

    Returns
    -------
    hyper_parameters : dict
        The hyperparameters.
    """
    print("Loss Function: The loss function to be used. Choose from 'log_loss', 'modified_huber'.")
    losses = ["log_loss", "modified_huber"]
    loss = str_input(losses, SECTION[2])

    print("Penalty: The penalty (regularization term) to be used. Choose from 'l2', 'l1', 'elasticnet', or None. The default is 'l2'.")
    penalties = ["l2", "l1", "elasticnet", "None"]
    penalty = str_input(penalties, SECTION[2])
    if penalty == "None":
        penalty = None

    l1_ratio = None
    if penalty == "elasticnet":
        print("L1 Ratio: The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1. The default is 0.15.")
        l1_ratio = float_input(0.15, SECTION[2], "@L1 Ratio: ")

    print("Alpha: The regularization strength. A good starting value could be between 0.0001 and 1, such as 0.01. The default is 0.0001.")
    alpha = float_input(0.0001, SECTION[2], "@Alpha: ")

    print("Fit Intercept: Whether to fit an intercept term. It is generally recommended to set it as True.")
    fit_intercept = bool_input(SECTION[2])

    print("Maximum Number of Iterations: The maximum number of iterations for training. The default is 1000.")
    max_iter = num_input(SECTION[2], "@Max Iterations: ")

    print("Tolerance: The tolerance for stopping criteria. A good starting value could be between 0.000001 and 0.001, such as 0.0001. The default is 0.001.")
    tol = float_input(0.001, SECTION[2], "@Tolerance: ")

    print("Shuffle: Whether or not to shuffle the training data after each epoch. The default is 'True'.")
    shuffle = bool_input(SECTION[2])

    print("Learning Rate: The learning rate schedule. Choose from 'constant', 'optimal', 'invscaling', or 'adaptive'. The default is 'optimal'.")
    learning_rates = ["constant", "optimal", "invscaling", "adaptive"]
    learning_rate = str_input(learning_rates, SECTION[2])

    print("Initial Learning Rate: The initial learning rate for the 'constant', 'invscaling' or 'adaptive' schedules. The default value is 0.0.")
    eta0 = float_input(0.0, SECTION[2], "@Initial Learning Rate: ")

    print("Power T: The exponent for inverse scaling learning rate. The default is 0.5.")
    power_t = float_input(0.5, SECTION[2], "@Power T: ")

    print("Early Stopping: Whether to use early stopping to terminate training when validation score is not improving. The default is 'False'.")
    early_stopping = bool_input(SECTION[2])

    print("Validation Fraction: The proportion of training data to set aside as validation set for early stopping.")
    print("It must be in range (0, 1). A good starting value could be between 0.000001 and 1, such as 0.1. The default is 0.1.")
    is_valid = False
    while not is_valid:
        validation_fraction = float_input(0.1, SECTION[2], "@Validation Fraction: ")
        if 0 < validation_fraction < 1:
            is_valid = True
        else:
            print("The validation fraction must be in range (0, 1).")
    print("Number of Iterations With No Improvement: Number of iterations with no improvement to wait before stopping fitting.")
    print("A good starting value could be between 1 and maximum number of iterations, such as 5. The default is 5.")
    n_iter_no_change = num_input(SECTION[2], "@Iterations With No Improvement: ")

    hyper_parameters = {
        "loss": loss,
        "penalty": penalty,
        "alpha": alpha,
        "fit_intercept": fit_intercept,
        "max_iter": max_iter,
        "tol": tol,
        "shuffle": shuffle,
        "learning_rate": learning_rate,
        "eta0": eta0,
        "power_t": power_t,
        "early_stopping": early_stopping,
        "validation_fraction": validation_fraction,
        "n_iter_no_change": n_iter_no_change,
    }
    if not l1_ratio:
        # Use the default value provided by sklearn.ensemble.SGDClassifier.
        hyper_parameters["l1_ratio"] = 0.15
    else:
        hyper_parameters["l1_ratio"] = l1_ratio

    return hyper_parameters
