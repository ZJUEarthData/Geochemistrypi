from typing import Dict

from rich import print

from ....constants import SECTION
from ....data.data_readiness import float_input, num_input, str_input


def ada_boost_manual_hyper_parameters() -> Dict:
    """Manually set hyperparameters for AdaBoost.

    Returns
    -------
    hyper_parameters : dict
        The hyperparameters.
    """

    print("N Estimators: The maximum number of estimators at which boosting is terminated. A good starting range could be between 50 and 500, such as 100.")
    n_estimators = num_input(SECTION[2], "@N Estimators: ")
    print("Learning Rate: A higher learning rate increases the contribution of each regressor. A good starting range could be between 0.01 and 1.0, such as 0.1.")
    learning_rate = float_input(0.1, SECTION[2], "@Learning Rate: ")
    print("Loss:  The loss function to use when updating the weights after each boosting iteration. It is generally recommended to leave it as 'linear'.")
    losses = ["linear", "square", "exponential"]
    loss = str_input(losses, SECTION[2])

    hyper_parameters = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "loss": loss,
    }

    return hyper_parameters
