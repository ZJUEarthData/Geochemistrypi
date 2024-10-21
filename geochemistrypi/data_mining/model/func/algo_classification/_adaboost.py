from typing import Dict

from rich import print

from ....constants import SECTION
from ....data.data_readiness import num_input, str_input, float_input


def adaboost_manual_hyper_parameters() -> Dict:
    """
    Manually set hyperparameters.

        Returns
        -------
        hyper_parameters : dict
    """
    print("N Estimators: The number of trees in the AdaBoost.")
    print("Please specify the number of trees in the forest. A good starting range could be between 50 and 500, such as 100.")
    n_estimators = num_input(SECTION[2], "@N Estimators: ")
    print("Learning Rate: It controls the step-size in updating the weights. It shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators.")
    print("Please specify the initial learning rate of Xgboost, such as 0.1.")
    learning_rate = float_input(0.01, SECTION[2], "@Learning Rate: ")
    hyper_parameters = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
    }
    return hyper_parameters
