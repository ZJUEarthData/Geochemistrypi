from typing import Dict

from rich import print

from ....constants import SECTION
from ....data.data_readiness import float_input, num_input, str_input


def gradient_boosting_manual_hyper_parameters() -> Dict:
    """Manually set hyperparameters.

    Returns
    -------
    hyper_parameters : dict
    """
    print("N Estimators: The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.")
    print("Please specify the number of boosting stages to perform. A good starting value could be between 50 and 500, such as 100.")
    n_estimators = num_input(SECTION[2], "@N Estimators: ")

    print("Learning Rate: Learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators.")
    print("Please specify the learning rate. A good starting value could be between 0.01 and 0.2, such as 0.1.")
    learning_rate = float_input(0.1, SECTION[2], "@Learning Rate: ")

    print("Max Depth: The maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.")
    print("Please specify the maximum depth of a tree. A good starting value could be between 5 and 10, such as 4.")
    max_depth = num_input(SECTION[2], "@Max Depth: ")

    print("Min Samples Split: The minimum number of samples required to split an internal node.")
    print("Please specify the minimum number of samples required to split an internal node. A good starting value could be between 2 and 10, such as 2.")
    min_samples_split = num_input(SECTION[2], "@Min Samples Split: ")

    print("Min Samples Leaf: The minimum number of samples required to be at a leaf node.")
    print("Please specify the minimum number. A good starting value could be between 1 and 10, such as 1.")
    min_samples_leaf = num_input(SECTION[2], "@Min Samples Leaf: ")

    print("Max Features: The number of features to consider when looking for the best split.")
    print("Please specify the number of features. A good starting value could be between 1 and the total number of features in the dataset.")
    print("It's recommended to start with a value of sqrt(n_features) or log2(n_features). Make sure to use integer values.")
    max_features = num_input(SECTION[2], "@Max Features: ")

    print("Subsample: The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in Stochastic Gradient Boosting.")
    print("Please specify the fraction of samples. A good starting value could be between 0.5 and 1.0, such as 1.0.")
    subsample = float_input(1.0, SECTION[2], "@Subsample: ")

    print("Loss: It refers to the loss function to be minimized in each split.")
    print("Please specify the loss function. It is generally recommended to leave it to 'log_loss'.")
    losses = ["log_loss", "exponential"]
    loss = str_input(losses, SECTION[2])
    hyper_parameters = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "subsample": subsample,
        "loss": loss,
    }
    return hyper_parameters
