from typing import Dict

from rich import print

from ....constants import SECTION
from ....data.data_readiness import bool_input, float_input, num_input


def extra_trees_manual_hyper_parameters() -> Dict:
    """Manually set hyperparameters.

    Returns
    -------
    hyper_parameters : dict
    """
    print("N Estimators: The number of trees in the forest.")
    print("Please specify the number of trees in the forest. A good starting range could be between 50 and 500, such as 100.")
    print("For small datasets, a range of 50 to 200 trees may be sufficient, while for larger datasets, a range of 200 to 500 trees may be appropriate.")
    n_estimators = num_input(SECTION[2], "@N Estimators: ")
    print("Max Depth: The maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.")
    print("Please specify the maximum depth of a tree. A good starting range could be between 5 and 10, such as 4.")
    max_depth = num_input(SECTION[2], "@Max Depth: ")
    print("Min Samples Split: The minimum number of samples required to split an internal node.")
    print("Please specify the minimum number of samples required to split an internal node. A good starting range could be between 2 and 10, such as 2.")
    min_samples_split = num_input(SECTION[2], "@Min Samples Split: ")
    print("Min Samples Leaf: The minimum number of samples required to be at a leaf node.")
    print("Please specify the minimum number of samples required to be at a leaf node. A good starting range could be between 1 and 10, such as 1.")
    min_samples_leaf = num_input(SECTION[2], "@Min Samples Leaf: ")
    print("Max Features: The number of features to consider when looking for the best split.")
    print("Please specify the number of features to consider when looking for the best split. A good starting range could be between 1 and the total number of features in the dataset.")
    print("It's recommended to start with a value of sqrt(n_features) or log2(n_features). Make sure to use integer values.")
    max_features = num_input(SECTION[2], "@Max Features: ")
    print(
        "Bootstrap: Whether bootstrap samples are used when building trees. Bootstrapping is a technique where a random subset of the data is sampled with replacement to create a new dataset"
        "  of the same size as the original. This new dataset is then used to construct a decision tree in the ensemble. If False, the whole dataset is used to build each tree."
    )
    print("Please specify whether bootstrap samples are used when building trees. It is generally recommended to leave it set to True.")
    bootstrap = bool_input(SECTION[2])
    max_samples = None
    if bootstrap:
        print("Max Samples: The number of samples to draw from X_train to train each base estimator.")
        print("Please specify the ratio. A good starting range could be between 0.5 and 1, such as 0.8.")
        max_samples = float_input(0.8, SECTION[2], "@Max Samples: ")
    print(
        "oob_score: Whether to use out-of-bag samples to estimate the generalization accuracy. When the oob_score hyperparameter is set to True, Extra Trees will use a random subset of the data"
        " to train each decision tree in the ensemble, and the remaining data that was not used for training (the out-of-bag samples) will be used to calculate the OOB score. "
    )
    print("Please specify whether to use out-of-bag samples to estimate the generalization accuracy. It is generally recommended to leave it set to True.")
    oob_score = bool_input(SECTION[2])
    hyper_parameters = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "bootstrap": bootstrap,
        "oob_score": oob_score,
    }
    if not max_samples:
        # Use the default value provided by sklearn.ensemble.ExtraTreesClassifier.
        hyper_parameters["max_samples"] = None
    else:
        hyper_parameters["max_samples"] = max_samples
    return hyper_parameters
