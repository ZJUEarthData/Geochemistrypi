from typing import Dict

from rich import print

from ....constants import SECTION
from ....data.data_readiness import num_input, str_input


def decision_tree_manual_hyper_parameters() -> Dict:
    """Manually set hyperparameters.

    Returns
    -------
    hype_parameters : dict
    """
    print("Criterion: The function to measure the quality of a split. Supported criteria are 'gini' for the Gini impurity and 'log_loss' and 'entropy' both for the Shannon information gain.")
    print("The default value is 'gini'. Optional criterions are 'entropy' and 'log_loss'.")
    criterions = ["gini", "entropy", "log_loss"]
    criterion = str_input(criterions, SECTION[2])
    print("Max Depth: The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.")
    print("Please specify the maximum depth of the tree. A good starting range could be between 3 and 15, such as 4.")
    max_depth = num_input(SECTION[2], "@Max Depth: ")
    print("Min Samples Split: The minimum number of samples required to split an internal node.")
    print("Please specify the minimum number of samples required to split an internal node. A good starting range could be between 2 and 10, such as 3.")
    min_samples_split = num_input(SECTION[2], "@Min Samples Split: ")
    print("Min Samples Leaf: The minimum number of samples required to be at a leaf node.")
    print("Please specify the minimum number of samples required to be at a leaf node. A good starting range could be between 1 and 10, such as 2.")
    min_samples_leaf = num_input(SECTION[2], "@Min Samples Leaf: ")
    print("Max Features: The number of features to consider when looking for the best split.")
    print("Please specify the number of features to consider when looking for the best split. A good starting range could be between 1 and the total number of features in the dataset.")
    max_features = num_input(SECTION[2], "@Max Features: ")
    hyper_parameters = {
        "criterion": criterion,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
    }
    return hyper_parameters
