from typing import Dict

from rich import print

from ....constants import SECTION
from ....data.data_readiness import num_input, str_input


def knn_manual_hyper_parameters() -> Dict:
    """Manually set hyperparameters.

    Returns
    -------
    hyper_parameters : dict
    """
    print("N Neighbors: The number of neighbors to use by default when making predictions.")
    print(
        "Please specify the number of neighbors to use by default for kneighbors queries. A good starting value could be between 1 and"
        " the square root of the number of samples in your dataset, such as 5."
    )
    n_neighbors = num_input(SECTION[2], "@N Neighbors: ")
    print("Weights: weight function used in prediction.")
    print("Please specify the weight function used in prediction. It is generally recommended to leave it as 'uniform'.")
    weights = ["uniform", "distance"]
    weight = str_input(weights, SECTION[2])
    print("Algorithm: Algorithm used to compute the nearest neighbors.")
    print("Please specify the algorithm used to compute the nearest neighbors. It is generally recommended to leave it as 'auto'.")
    algorithms = ["auto", "ball_tree", "kd_tree", "brute"]
    algorithm = str_input(algorithms, SECTION[2])
    leaf_size = None
    if algorithm in ["ball_tree", "kd_tree"]:
        print("Leaf Size: Leaf size controls the size of leaf nodes in the tree-based algorithms ('ball_tree' and 'kd_tree')")
        print("Please specify the leaf size passed to BallTree or KDTree. A good starting value could be between 10 and 100, such as 30.")
        leaf_size = num_input(SECTION[2], "@Leaf Size: ")
    print("Metric: The distance metric to use for distance computation.")
    print("Please specify the distance metric to use for the tree. It is generally recommended to leave it as 'minkowski'.")
    metrics = ["euclidean", "manhattan", "minkowski"]
    metric = str_input(metrics, SECTION[2])
    p = None
    if metric == "minkowski":
        print("P: Power parameter for the Minkowski metric.")
        print("Please specify the power parameter for the Minkowski metric. A good starting value could be between 1 and 5, such as 2.")
        p = num_input(SECTION[2], "@P: ")
    hyper_parameters = {
        "n_neighbors": n_neighbors,
        "weights": weight,
        "algorithm": algorithm,
        "metric": metric,
    }
    if not leaf_size:
        # Use the default value provided by sklearn.neighbors.KNeighborsClassifier
        hyper_parameters["leaf_size"] = 30
    else:
        hyper_parameters["leaf_size"] = leaf_size
    if not p:
        # Use the default value provided by sklearn.neighbors.KNeighborsClassifier
        hyper_parameters["p"] = 2
    else:
        hyper_parameters["p"] = p
    return hyper_parameters
