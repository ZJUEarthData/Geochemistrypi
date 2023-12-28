from typing import Dict

from rich import print

from ....constants import SECTION
from ....data.data_readiness import float_input, num_input, str_input


def dbscan_manual_hyper_parameters() -> Dict:
    """Manually set hyperparameters.

    Returns
    -------
    hyper_parameters : dict
    """
    print("Eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.")
    print("Please specify the maximum distance. A good starting range could be between 0.1 and 1.0, such as 0.5.")
    eps = float_input(0.5, SECTION[2], "Eps: ")
    print("Min Samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.")
    print("Please specify the number of samples. A good starting range could be between 5 and 20, such as 5.")
    min_samples = num_input(SECTION[2], "Min Samples: ")
    print("Metric: The metric to use when calculating distance between instances in a feature array.")
    print("Please specify the metric to use when calculating distance between instances in a feature array. It is generally recommended to leave it as 'euclidean'.")
    metrics = ["euclidean", "manhattan", "chebyshev", "minkowski", "cosine", "correlation"]
    metric = str_input(metrics, SECTION[2])
    print("Algorithm: The algorithm to be used by the NearestNeighbors module to compute pointwise distances and find nearest neighbors.")
    print("Please specify the algorithm. It is generally recommended to leave it as 'auto'.")
    algorithms = ["auto", "ball_tree", "kd_tree", "brute"]
    algorithm = str_input(algorithms, SECTION[2])
    print("Leaf Size: Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree.")
    print("Please specify the leaf size. A good starting range could be between 10 and 30, such as 30.")
    leaf_size = num_input(SECTION[2], "Leaf Size: ")
    p = None
    if metric == "minkowski":
        print("P: The power of the Minkowski metric to be used to calculate distance between points.")
        print("Please specify the power of the Minkowski metric. A good starting range could be between 1 and 2, such as 2.")
        p = num_input(SECTION[2], "P: ")
    hyper_parameters = {
        "eps": eps,
        "min_samples": min_samples,
        "metric": metric,
        "algorithm": algorithm,
        "leaf_size": leaf_size,
        "p": p,
    }
    return hyper_parameters
