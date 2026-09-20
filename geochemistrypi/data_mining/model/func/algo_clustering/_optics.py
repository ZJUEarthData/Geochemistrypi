from typing import Dict

import numpy as np
from rich import print

from ....constants import SECTION
from ....data.data_readiness import float_input, int_input, num_input, str_input


def OPTICS_manual_hyper_parameters() -> Dict:
    """Manually set hyperparameters.

    Returns
    -------
    hyper_parameters : dict

    """
    print("max_eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.")
    print("Default value of ``np.inf`` will identify clusters across all scales; reducing ``max_eps`` will result in shorter run times.")
    max_eps = float_input(np.inf, SECTION[2], "max_eps: ")

    print("min_samples: The number of samples in a neighborhood for a point to be considered as a core point")
    print("A good starting value could be int > 1, such as 5.")
    min_samples = int_input(5, SECTION[2], "min_samples: ")

    print("algorithm: Algorithm used to compute the nearest neighbors")
    print("Please specify the algorithm. It is generally recommended to leave it as 'auto'.")
    algorithms = ["auto", "ball_tree", "kd_tree", "brute"]
    algorithm = str_input(algorithms, SECTION[2])

    print("metric: The metric to use when calculating distance between instances in a feature array.")
    print("Please specify the metric to use when calculating distance between instances in a feature array. It is generally recommended to leave it as 'minkowski'.")
    if algorithm == "kd_tree":
        metrics = ["euclidean", "l2", "minkowski", "p", "manhattan", "cityblock", "l1", "chebyshev", "infinity"]
    elif algorithm == "ball_tree":
        metrics = [
            "euclidean",
            "l2",
            "minkowski",
            "p",
            "manhattan",
            "cityblock",
            "l1",
            "chebyshev",
            "infinity",
            "seuclidean",
            "mahalanobis",
            "hamming",
            "canberra",
            "braycurtis",
            "jaccard",
            "dice",
            "rogerstanimoto",
            "russellrao",
            "sokalmichener",
            "sokalsneath",
            "haversine",
        ]
    else:
        metrics = ["euclidean", "manhattan", "chebyshev", "minkowski", "cosine", "correlation"]
    metric = str_input(metrics, SECTION[2])

    print("cluster_method: The extraction method used to extract clusters using the calculated reachability and ordering.")
    print("Please specify the method. It is generally recommended to leave it as 'xi'.")
    cluster_methods = ["xi", "dbscan"]
    cluster_method = str_input(cluster_methods, SECTION[2])

    print("Leaf Size: Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree.")
    print("Please specify the leaf size. A good starting range could be between 10 and 30, such as 30.")
    leaf_size = num_input(SECTION[2], "Leaf Size: ")

    p = None
    if metric == "minkowski":
        print("P: The power of the Minkowski metric to be used to calculate distance between points.")
        print("Please specify the power of the Minkowski metric. A good starting range could be between 1 and 2, such as 2.")
        p = num_input(SECTION[2], "P: ")

    eps = None
    xi = None

    if cluster_method == "dbscan":
        print("Eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.")
        print("Please specify the maximum distance. A good starting range could be between 0.1 and 1.0, such as 0.5.")
        eps = float_input(0.5, SECTION[2], "Eps: ")

        predecessor_correction = None
        min_cluster_size = None

    if cluster_method == "xi":
        print("xi: minimum steepness on the reachability plot that constitutes a cluster boundary.")
        print("A good starting range would be float between 0 and 1, such as 0.05.")
        xi = float_input(0.05, SECTION[2], "xi: ")

        """
        print("predecessor_correction: Correct clusters according to the predecessors calculated by OPTICS")
        print("It is generally recommended to leave it as True")
        predecessor_correction = bool_input(SECTION[2], "predecessor_correction: ")
        """
        predecessor_correction = True

        print("min_cluster_size: Minimum number of samples in an OPTICS cluster, expressed as an absolute number or a fraction of the number of samples")
        print("A good starting range would be int > 1 or float between 0 and 1, such as None")
        min_cluster_size = int_input(None, SECTION[2], "min_cluster_size: ")

    # Reference:

    hyper_parameters = {
        "min_samples": min_samples,
        "max_eps": max_eps,
        "metric": metric,
        "p": p,
        "cluster_method": cluster_method,
        "eps": eps,
        "xi": xi,
        "predecessor_correction": predecessor_correction,
        "min_cluster_size": min_cluster_size,
        "algorithm": algorithm,
        "leaf_size": leaf_size,
    }
    return hyper_parameters
