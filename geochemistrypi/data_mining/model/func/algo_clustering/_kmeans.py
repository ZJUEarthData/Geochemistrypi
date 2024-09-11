from typing import Dict

from rich import print

from ....constants import SECTION
from ....data.data_readiness import float_input, num_input, str_input


def kmeans_manual_hyper_parameters() -> Dict:
    """Manually set hyperparameters.

    Returns
    -------
    hyper_parameters : dict
    """
    print("N Clusters: The number of clusters to form as well as the number of centroids to generate.")
    print("Please specify the number of clusters for KMeans. A good starting range could be between 2 and 10, such as 4.")
    n_clusters = num_input(SECTION[2], "N Clusters: ")
    print("Init: Method for initialization of centroids. The centroids represent the center points of the clusters in the dataset.")
    print("Please specify the method for initialization of centroids. It is generally recommended to leave it as 'k-means++'.")
    inits = ["k-means++", "random"]
    init = str_input(inits, SECTION[2])
    print("Max Iter: Maximum number of iterations of the k-means algorithm for a single run.")
    print("Please specify the maximum number of iterations of the k-means algorithm for a single run. A good starting range could be between 100 and 500, such as 300.")
    max_iters = num_input(SECTION[2], "Max Iter: ")
    print("Tolerance: Relative tolerance with regards to inertia to declare convergence.")
    print("Please specify the relative tolerance with regards to inertia to declare convergence. A good starting range could be between 0.0001 and 0.001, such as 0.0005.")
    tol = float_input(0.0005, SECTION[2], "Tolerance: ")
    print("Algorithm: The algorithm to use for the computation.")
    print("Please specify the algorithm to use for the computation. It is generally recommended to leave it as 'auto'.")
    print("Auto: selects 'elkan' for dense data and 'full' for sparse data. 'elkan' is generally faster on data with lower dimensionality, while 'full' is faster on data with higher dimensionality")
    algorithms = ["auto", "full", "elkan"]
    algorithm = str_input(algorithms, SECTION[2])
    hyper_parameters = {"n_clusters": n_clusters, "init": init, "max_iter": max_iters, "tol": tol, "algorithm": algorithm}
    return hyper_parameters
