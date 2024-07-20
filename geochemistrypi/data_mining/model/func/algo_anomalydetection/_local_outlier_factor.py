# -*- coding: utf-8 -*-
from typing import Dict

from rich import print

from ....constants import SECTION
from ....data.data_readiness import float_input, num_input


def local_outlier_factor_manual_hyper_parameters() -> Dict:
    """Manually set hyperparameters.

    Returns
    -------
    hyper_parameters : dict
    """
    print("N neighbors: The number of neighbors to use.")
    print("Please specify the number of neighbors. A good starting range could be between 10 and 50, such as 20.")
    n_neighbors = num_input(SECTION[2], "@N Neighbors: ")
    print("Leaf size: The leaf size used in the ball tree or KD tree.")
    print("Please specify the leaf size. A good starting range could be between 20 and 50, such as 30.")
    leaf_size = num_input(SECTION[2], "@Leaf Size: ")
    print("P: The power parameter for the Minkowski metric.")
    print("Please specify the power parameter. When p = 1, this is equivalent to using manhattan_distance, and when p = 2 euclidean_distance is applied. For arbitrary p, minkowski_distance is used.")
    p = float_input(2.0, SECTION[2], "@P: ")
    print("Contamination: The amount of contamination of the data set.")
    print("Please specify the contamination of the data set. A good starting range could be between 0.1 and 0.5, such as 0.3.")
    contamination = float_input(0.3, SECTION[2], "@Contamination: ")
    print("N jobs: The number of parallel jobs to run.")
    print("Please specify the number of jobs. Use -1 to use all available CPUs, 1 for no parallelism, or specify the number of CPUs to use. A good starting value is 1.")
    n_jobs = num_input(SECTION[2], "@N Jobs: ")
    hyper_parameters = {
        "n_neighbors": n_neighbors,
        "leaf_size": leaf_size,
        "p": p,
        "contamination": contamination,
        "n_jobs": n_jobs,
    }
    return hyper_parameters
