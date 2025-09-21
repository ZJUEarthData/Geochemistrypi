from typing import Dict

from rich import print

from ....constants import SECTION
from ....data.data_readiness import float_input, num_input, str_input


def affinitypropagation_manual_hyper_parameters() -> Dict:
    """Manually set hyperparameters.

    Returns
    -------
    hyper_parameters : dict
    """
    print("damping: The extent to which the current value is maintained relative to incoming values ")
    print("Please specify the number of clusters for AffinityPropagation. A good starting range could be between 0.5 and 1, such as 0.5.")
    damping = float_input(0.5, SECTION[2], "damping: ")
    print("Max Iter: Maximum number of iterations of the algorithm for a single run.")
    print("Please specify the maximum number of iterations of the affinitypropagation algorithm for a single run. A good starting range could be between 100 and 400, such as 200.")
    max_iter = num_input(SECTION[2], "Max Iter: ")
    print("convergence_iter: Number of iterations with no change in the number of estimated clusters that stops the convergence.")
    print("Please specify the convergence number of iterations of the affinitypropagation algorithm. A good starting range could be between 10 and 15, such as 15.")
    convergence_iter = num_input(SECTION[2], "convergence_iter: ")
    print("affinity: Different affinity methods for affinitypropagation in clustering.")
    print("Please specify the affinity to use for the computation. It is generally recommended to leave it as 'euclidean'.")
    affinity = ["euclidean", "precomputed"]
    affinity = str_input(affinity, SECTION[2])
    hyper_parameters = {"damping": damping, "max_iter": max_iter, "convergence_iter": convergence_iter, "affinity": affinity}
    return hyper_parameters
