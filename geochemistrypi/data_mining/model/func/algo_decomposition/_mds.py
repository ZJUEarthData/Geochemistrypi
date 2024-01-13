from typing import Dict

from rich import print

from ....constants import SECTION
from ....data.data_readiness import bool_input, num_input


def mds_manual_hyper_parameters() -> Dict:
    """Manually set hyperparameters.

    Returns
    -------
    hyper_parameters : dict
    """
    print("N Components: This parameter specifies the number of components to retain after dimensionality reduction.")
    print("Please specify the number of components to retain. A good starting range could be between 2 and 10, such as 4.")
    n_components = num_input(SECTION[2], "N Components: ")
    print("Metric: This parameter specifies the metric to be used when calculating distance between instances in a feature array.")
    print("Please specify whether the metric is used when measuring the pairwise distances between data points in the input space. It is generally recommended to leave it set to True.")
    metric = bool_input(SECTION[2])
    print("N Init: This parameter specifies the number of times the SMACOF algorithm will be run with different initializations.")
    print("Please specify the number of times. A good starting range could be between 1 and 10, such as 4.")
    n_init = num_input(SECTION[2], "N Init: ")
    print("Max Iter: This parameter specifies the maximum number of iterations of the SMACOF algorithm for a single run.")
    print("Please specify the maximum number of iterations. A good starting range could be between 100 and 1000, such as 300.")
    max_iter = num_input(SECTION[2], "Max Iter: ")
    hyper_parameters = {"n_components": n_components, "metric": metric, "n_init": n_init, "max_iter": max_iter}
    return hyper_parameters
