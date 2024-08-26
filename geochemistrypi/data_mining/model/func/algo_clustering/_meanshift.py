from typing import Dict

from rich import print

from ....constants import SECTION
from ....data.data_readiness import num_input, str_input


def meanshift_manual_hyper_parameters() -> Dict:
    """Manually set hyperparameters for MeanShift algorithm.

    Returns
    -------
    hyper_parameters : dict
        Dictionary containing the manually set hyperparameters.
    """
    print("Bandwidth: The bandwidth of the kernel used in the algorithm. This parameter can greatly influence the results.")
    print("If you do not have a specific value in mind, you can leave this as 0, and the algorithm will estimate it automatically.")
    print("A good starting point could be around 0.5 to 1.5, depending on your data's scale.")
    bandwidth_input = num_input(SECTION[2], "Enter Bandwidth (or 0 for automatic estimation): ")
    bandwidth = None if bandwidth_input == 0 else bandwidth_input

    print("Cluster All: By default, only points at least as close to a cluster center as the given bandwidth are assigned to that cluster.")
    print("Setting this to False will prevent points from being assigned to any cluster if they are too far away. Leave it True if you want all data points to be part of some cluster.")
    print("For most use cases, 'True' is recommended to ensure all points are clustered.")
    cluster_all = str_input(["True", "False"], SECTION[2])

    print("Bin Seeding: If true, initial kernel locations are binned points, speeding up the algorithm with fewer seeds. Default is False.")
    print("Setting this to True can be useful for large datasets to speed up computation. Consider using True if your dataset is large.")
    bin_seeding = str_input(["True", "False"], SECTION[2])

    print("Min Bin Frequency: To speed up the algorithm, accept only those bins with at least min_bin_freq points as seeds.")
    print("A typical value is 1, but you might increase this for very large datasets to reduce the number of seeds.")
    min_bin_freq = num_input(SECTION[2], "Enter Min Bin Frequency (default is 1): ")

    print("Number of Jobs: The number of jobs to use for the computation. 1 means using all processors.")
    print("If you are unsure, use 1 to utilize all available processors.")
    n_jobs = num_input(SECTION[2], "Enter Number of Jobs (or None): ")
    n_jobs = -1 if n_jobs == 1 else int(n_jobs)

    print("Max Iterations: Maximum number of iterations, per seed point before the clustering operation terminates (for that seed point), if has not converged yet.")
    print("The default value is 300, which is sufficient for most use cases. You might increase this for very complex data.")
    max_iter = num_input(SECTION[2], "Enter Max Iterations (default is 300): ")

    hyper_parameters = {
        "bandwidth": bandwidth,
        "cluster_all": cluster_all == "True",
        "bin_seeding": bin_seeding == "True",
        "min_bin_freq": min_bin_freq,
        "n_jobs": n_jobs if n_jobs != "None" else None,
        "max_iter": max_iter,
    }
    return hyper_parameters
