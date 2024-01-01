from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print

from ....constants import SECTION
from ....data.data_readiness import float_input, num_input, str_input

def agglomerative_manual_hyper_parameters() -> Dict:
    """Manually set hyperparameters.

    Returns
    -------
    hyper_parameters : dict
    """
    print("N Clusters: The number of clusters to form as well as the number of centroids to generate.")
    print("Please specify the number of clusters for agglomerative. A good starting range could be between 2 and 10, such as '4'.")
    n_clusters = num_input(SECTION[2], "N Clusters: ")
    print("linkage: The linkage criterion determines which distance to use between sets of observation. ")
    print("Please specify the linkage criterion. It is generally recommended to leave it set to 'ward'.")
    linkages = ["ward", "complete", "average", "single"]
    linkage = str_input(linkages, SECTION[2])
    hyper_parameters = {
        "n_clusters": n_clusters,
        "linkage": linkage,
    }
    return hyper_parameters