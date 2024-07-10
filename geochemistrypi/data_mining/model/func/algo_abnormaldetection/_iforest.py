# -*- coding: utf-8 -*-
from typing import Dict

from rich import print

from ....constants import SECTION
from ....data.data_readiness import bool_input, float_input, num_input


def isolation_forest_manual_hyper_parameters() -> Dict:
    """Manually set hyperparameters.

    Returns
    -------
    hyper_parameters : dict
    """
    print("N Estimators: The number of trees in the forest.")
    print("Please specify the number of trees in the forest. A good starting range could be between 50 and 500, such as 100.")
    n_estimators = num_input(SECTION[2], "@N Estimators: ")
    print("Contamination: The amount of contamination of the data set.")
    print("Please specify the contamination of the data set. A good starting range could be between 0.1 and 0.5, such as 0.3.")
    contamination = float_input(0.3, SECTION[2], "@Contamination: ")
    print("Max Features: The number of features to draw from X to train each base estimator.")
    print("Please specify the number of features. A good starting range could be between 1 and the total number of features in the dataset.")
    max_features = num_input(SECTION[2], "@Max Features: ")
    print(
        "Bootstrap: Whether bootstrap samples are used when building trees. Bootstrapping is a technique where a random subset of the data is sampled with replacement"
        " to create a new dataset of the same size as the original. This new dataset is then used to construct a decision tree in the ensemble. If False, the whole dataset is used to build each tree."
    )
    print("Please specify whether bootstrap samples are used when building trees. It is generally recommended to leave it as True.")
    bootstrap = bool_input(SECTION[2])
    max_samples = None
    if bootstrap:
        print("Max Samples: The number of samples to draw from X_train to train each base estimator.")
        print("Please specify the number of samples. A good starting range could be between 256 and the number of dataset.")
        max_samples = num_input(SECTION[2], "@@Max Samples: ")
    hyper_parameters = {
        "n_estimators": n_estimators,
        "contamination": contamination,
        "max_features": max_features,
        "bootstrap": bootstrap,
    }
    if not max_samples:
        # Use the default value provided by sklearn.ensemble.RandomForestClassifier.
        hyper_parameters["max_samples"] = None
    else:
        hyper_parameters["max_samples"] = max_samples
    return hyper_parameters


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
    print("Please specify the power parameter. A good starting range could be between 1 and 3, such as 2.0.")
    p = float_input(2.0, SECTION[2], "@P: ")

    print("Contamination: The amount of contamination of the data set.")
    print("Please specify the contamination of the data set. A good starting range could be between 0.1 and 0.5, such as 0.3.")
    contamination = float_input(0.3, SECTION[2], "@Contamination: ")

    print("N jobs: The number of parallel jobs to run.")
    print("Please specify the number of jobs. Use -1 to use all available CPUs, 1 for no parallelism, or specify the number of CPUs to use. A good starting value is None.")
    n_jobs = num_input(SECTION[2], "@N Jobs: ")

    hyper_parameters = {
        "n_neighbors": n_neighbors,
        "leaf_size": leaf_size,
        "p": p,
        "contamination": contamination,
        "n_jobs": n_jobs,
    }

    return hyper_parameters
