# -*- coding: utf-8 -*-
from typing import Dict

from rich import print

from ....constants import SECTION
from ....data.data_readiness import float_input, num_input


def tsne_manual_hyper_parameters() -> Dict:
    """Manually set hyperparameters.

    Returns
    -------
    hyper_parameters : dict
    """
    print("N Components: This parameter specifies the number of components to retain after dimensionality reduction.")
    print("Please specify the number of components to retain. A good starting range could be between 2 and 10, such as 4.")
    n_components = num_input(SECTION[2], "N Components: ")
    print("Perplexity: This parameter is related to the number of nearest neighbors that each point considers when computing the probabilities.")
    print("Please specify the perplexity. A good starting range could be between 5 and 50, such as 30.")
    perplexity = num_input(SECTION[2], "Perplexity: ")
    print("Learning Rate: This parameter controls the step size during the optimization process.")
    print("Please specify the learning rate. A good starting range could be between 10 and 1000, such as 200.")
    learning_rate = float_input(200, SECTION[2], "Learning Rate: ")
    print("Number of Iterations: This parameter controls how many iterations the optimization will run for.")
    print("Please specify the number of iterations. A good starting range could be between 250 and 1000, such as 500.")
    n_iter = num_input(SECTION[2], "Number of Iterations: ")
    print("Early Exaggeration: This parameter controls how tight natural clusters in the original space are in the embedded space and how much space will be between them.")
    print("Please specify the early exaggeration. A good starting range could be between 5 and 50, such as 12.")
    early_exaggeration = float_input(12, SECTION[2], "Early Exaggeration: ")
    hyper_parameters = {"n_components": n_components, "perplexity": perplexity, "learning_rate": learning_rate, "n_iter": n_iter, "early_exaggeration": early_exaggeration}
    return hyper_parameters
