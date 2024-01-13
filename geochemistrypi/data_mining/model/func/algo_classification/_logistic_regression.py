from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print

from ....constants import SECTION
from ....data.data_readiness import float_input, num_input, str_input


def logistic_regression_manual_hyper_parameters() -> Dict:
    """Manually set hyperparameters.

    Returns
    -------
    hyper_parameters : dict
    """
    print("Penalty: This hyperparameter specifies the norm used in the penalization.")
    print("Please specify the norm used in the penalization. It is generally recommended to leave it set to l2.")
    penalties = ["l1", "l2", "elasticnet", "None"]
    penalty = str_input(penalties, SECTION[2])
    if penalty == "None":
        penalty = None
    print("C: This hyperparameter specifies the inverse of regularization strength. A smaller value of C indicates stronger regularization, whereas a larger value indicates weaker regularization.")
    print("Please specify the inverse of regularization strength. A good starting range could be between 0.001 and 1000, such as 1.0.")
    C = float_input(1, SECTION[2], "@C: ")
    l1_ratio = None
    if penalty == "l1":
        print("Solver: This hyperparameter specifies the algorithm to use in the optimization problem.")
        print("Please specify the algorithm to use in the optimization problem. It is generally recommended to leave it set to liblinear.")
        solvers = ["liblinear", "saga"]
        solver = str_input(solvers, SECTION[2])
    elif penalty == "l2" or penalty == "none":
        print("Solver: This hyperparameter specifies the algorithm to use in the optimization problem.")
        print("Please specify the algorithm to use in the optimization problem. It is generally recommended to leave it set to lbfgs.")
        solvers = ["newton-cg", "lbfgs", "sag", "saga"]
        solver = str_input(solvers, SECTION[2])
    elif penalty == "elasticnet":
        solver = "saga"
        print("L1 Ratio: This hyperparameter specifies the Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1.")
        print("Please specify the Elastic-Net mixing parameter. A good starting range could be between 0.0 and 1.0, such as 0.5.")
        l1_ratio = float_input(0.5, SECTION[2], "@L1 Ratio: ")
    print("Max Iter: This hyperparameter specifies the maximum number of iterations taken for the solvers to converge.")
    print("Please specify the maximum number of iterations taken for the solvers to converge. A good starting range could be between 100 and 1000, such as 100.")
    max_iter = num_input(SECTION[2], "@Max Iter: ")
    print(
        "Class Weight: This hyperparameter specifies the weights associated with classes. It can be set to 'balanced'"
        " to automatically adjust the weights inversely proportional to the class frequencies in the input data."
    )
    print("Please specify the weights associated with classes. It is generally recommended to leave it set to None.")
    class_weights = ["None", "balanced"]
    class_weight = str_input(class_weights, SECTION[2])
    if class_weight == "None":
        class_weight = None
    hyper_parameters = {
        "penalty": penalty,
        "C": C,
        "solver": solver,
        "max_iter": max_iter,
        "class_weight": class_weight,
        "l1_ratio": l1_ratio,
    }
    return hyper_parameters


def plot_logistic_importance(columns_name: np.ndarray, trained_model: object) -> pd.DataFrame:
    """Draw the feature importance diagram for analysis.

    Parameters
    ----------
    data: pd.DataFrame (n_samples, n_components)
        Data for silhouette.

    trained_model: any
        The algorithm which to be used.

    References
    ----------
    Logistic regression, despite its name, is a linear model for classification rather than regression.
    Logistic regression is also known in the literature as logit regression, maximum-entropy classific
    ation (MaxEnt) or the log-linear classifier. In this model, the probabilities describing the possible
    outcomes of a single trial are modeled using a logistic function.

    https://scikit-learn.org/stable/modules/linear_model.html/logistic-regression
    """
    for feature_name, score in zip(list(columns_name), trained_model.coef_.flatten()):
        print(feature_name, ":", score)

    # feature importance map ranked by coefficient
    coef_lr = pd.DataFrame({"var": columns_name, "coef": trained_model.coef_.flatten()})
    index_sort = np.abs(coef_lr["coef"]).sort_values().index
    coef_lr_sort = coef_lr.loc[index_sort, :]

    # Horizontal column chart plot
    fig, ax = plt.subplots(figsize=(14, 8))
    x, y = coef_lr_sort["var"], coef_lr_sort["coef"]
    rects = plt.barh(x, y, color="dodgerblue")
    plt.grid(linestyle="-.", axis="y", alpha=0.4)
    plt.tight_layout()

    # Add data labels
    for rect in rects:
        w = rect.get_width()
        ax.text(w, rect.get_y() + rect.get_height() / 2, "%.2f" % w, ha="left", va="center")
        plt.title("Feature Importance Map Ranked by Coefficient")

    return coef_lr_sort
