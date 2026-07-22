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
    print("Please specify the norm used in the penalization. It is generally recommended to leave it as 'l2'.")
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
        print("Please specify the algorithm to use in the optimization problem. It is generally recommended to leave it as 'liblinear'.")
        solvers = ["liblinear", "saga"]
        solver = str_input(solvers, SECTION[2])
    elif penalty == "l2" or penalty == "none":
        print("Solver: This hyperparameter specifies the algorithm to use in the optimization problem.")
        print("Please specify the algorithm to use in the optimization problem. It is generally recommended to leave it as 'lbfgs'.")
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
    print("Please specify the weights associated with classes. It is generally recommended to leave it as None.")
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
    columns_name = list(columns_name)
    coefficients = np.asarray(trained_model.coef_)
    if coefficients.ndim == 1:
        coefficients = coefficients.reshape(1, -1)
    if coefficients.shape[1] != len(columns_name):
        raise ValueError("The number of logistic regression coefficients does not match the number of feature columns.")

    if coefficients.shape[0] == 1:
        for feature_name, score in zip(columns_name, coefficients[0]):
            print(feature_name, ":", score)
        coef_lr = pd.DataFrame({"var": columns_name, "coef": coefficients[0]})
        coef_lr_sort = coef_lr.assign(abs_coef=lambda data: np.abs(data["coef"])).sort_values("abs_coef").drop(columns=["abs_coef"])
        y_labels = coef_lr_sort["var"]
    else:
        class_labels = getattr(trained_model, "classes_", np.arange(coefficients.shape[0]))
        if len(class_labels) != coefficients.shape[0]:
            class_labels = np.arange(coefficients.shape[0])
        records = []
        for class_label, class_coefficients in zip(class_labels, coefficients):
            for feature_name, score in zip(columns_name, class_coefficients):
                print(f"class {class_label} - {feature_name}", ":", score)
                records.append({"class_label": class_label, "var": feature_name, "coef": score, "abs_coef": abs(score)})
        coef_lr_sort = pd.DataFrame(records).sort_values("abs_coef")
        y_labels = coef_lr_sort["class_label"].astype(str) + " | " + coef_lr_sort["var"].astype(str)

    # Horizontal column chart plot
    fig, ax = plt.subplots(figsize=(14, 8))
    rects = plt.barh(y_labels, coef_lr_sort["coef"], color="dodgerblue")
    plt.grid(linestyle="-.", axis="y", alpha=0.4)
    plt.tight_layout()

    # Add data labels
    for rect in rects:
        w = rect.get_width()
        ax.text(w, rect.get_y() + rect.get_height() / 2, "%.2f" % w, ha="left", va="center")
        plt.title("Feature Importance Map Ranked by Coefficient")

    return coef_lr_sort
