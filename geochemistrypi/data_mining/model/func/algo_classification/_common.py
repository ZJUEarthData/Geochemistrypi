# -*- coding: utf-8 -*-
from typing import Dict

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from rich import print
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_curve
from sklearn.model_selection import cross_val_predict, cross_validate


def score(y_true: pd.DataFrame, y_predict: pd.DataFrame) -> Dict:
    """Calculate the scores of the classification model.

    Parameters
    ----------
    y_true : pd.DataFrame (n_samples, n_components)
        The true target values.

    y_predict : pd.DataFrame (n_samples, n_components)
        The predicted target values.

    Returns
    -------
    scores : dict
        The scores of the classification model.
    """
    accuracy = accuracy_score(y_true, y_predict)
    precision = precision_score(y_true, y_predict)
    recall = recall_score(y_true, y_predict)
    f1 = f1_score(y_true, y_predict)
    print("Accuracy: ", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    scores = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
    }
    return scores


def plot_confusion_matrix(y_test: pd.DataFrame, y_test_predict: pd.DataFrame, trained_model: object) -> np.ndarray:
    """Plot the confusion matrix.

    Parameters
    ----------
    y_test : pd.DataFrame (n_samples, n_components)
        The testing target values.

    y_test_predict : pd.DataFrame (n_samples, n_components)
        The predicted target values.

    trained_model : sklearn algorithm model
        The sklearn algorithm model trained with X_train data.

    Returns
    -------
    cm : np.ndarray
        The confusion matrix.
    """
    cm = confusion_matrix(y_test, y_test_predict)
    print(cm)
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=trained_model.classes_)
    disp.plot()
    return cm


def display_cross_validation_scores(scores: np.ndarray, score_name: str) -> Dict:
    """Display the scores of cross-validation.

    Parameters
    ----------
    scores : np.ndarray
        The scores of cross-validation.

    score_name : str
        The name of the score.

    Returns
    -------
    cv_scores : dict
        The scores of cross-validation.
    """
    cv_scores = {
        "Fold Scores": str(scores.tolist()),
        "Mean": scores.mean(),
        "Standard Deviation": scores.std(),
    }
    print("Scores:", cv_scores["Fold Scores"])
    print("Mean:", cv_scores["Mean"])
    print("Standard deviation:", cv_scores["Standard Deviation"])
    mlflow.log_metric(f"CV - {score_name} - Mean", cv_scores["Mean"])
    mlflow.log_metric(f"CV - {score_name} - Standard Deviation", cv_scores["Standard Deviation"])
    return cv_scores


def cross_validation(trained_model: object, X_train: pd.DataFrame, y_train: pd.DataFrame, cv_num: int = 10) -> Dict:
    """Evaluate metric(s) by cross-validation and also record fit/score times.

    Parameters
    ----------
    trained_model : object
        The model trained.

    X_train : pd.DataFrame (n_samples, n_components)
        The training feature data.

    y_train : pd.DataFrame (n_samples, n_components)
        The training target values.

    cv_num : int
        Determines the cross-validation splitting strategy.

    Returns
    -------
    scores_result : dict
        The scores of cross-validation.
    """

    scores = cross_validate(trained_model, X_train, y_train, scoring=("accuracy", "precision", "recall", "f1"), cv=cv_num)
    del scores["fit_time"]
    del scores["score_time"]
    # the keys follow the returns of cross_validate in scikit-learn
    scores2display = {
        "test_accuracy": "Accuracy",
        "test_precision": "Precision",
        "test_recall": "Recall",
        "test_f1": "F1 Score",
    }
    scores_result = {"K-Fold": cv_num}
    for key, values in scores.items():
        print("*", scores2display[key], "*")
        cv_scores = display_cross_validation_scores(values, scores2display[key])
        scores_result[scores2display[key]] = cv_scores
        print("-------------")
    return scores_result


# def contour_data(X: pd.DataFrame, trained_model: object) -> Tuple[List[np.ndarray], np.ndarray]:
#     """Build up coordinate matrices as the data of contour plot.

#     Parameters
#     ----------
#     X : pd.DataFrame (n_samples, n_components)
#         The complete feature data.

#     trained_model : object
#         Te algorithm model class from sklearn is trained.

#     Returns
#     -------
#     matrices : List[np.ndarray]
#         Coordinate matrices.

#     labels : np.ndarray
#         Predicted value by the trained model with coordinate data as input data.
#     """

#     # build up coordinate matrices from coordinate vectors.
#     xi = [np.arange(X.iloc[:, i].min(), X.iloc[:, i].max(), (X.iloc[:, i].max() - X.iloc[:, i].min()) / 50) for i in range(X.shape[1])]
#     ndim = len(xi)
#     s0 = (1,) * ndim
#     matrices = [np.asanyarray(x).reshape(s0[:i] + (-1,) + s0[i + 1 :]) for i, x in enumerate(xi)]
#     matrices[0].shape = (1, -1) + s0[2:]
#     matrices[1].shape = (-1, 1) + s0[2:]
#     matrices = np.broadcast_arrays(*matrices, subok=True)

#     # get the labels of the coordinate matrices through the trained model
#     input_array = np.column_stack((i.ravel() for i in matrices))
#     labels = trained_model.predict(input_array).reshape(matrices[0].shape)

#     return matrices, labels


def plot_precision_recall(X_train: pd.DataFrame, y_train: pd.DataFrame, trained_model: object, algorithm_name: str) -> None:
    """Plot the precision-recall curve.

    Parameters
    ----------
    X_train : pd.DataFrame (n_samples, n_components)
        The training feature data.

    y_train : pd.DataFrame (n_samples, n_components)
        The training target values.

    trained_model : object
        The model trained.

    algorithm_name : str
        The name of the algorithm.
    """
    y_scores = cross_val_predict(trained_model, X_train, y_train.iloc[:, 0], cv=3, method="decision_function")
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
    plt.figure()
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.legend(labels=["Precision", "Recall"], loc="best")
    plt.title(f"Precision_Recall_Curve - {algorithm_name}")


def plot_ROC(X_train: pd.DataFrame, y_train: pd.DataFrame, trained_model: object, algorithm_name: str) -> None:
    """Plot the ROC curve.

    Parameters
    ----------
    X_train : pd.DataFrame (n_samples, n_components)
        The training feature data.

    y_train : pd.DataFrame (n_samples, n_components)
        The training target values.

    trained_model : object
        The model trained.

    algorithm_name : str
        The name of the algorithm.
    """
    y_scores = cross_val_predict(trained_model, X_train, y_train.iloc[:, 0], cv=3, method="decision_function")
    fpr, tpr, thresholds = roc_curve(y_train, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title(f"ROC_Curve - {algorithm_name}")
