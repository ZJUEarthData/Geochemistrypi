# -*- coding: utf-8 -*-
from typing import Dict

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from rich import print
from scipy.stats import gaussian_kde
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate


def score(y_true: pd.DataFrame, y_predict: pd.DataFrame) -> Dict:
    """Calculate the scores of the regression model.

    Parameters
    ----------
    y_true : pd.DataFrame (n_samples, n_components)
        The true target values.

    y_predict : pd.DataFrame (n_samples, n_components)
        The predicted target values.

    Returns
    -------
    scores : dict
        The scores of the regression model.
    """
    mse = mean_squared_error(y_true, y_predict)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_predict)
    r2 = r2_score(y_true, y_predict)
    evs = explained_variance_score(y_true, y_predict)
    print("Mean Square Error: ", mse)
    print("Root Mean Square Error:", rmse)
    print("Mean Absolute Error:", mae)
    print("R2 Score:", r2)
    print("Explained Variance Score:", evs)
    scores = {
        "Root Mean Square Error": rmse,
        "Mean Absolute Error": mae,
        "R2 Score": r2,
        "Explained Variance Score": evs,
    }
    if y_true.shape[1] > 1:
        mse_by_target = mean_squared_error(y_true, y_predict, multioutput="raw_values")
        mae_by_target = mean_absolute_error(y_true, y_predict, multioutput="raw_values")
        r2_by_target = r2_score(y_true, y_predict, multioutput="raw_values")
        evs_by_target = explained_variance_score(y_true, y_predict, multioutput="raw_values")
        scores["Per Target"] = {
            str(column): {
                "Root Mean Square Error": float(np.sqrt(mse_by_target[index])),
                "Mean Absolute Error": float(mae_by_target[index]),
                "R2 Score": float(r2_by_target[index]),
                "Explained Variance Score": float(evs_by_target[index]),
            }
            for index, column in enumerate(y_true.columns)
        }
    return scores


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
        "Fold Scores": scores.tolist(),
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

    scores = cross_validate(
        trained_model,
        X_train,
        y_train,
        scoring=("neg_root_mean_squared_error", "neg_mean_absolute_error", "r2", "explained_variance"),
        cv=cv_num,
    )
    del scores["fit_time"]
    del scores["score_time"]
    # the keys follow the returns of cross_validate in scikit-learn
    scores2display = {
        "test_neg_root_mean_squared_error": "Root Mean Square Error",
        "test_neg_mean_absolute_error": "Mean Absolute Error",
        "test_r2": "R2 Score",
        "test_explained_variance": "Explained Variance Score",
    }
    scores_result = {"K-Fold": cv_num}
    for key, values in scores.items():
        print("*", scores2display[key], "*")
        if (key == "test_neg_root_mean_squared_error") or (key == "test_neg_mean_absolute_error"):
            cv_scores = display_cross_validation_scores(-values, scores2display[key])
        else:
            cv_scores = display_cross_validation_scores(values, scores2display[key])
        scores_result[scores2display[key]] = cv_scores
        print("-------------")
    return scores_result


def plot_predicted_vs_actual(y_test_predict: pd.DataFrame, y_test: pd.DataFrame, algorithm_name: str) -> None:
    """Plot the testing predict values of the trained model and the testing target values.

    Parameters
    ----------
    y_test_predict : pd.DataFrame (n_samples, n_components)
        The testing predict values.

    y_test : pd.DataFrame (n_samples, n_components)
        The testing target values.

    algorithm_name : str
        The name of the algorithm model.
    """
    if y_test.shape[1] == 1:
        plt.scatter(y_test_predict, y_test, color="b")
        plt.plot(y_test_predict, y_test_predict, color="r", linestyle="--", label="Perfect Prediction Line")
    else:
        for column in y_test.columns:
            predicted = y_test_predict[column]
            actual = y_test[column]
            plt.scatter(predicted, actual, label=str(column))
        minimum = min(float(y_test_predict.min().min()), float(y_test.min().min()))
        maximum = max(float(y_test_predict.max().max()), float(y_test.max().max()))
        plt.plot([minimum, maximum], [minimum, maximum], color="r", linestyle="--", label="Perfect Prediction Line")
    plt.xlabel("Predicted Values")
    plt.ylabel("Actual Values")
    plt.legend()
    plt.title(f"Predicted vs. Actual Diagram - {algorithm_name}")


def plot_predicted_actual_density(
    y_train_predict: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test_predict: pd.DataFrame,
    y_test: pd.DataFrame,
    algorithm_name: str,
) -> None:
    """Plot auditable train/test predicted-versus-actual density panels."""

    if y_train.shape[1] != 1 or y_test.shape[1] != 1:
        raise ValueError("Predicted-versus-actual density output requires one regression target.")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    panels = (
        (axes[0], y_train, y_train_predict, "Training", "Reds"),
        (axes[1], y_test, y_test_predict, "Testing", "Blues"),
    )
    for axis, observed_frame, predicted_frame, label, color_map in panels:
        observed = np.asarray(observed_frame).reshape(-1).astype(float)
        predicted = np.asarray(predicted_frame).reshape(-1).astype(float)
        finite = np.isfinite(observed) & np.isfinite(predicted)
        observed = observed[finite]
        predicted = predicted[finite]
        if observed.size < 2:
            raise ValueError(f"{label} density output requires at least two finite observations.")
        coordinates = np.vstack((observed, predicted))
        try:
            density = gaussian_kde(coordinates)(coordinates)
        except np.linalg.LinAlgError:
            density = np.ones(observed.shape, dtype=float)
        order = np.argsort(density)
        observed = observed[order]
        predicted = predicted[order]
        density = density[order]
        points = axis.scatter(observed, predicted, c=density, cmap=color_map, s=12, alpha=0.85)
        lower = min(float(observed.min()), float(predicted.min()))
        upper = max(float(observed.max()), float(predicted.max()))
        axis.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1)
        rmse = float(np.sqrt(mean_squared_error(observed, predicted)))
        r2 = float(r2_score(observed, predicted))
        axis.text(
            0.04,
            0.96,
            f"R² = {r2:.4f}\nRMSE = {rmse:.4f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )
        axis.set_title(label)
        axis.set_xlabel("Actual Values")
        axis.set_ylabel("Predicted Values")
        figure.colorbar(points, ax=axis, label="Point density")
    figure.suptitle(f"Predicted vs. Actual Density - {algorithm_name}")


def plot_residuals(y_test_predict: pd.DataFrame, y_test: pd.DataFrame, algorithm_name: str) -> pd.DataFrame:
    """Plot the residuals of the testing predict values and the testing target values.

    Parameters
    ----------
    y_test_predict : pd.DataFrame (n_samples, n_components)
        The testing predict values.

    y_test : pd.DataFrame (n_samples, n_components)
        The testing target values.

    algorithm_name : str
        The name of the algorithm model.

    Returns
    -------
    residuals : pd.DataFrame (n_samples, n_components)
        The residuals of the testing predict values and the testing target values.
    """
    residuals = y_test.values - y_test_predict.values
    # Support multiple Y columns: create column names based on the actual number of columns
    if y_test.shape[1] == 1:
        residuals = pd.DataFrame(residuals, columns=["Residuals"])
    else:
        # Support multiple Y columns: create column names based on the actual number of columns
        residual_columns = [f"Residuals_{col}" for col in y_test.columns]
        residuals = pd.DataFrame(residuals, columns=residual_columns)

    if y_test.shape[1] == 1:
        plt.scatter(y_test_predict, residuals, color="b")
    else:
        for index, column in enumerate(y_test.columns):
            plt.scatter(y_test_predict[column], residuals.iloc[:, index], label=str(column))
    plt.axhline(0, color="r", linestyle="--", label="Zero Residual Line")
    plt.title(f"Residuals Diagram - {algorithm_name}")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals (Actual - Predicted)'")
    plt.legend()
    return residuals
