# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.model_selection import cross_validate
import numpy as np


def score(y_true: pd.DataFrame, y_predict: pd.DataFrame) -> None:
    mse = mean_squared_error(y_true, y_predict)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_predict)
    r2 = r2_score(y_true, y_predict)
    evs = explained_variance_score(y_true, y_predict)
    print("RMSE score:", rmse)
    print("MAE score:", mae)
    print("R2 score:", r2)
    print("Explained Variance Score:", evs)


def display_cross_validation_scores(scores: np.ndarray) -> None:
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


def cross_validation(trained_model: object, X_train: pd.DataFrame, y_train: pd.DataFrame, cv_num: int = 10) -> None:
    # self.model comes from the subclass of every regression algorithm
    scores = cross_validate(trained_model, X_train, y_train,
                            scoring=('neg_root_mean_squared_error',
                                     'neg_mean_absolute_error',
                                     'r2',
                                     'explained_variance'),
                            cv=cv_num)
    for key, values in scores.items():
        print("*", key.upper(), "*")
        display_cross_validation_scores(values)
        print('-------------')
    return scores


def plot_predicted_value_evaluation(y_test: pd.DataFrame, y_test_predict: pd.DataFrame) -> None:
    plt.figure(figsize=(4, 4))
    plt.scatter(y_test, y_test_predict, color='gold', alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test_predict.min(),y_test_predict.max()], '-r', linewidth=1)
    plt.title('Predicted image')
    plt.xlabel('y_test')
    plt.ylabel('y_test_predict')


def plot_true_vs_predicted(y_test_predict: pd.DataFrame, y_test: pd.DataFrame, algorithm_name: str):
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

    y_test.index = range(len(y_test))
    plt.figure(figsize=(8, 6))
    plt.plot(y_test_predict, color='red', label='predict')
    plt.plot(y_test, color='gold', label='true')
    plt.legend()
    plt.title(f'Ground Truth v.s. Prediction - {algorithm_name}')