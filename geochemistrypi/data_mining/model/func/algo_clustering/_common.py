# -*- coding: utf-8 -*-
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rich import print
from sklearn.metrics import calinski_harabasz_score, silhouette_samples, silhouette_score


def score(X: pd.DataFrame, labels: pd.DataFrame) -> Dict:
    """Calculate the scores of the clustering model.

    Parameters
    ----------
    X : pd.DataFrame (n_samples, n_components)
        The true values.

    label : pd.DataFrame (n_samples, n_components)
        The labels values.

    Returns
    -------
    scores : dict
        The scores of the clustering model.
    """
    silhouette = silhouette_score(X, labels)
    calinski_harabaz = calinski_harabasz_score(X, labels)
    print("silhouette_score: ", silhouette)
    print("calinski_harabasz_score:", calinski_harabaz)
    scores = {
        "silhouette_score": silhouette,
        "calinski_harabasz_score": calinski_harabaz,
    }
    return scores


def plot_results(X, labels, algorithm_name: str, cluster_centers_=None) -> None:
    """Plot clustering results of the clustering model.

    Parameters
    ----------
    X : pd.DataFrame (n_samples, n_components)
        The true values.

    label : pd.DataFrame (n_samples, n_components)
        The labels values.

    algorithm_name : str
        The name of the algorithm model.

    cluster_centers
        The center of the algorithm model.
    """
    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=labels, palette="viridis", s=50, alpha=0.8)
    if not isinstance(cluster_centers_, str):
        plt.scatter(cluster_centers_[:, 0], cluster_centers_[:, 1], c="red", marker="X", s=200, label="Cluster Centers")
    plt.title(f"results - {algorithm_name}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()


def plot_silhouette_diagram(X, labels, algorithm_name: str):
    """Calculate the scores of the clustering model.

    Parameters
    ----------
    X : pd.DataFrame (n_samples, n_components)
        The true values.

    label : pd.DataFrame (n_samples, n_components)
        The labels values.

    algorithm_name : str
        The name of the algorithm model.
    """
    silhouette_values = silhouette_samples(X, labels)
    sns.histplot(silhouette_values, bins=30, kde=True)
    plt.title(f"Silhouette Diagram - {algorithm_name}")
    plt.xlabel("Silhouette Coefficient")
    plt.ylabel("Frequency")
    plt.legend()
