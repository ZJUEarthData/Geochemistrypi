# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd


def density_estimation(data: pd.DataFrame, labels: pd.DataFrame, algorithm_name: str) -> None:
    """Generate a density estimation plot for anomaly detection."""
    # Assuming the labels contain '0' for normal and '1' for anomalies.
    normal_data = data[labels == 0]
    anomaly_data = data[labels == 1]

    # Using Kernel Density Estimation (KDE) for density estimation
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 6))

    sns.kdeplot(data=normal_data, fill=True, label="Normal Data", color="blue")
    sns.kdeplot(data=anomaly_data, fill=True, label="Anomaly Data", color="red")

    plt.title(f"Density Estimation for {algorithm_name}")
    plt.xlabel("Feature Space")
    plt.ylabel("Density")
    plt.legend()


def scatter2d(data: pd.DataFrame, labels: pd.DataFrame, algorithm_name: str) -> None:
    """
    Draw the 2D scatter plot for anomaly detection results.

    Parameters
    ----------
    data : pd.DataFrame (n_samples, n_components)
       The features of the data.

    labels : pd.DataFrame (n_samples,)
        Labels of each point (1 for normal, -1 for anomaly).

    algorithm_name : str
        The name of the algorithm
    """
    markers = ["o", "x"]
    colors = ["#1f77b4", "#d62728"]

    fig = plt.figure()
    fig.set_size_inches(18, 10)
    plt.subplot(111)

    for i, label in enumerate([-1, 1]):
        anomaly_data = data[labels == label]
        color = colors[i]
        marker = markers[i]
        plt.scatter(anomaly_data.iloc[:, 0], anomaly_data.iloc[:, 1], c=color, marker=marker, label="Anomaly" if label == -1 else "Normal")

    plt.xlabel(f"{data.columns[0]}")
    plt.ylabel(f"{data.columns[1]}")
    plt.title(f"Anomaly Detection 2D Scatter Plot - {algorithm_name}")
    plt.legend()


def scatter3d(data: pd.DataFrame, labels: pd.DataFrame, algorithm_name: str) -> None:
    """
    Draw the 3D scatter plot for anomaly detection results.

    Parameters
    ----------
    data : pd.DataFrame (n_samples, n_components)
       The features of the data.

    labels : pd.DataFrame (n_samples,)
        Labels of each point (1 for normal, -1 for anomaly).

    algorithm_name : str
        The name of the algorithm
    """
    fig = plt.figure(figsize=(12, 6), facecolor="w")
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9)

    ax = fig.add_subplot(121, projection="3d")
    ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], alpha=0.3, c="#FF0000", marker=".")
    ax.set_xlabel(data.columns[0])
    ax.set_ylabel(data.columns[1])
    ax.set_zlabel(data.columns[2])
    plt.grid(True)

    ax2 = fig.add_subplot(122, projection="3d")
    markers = ["o", "x"]
    colors = ["#1f77b4", "#d62728"]

    for i, label in enumerate([-1, 1]):
        anomaly_data = data[labels == label]
        color = colors[i]
        marker = markers[i]
        ax2.scatter(
            anomaly_data.iloc[:, 0], anomaly_data.iloc[:, 1], anomaly_data.iloc[:, 2], c=color, marker=marker, s=6, cmap=plt.cm.Paired, edgecolors="none", label="Anomaly" if label == -1 else "Normal"
        )

    ax2.set_xlabel(data.columns[0])
    ax2.set_ylabel(data.columns[1])
    ax2.set_zlabel(data.columns[2])
    plt.grid(True)
    ax.set_title(f"Base Data 3D Plot - {algorithm_name}")
    ax2.set_title(f"Anomaly Detection 3D Plot - {algorithm_name}")
    plt.legend()
