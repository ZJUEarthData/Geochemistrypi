# -*- coding: utf-8 -*-
from utils.base import save_fig
import matplotlib.pyplot as plt
import numpy as np


def DBSCAN_Plot_2D(data: np.ndarray, labels: np.ndarray, db, store_path: str) -> None:
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print("Estimated number of clusters: %d\n" % n_clusters_)
    unique_labels = set(labels)
    plt.figure()
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]
        class_member_mask = labels == k
        xy = data[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="w", markersize=5.5)
        xy = data[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="w", markersize=3, alpha=0.75)
    plt.title("cluster Numbers =" + str(n_clusters_), y=-0.2)
    save_fig("Plot_DBSCAN", store_path)