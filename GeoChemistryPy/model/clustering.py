# -*- coding: utf-8 -*-
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib as mpl
from matplotlib.ticker import FixedLocator, FixedFormatter
from core.base import *
from global_variable import DATASET_OUTPUT_PATH, MODEL_OUTPUT_IMAGE_PATH


class ClusteringWorkflowBase(object):

    name = None
    # TODO: build virtualization in 2D and 3D graph
    common_function = ['Cluster Centers',
                       'Cluster Labels',
                       'Virtualization in 2D graph',
                       'Virtualization in 3D graph',
                       'Silhouette Plot']
    special_function = None

    @classmethod
    def show_info(cls):
        print("*-*" * 2, cls.name, "is running ...", "*-*" * 2)
        print("Expected Functionality:")
        function = cls.common_function + cls.special_function
        for i in range(len(function)):
            print("+ ", function[i])

    def __init__(self):
        self.model = None
        self.X = None
        self.naming = None

    def fit(self, X, y=None):
        self.X = X
        self.model.fit(X)

    def get_cluster_centers(self):
        print("-----* Clustering Centers *-----")
        print(self.model.cluster_centers_)

    def get_labels(self):
        print("-----* Clustering Labels *-----")
        self.X['clustering result'] = self.model.labels_
        print(self.X)
        save_data(self.X, f"{self.naming}", DATASET_OUTPUT_PATH)

    # FIXME: code silhouette diagram
    # def plot_silhouette_diagram(self, show_xlabels=True,
    #                             show_ylabels=True, show_title=True):
    #     """Plot silhouette diagram
    #
    #     :param show_xlabels: If true, add abscissa information
    #     :param show_ylabels: If true, add ordinate information
    #     :param show_title: If true, add the figure name
    #     """
    #
    #     plt.figure(1)
    #     y_pred = self.model.labels_
    #     silhouette_coefficients = silhouette_samples(self.X, y_pred)
    #     silhouette_average = silhouette_score(self.X, y_pred)
    #
    #     padding = len(self.X) // 30
    #     pos = padding
    #     ticks = []
    #     for i in range(self.model.n_clusters):
    #         coeffs = silhouette_coefficients[y_pred == i]
    #         coeffs.sort()
    #
    #         color = mpl.cm.Spectral(i / self.model.n_clusters)
    #         plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
    #                           facecolor=color, edgecolor=color, alpha=0.7)
    #         ticks.append(pos + len(coeffs) // 2)
    #         pos += len(coeffs) + padding
    #
    #     plt.axvline(x=silhouette_average, color="red", linestyle="--")
    #
    #     plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
    #     plt.gca().yaxis.set_major_formatter(FixedFormatter(range(self.model.n_clusters)))
    #
    #     if show_xlabels:
    #         plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    #         plt.xlabel("Silhouette Coefficient")
    #     else:
    #         plt.tick_params(labelbottom=False)
    #     if show_ylabels:
    #         plt.ylabel("Cluster")
    #     if show_title:
    #         plt.title("init:{}  n_cluster:{}".format(self.model.init, self.model.n_clusters))
    #     print("Successfully graph the Silhouette Diagram.")
    #     plt.show()
    #     save_fig(f"Silhouette Plot - {self.naming}", MODEL_OUTPUT_IMAGE_PATH)


class KMeansClustering(ClusteringWorkflowBase):

    name = "KMeans"
    special_function = ['KMeans Score']

    def __init__(self,
                 n_clusters=8,
                 init="k-means++",
                 n_init=10,
                 max_iter=300,
                 tol=1e-4,
                 verbose=0,
                 random_state=None,
                 copy_x=True,
                 algorithm="auto"):

        super().__init__()
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.algorithm = algorithm

        self.model = KMeans(n_clusters=self.n_clusters,
                            init=self.init,
                            n_init=self.n_init,
                            max_iter=self.max_iter,
                            tol=self.tol,
                            verbose=self.verbose,
                            random_state=self.random_state,
                            copy_x=self.copy_x,
                            algorithm=self.algorithm)
        self.naming = KMeansClustering.name

    def _get_scores(self):
        print("-----* KMeans Scores *-----")
        print("Inertia Score: ", self.model.inertia_)
        print("Calinski Harabasz Score: ", metrics.calinski_harabasz_score(self.X, self.model.labels_))
        print("Silhouette Score: ", metrics.silhouette_score(self.X, self.model.labels_))

    def special_components(self):
        self._get_scores()
