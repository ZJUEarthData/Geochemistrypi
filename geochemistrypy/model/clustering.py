# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# from matplotlib.ticker import FixedLocator, FixedFormatter
import numpy as np
from utils.base import save_data
from utils.base import save_fig
from global_variable import MODEL_OUTPUT_IMAGE_PATH
from global_variable import DATASET_OUTPUT_PATH
from typing import Optional, Union, Dict
from abc import ABCMeta, abstractmethod
from ._base import WorkflowBase
from .func.algo_clustering._cluster import plot_silhouette_diagram, scatter2d


class ClusteringWorkflowBase(WorkflowBase):
    """Base class for Cluster.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    common_function = ['Cluster Centers',
                       'Cluster Labels',
                       ]

    def __init__(self):
        super().__init__()

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> None:
        self.X = X
        self.model.fit(X)

    def get_cluster_centers(self) -> np.ndarray:
        print("-----* Clustering Centers *-----")
        print(getattr(self.model, 'cluster_centers_', 'This class don not have cluster_centers_'))
        return getattr(self.model, 'cluster_centers_', 'This class don not have cluster_centers_')

    def get_labels(self):
        print("-----* Clustering Labels *-----")
        self.X['clustering result'] = self.model.labels_
        print(self.X)
        save_data(self.X, f"{self.naming}", DATASET_OUTPUT_PATH)


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
        """

        Parameters
        ----------

        n_clusters : int, default=8
            The number of clusters to form as well as the number of
            centroids to generate.

        init : {'k-means++', 'random'}, callable or array-like of shape \
                (n_clusters, n_features), default='k-means++'
            Method for initialization:

            'k-means++' : selects initial cluster centers for k-mean
            clustering in a smart way to speed up convergence. See section
            Notes in k_init for more details.

            'random': choose `n_clusters` observations (rows) at random from data
            for the initial centroids.

            If an array is passed, it should be of shape (n_clusters, n_features)
            and gives the initial centers.

            If a callable is passed, it should take arguments X, n_clusters and a
            random state and return an initialization.

        n_init : int, default=10
            Number of time the k-means algorithm will be run with different
            centroid seeds. The final results will be the best output of
            n_init consecutive runs in terms of inertia.

        max_iter : int, default=300
            Maximum number of iterations of the k-means algorithm for a
            single run.

        tol : float, default=1e-4
            Relative tolerance with regards to Frobenius norm of the difference
            in the cluster centers of two consecutive iterations to declare
            convergence.

        verbose : int, default=0
            Verbosity mode.

        random_state : int, RandomState instance or None, default=None
            Determines random number generation for centroid initialization. Use
            an int to make the randomness deterministic.
            See :term:`Glossary <random_state>`.

        copy_x : bool, default=True
            When pre-computing distances it is more numerically accurate to center
            the data first. If copy_x is True (default), then the original data is
            not modified. If False, the original data is modified, and put back
            before the function returns, but small numerical differences may be
            introduced by subtracting and then adding the data mean. Note that if
            the original data is not C-contiguous, a copy will be made even if
            copy_x is False. If the original data is sparse, but not in CSR format,
            a copy will be made even if copy_x is False.

        algorithm : {"auto", "full", "elkan"}, default="auto"
            K-means algorithm to use. The classical EM-style algorithm is "full".
            The "elkan" variation is more efficient on data with well-defined
            clusters, by using the triangle inequality. However it's more memory
            intensive due to the allocation of an extra array of shape
            (n_samples, n_clusters).

            For now "auto" (kept for backward compatibility) chooses "elkan" but it
            might change in the future for a better heuristic.

        References
        ----------------------------------------
        Read more in the :ref:`User Guide <k_means>`.
        https://scikit-learn.org/stable/modules/clustering.html#k-means
        """

    def _get_scores(self):
        print("-----* KMeans Scores *-----")
        print("Inertia Score: ", self.model.inertia_)
        print("Calinski Harabasz Score: ", metrics.calinski_harabasz_score(self.X, self.model.labels_))
        print("Silhouette Score: ", metrics.silhouette_score(self.X, self.model.labels_))

    def _plot_silhouette_diagram(self) -> None:
        print("-----* Silhouette Diagram *-----")
        print(type(self.get_cluster_centers()))
        plot_silhouette_diagram(self.X, self.X['clustering result'],
                                self.get_cluster_centers(), self.n_clusters, MODEL_OUTPUT_IMAGE_PATH)

    def _scatter2d(self) -> None:
        scatter2d(self.X, self.X['clustering result'], MODEL_OUTPUT_IMAGE_PATH)

    def special_components(self, **kwargs: Union[Dict, np.ndarray, int]) -> None:
        self._get_scores()
        self._plot_silhouette_diagram()
        self._scatter2d()

        # Draw graphs when the number of principal components > 3
        if kwargs['components_num'] > 3:
            pass
        elif kwargs['components_num'] == 3:
            pass
        elif kwargs['components_num'] == 2:
            self._scatter2d()
        else:
            pass

class AffinityPropagationClustering(ClusteringWorkflowBase):
    name = "AffinityPropagation"

    def __init__(self,
                 *,
                 damping=0.5,
                 max_iter=200,
                 convergence_iter=15,
                 copy=True,
                 preference=None,
                 affinity="euclidean",
                 verbose=False,
                 random_state=None,
    ):

        super().__init__()
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.copy = copy
        self.verbose = verbose
        self.preference = preference
        self.affinity = affinity
        self.random_state = random_state
        self.model = AffinityPropagation(damping = self.damping,
                                         max_iter = self.max_iter,
                                         convergence_iter = self.convergence_iter,
                                         copy = self.copy,
                                         preference=None,
                                         affinity="euclidean",
                                         verbose=False,
                                         random_state=None,

        )
        self.naming = AffinityPropagationClustering.name

    pass


class MeanShiftClustering(ClusteringWorkflowBase):
    name = "MeanShift"
    pass


class SpectralClustering(ClusteringWorkflowBase):
    name = "Spectral"
    pass


class WardHierarchicalClustering(ClusteringWorkflowBase):
    name = "WardHierarchical"
    pass


class AgglomerativeClustering(ClusteringWorkflowBase):
    name = "Agglomerative"
    pass


class DBSCANClustering(ClusteringWorkflowBase):

    name = "DBSCAN"
    special_function = []

    def __init__(self,
                 eps=0.5,
                 min_samples=5,
                 metric="euclidean",
                 metric_params=None,
                 algorithm="auto",
                 leaf_size=30,
                 p=None,
                 n_jobs=None,
                 ):

        super().__init__()
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs

        self.model = DBSCAN(eps=self.eps,
                            min_samples=self.min_samples,
                            metric=self.metric,
                            metric_params=self.min_samples,
                            algorithm=self.algorithm,
                            leaf_size=self.leaf_size,
                            p=self.p,
                            n_jobs=self.n_jobs)

    def special_components(self):
        pass


class OPTICSClustering(ClusteringWorkflowBase):
    name = "OPTICS"
    pass


class GaussianMixturesClustering(ClusteringWorkflowBase):
    name = "GaussianMixtures"
    pass


class BIRCHClusteringClustering(ClusteringWorkflowBase):
    name = "BIRCHClustering"
    pass


class BisectingKMeansClustering(ClusteringWorkflowBase):
    name = "BisectingKMeans"
    pass