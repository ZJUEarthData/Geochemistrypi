# -*- coding: utf-8 -*-
import os
from typing import Dict, Optional, Union

import mlflow
import numpy as np
import pandas as pd
from rich import print
from sklearn import metrics
from sklearn.cluster import DBSCAN, AffinityPropagation, KMeans

from ..constants import MLFLOW_ARTIFACT_DATA_PATH, MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH
from ..utils.base import clear_output, save_data, save_fig
from ._base import WorkflowBase
from .func.algo_clustering._dbscan import dbscan_manual_hyper_parameters, dbscan_result_plot
from .func.algo_clustering._kmeans import kmeans_manual_hyper_parameters, plot_silhouette_diagram, scatter2d, scatter3d


class ClusteringWorkflowBase(WorkflowBase):
    """The base workflow class of clustering algorithms."""

    common_function = ["Cluster Centers", "Cluster Labels", "Model Persistence"]

    def __init__(self):
        super().__init__()
        self.clustering_result = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> None:
        """Fit the model according to the given training data."""
        self.X = X
        self.model.fit(X)
        mlflow.log_params(self.model.get_params())

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        return dict()

    # TODO(Samson 1057266013@qq.com): This function might need to be rethought.
    def get_cluster_centers(self) -> np.ndarray:
        """Get the cluster centers."""
        print("-----* Clustering Centers *-----")
        print(getattr(self.model, "cluster_centers_", "This class don not have cluster_centers_"))
        return getattr(self.model, "cluster_centers_", "This class don not have cluster_centers_")

    def get_labels(self):
        """Get the cluster labels."""
        print("-----* Clustering Labels *-----")
        # self.X['clustering result'] = self.model.labels_
        self.clustering_result = pd.DataFrame(self.model.labels_, columns=["clustering result"])
        print(self.clustering_result)
        GEOPI_OUTPUT_ARTIFACTS_DATA_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_DATA_PATH")
        save_data(self.clustering_result, f"{self.naming} Result", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)


class KMeansClustering(ClusteringWorkflowBase):
    """The automation workflow of using KMeans algorithm to make insightful products."""

    name = "KMeans"
    special_function = ["KMeans Score"]

    def __init__(
        self,
        n_clusters: int = 8,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        verbose: int = 0,
        random_state: Optional[int] = None,
        copy_x: bool = True,
        algorithm: str = "auto",
    ) -> None:
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
        Scikit-learn API: sklearn.cluster.KMeans
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        """

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

        self.model = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
            random_state=self.random_state,
            copy_x=self.copy_x,
            algorithm=self.algorithm,
        )

        self.naming = KMeansClustering.name

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        print(f"-*-*- {cls.name} - Hyper-parameters Specification -*-*-")
        hyper_parameters = kmeans_manual_hyper_parameters()
        clear_output()
        return hyper_parameters

    def _get_scores(self):
        """Get the scores of the clustering result."""
        print("-----* KMeans Scores *-----")
        print("Inertia Score: ", self.model.inertia_)
        print("Calinski Harabasz Score: ", metrics.calinski_harabasz_score(self.X, self.model.labels_))
        print("Silhouette Score: ", metrics.silhouette_score(self.X, self.model.labels_))
        mlflow.log_metric("Inertia Score", self.model.inertia_)
        mlflow.log_metric("Calinski Harabasz Score", metrics.calinski_harabasz_score(self.X, self.model.labels_))
        mlflow.log_metric("Silhouette Score", metrics.silhouette_score(self.X, self.model.labels_))

    @staticmethod
    def _plot_silhouette_diagram(
        data: pd.DataFrame,
        cluster_labels: pd.DataFrame,
        cluster_centers_: np.ndarray,
        n_clusters: int,
        algorithm_name: str,
        local_path: str,
        mlflow_path: str,
    ) -> None:
        """Plot the silhouette diagram of the clustering result."""
        print("-----* Silhouette Diagram *-----")
        plot_silhouette_diagram(data, cluster_labels, cluster_centers_, n_clusters, algorithm_name)
        save_fig(f"Silhouette Diagram - {algorithm_name}", local_path, mlflow_path)
        data_with_labels = pd.concat([data, cluster_labels], axis=1)
        save_data(data_with_labels, "Silhouette Diagram - Data With Labels", local_path, mlflow_path)
        cluster_center_data = pd.DataFrame(cluster_centers_, columns=data.columns)
        save_data(cluster_center_data, "Silhouette Diagram - Cluster Centers", local_path, mlflow_path)

    @staticmethod
    def _scatter2d(data: pd.DataFrame, cluster_labels: pd.DataFrame, algorithm_name: str, local_path: str, mlflow_path: str) -> None:
        """Plot the two-dimensional diagram of the clustering result."""
        print("-----* Cluster Two-Dimensional Diagram *-----")
        scatter2d(data, cluster_labels, algorithm_name)
        save_fig(f"Cluster Two-Dimensional Diagram - {algorithm_name}", local_path, mlflow_path)
        data_with_labels = pd.concat([data, cluster_labels], axis=1)
        save_data(data_with_labels, f"Cluster Two-Dimensional Diagram - {algorithm_name}", local_path, mlflow_path)

    @staticmethod
    def _scatter3d(data: pd.DataFrame, cluster_labels: pd.DataFrame, algorithm_name: str, local_path: str, mlflow_path: str) -> None:
        """Plot the three-dimensional diagram of the clustering result."""
        print("-----* Cluster Three-Dimensional Diagram *-----")
        scatter3d(data, cluster_labels, algorithm_name)
        save_fig(f"Cluster Three-Dimensional Diagram - {algorithm_name}", local_path, mlflow_path)
        data_with_labels = pd.concat([data, cluster_labels], axis=1)
        save_data(data_with_labels, f"Cluster Two-Dimensional Diagram - {algorithm_name}", local_path, mlflow_path)

    def special_components(self, **kwargs: Union[Dict, np.ndarray, int]) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        self._get_scores()
        self._plot_silhouette_diagram(
            data=self.X,
            cluster_labels=self.clustering_result["clustering result"],
            cluster_centers_=self.get_cluster_centers(),
            n_clusters=self.n_clusters,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )
        # Draw graphs when the number of principal components > 3
        if self.X.shape[1] >= 3:
            # choose two of dimensions to draw
            two_dimen_axis_index, two_dimen_data = self.choose_dimension_data(self.X, 2)
            self._scatter2d(
                data=two_dimen_data,
                cluster_labels=self.clustering_result["clustering result"],
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )

            # choose three of dimensions to draw
            three_dimen_axis_index, three_dimen_data = self.choose_dimension_data(self.X, 3)
            self._scatter3d(
                data=three_dimen_data,
                cluster_labels=self.clustering_result["clustering result"],
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
        elif self.X.shape[1] == 3:
            # choose two of dimensions to draw
            two_dimen_axis_index, two_dimen_data = self.choose_dimension_data(self.X, 2)
            self._scatter2d(
                data=two_dimen_data,
                cluster_labels=self.clustering_result["clustering result"],
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )

            # no need to choose
            self._scatter3d(
                data=self.X,
                cluster_labels=self.clustering_result["clustering result"],
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
        elif self.X.shape[1] == 2:
            self._scatter2d(
                data=self.X,
                cluster_labels=self.clustering_result["clustering result"],
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
        else:
            pass


class DBSCANClustering(ClusteringWorkflowBase):
    """The automation workflow of using DBSCAN algorithm to make insightful products."""

    name = "DBSCAN"
    special_function = ["Virtualization of Result in 2D Graph"]

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
        metric_params: Optional[Dict] = None,
        algorithm: str = "auto",
        leaf_size: int = 30,
        p: float = None,
        n_jobs: int = None,
    ) -> None:
        """
        Parameters
        ----------
        eps : float, default=0.5
            The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is not a maximum bound on the distances of points within a cluster.
            This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.

        min_samples : int, default=5
            The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.

        metric : str, or callable, default=`euclidean`
            The metric to use when calculating distance between instances in a feature array. If metric is a string or callable, it must be one of the options allowed
              by sklearn.metrics.pairwise_distances for its metric parameter.
            If metric is “precomputed”, X is assumed to be a distance matrix and must be square. X may be a sparse graph, in which case only “nonzero” elements may be considered neighbors for DBSCAN.

            New in version 0.17: metric precomputed to accept precomputed sparse matrix.

        metric_params : dict, default=None
            Additional keyword arguments for the metric function.

            New in version 0.19.

        algorithm : {`auto`, `ball_tree`, `kd_tree`, `brute`}, default=`auto`
            The algorithm to be used by the NearestNeighbors module to compute pointwise distances and find nearest neighbors. See NearestNeighbors module documentation for details.

        leaf_size : int, default=30
            Leaf size passed to BallTree or cKDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends
            on the nature of the problem.

        p : float, default=None
            The power of the Minkowski metric to be used to calculate distance between points. If None, then p=2 (equivalent to the Euclidean distance).

        n_jobs : int, default=None
            The number of parallel jobs to run. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.

        References
        ----------------------------------------
        Scikit-learn API: sklearn.cluster.DBSCAN
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
        """

        super().__init__()
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs

        self.model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            metric_params=self.metric_params,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            p=self.p,
            n_jobs=self.n_jobs,
        )

        self.naming = DBSCANClustering.name

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        print(f"-*-*- {cls.name} - Hyper-parameters Specification -*-*-")
        hyper_parameters = dbscan_manual_hyper_parameters()
        clear_output()
        return hyper_parameters

    @staticmethod
    def _clustering_result_plot(X: pd.DataFrame, trained_model: any, algorithm_name: str, imag_config: dict, local_path: str, mlflow_path: str) -> None:
        """Plot the clustering result in 2D graph."""
        print("-------** Cluster Two-Dimensional Diagram **----------")
        dbscan_result_plot(X, trained_model, imag_config, algorithm_name)
        save_fig(f"Cluster Two-Dimensional Diagram - {algorithm_name}", local_path, mlflow_path)
        save_data(X, f"Cluster Two-Dimensional Diagram - {algorithm_name}", local_path, mlflow_path)

    def special_components(self, **kwargs: Union[Dict, np.ndarray, int]) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        self._clustering_result_plot(
            X=self.X,
            trained_model=self.model,
            algorithm_name=self.naming,
            imag_config=self.image_config,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )


class AffinityPropagationClustering(ClusteringWorkflowBase):
    name = "AffinityPropagation"

    def __init__(
        self,
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
        self.model = AffinityPropagation(
            damping=self.damping,
            max_iter=self.max_iter,
            convergence_iter=self.convergence_iter,
            copy=self.copy,
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
