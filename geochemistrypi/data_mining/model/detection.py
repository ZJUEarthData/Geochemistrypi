# -*- coding: utf-8 -*-
import os
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from rich import print
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from ..constants import MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH
from ..utils.base import clear_output, save_data, save_fig
from ._base import WorkflowBase
from .func.algo_anomalydetection._common import density_estimation, scatter2d, scatter3d
from .func.algo_anomalydetection._enum import AnormalyDetectionCommonFunction, LocalOutlierFactorSpecialFunction
from .func.algo_anomalydetection._iforest import isolation_forest_manual_hyper_parameters
from .func.algo_anomalydetection._local_outlier_factor import local_outlier_factor_manual_hyper_parameters, plot_lof_scores


class AnomalyDetectionWorkflowBase(WorkflowBase):
    """The base workflow class of anomaly detection algorithms."""

    # common_function = []

    def __init__(self) -> None:
        super().__init__()
        self.mode = "Anomaly Detection"
        self.anomaly_detection = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> None:
        """Fit the model by Scikit-learn framework."""
        self.X = X
        self.model.fit(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Perform Anomaly Detection on samples in X by Scikit-learn framework."""
        y_predict = self.model.predict(X)
        return y_predict

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        return dict()

    @staticmethod
    def _detect_data(X: pd.DataFrame, detect_label: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Merge the detection results into the source data.

        Parameters
        ----------
        X : pd.DataFrame
            The original data.

        detect_label : np.ndarray
            The detection labels for each data point.

        Returns
        -------
        X_anomaly_detection : pd.DataFrame
            DataFrame containing the original data with detection results.

        X_normal : pd.DataFrame
            DataFrame containing the normal data points.

        X_anomaly : pd.DataFrame
            DataFrame containing the anomaly data points.
        """
        X_anomaly_detection = X.copy()
        # Merge detection results into the source data
        X_anomaly_detection["is_anomaly"] = detect_label
        X_normal = X_anomaly_detection[X_anomaly_detection["is_anomaly"] == 1]
        X_anomaly = X_anomaly_detection[X_anomaly_detection["is_anomaly"] == -1]

        return X_anomaly_detection, X_normal, X_anomaly

    @staticmethod
    def _density_estimation(data: pd.DataFrame, labels: pd.DataFrame, graph_name: str, algorithm_name: str, local_path: str, mlflow_path: str) -> None:
        """Plot the density estimation diagram of the anomaly detection result."""
        print(f"-----* {graph_name} *-----")
        density_estimation(data, labels, algorithm_name=algorithm_name)
        save_fig(f"{graph_name} - {algorithm_name}", local_path, mlflow_path)
        data_with_labels = pd.concat([data, labels], axis=1)
        save_data(data_with_labels, f"{graph_name} - {algorithm_name}", local_path, mlflow_path)

    @staticmethod
    def _scatter2d(data: pd.DataFrame, labels: pd.DataFrame, algorithm_name: str, graph_name: str, local_path: str, mlflow_path: str) -> None:
        """Plot the two-dimensional diagram of the anomaly detection result."""
        print(f"-----* {graph_name} *-----")
        scatter2d(data, labels, algorithm_name=algorithm_name)
        save_fig(f"{graph_name} - {algorithm_name}", local_path, mlflow_path)
        data_with_labels = pd.concat([data, labels], axis=1)
        save_data(data_with_labels, f"{graph_name} - {algorithm_name}", local_path, mlflow_path)

    @staticmethod
    def _scatter3d(data: pd.DataFrame, labels: pd.DataFrame, algorithm_name: str, graph_name: str, local_path: str, mlflow_path: str) -> None:
        """Plot the three-dimensional diagram of the anomaly detection result."""
        print(f"-----* {graph_name} *-----")
        scatter3d(data, labels, algorithm_name=algorithm_name)
        save_fig(f"{graph_name} - {algorithm_name}", local_path, mlflow_path)
        data_with_labels = pd.concat([data, labels], axis=1)
        save_data(data_with_labels, f"{graph_name} - {algorithm_name}", local_path, mlflow_path)

    def common_components(self) -> None:
        """Invoke all common application functions for anomaly detection algorithms by Scikit-learn framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        if self.X.shape[1] >= 3:
            two_dimen_axis_index, two_dimen_data = self.choose_dimension_data(self.X, 2)
            self._scatter2d(
                data=two_dimen_data,
                labels=self.y_test,
                algorithm_name=self.naming,
                graph_name=AnormalyDetectionCommonFunction.PLOT_SCATTER_2D.value,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )

            three_dimen_axis_index, three_dimen_data = self.choose_dimension_data(self.X, 3)
            self._scatter3d(
                data=three_dimen_data,
                labels=self.y_test,
                algorithm_name=self.naming,
                graph_name=AnormalyDetectionCommonFunction.PLOT_SCATTER_3D.value,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )

        self._density_estimation(
            data=self.X,
            labels=self.y_test,
            algorithm_name=self.naming,
            graph_name=AnormalyDetectionCommonFunction.DENSITY_ESTIMATION.value,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )


class IsolationForestAnomalyDetection(AnomalyDetectionWorkflowBase):
    """The automation workflow of using Isolation Forest algorithm to make insightful products."""

    name = "Isolation Forest"
    # special_function = []

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: Union[str, int, float] = "auto",
        contamination: Union[str, float] = "auto",
        max_features: Union[int, float] = 1.0,
        bootstrap: bool = False,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: int = 0,
        warm_start: bool = False,
    ) -> None:
        """
        Isolation Forest Algorithm.

        Return the anomaly score of each sample using the IsolationForest algorithm

        The IsolationForest 'isolates' observations by randomly selecting a feature
        and then randomly selecting a split value between the maximum and minimum
        values of the selected feature.

        Since recursive partitioning can be represented by a tree structure, the
        number of splittings required to isolate a sample is equivalent to the path
        length from the root node to the terminating node.

        This path length, averaged over a forest of such random trees, is a
        measure of normality and our decision function.

        Random partitioning produces noticeably shorter paths for anomalies.
        Hence, when a forest of random trees collectively produce shorter path
        lengths for particular samples, they are highly likely to be anomalies.

        Read more in the :ref:`User Guide <isolation_forest>`.

        .. versionadded:: 0.18

        Parameters
        ----------
        n_estimators : int, default=100
            The number of base estimators in the ensemble.

        max_samples : "auto", int or float, default="auto"
            The number of samples to draw from X to train each base estimator.
                - If int, then draw `max_samples` samples.
                - If float, then draw `max_samples * X.shape[0]` samples.
                - If "auto", then `max_samples=min(256, n_samples)`.

            If max_samples is larger than the number of samples provided,
            all samples will be used for all trees (no sampling).

        contamination : 'auto' or float, default='auto'
            The amount of contamination of the data set, i.e. the proportion
            of outliers in the data set. Used when fitting to define the threshold
            on the scores of the samples.

                - If 'auto', the threshold is determined as in the
                original paper.
                - If float, the contamination should be in the range (0, 0.5].

            .. versionchanged:: 0.22
            The default value of ``contamination`` changed from 0.1
            to ``'auto'``.

        max_features : int or float, default=1.0
            The number of features to draw from X to train each base estimator.

                - If int, then draw `max_features` features.
                - If float, then draw `max(1, int(max_features * n_features_in_))` features.

            Note: using a float number less than 1.0 or integer less than number of
            features will enable feature subsampling and leads to a longer runtime.

        bootstrap : bool, default=False
            If True, individual trees are fit on random subsets of the training
            data sampled with replacement. If False, sampling without replacement
            is performed.

        n_jobs : int, default=None
            The number of jobs to run in parallel for both :meth:`fit` and
            :meth:`predict`. ``None`` means 1 unless in a
            :obj:`joblib.parallel_backend` context. ``-1`` means using all
            processors. See :term:`Glossary <n_jobs>` for more details.

        random_state : int, RandomState instance or None, default=None
            Controls the pseudo-randomness of the selection of the feature
            and split values for each branching step and each tree in the forest.

            Pass an int for reproducible results across multiple function calls.
            See :term:`Glossary <random_state>`.

        verbose : int, default=0
            Controls the verbosity of the tree building process.

        warm_start : bool, default=False
            When set to ``True``, reuse the solution of the previous call to fit
            and add more estimators to the ensemble, otherwise, just fit a whole
            new forest. See :term:`the Glossary <warm_start>`.

            .. versionadded:: 0.21

        References
        ----------
        Scikit-learn API: sklearn.ensemble.IsolationForest
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#
        """

        super().__init__()
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.warm_start = warm_start

        if random_state:
            self.random_state = random_state

        # If 'random_state' is None, 'self.random_state' comes from the parent class 'WorkflowBase'
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
            warm_start=self.warm_start,
        )

        self.naming = IsolationForestAnomalyDetection.name

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        print(f"-*-*- {cls.name} - Hyper-parameters Specification -*-*-")
        hyper_parameters = isolation_forest_manual_hyper_parameters()
        clear_output()
        return hyper_parameters

    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        pass


class LocalOutlierFactorAnomalyDetection(AnomalyDetectionWorkflowBase):
    """The automation workflow of using Local Outlier Factor algorithm to make insightful products."""

    name = "Local Outlier Factor"
    # special_function = []

    def __init__(
        self,
        n_neighbors: int = 20,
        algorithm: str = "auto",
        leaf_size: int = 30,
        metric: Union[str, callable] = "minkowski",
        p: float = 2.0,
        metric_params: dict = None,
        contamination: Union[str, float] = "auto",
        novelty: bool = True,  # Change this variable from False to True inorder to make this function work
        n_jobs: int = None,
    ) -> None:
        """
        Unsupervised Outlier Detection using the Local Outlier Factor (LOF).

        The anomaly score of each sample is called the Local Outlier Factor.
        It measures the local deviation of the density of a given sample with respect
        to its neighbors.
        It is local in that the anomaly score depends on how isolated the object
        is with respect to the surrounding neighborhood.
        More precisely, locality is given by k-nearest neighbors, whose distance
        is used to estimate the local density.
        By comparing the local density of a sample to the local densities of its
        neighbors, one can identify samples that have a substantially lower density
        than their neighbors. These are considered outliers.

        .. versionadded:: 0.19

        Parameters
        ----------
        n_neighbors : int, default=20
            Number of neighbors to use by default for :meth:`kneighbors` queries.
            If n_neighbors is larger than the number of samples provided,
            all samples will be used.

        algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
            Algorithm used to compute the nearest neighbors:

            - 'ball_tree' will use :class:`BallTree`
            - 'kd_tree' will use :class:`KDTree`
            - 'brute' will use a brute-force search.
            - 'auto' will attempt to decide the most appropriate algorithm
            based on the values passed to :meth:`fit` method.

            Note: fitting on sparse input will override the setting of
            this parameter, using brute force.

        leaf_size : int, default=30
            Leaf is size passed to :class:`BallTree` or :class:`KDTree`. This can
            affect the speed of the construction and query, as well as the memory
            required to store the tree. The optimal value depends on the
            nature of the problem.

        metric : str or callable, default='minkowski'
            Metric to use for distance computation. Default is "minkowski", which
            results in the standard Euclidean distance when p = 2. See the
            documentation of `scipy.spatial.distance
            <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>`_ and
            the metrics listed in
            :class:`~sklearn.metrics.pairwise.distance_metrics` for valid metric
            values.

            If metric is "precomputed", X is assumed to be a distance matrix and
            must be square during fit. X may be a :term:`sparse graph`, in which
            case only "nonzero" elements may be considered neighbors.

            If metric is a callable function, it takes two arrays representing 1D
            vectors as inputs and must return one value indicating the distance
            between those vectors. This works for Scipy's metrics, but is less
            efficient than passing the metric name as a string.

        p : float, default=2
            Parameter for the Minkowski metric from
            :func:`sklearn.metrics.pairwise_distances`. When p = 1, this
            is equivalent to using manhattan_distance (l1), and euclidean_distance
            (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

        metric_params : dict, default=None
            Additional keyword arguments for the metric function.

        contamination : 'auto' or float, default='auto'
            The amount of contamination of the data set, i.e. the proportion
            of outliers in the data set. When fitting this is used to define the
            threshold on the scores of the samples.

            - if 'auto', the threshold is determined as in the
            original paper,
            - if a float, the contamination should be in the range (0, 0.5].

            .. versionchanged:: 0.22
            The default value of ``contamination`` changed from 0.1
            to ``'auto'``.

        novelty : bool, default=False
            By default, LocalOutlierFactor is only meant to be used for outlier
            detection (novelty=False). Set novelty to True if you want to use
            LocalOutlierFactor for novelty detection. In this case be aware that
            you should only use predict, decision_function and score_samples
            on new unseen data and not on the training set; and note that the
            results obtained this way may differ from the standard LOF results.

            .. versionadded:: 0.20

        n_jobs : int, default=None
            The number of parallel jobs to run for neighbors search.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

        References
        ----------
        Scikit-learn API: sklearn.neighbors.LocalOutlierFactor
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#
        """

        super().__init__()
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.contamination = contamination
        self.novelty = novelty
        self.n_jobs = n_jobs

        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            metric_params=self.metric_params,
            contamination=self.contamination,
            novelty=self.novelty,
            n_jobs=self.n_jobs,
        )

        self.naming = LocalOutlierFactorAnomalyDetection.name

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        print(f"-*-*- {cls.name} - Hyper-parameters Specification -*-*-")
        hyper_parameters = local_outlier_factor_manual_hyper_parameters()
        clear_output()
        return hyper_parameters

    @staticmethod
    def _plot_lof_scores(X_train: pd.DataFrame, lof_scores: np.ndarray, graph_name: str, image_config: dict, algorithm_name: str, local_path: str, mlflow_path: str) -> None:
        """Draw the LOF scores bar diagram."""
        print(f"-----* {graph_name} *-----")
        columns_name = X_train.index
        data = plot_lof_scores(columns_name, lof_scores, image_config)
        save_fig(f"{graph_name} - {algorithm_name}", local_path, mlflow_path)
        save_data(data, f"{graph_name} - {algorithm_name}", local_path, mlflow_path, True)

    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        lof_scores = self.model.negative_outlier_factor_
        self._plot_lof_scores(
            X_train=self.X_train,
            lof_scores=lof_scores,
            image_config=self.image_config,
            algorithm_name=self.naming,
            graph_name=LocalOutlierFactorSpecialFunction.PLOT_LOF_SCORE.value,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )
