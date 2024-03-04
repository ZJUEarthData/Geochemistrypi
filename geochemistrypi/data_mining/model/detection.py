# -*- coding: utf-8 -*-

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from rich import print
from sklearn.ensemble import IsolationForest

from ..utils.base import clear_output
from ._base import WorkflowBase
from .func.algo_abnormaldetection._iforest import isolation_forest_manual_hyper_parameters


class AbnormalDetectionWorkflowBase(WorkflowBase):
    """The base workflow class of abnormal detection algorithms."""

    # common_function = []

    def __init__(self) -> None:
        super().__init__()
        self.mode = "Abnormal Detection"

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> None:
        """Fit the model by Scikit-learn framework."""
        self.X = X
        self.model.fit(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Perform Abnormal Detection on samples in X by Scikit-learn framework."""
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
        X_abnormal_detection : pd.DataFrame
            DataFrame containing the original data with detection results.

        X_normal : pd.DataFrame
            DataFrame containing the normal data points.

        X_abnormal : pd.DataFrame
            DataFrame containing the abnormal data points.
        """
        X_abnormal_detection = X.copy()
        # Merge detection results into the source data
        X_abnormal_detection["is_abnormal"] = detect_label
        X_normal = X_abnormal_detection[X_abnormal_detection["is_abnormal"] == 1]
        X_abnormal = X_abnormal_detection[X_abnormal_detection["is_abnormal"] == -1]

        return X_abnormal_detection, X_normal, X_abnormal

    def common_components(self) -> None:
        """Invoke all common application functions for abnormal detection algorithms by Scikit-learn framework."""
        pass


class IsolationForestAbnormalDetection(AbnormalDetectionWorkflowBase):
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

        self.naming = IsolationForestAbnormalDetection.name

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
