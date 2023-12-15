# -*- coding: utf-8 -*-
import os
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from rich import print
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE

from ..constants import MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH
from ..utils.base import clear_output, save_data, save_fig
from ._base import WorkflowBase
from .func.algo_decomposition._mds import mds_manual_hyper_parameters
from .func.algo_decomposition._pca import biplot, pca_manual_hyper_parameters, triplot
from .func.algo_decomposition._tsne import tsne_manual_hyper_parameters


class DecompositionWorkflowBase(WorkflowBase):
    """The base workflow class of decomposition algorithms."""

    common_function = ["Model Persistence"]  # 'Decomposition Result',

    def __init__(self) -> None:
        super().__init__()

        # the extra attributes that decomposition algorithm needs
        self.X_reduced = None
        self.mode = "Decomposition"

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> None:
        """Fit the model."""
        self.model.fit(X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply dimensionality reduction to X."""
        return self.model.transform(X)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Fit the model with X and apply the dimensionality reduction on X."""
        # TODO(Can He(Sany) sanhe1097618435@163.com): check if we need to put this function in the base class
        return self.model.fit_transform(X)

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        return dict()

    def _reduced_data2pd(self, reduced_data: np.ndarray, components_num: int) -> None:
        """Transform reduced data into the format of pd.DataFrame.

        Parameters
        ----------
        reduced_data : np.ndarray
            The data X after dimension reduction.

        components_num : int
            The numbers of the principal components.
        """
        pa_name = []
        for i in range(components_num):
            pa_name.append(f"Principal Axis {i+1}")
        self.X_reduced = pd.DataFrame(reduced_data)
        self.X_reduced.columns = pa_name


class PCADecomposition(DecompositionWorkflowBase):
    """The automation workflow of using PCA algorithm to make insightful products."""

    name = "PCA"
    special_function = ["Principal Components", "Explained Variance Ratio", "Compositional Bi-plot", "Compositional Tri-plot"]

    def __init__(
        self,
        n_components: Optional[int] = None,
        copy: bool = True,
        whiten: bool = False,
        svd_solver: str = "auto",
        tol: float = 0.0,
        iterated_power: Union[int, str] = "auto",
        n_oversamples: int = 10,
        power_iteration_normalizer: str = "auto",
        random_state: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        n_components : int, float or 'mle', default=None
            Number of components to keep.
            if n_components is not set all components are kept::

                n_components == min(n_samples, n_features)

            If ``n_components == 'mle'`` and ``svd_solver == 'full'``, Minka's
            MLE is used to guess the dimension. Use of ``n_components == 'mle'``
            will interpret ``svd_solver == 'auto'`` as ``svd_solver == 'full'``.

            If ``0 < n_components < 1`` and ``svd_solver == 'full'``, select the
            number of components such that the amount of variance that needs to be
            explained is greater than the percentage specified by n_components.

            If ``svd_solver == 'arpack'``, the number of components must be
            strictly less than the minimum of n_features and n_samples.

            Hence, the None case results in::

                n_components == min(n_samples, n_features) - 1

        copy : bool, default=True
            If False, data passed to fit are overwritten and running
            fit(X).transform(X) will not yield the expected results,
            use fit_transform(X) instead.

        whiten : bool, default=False
            When True (False by default) the `components_` vectors are multiplied
            by the square root of n_samples and then divided by the singular values
            to ensure uncorrelated outputs with unit component-wise variances.
            Whitening will remove some information from the transformed signal
            (the relative variance scales of the components) but can sometime
            improve the predictive accuracy of the downstream estimators by
            making their data respect some hard-wired assumptions.

        svd_solver : {'auto', 'full', 'arpack', 'randomized'}, default='auto'
            If auto :
                The solver is selected by a default policy based on `X.shape` and
                `n_components`: if the input data is larger than 500x500 and the
                number of components to extract is lower than 80% of the smallest
                dimension of the data, then the more efficient 'randomized'
                method is enabled. Otherwise the exact full SVD is computed and
                optionally truncated afterwards.
            If full :
                run exact full SVD calling the standard LAPACK solver via
                `scipy.linalg.svd` and select the components by postprocessing
            If arpack :
                run SVD truncated to n_components calling ARPACK solver via
                `scipy.sparse.linalg.svds`. It requires strictly
                0 < n_components < min(X.shape)
            If randomized :
                run randomized SVD by the method of Halko et al.

            .. versionadded:: 0.18.0

        tol : float, default=0.0
            Tolerance for singular values computed by svd_solver == 'arpack'.
            Must be of range [0.0, infinity).

            .. versionadded:: 0.18.0

        iterated_power : int or 'auto', default='auto'
            Number of iterations for the power method computed by
            svd_solver == 'randomized'.
            Must be of range [0, infinity).

            .. versionadded:: 0.18.0

        n_oversamples : int, default=10
            This parameter is only relevant when `svd_solver="randomized"`.
            It corresponds to the additional number of random vectors to sample the
            range of `X` so as to ensure proper conditioning. See
            :func:`~sklearn.utils.extmath.randomized_svd` for more details.

            .. versionadded:: 1.1

        power_iteration_normalizer : {'auto', 'QR', 'LU', 'none'}, default='auto'
            Power iteration normalizer for randomized SVD solver.
            Not used by ARPACK. See :func:`~sklearn.utils.extmath.randomized_svd`
            for more details.

            .. versionadded:: 1.1

        random_state : int, RandomState instance or None, default=None
            Used when the 'arpack' or 'randomized' solvers are used. Pass an int
            for reproducible results across multiple function calls.
            See :term:`Glossary <random_state>`.

            .. versionadded:: 0.18.0

        References
        ----------
        Scikit-learn API: sklearn.decomposition.PCA
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        """
        super().__init__()
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        # self.n_oversamples = n_oversamples
        # self.power_iteration_normalizer = power_iteration_normalizer
        self.random_state = random_state

        self.model = PCA(
            n_components=self.n_components,
            copy=self.copy,
            whiten=self.whiten,
            svd_solver=self.svd_solver,
            tol=self.tol,
            iterated_power=self.iterated_power,
            # n_oversamples=self.n_oversamples,
            # power_iteration_normalizer=self.power_iteration_normalizer,
            random_state=self.random_state,
        )

        self.naming = PCADecomposition.name

        # special attributes
        self.pc_data = None

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        print(f"-*-*- {cls.name} - Hyper-parameters Specification -*-*-")
        hyper_parameters = pca_manual_hyper_parameters()
        clear_output()
        return hyper_parameters

    def _get_principal_components(self) -> None:
        """Get principal components."""
        print("-----* Principal Components *-----")
        print("Every column represents one principal component respectively.")
        print("Every row represents how much that row feature contributes to each principal component respectively.")
        print("The tabular data looks like in format: 'rows x columns = 'features x principal components'.")
        pc_name = []
        for i in range(self.n_components):
            pc_name.append(f"PC{i+1}")
        self.pc_data = pd.DataFrame(self.model.components_.T)
        self.pc_data.columns = pc_name
        self.pc_data.set_index(DecompositionWorkflowBase.X.columns, inplace=True)
        print(self.pc_data)

    def _get_explained_variance_ratio(self) -> None:
        """Get explained variance ratio."""
        print("-----* Explained Variance Ratio *-----")
        print(self.model.explained_variance_ratio_)

    @staticmethod
    def _biplot(reduced_data: pd.DataFrame, pc_data: pd.DataFrame, algorithm_name: str, local_path: str, mlflow_path: str) -> None:
        """Draw bi-plot."""
        print("-----* Compositional Bi-plot *-----")
        biplot(reduced_data, pc_data, algorithm_name)
        save_fig(f"Compositional Bi-plot - {algorithm_name}", local_path, mlflow_path)
        save_data(reduced_data, "Compositional Bi-plot - Reduced Data", local_path, mlflow_path)
        save_data(pc_data, "Compositional Bi-plot - PC Data", local_path, mlflow_path)

    @staticmethod
    def _triplot(reduced_data: pd.DataFrame, pc_data: pd.DataFrame, algorithm_name: str, local_path: str, mlflow_path: str) -> None:
        """Draw tri-plot."""
        print("-----* Compositional Tri-plot *-----")
        triplot(reduced_data, pc_data, algorithm_name)
        save_fig(f"Compositional Tri-plot - {algorithm_name}", local_path, mlflow_path)
        save_data(reduced_data, "Compositional Tri-plot - Reduced Data", local_path, mlflow_path)
        save_data(pc_data, "Compositional Tri-plot - PC Data", local_path, mlflow_path)

    def special_components(self, **kwargs: Union[Dict, np.ndarray, int]) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        self._reduced_data2pd(kwargs["reduced_data"], kwargs["components_num"])
        self._get_principal_components()
        self._get_explained_variance_ratio()

        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        # Draw graphs when the number of principal components > 3
        if kwargs["components_num"] > 3:
            # choose two of dimensions to draw
            two_dimen_axis_index, two_dimen_pc_data = self.choose_dimension_data(self.pc_data, 2)
            two_dimen_reduced_data = self.X_reduced.iloc[:, two_dimen_axis_index]
            self._biplot(
                reduced_data=two_dimen_reduced_data,
                pc_data=two_dimen_pc_data,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )

            # choose three of dimensions to draw
            three_dimen_axis_index, three_dimen_pc_data = self.choose_dimension_data(self.pc_data, 3)
            three_dimen_reduced_data = self.X_reduced.iloc[:, three_dimen_axis_index]
            self._triplot(
                reduced_data=three_dimen_reduced_data,
                pc_data=three_dimen_pc_data,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
        elif kwargs["components_num"] == 3:
            # choose two of dimensions to draw
            two_dimen_axis_index, two_dimen_pc_data = self.choose_dimension_data(self.pc_data, 2)
            two_dimen_reduced_data = self.X_reduced.iloc[:, two_dimen_axis_index]
            self._biplot(
                reduced_data=two_dimen_reduced_data,
                pc_data=two_dimen_pc_data,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
            # no need to choose
            self._triplot(
                reduced_data=self.X_reduced,
                pc_data=self.pc_data,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
        elif kwargs["components_num"] == 2:
            self._biplot(
                reduced_data=self.X_reduced,
                pc_data=self.pc_data,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
        else:
            pass


class TSNEDecomposition(DecompositionWorkflowBase):
    """The automation workflow of using T-SNE algorithm to make insightful products."""

    name = "T-SNE"
    special_function = []

    def __init__(
        self,
        n_components: int = 2,
        *,
        perplexity: float = 30.0,
        early_exaggeration: float = 12.0,
        learning_rate: Union[float, str] = "auto",
        n_iter: int = 1000,
        n_iter_without_progress: int = 300,
        min_grad_norm: float = 1e-7,
        metric: str = "euclidean",
        metric_params: Optional[Dict] = None,
        init: str = "pca",
        verbose: int = 0,
        random_state: Optional[int] = None,
        method: str = "exact",
        angle: float = 0.5,
        n_jobs: int = None,
        square_distances: Union[bool, str] = "deprecated",
    ) -> None:
        """
        Parameters
        ----------
        n_components : int, default=2
            Dimension of the embedded space.

        perplexity : float, default=30.0
            The perplexity is related to the number of nearest neighbors that
            is used in other manifold learning algorithms. Larger datasets
            usually require a larger perplexity. Consider selecting a value
            between 5 and 50. Different values can result in significantly
            different results. The perplexity must be less than the number
            of samples.

        early_exaggeration : float, default=12.0
            Controls how tight natural clusters in the original space are in
            the embedded space and how much space will be between them. For
            larger values, the space between natural clusters will be larger
            in the embedded space. Again, the choice of this parameter is not
            very critical. If the cost function increases during initial
            optimization, the early exaggeration factor or the learning rate
            might be too high.

        learning_rate : float or "auto", default="auto"
            The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
            the learning rate is too high, the data may look like a 'ball' with any
            point approximately equidistant from its nearest neighbours. If the
            learning rate is too low, most points may look compressed in a dense
            cloud with few outliers. If the cost function gets stuck in a bad local
            minimum increasing the learning rate may help.
            Note that many other t-SNE implementations (bhtsne, FIt-SNE, openTSNE,
            etc.) use a definition of learning_rate that is 4 times smaller than
            ours. So our learning_rate=200 corresponds to learning_rate=800 in
            those other implementations. The 'auto' option sets the learning_rate
            to `max(N / early_exaggeration / 4, 50)` where N is the sample size,
            following [4] and [5].

            .. versionchanged:: 1.2
            The default value changed to `"auto"`.

        n_iter : int, default=1000
            Maximum number of iterations for the optimization. Should be at
            least 250.

        n_iter_without_progress : int, default=300
            Maximum number of iterations without progress before we abort the
            optimization, used after 250 initial iterations with early
            exaggeration. Note that progress is only checked every 50 iterations so
            this value is rounded to the next multiple of 50.

            .. versionadded:: 0.17
            parameter *n_iter_without_progress* to control stopping criteria.

        min_grad_norm : float, default=1e-7
            If the gradient norm is below this threshold, the optimization will
            be stopped.

        metric : str or callable, default='euclidean'
            The metric to use when calculating distance between instances in a
            feature array. If metric is a string, it must be one of the options
            allowed by scipy.spatial.distance.pdist for its metric parameter, or
            a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
            If metric is "precomputed", X is assumed to be a distance matrix.
            Alternatively, if metric is a callable function, it is called on each
            pair of instances (rows) and the resulting value recorded. The callable
            should take two arrays from X as input and return a value indicating
            the distance between them. The default is "euclidean" which is
            interpreted as squared euclidean distance.

        metric_params : dict, default=None
            Additional keyword arguments for the metric function.

            .. versionadded:: 1.1

        init : {"random", "pca"} or ndarray of shape (n_samples, n_components), \
                default="pca"
            Initialization of embedding.
            PCA initialization cannot be used with precomputed distances and is
            usually more globally stable than random initialization.

            .. versionchanged:: 1.2
            The default value changed to `"pca"`.

        verbose : int, default=0
            Verbosity level.

        random_state : int, RandomState instance or None, default=None
            Determines the random number generator. Pass an int for reproducible
            results across multiple function calls. Note that different
            initializations might result in different local minima of the cost
            function. See :term:`Glossary <random_state>`.

        method : {'barnes_hut', 'exact'}, default='barnes_hut'
            By default the gradient calculation algorithm uses Barnes-Hut
            approximation running in O(NlogN) time. method='exact'
            will run on the slower, but exact, algorithm in O(N^2) time. The
            exact algorithm should be used when nearest-neighbor errors need
            to be better than 3%. However, the exact method cannot scale to
            millions of examples.

            .. versionadded:: 0.17
            Approximate optimization *method* via the Barnes-Hut.

        angle : float, default=0.5
            Only used if method='barnes_hut'
            This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
            'angle' is the angular size (referred to as theta in [3]) of a distant
            node as measured from a point. If this size is below 'angle' then it is
            used as a summary node of all points contained within it.
            This method is not very sensitive to changes in this parameter
            in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
            computation time and angle greater 0.8 has quickly increasing error.

        n_jobs : int, default=None
            The number of parallel jobs to run for neighbors search. This parameter
            has no impact when ``metric="precomputed"`` or
            (``metric="euclidean"`` and ``method="exact"``).
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

            .. versionadded:: 0.22

        square_distances : True, default='deprecated'
            This parameter has no effect since distance values are always squared
            since 1.1.

            .. deprecated:: 1.1
                `square_distances` has no effect from 1.1 and will be removed in
                1.3.

        References
        ----------
        Scikit-learn API: sklearn.manifold.TSNE
        https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
        """
        super().__init__()
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.metric_params = metric_params
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.method = method
        self.angle = angle
        self.n_jobs = n_jobs
        self.square_distances = square_distances

        self.model = TSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            early_exaggeration=self.early_exaggeration,
            learning_rate=self.learning_rate,
            n_iter=self.n_iter,
            n_iter_without_progress=self.n_iter_without_progress,
            min_grad_norm=self.min_grad_norm,
            metric=self.metric,
            metric_params=self.metric_params,
            init=self.init,
            verbose=self.verbose,
            random_state=self.random_state,
            method=self.method,
            angle=self.angle,
            n_jobs=self.n_jobs,
            square_distances=self.square_distances,
        )

        self.naming = TSNEDecomposition.name

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        print(f"-*-*- {cls.name} - Hyper-parameters Specification -*-*-")
        hyper_parameters = tsne_manual_hyper_parameters()
        clear_output()
        return hyper_parameters

    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        pass


class MDSDecomposition(DecompositionWorkflowBase):
    """The automation workflow of using MDS algorithm to make insightful products."""

    name = "MDS"
    special_function = []

    def __init__(
        self,
        n_components: int = 2,
        *,
        metric: bool = True,
        n_init: int = 4,
        max_iter: int = 300,
        verbose: int = 0,
        eps: float = 1e-3,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
        dissimilarity: str = "euclidean",
        # normalized_stress="warn",
    ) -> None:
        """
        Parameters
        ----------
        n_components : int, default=2
            Number of dimensions in which to immerse the dissimilarities.

        metric : bool, default=True
            If ``True``, perform metric MDS; otherwise, perform nonmetric MDS.
            When ``False`` (i.e. non-metric MDS), dissimilarities with 0 are considered as
            missing values.

        n_init : int, default=4
            Number of times the SMACOF algorithm will be run with different
            initializations. The final results will be the best output of the runs,
            determined by the run with the smallest final stress.

        max_iter : int, default=300
            Maximum number of iterations of the SMACOF algorithm for a single run.

        verbose : int, default=0
            Level of verbosity.

        eps : float, default=1e-3
            Relative tolerance with respect to stress at which to declare
            convergence. The value of `eps` should be tuned separately depending
            on whether or not `normalized_stress` is being used.

        n_jobs : int, default=None
            The number of jobs to use for the computation. If multiple
            initializations are used (``n_init``), each run of the algorithm is
            computed in parallel.

            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

        random_state : int, RandomState instance or None, default=None
            Determines the random number generator used to initialize the centers.
            Pass an int for reproducible results across multiple function calls.
            See :term:`Glossary <random_state>`.

        dissimilarity : {'euclidean', 'precomputed'}, default='euclidean'
            Dissimilarity measure to use:

            - 'euclidean':
                Pairwise Euclidean distances between points in the dataset.

            - 'precomputed':
                Pre-computed dissimilarities are passed directly to ``fit`` and
                ``fit_transform``.

        normalized_stress : bool or "auto" default=False
            Whether use and return normed stress value (Stress-1) instead of raw
            stress calculated by default. Only supported in non-metric MDS.

            .. versionadded:: 1.2

        References
        ----------
        Scikit-learn API: sklearn.manifold.MDS
        https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html
        """
        super().__init__()
        self.n_components = n_components
        self.metric = metric
        self.n_init = n_init
        self.max_iter = max_iter
        self.verbose = verbose
        self.eps = eps
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.dissimilarity = dissimilarity
        # self.normalized_stress = normalized_stress

        self.model = MDS(
            n_components=self.n_components,
            metric=self.metric,
            n_init=self.n_init,
            max_iter=self.max_iter,
            verbose=self.verbose,
            eps=self.eps,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            dissimilarity=self.dissimilarity,
            # normalized_stress=self.normalized_stress,
        )

        self.naming = MDSDecomposition.name

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        print(f"-*-*- {cls.name} - Hyper-parameters Specification -*-*-")
        hyper_parameters = mds_manual_hyper_parameters()
        clear_output()
        return hyper_parameters

    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        pass
