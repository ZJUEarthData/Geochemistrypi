# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from global_variable import MODEL_OUTPUT_IMAGE_PATH
from typing import Optional, Union, Dict
from ._base import WorkflowBase
from .func.algo_decomposition._pca import biplot, triplot
from utils.base import save_fig


class DecompositionWorkflowBase(WorkflowBase):
    """The base workflow class of decomposition algorithms."""

    def __init__(self) -> None:
        super().__init__()

        # the extra attributes that decomposition algorithm needs
        self.X_reduced = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> None:
        """Fit the model."""
        self.model.fit(X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply dimensionality reduction to X."""
        return self.model.transform(X)

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
            pa_name.append(f'Principal Axis {i+1}')
        self.X_reduced = pd.DataFrame(reduced_data)
        self.X_reduced.columns = pa_name


class PCADecomposition(DecompositionWorkflowBase):
    """The automation workflow of using PCA algorithm to make insightful products."""

    name = 'PCA'
    special_function = ["Principal Components", "Explained Variance Ratio",
                        "Compositional Bi-plot", "Compositional Tri-plot"]

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
        scikit API: sklearn.decomposition.PCA
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

        self.model = PCA(n_components=self.n_components,
                         copy=self.copy,
                         whiten=self.whiten,
                         svd_solver=self.svd_solver,
                         tol=self.tol,
                         iterated_power=self.iterated_power,
                         # n_oversamples=self.n_oversamples,
                         # power_iteration_normalizer=self.power_iteration_normalizer,
                         random_state=self.random_state)
        self.naming = PCADecomposition.name

        # special attributes
        self.pc_data = None

    def _get_principal_components(self) -> None:
        print("-----* Principal Components *-----")
        print("Every column represents one principal component respectively.")
        print("Every row represents how much that row feature contributes to each principal component respectively.")
        print("The tabular data looks like in format: 'rows x columns = 'features x principal components'.")
        pc_name = []
        for i in range(self.n_components):
            pc_name.append(f'PC{i+1}')
        self.pc_data = pd.DataFrame(self.model.components_.T)
        self.pc_data.columns = pc_name
        self.pc_data.set_index(DecompositionWorkflowBase.X.columns, inplace=True)
        print(self.pc_data)

    def _get_explained_variance_ratio(self) -> None:
        print("-----* Explained Variance Ratio *-----")
        print(self.model.explained_variance_ratio_)

    @staticmethod
    def _biplot(reduced_data: pd.DataFrame, pc_data: pd.DataFrame, algorithm_name: str, store_path: str) -> None:
        print("-----* Compositional Bi-plot *-----")
        biplot(reduced_data, pc_data, algorithm_name)
        save_fig(f"Compositional Bi-plot - {algorithm_name}", store_path)

    @staticmethod
    def _triplot(reduced_data: pd.DataFrame, pc_data: pd.DataFrame, algorithm_name: str, store_path: str) -> None:
        print("-----* Compositional Tri-plot *-----")
        triplot(reduced_data, pc_data, algorithm_name)
        save_fig(f"Compositional Tri-plot - {algorithm_name}", store_path)

    def special_components(self, **kwargs: Union[Dict, np.ndarray, int]) -> None:
        self._reduced_data2pd(kwargs['reduced_data'], kwargs['components_num'])
        self._get_principal_components()
        self._get_explained_variance_ratio()

        # Draw graphs when the number of principal components > 3
        if kwargs['components_num'] > 3:
            # choose two of dimensions to draw
            two_dimen_axis_index, two_dimen_pc_data = self.choose_dimension_data(self.pc_data, 2)
            two_dimen_reduced_data = self.X_reduced.iloc[:, two_dimen_axis_index]
            self._biplot(reduced_data=two_dimen_reduced_data, pc_data=two_dimen_pc_data,
                         algorithm_name=self.naming, store_path=MODEL_OUTPUT_IMAGE_PATH)

            # choose three of dimensions to draw
            three_dimen_axis_index, three_dimen_pc_data = self.choose_dimension_data(self.pc_data, 3)
            three_dimen_reduced_data = self.X_reduced.iloc[:, three_dimen_axis_index]
            self._triplot(reduced_data=three_dimen_reduced_data, pc_data=three_dimen_pc_data,
                          algorithm_name=self.naming, store_path=MODEL_OUTPUT_IMAGE_PATH)
        elif kwargs['components_num'] == 3:
            # choose two of dimensions to draw
            two_dimen_axis_index, two_dimen_pc_data = self.choose_dimension_data(self.pc_data, 2)
            two_dimen_reduced_data = self.X_reduced.iloc[:, two_dimen_axis_index]
            self._biplot(reduced_data=two_dimen_reduced_data, pc_data=two_dimen_pc_data,
                         algorithm_name=self.naming, store_path=MODEL_OUTPUT_IMAGE_PATH)
            # no need to choose
            self._triplot(reduced_data=self.X_reduced, pc_data=self.pc_data,
                          algorithm_name=self.naming, store_path=MODEL_OUTPUT_IMAGE_PATH)
        elif kwargs['components_num'] == 2:
            self._biplot(reduced_data=self.X_reduced, pc_data=self.pc_data,
                         algorithm_name=self.naming, store_path=MODEL_OUTPUT_IMAGE_PATH)
        else:
            pass
