# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from global_variable import MODEL_OUTPUT_IMAGE_PATH
from typing import Optional, Union, Dict
from ._base import WorkflowBase
from .func.algo_decomposition._pca import biplot, triplot


class DecompositionWorkflowBase(WorkflowBase):
    def __init__(self) -> None:
        super().__init__()
        self.X_reduced = None

    def _reduced_data2pd(self, reduced_data: np.ndarray, components_num: int) -> None:
        pa_name = []
        for i in range(components_num):
            pa_name.append(f'Principal Axis {i+1}')
        self.X_reduced = pd.DataFrame(reduced_data)
        self.X_reduced.columns = pa_name


class PCADecomposition(DecompositionWorkflowBase):

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
        iterated_power: str = "auto",
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state

        self.model = PCA(n_components=self.n_components,
                         copy=self.copy,
                         whiten=self.whiten,
                         svd_solver=self.svd_solver,
                         tol=self.tol,
                         iterated_power=self.iterated_power,
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

    def _biplot(self, reduced_data, pc_data):
        print("-----* Compositional Bi-plot *-----")
        biplot(reduced_data, pc_data, self.naming, MODEL_OUTPUT_IMAGE_PATH)

    def _triplot(self, reduced_data, pc_data):
        print("-----* Compositional Tri-plot *-----")
        triplot(reduced_data, pc_data, self.naming, MODEL_OUTPUT_IMAGE_PATH)

    def special_components(self, **kwargs: Union[Dict, np.ndarray, int]) -> None:
        self._reduced_data2pd(kwargs['reduced_data'], kwargs['components_num'])
        self._get_principal_components()
        self._get_explained_variance_ratio()

        # Draw graphs when the number of principal components > 3
        if kwargs['components_num'] > 3:
            # choose two of dimensions to draw
            two_dimen_axis_index, two_dimen_pc_data = self.choose_dimension_data(self.pc_data, 2)
            two_dimen_reduced_data = self.X_reduced.iloc[:, two_dimen_axis_index]
            self._biplot(reduced_data=two_dimen_reduced_data, pc_data=two_dimen_pc_data)
            # choose three of dimensions to draw
            three_dimen_axis_index, three_dimen_pc_data = self.choose_dimension_data(self.pc_data, 3)
            three_dimen_reduced_data = self.X_reduced.iloc[:, three_dimen_axis_index]
            self._triplot(reduced_data=three_dimen_reduced_data, pc_data=three_dimen_pc_data)
        elif kwargs['components_num'] == 3:
            # choose two of dimensions to draw
            two_dimen_axis_index, two_dimen_pc_data = self.choose_dimension_data(self.pc_data, 2)
            two_dimen_reduced_data = self.X_reduced.iloc[:, two_dimen_axis_index]
            self._biplot(reduced_data=two_dimen_reduced_data, pc_data=two_dimen_pc_data)
            # no need to choose
            self._triplot(reduced_data=self.X_reduced, pc_data=self.pc_data)
        elif kwargs['components_num'] == 2:
            self._biplot(reduced_data=self.X_reduced, pc_data=self.pc_data)
        else:
            pass





