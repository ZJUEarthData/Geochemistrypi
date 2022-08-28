# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from utils.base import save_fig
from global_variable import MODEL_OUTPUT_IMAGE_PATH
from typing import Optional, Union, List, Dict
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

    # class DecompositionWorkflowBase(object):
#
#     name = None
#     common_function = []
#     special_function = None
#     X, y = None, None
#     X_train, X_test, y_train, y_test = None, None, None, None
#
#     @classmethod
#     def show_info(cls):
#         print("*-*" * 2, cls.name, "is running ...", "*-*" * 2)
#         print("Expected Functionality:")
#         function = cls.common_function + cls.special_function
#         for i in range(len(function)):
#             print("+ ", function[i])
#
#     def __init__(self):
#         self.model = None
#         self.naming = None
#         self.X_reduced = None
#
#     def fit(self, X, y=None):
#         self.model.fit(X)
#
#     def transform(self, X):
#         return self.model.transform(X)
#
#     @staticmethod
#     def data_upload(X, y=None, X_train=None, X_test=None, y_train=None, y_test=None):
#         DecompositionWorkflowBase.X = X
#         DecompositionWorkflowBase.y = y
#         DecompositionWorkflowBase.X_train = X_train
#         DecompositionWorkflowBase.X_test = X_test
#         DecompositionWorkflowBase.y_train = y_train
#         DecompositionWorkflowBase.y_test = y_test


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

    # def _biplot(self, reduced_data: pd.DataFrame, labels: Optional[List[str]] = None) -> None:
    #     """plot a compositional bi-plot for two principal components
    #
    #     Parameters
    #     ----------
    #     reduced_data : pd.DataFrame
    #         Data processed by PCA.
    #     labels : List[str]
    #         The type of tag of the samples in the data set.
    #     """
    #     print("-----* Compositional Bi-plot *-----")
    #     plt.figure(figsize=(14, 10))
    #
    #     # principal component's weight coefficient
    #     # it can obtain the expression of principal component in feature space
    #     n = self.model.components_.shape[1]
    #
    #     x = reduced_data[:, 0]  # variable contributions for PC1
    #     y = reduced_data[:, 1]  # variable contributions for PC2
    #     scalex = 1.0/(x.max() - x.min())
    #     scaley = 1.0/(y.max() - y.min())
    #
    #     # Draw a data point projection plot that is projected to
    #     # a two-dimensional plane using normal PCA
    #     if labels:
    #         legend = []
    #         classes = np.unique(labels)
    #         for i, label in enumerate(classes):
    #             plt.scatter(x[labels == label] * scalex,
    #                         y[labels == label] * scaley,
    #                         linewidth=0.01)
    #             legend.append("Label: {}".format(label))
    #         plt.legend(legend)
    #     else:
    #         plt.scatter(x * scalex, y * scaley, linewidths=0.01)
    #
    #     # plot arrows as the variable contribution,
    #     # each variable has a score for PC1 and for PC2 respectively
    #     for i in range(n):
    #         plt.arrow(0, 0, self.model.components_[0, i], self.model.components_[1, i],
    #                   color='k', alpha=0.7, linewidth=1, )
    #         plt.text(self.model.components_[0, i]*1.01, self.model.components_[1, i]*1.01,
    #                  WorkflowBase.X.columns[i],
    #                  ha='center', va='center', color='k', fontsize=12)
    #
    #     plt.xlabel("$PC1$")
    #     plt.ylabel("$PC2$")
    #     plt.title("Compositional Bi-plot")
    #     plt.grid()
    #     save_fig(f"Compositional Bi-plot - {self.naming}", MODEL_OUTPUT_IMAGE_PATH)

    def _biplot(self, reduced_data, pc_data):
        print("-----* Compositional Bi-plot *-----")
        biplot(reduced_data, pc_data, self.naming, MODEL_OUTPUT_IMAGE_PATH)

    def _triplot(self, reduced_data, pc_data):
        print("-----* Compositional Tri-plot *-----")
        triplot(reduced_data, pc_data, self.naming, MODEL_OUTPUT_IMAGE_PATH)

    def special_components(self, **kwargs: Union[Dict, pd.DataFrame, int]) -> None:
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





