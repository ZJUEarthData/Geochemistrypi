# -*- coding: utf-8 -*-
from sklearn.decomposition import PCA


class DecompositionWorkflowBase(object):

    name = None
    common_function = []
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


class PrincipalComponentAnalysis(DecompositionWorkflowBase):

    name = 'PCA'
    special_function = ["Principal Components", "Explained Variance Ratio",
                        "Biplot", "Triplot"]

    def __init__(
        self,
        n_components=None,
        copy=True,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        random_state=None,
    ):
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

    def _get_principal_components(self):
        print("-----* Principal Components *-----")
        print(self.model.components_)

    def _get_explained_variance_ratio(self):
        print("-----* Explained Variance Ratio *-----")
        print(self.model.explained_variance_ratio_)

    def special_components(self):
        self._get_principal_components()
        self._get_explained_variance_ratio()
