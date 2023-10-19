# -*- coding: utf-8 -*-
import os
from typing import Optional

import pandas as pd

from ..constants import MLFLOW_ARTIFACT_DATA_PATH
from ..model.decomposition import DecompositionWorkflowBase, MDSDecomposition, PCADecomposition, TSNEDecomposition
from ._base import ModelSelectionBase


class DecompositionModelSelection(ModelSelectionBase):
    """Simulate the normal way of invoking scikit-learn decomposition algorithms."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.dcp_workflow = DecompositionWorkflowBase()
        self.transformer_config = {}

    def activate(
        self,
        X: pd.DataFrame,
        y: Optional[pd.DataFrame] = None,
        X_train: Optional[pd.DataFrame] = None,
        X_test: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.DataFrame] = None,
    ) -> None:
        """Train by Scikit-learn framework."""

        self.dcp_workflow.data_upload(X=X, y=y, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

        if self.model_name == "PCA":
            hyper_parameters = PCADecomposition.manual_hyper_parameters()
            self.dcp_workflow = PCADecomposition(n_components=hyper_parameters["n_components"], svd_solver=hyper_parameters["svd_solver"])
        elif self.model_name == "T-SNE":
            hyper_parameters = TSNEDecomposition.manual_hyper_parameters()
            self.dcp_workflow = TSNEDecomposition(
                n_components=hyper_parameters["n_components"],
                perplexity=hyper_parameters["perplexity"],
                learning_rate=hyper_parameters["learning_rate"],
                n_iter=hyper_parameters["n_iter"],
                early_exaggeration=hyper_parameters["early_exaggeration"],
            )
        elif self.model_name == "MDS":
            hyper_parameters = MDSDecomposition.manual_hyper_parameters()
            self.dcp_workflow = MDSDecomposition(
                n_components=hyper_parameters["n_components"],
                metric=hyper_parameters["metric"],
                n_init=hyper_parameters["n_init"],
                max_iter=hyper_parameters["max_iter"],
            )

        self.dcp_workflow.show_info()

        # Use Scikit-learn style API to process input data
        X_reduced = self.dcp_workflow.fit_transform(X)
        X_reduced = self.dcp_workflow.np2pd(X_reduced, [f"Dimension {i+1}" for i in range(X_reduced.shape[1])])
        self.dcp_workflow.data_upload(X=X)

        # Save the model hyper-parameters
        self.dcp_workflow.save_hyper_parameters(hyper_parameters, self.model_name, os.getenv("GEOPI_OUTPUT_PARAMETERS_PATH"))

        # special components of different algorithms
        self.dcp_workflow.special_components(components_num=hyper_parameters["n_components"], reduced_data=X_reduced)

        # Save decomposition result
        self.dcp_workflow.data_save(X_reduced, "X Reduced", os.getenv("GEOPI_OUTPUT_ARTIFACTS_DATA_PATH"), MLFLOW_ARTIFACT_DATA_PATH, "Reduced Data")

        # Save the trained model
        self.dcp_workflow.model_save()
