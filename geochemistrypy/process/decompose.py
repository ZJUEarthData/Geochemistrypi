# -*- coding: utf-8 -*-
# import sys
import pandas as pd
from model.decomposition import DecompositionWorkflowBase, PCADecomposition
from data.data_readiness import num_input
from global_variable import SECTION
from typing import Optional
# sys.path.append("..")


class DecompositionModelSelection(object):
    """Simulate the normal way of invoking scikit-learn decomposition algorithms."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.dcp_workflow = DecompositionWorkflowBase()
        self.components_num = None

    def activate(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> None:
        if self.model == "Principal Component Analysis":
            print("Decide the component numbers to keep:")
            self.components_num = num_input(SECTION[2])
            self.dcp_workflow = PCADecomposition(n_components=self.components_num)

        # common components for every decomposition algorithm
        self.dcp_workflow.show_info()
        self.dcp_workflow.fit(X)
        X_reduced = self.dcp_workflow.transform(X)
        self.dcp_workflow.data_upload(X=X)

        # special components of different algorithms
        self.dcp_workflow.special_components(components_num=self.components_num, reduced_data=X_reduced)