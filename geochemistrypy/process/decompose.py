# -*- coding: utf-8 -*-
# import sys
from model.decomposition import DecompositionWorkflowBase, PrincipalComponentAnalysis

# sys.path.append("..")


class DecompositionModelSelection(object):

    def __init__(self, model):
        self.model = model
        self.dcp_workflow = DecompositionWorkflowBase()

    def activate(self, X, y=None):
        if self.model == "Principal Component Analysis":
            self.dcp_workflow = PrincipalComponentAnalysis()

        # common components for every decomposition algorithm
        self.dcp_workflow.show_info()
        self.dcp_workflow.fit(X)

        # special components of different algorithms
        self.dcp_workflow.special_components()
