# -*- coding: utf-8 -*-

import pandas as pd


from ..model.network import (
    NetworkAnalysisWorkflowBase,
    bron_kerbosch,
    louvain_method
)
from ._base import ModelSelectionBase
class NetworkAnalysisModelSelection(ModelSelectionBase):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        if model_name == "Bron Kerbosch Community Detection":
            self.net_workflow = BronKerboschWorkflow()
        elif model_name == "Louvain Method Community Detection":
            self.net_workflow = LouvainMethodWorkflow()

    def activate(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self.net_workflow.community_detection(X, y)

class BronKerboschWorkflow(NetworkAnalysisWorkflowBase):
    def community_detection(self, X, y):
        instance = bron_kerbosch(X,y)
        instance.community_detection()
        pass

class LouvainMethodWorkflow(NetworkAnalysisWorkflowBase):
    def community_detection(self, X, y):
        instance = louvain_method(X,y)
        instance.community_detection()
        pass

