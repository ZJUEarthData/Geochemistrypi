# -*- coding: utf-8 -*-
import pandas as pd

from ..model.network import BronKerboschWorkflow, LouvainMethodWorkflow
from ._base import ModelSelectionBase


class NetworkAnalysisModelSelection(ModelSelectionBase):
    def __init__(self, model_name: str) -> None:
        # Initialize the model name
        self.model_name = model_name
        # Initialize the corresponding network analysis workflow based on the model name
        if model_name == "Bron Kerbosch Community Detection":
            self.net_workflow = BronKerboschWorkflow()
        elif model_name == "Louvain Method Community Detection":
            self.net_workflow = LouvainMethodWorkflow()

    # Define the activate method, accepting input data X and y, and calling the community detection method of the corresponding workflow
    def activate(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self.net_workflow.community_detection(X, y)
