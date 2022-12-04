# -*- coding: utf-8 -*-
# import sys
import pandas as pd
from data.data_readiness import num_input
from model.clustering import KMeansClustering, DBSCANClustering, ClusteringWorkflowBase
from global_variable import SECTION
from typing import Optional
# sys.path.append("..")


class ClusteringModelSelection(object):
    """Simulate the normal way of invoking scikit-learn clustering algorithms."""

    def __init__(self, model):
        self.model = model
        self.clt_workflow = ClusteringWorkflowBase()
        self.cluster_num = None

    def activate(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> None:
        if self.model == "KMeans":
            print("Designate the clustering number in advance:")
            self.cluster_num = num_input(SECTION[2])
            self.clt_workflow = KMeansClustering(n_clusters=self.cluster_num)
        elif self.model == "DBSCAN":
            # cluster_num = num_input("Designate the clustering number in advance:\n@Number: ")
            self.clt_workflow = DBSCANClustering()
        elif self.model == "":
            pass

        # common components for every clustering algorithm
        self.clt_workflow.show_info()
        self.clt_workflow.fit(X)
        self.clt_workflow.get_cluster_centers()
        self.clt_workflow.get_labels()
        
        # special components of different algorithms
        self.clt_workflow.special_components()

