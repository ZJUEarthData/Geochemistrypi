# -*- coding: utf-8 -*-
import pandas as pd
from typing import Optional

from ..data.data_readiness import num_input, float_input, str_input
from ..model.clustering import KMeansClustering, DBSCANClustering, ClusteringWorkflowBase
from ..global_variable import SECTION



class ClusteringModelSelection(object):
    """Simulate the normal way of invoking scikit-learn clustering algorithms."""

    def __init__(self, model):
        self.model = model
        self.clt_workflow = ClusteringWorkflowBase()

    def activate(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None, X_train: Optional[pd.DataFrame] = None,
                 X_test: Optional[pd.DataFrame] = None, y_train: Optional[pd.DataFrame] = None,
                 y_test: Optional[pd.DataFrame] = None) -> None:
        """Train by Scikit-learn framework."""

        self.clt_workflow.data_upload(X=X, y=y, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

        if self.model == "KMeans":
            hyper_parameters = KMeansClustering.manual_hyper_parameters()
            self.clt_workflow = KMeansClustering(n_clusters=hyper_parameters["n_clusters"], init=hyper_parameters["init"], max_iter=hyper_parameters["max_iter"], tol=hyper_parameters["tol"], algorithm=hyper_parameters["algorithm"])
        elif self.model == "DBSCAN":
            hyper_parameters = DBSCANClustering.manual_hyper_parameters()
            self.clt_workflow = DBSCANClustering(eps=hyper_parameters["eps"], min_samples=hyper_parameters["min_samples"], metric=hyper_parameters["metric"], algorithm=hyper_parameters["algorithm"], leaf_size=hyper_parameters["leaf_size"], p=hyper_parameters["p"])
        elif self.model == "":
            pass

        # common components for every clustering algorithm
        self.clt_workflow.show_info()
        self.clt_workflow.fit(X)
        self.clt_workflow.get_cluster_centers()
        self.clt_workflow.get_labels()

        # special components of different algorithms
        self.clt_workflow.special_components()

        # Save the trained model
        self.clt_workflow.save_model()


