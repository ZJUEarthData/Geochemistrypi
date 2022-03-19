# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
from data.data_readiness import num_input
from model.clustering import *


class ClusteringModelSelection(object):

    def __init__(self, model):
        self.model = model
        self.clt_workflow = ClusteringWorkflowBase()

    def activate(self, X, y=None):
        if self.model == "KMeans":
            cluster_num = num_input("Designate the clustering number in advance:\n@Number: ")
            self.clt_workflow = KMeansClustering(n_clusters=cluster_num)

        # common components for every clustering algorithm
        self.clt_workflow.show_info()
        self.clt_workflow.fit(X)
        self.clt_workflow.get_cluster_centers()
        self.clt_workflow.get_labels()
        # self.clt_workflow.plot_silhouette_diagram()

        # special components of different algorithms
        self.clt_workflow.special_components()



