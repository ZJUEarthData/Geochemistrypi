# -*- coding: utf-8 -*-
# import sys
from data.data_readiness import num_input
from model.clustering import KMeansClustering, DBSCANClustering, ClusteringWorkflowBase
from global_variable import SECTION
# sys.path.append("..")

from utils.Delegate import MixinDelegator


class ClusteringModelSelection(MixinDelegator):
    DELEGATED_METHODS = {
        "KMeans" : [
            'show_info',
            'fit',
            'get_cluster_centers',
            'get_labels',
            'plot_silhouette_diagram',
            'plot_2d_graph',
            'plot_3d_graph',
            'special_components'
        ]
    }

    def __init__(self, model):
        self.KMeans = KMeansClustering()

        self.model = model
        self.clt_workflow = ClusteringWorkflowBase()
        self.cluster_num = None

    def activate(self, X, y=None):

        if self.model == "KMeans":
            print("Designate the clustering number in advance:")
            self.cluster_num = num_input(SECTION[2])
            #self.clt_workflow = KMeansClustering(n_clusters=self.cluster_num)
            self.clt_workflow = self.KMeans
        elif self.model == "AffinityPropagation":
            print("AffinityPropagation")
        elif self.model == "MeanShift":
            pass
        elif self.model == "Spectral":
            pass
        elif self.model == "WardHierarchical":
            pass
        elif self.model == "Agglomerative":
            pass
        elif self.model == "DBSCAN":
            # cluster_num = num_input("Designate the clustering number in advance:\n@Number: ")
            self.clt_workflow = DBSCANClustering()
        elif self.model == "OPTICS":
            pass
        elif self.model == "GaussianMixtures":
            pass
        elif self.model == "BIRCHClustering":
            pass
        elif self.model == "BisectingKMeans":
            pass

        self.KMeans.__init__(n_clusters=self.cluster_num)
        '''
        # common components for every clustering algorithm
        self.clt_workflow.show_info()
        self.clt_workflow.fit(X)
        self.clt_workflow.get_cluster_centers()
        self.clt_workflow.get_labels()
        self.clt_workflow.plot_silhouette_diagram(self.cluster_num)
        self.clt_workflow.plot_2d_graph()
        self.clt_workflow.plot_3d_graph()

        # special components of different algorithms
        self.clt_workflow.special_components()
        '''
        print("---------Test---------")
        self.show_info()
        self.fit(X)
        self.get_cluster_centers()
        #self.get_labels()
        self.plot_silhouette_diagram(self.cluster_num)
        self.plot_2d_graph()
        self.plot_3d_graph()
        self.special_components()
        print("---------Test---------")
