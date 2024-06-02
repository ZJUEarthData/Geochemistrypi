import os
import pandas as pd
from ._base import WorkflowBase
from ..constants import MLFLOW_ARTIFACT_DATA_PATH, MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH
from ..utils.base import save_data
from .func.algo_network._distance import mahalanobis_distance_calculator
from .func.algo_network._distance import euclidean_distance_calcular
from .func.algo_network._common import  pair_dataframes
from .func.algo_network._nearest import top_k_nearest_neighbors
from .func.algo_network._common import convert_to_triplets
from .func.algo_network._common import triplets_df_clean
from .func.algo_network._common import construct_adj_matrix
from .func.algo_network._community import bron_kerbosch_algo
from .func.algo_network._community import louvain_method_algo
from .func.algo_network._common import accurate_statistic_algo
class NetworkAnalysisWorkflowBase(WorkflowBase):
    def __init__(self) -> None:
        self.distance_calculator = "EU"
        self.community_detection_algo = "BK"
        self.minerals = [] # df list
        self.ids = []
        self.labels = []
        self.k = 1
        self.distances = pd.DataFrame   #graph
        self.communities_sample= pd.DataFrame #result
        self.communities_gruop= pd.DataFrame  #result after statistic
    def fit(self) -> None:
        merged_df = pd.concat([self.X, self.y], axis=1)
        split_dfs = []
        for mineral_type, group in merged_df.groupby('mineral_type'):
            split_dfs.append(group)
        extracted_dfs = []
        for mineral_type, df in  enumerate(split_dfs):
            last_column = df.iloc[:, -1:]
            extracted_dfs.append(last_column)
        for mineral_type, df in  enumerate(split_dfs):
            split_dfs[mineral_type] = df.iloc[:, :-1]
        self.minerals = split_dfs
        self.labels = extracted_dfs
    def manual_hyper_parameters(cls) -> None:
        """Manual hyper-parameters specification."""
        return dict()
    def generate_ids(self):
        offset = 0
        for df in self.minerals:
            self.ids.append(list(range(offset, offset + len(df))))
            offset += len(df)
    def compute_distance(self):
        all_triplets = []
        pair_combinations = pair_dataframes(self.minerals)
        if self.distance_calculator == 'EU':
            distance_func = euclidean_distance_calcular
        elif self.distance_calculator == 'MA':
            distance_func = mahalanobis_distance_calculator
        for pair in pair_combinations:
            mineral1, mineral2, index1, index2 = pair
            a_to_b_indices, a_to_b_distances=top_k_nearest_neighbors(distance_func(mineral1, mineral2),self.k)
            all_triplets +=convert_to_triplets(a_to_b_indices, a_to_b_distances,self.ids[index1], self.ids[index2])
        self.distances = triplets_df_clean(pd.DataFrame(all_triplets, columns=['Node1', 'Node2', 'Distance']))
        GEOPI_OUTPUT_ARTIFACTS_DATA_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_DATA_PATH")
        save_data(self.distances, f"{self.naming} Result", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)

    def accuracy_statistic(self):
        print("----------accuracy---------")
        self.communities_gruop=accurate_statistic_algo(self.communities_sample,self.ids,self.labels)
        GEOPI_OUTPUT_ARTIFACTS_DATA_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_DATA_PATH")
        save_data(self.communities_gruop, f"{self.naming} Result Accuracy Statistic", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)

class bron_kerbosch(NetworkAnalysisWorkflowBase):
    def __init__(self,X,y) -> None:
        super().__init__()
        self.naming = "Bron_kerbosch"
        self.X = X
        self.y = y
        self.community_detection_algo = "BK"

    def community_detection(self):
        self.fit()
        self.generate_ids()
        self.compute_distance()
        adj_matrix,mapping_df=construct_adj_matrix(self.distances)
        self.communities_sample=bron_kerbosch_algo(adj_matrix,mapping_df)
        GEOPI_OUTPUT_ARTIFACTS_DATA_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_DATA_PATH")
        save_data(self.communities_sample, f"{self.naming} Result", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
        self.accuracy_statistic()


class louvain_method(NetworkAnalysisWorkflowBase):
    def __init__(self,X,y) -> None:
        super().__init__()
        self.naming = "Louvain_method"
        self.X = X
        self.y = y
        self.community_detection_algo = "LU"

    def community_detection(self):
        self.fit()
        self.generate_ids()
        self.compute_distance()
        adj_matrix, mapping_df = construct_adj_matrix(self.distances)
        self.communities_sample=louvain_method_algo(adj_matrix,mapping_df)
        GEOPI_OUTPUT_ARTIFACTS_DATA_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_DATA_PATH")
        save_data(self.communities_sample, f"{self.naming} Result", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
        self.accuracy_statistic()


