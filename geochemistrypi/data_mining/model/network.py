import os

import pandas as pd

from ..constants import MLFLOW_ARTIFACT_DATA_PATH
from ..utils.base import save_data
from ._base import WorkflowBase
from .func.algo_network._common import accurate_statistic_algo, construct_adj_matrix, convert_to_triplets, pair_dataframes, triplets_df_clean
from .func.algo_network._community import bron_kerbosch_algo, louvain_method_algo
from .func.algo_network._distance import euclidean_distance_calcular, mahalanobis_distance_calculator
from .func.algo_network._nearest import top_k_nearest_neighbors


class NetworkAnalysisWorkflowBase(WorkflowBase):
    def __init__(self) -> None:
        # Initialize default values
        self.distance_calculator = "EU"  # Default distance calculator
        self.community_detection_algo = "BK"  # Default community detection algorithm
        self.minerals = []  # List to store dataframes of minerals
        self.ids = []  # List to store ids
        self.labels = []  # List to store labels
        self.k = 1  # Number of nearest neighbors
        self.distances = pd.DataFrame  # DataFrame to store distances
        self.communities_sample = pd.DataFrame  # DataFrame to store sample communities
        self.communities_gruop = pd.DataFrame  # DataFrame to store grouped communities

    def fit(self) -> None:
        # Merge features and labels
        merged_df = pd.concat([self.X, self.y], axis=1)
        split_dfs = []
        # Group by mineral type and split
        for mineral_type, group in merged_df.groupby("mineral_type"):
            split_dfs.append(group)
        extracted_dfs = []
        # Extract last column from each group
        for mineral_type, df in enumerate(split_dfs):
            last_column = df.iloc[:, -1:]
            extracted_dfs.append(last_column)
        # Remove last column from original dataframes
        for mineral_type, df in enumerate(split_dfs):
            split_dfs[mineral_type] = df.iloc[:, :-1]
        self.minerals = split_dfs
        self.labels = extracted_dfs

    def manual_hyper_parameters(cls) -> None:
        """Manual hyper-parameters specification."""
        return dict()

    def generate_ids(self):
        # Generate unique ids for each mineral
        offset = 0
        for df in self.minerals:
            self.ids.append(list(range(offset, offset + len(df))))
            offset += len(df)

    def compute_distance(self):
        # Compute distances between minerals
        all_triplets = []
        pair_combinations = pair_dataframes(self.minerals)
        # Select distance calculator
        if self.distance_calculator == "EU":
            distance_func = euclidean_distance_calcular
        elif self.distance_calculator == "MA":
            distance_func = mahalanobis_distance_calculator
        # Compute distances for each pair
        for pair in pair_combinations:
            mineral1, mineral2, index1, index2 = pair
            a_to_b_indices, a_to_b_distances = top_k_nearest_neighbors(distance_func(mineral1, mineral2), self.k)
            all_triplets += convert_to_triplets(a_to_b_indices, a_to_b_distances, self.ids[index1], self.ids[index2])
        self.distances = triplets_df_clean(pd.DataFrame(all_triplets, columns=["Node1", "Node2", "Distance"]))
        # Save distances to file
        GEOPI_OUTPUT_ARTIFACTS_DATA_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_DATA_PATH")
        save_data(self.distances, f"{self.naming} Result", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)

    def accuracy_statistic(self):
        # Print accuracy statistics
        self.communities_gruop = accurate_statistic_algo(self.communities_sample, self.ids, self.labels)
        # Save accuracy statistics to file
        GEOPI_OUTPUT_ARTIFACTS_DATA_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_DATA_PATH")
        save_data(self.communities_gruop, f"{self.naming} Result Accuracy Statistic", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)


class bron_kerbosch(NetworkAnalysisWorkflowBase):
    def __init__(self, X, y) -> None:
        super().__init__()
        self.naming = "Bron_kerbosch"  # Name of the algorithm
        self.X = X  # Features
        self.y = y  # Labels
        self.community_detection_algo = "BK"  # Set community detection algorithm

    def community_detection(self):
        # Perform community detection using Bron-Kerbosch algorithm
        self.fit()
        self.generate_ids()
        self.compute_distance()
        adj_matrix, mapping_df = construct_adj_matrix(self.distances)
        self.communities_sample = bron_kerbosch_algo(adj_matrix, mapping_df)
        # Save communities to file
        GEOPI_OUTPUT_ARTIFACTS_DATA_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_DATA_PATH")
        save_data(self.communities_sample, f"{self.naming} Result", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
        self.accuracy_statistic()


class louvain_method(NetworkAnalysisWorkflowBase):
    def __init__(self, X, y) -> None:
        super().__init__()
        self.naming = "Louvain_method"  # Name of the algorithm
        self.X = X  # Features
        self.y = y  # Labels
        self.community_detection_algo = "LU"  # Set community detection algorithm

    def community_detection(self):
        # Perform community detection using Louvain method
        self.fit()
        self.generate_ids()
        self.compute_distance()
        adj_matrix, mapping_df = construct_adj_matrix(self.distances)
        self.communities_sample = louvain_method_algo(adj_matrix, mapping_df)
        # Save communities to file
        GEOPI_OUTPUT_ARTIFACTS_DATA_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_DATA_PATH")
        save_data(self.communities_sample, f"{self.naming} Result", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
        self.accuracy_statistic()


class BronKerboschWorkflow(NetworkAnalysisWorkflowBase):
    def community_detection(self, X, y):
        instance = bron_kerbosch(X, y)
        instance.community_detection()
        pass


class LouvainMethodWorkflow(NetworkAnalysisWorkflowBase):
    def community_detection(self, X, y):
        instance = louvain_method(X, y)
        instance.community_detection()
        pass
