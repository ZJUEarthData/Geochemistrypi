from itertools import combinations

import numpy as np
import pandas as pd


# Pair each dataframe with every other dataframe in a list of dataframes
def pair_dataframes(dataframes):
    pairs = []
    for pair in combinations(enumerate(dataframes), 2):
        idx1, df1 = pair[0]
        idx2, df2 = pair[1]
        pairs.append((df1, df2, idx1, idx2))
    return pairs


# Convert indices and distances into triplets of mineral IDs and their distance
def convert_to_triplets(indices, distances, mineral_a_ids, mineral_b_ids):
    triplets = []

    for i, (neighbors, distances_row) in enumerate(zip(indices, distances)):
        for neighbor, distance in zip(neighbors, distances_row):
            triplet = (mineral_a_ids[i], mineral_b_ids[neighbor], distance)
            triplets.append(triplet)

    return triplets


# Clean the dataframe of triplets by sorting, removing duplicates, and handling nulls or zeros in distance
def triplets_df_clean(triplets_df):
    triplets_df[["Node1", "Node2"]] = np.sort(triplets_df[["Node1", "Node2"]], axis=1)
    triplets_df = triplets_df.drop_duplicates(subset=["Node1", "Node2"])
    triplets_df["Distance"] = triplets_df["Distance"].apply(lambda x: 0.001 if pd.isnull(x) or x == 0 else x)
    triplets_df = triplets_df.dropna(subset=["Distance"])
    triplets_df = triplets_df.sort_values(by="Node1")
    return triplets_df


# Construct an adjacency matrix from graph data
def construct_adj_matrix(graph_data):
    nodes = np.unique(graph_data[["Node1", "Node2"]].values)
    num_nodes = len(nodes)
    adj_matrix = np.zeros((num_nodes, num_nodes))
    node_index_mapping = {node: idx for idx, node in enumerate(nodes)}
    for index, row in graph_data.iterrows():
        node1, node2, distance = int(row["Node1"]), int(row["Node2"]), row["Distance"]
        adj_matrix[node_index_mapping[node1], node_index_mapping[node2]] = distance
        adj_matrix[node_index_mapping[node2], node_index_mapping[node1]] = distance
    mapping_df = pd.DataFrame(list(node_index_mapping.items()), columns=["Original_Node", "Mapped_Node"])
    return adj_matrix, mapping_df


# Update community IDs based on a mapping and calculate unique and repeated counts
def accurate_statistic_algo(communities, ids, group_ids):
    result_df = communities.copy()
    flat_ids = np.array(ids).flatten()
    flat_group_ids = np.array(group_ids).flatten()

    for i, row in communities.iloc[1:].iterrows():
        for j, val in row.items():
            if val in flat_ids:
                idx = flat_ids.tolist().index(val)
                if idx < len(flat_group_ids):
                    result_df.at[i, j] = flat_group_ids[idx]
    repeated_counts = []
    unique_counts = []
    for index, row in result_df.iterrows():
        row_values = row[1:].dropna()
        seen_values = set()
        repeated_set = set()
        for num in row_values:
            if num in seen_values:
                repeated_set.add(num)
            else:
                seen_values.add(num)

        repeated_count = len(repeated_set)
        unique_count = len(seen_values) - repeated_count

        repeated_counts.append(repeated_count)
        unique_counts.append(unique_count)

    result_df["Repeated_Counts"] = repeated_counts
    result_df["Unique_Counts"] = unique_counts

    return result_df
