import communities.algorithms
import pandas as pd


# Function to apply the Bron-Kerbosch algorithm to find maximal cliques in an adjacency matrix
def bron_kerbosch_algo(adj_matrix, mapping_df):
    # Use the Bron-Kerbosch algorithm from the communities package, with pivoting enabled
    communities_list = communities.algorithms.bron_kerbosch(adj_matrix, pivot=True)
    # Retrieve the mapping from the dataframe to map back to original node IDs
    node_mapping_df = mapping_df
    mapping_dict = dict(zip(node_mapping_df["Mapped_Node"], node_mapping_df["Original_Node"]))

    # Prepare column names for the dataframe based on the largest community size
    column_names = ["Community"] + [f"Node{i + 1}" for i in range(len(max(communities_list, key=len)))]

    # Initialize community data with column names
    community_data = [["Community"] + [f"Node{i + 1}" for i in range(len(max(communities_list, key=len)))]]
    # Append community data with community ID and nodes in each community
    community_data += [[f"Community {idx + 1}"] + list(community) for idx, community in enumerate(communities_list)]

    # Create a dataframe from the community data
    bk_df = pd.DataFrame(community_data, columns=column_names)
    # Initialize a dataframe for mapped community data
    mapped_bk_df = pd.DataFrame(columns=column_names)
    # Map the community nodes back to their original IDs
    for index, row in bk_df.iterrows():
        mapped_row = []
        for column in bk_df.columns[1:]:
            community_nodes = row[column]

            if not isinstance(community_nodes, list):
                community_nodes = [community_nodes]

            original_nodes = [mapping_dict.get(node, float("nan")) for node in community_nodes]
            mapped_row.append(original_nodes)

        mapped_bk_df.loc[index] = [row["Community"]] + [item for sublist in mapped_row for item in sublist]
    return mapped_bk_df


# Function to apply the Louvain method to find communities in an adjacency matrix
def louvain_method_algo(adj_matrix, mapping_df):
    # Use the Louvain method from the communities package
    communities_list, _ = communities.algorithms.louvain_method(adj_matrix)
    # Retrieve the mapping from the dataframe to map back to original node IDs
    node_mapping_df = mapping_df
    mapping_dict = dict(zip(node_mapping_df["Mapped_Node"], node_mapping_df["Original_Node"]))
    # Prepare column names for the dataframe based on the largest community size
    column_names = ["Community"] + [f"Node{i + 1}" for i in range(len(max(communities_list, key=len)))]

    # Initialize community data with column names
    community_data = [["Community"] + [f"Node{i + 1}" for i in range(len(max(communities_list, key=len)))]]
    # Append community data with community ID and nodes in each community
    community_data += [[f"Community {idx + 1}"] + list(community) for idx, community in enumerate(communities_list)]
    # Create a dataframe from the community data
    louvain_df = pd.DataFrame(community_data, columns=column_names)
    # Initialize a dataframe for mapped community data
    mapped_louvain_df = pd.DataFrame(columns=column_names)
    # Map the community nodes back to their original IDs
    for index, row in louvain_df.iterrows():
        mapped_row = []

        for column in louvain_df.columns[1:]:
            community_nodes = row[column]

            if not isinstance(community_nodes, list):
                community_nodes = [community_nodes]

            original_nodes = [mapping_dict.get(node, float("nan")) for node in community_nodes]
            mapped_row.append(original_nodes)

        mapped_louvain_df.loc[index] = [row["Community"]] + [item for sublist in mapped_row for item in sublist]
    return mapped_louvain_df
