import communities.algorithms
import pandas as pd
def bron_kerbosch_algo(adj_matrix,mapping_df):
    communities_list = communities.algorithms.bron_kerbosch(adj_matrix, pivot=True)
    node_mapping_df = mapping_df
    mapping_dict = dict(zip(node_mapping_df['Mapped_Node'], node_mapping_df['Original_Node']))


    column_names = ['Community'] + [f'Node{i + 1}' for i in range(len(max(communities_list, key=len)))]

    community_data = [['Community'] + [f'Node{i + 1}' for i in range(len(max(communities_list, key=len)))]]
    community_data += [[f'Community {idx + 1}'] + list(community) for idx, community in
                       enumerate(communities_list)]

    bk_df = pd.DataFrame(community_data, columns=column_names)
    mapped_bk_df = pd.DataFrame(columns=column_names)
    for index, row in bk_df.iterrows():
        mapped_row = []
        for column in bk_df.columns[1:]:
            community_nodes = row[column]

            if not isinstance(community_nodes, list):
                community_nodes = [community_nodes]

            original_nodes = [mapping_dict.get(node, float('nan')) for node in community_nodes]
            mapped_row.append(original_nodes)

        mapped_bk_df.loc[index] = [row['Community']] + [item for sublist in mapped_row for item in sublist]
    return mapped_bk_df


def louvain_method_algo(adj_matrix,mapping_df):
    communities_list, _ =communities.algorithms.louvain_method(adj_matrix)
    node_mapping_df = mapping_df
    mapping_dict = dict(zip(node_mapping_df['Mapped_Node'], node_mapping_df['Original_Node']))
    column_names = ['Community'] + [f'Node{i + 1}' for i in range(len(max(communities_list, key=len)))]


    community_data = [['Community'] + [f'Node{i + 1}' for i in range(len(max(communities_list, key=len)))]]
    community_data += [[f'Community {idx + 1}'] + list(community) for idx, community in
                       enumerate(communities_list)]
    louvain_df=pd.DataFrame(community_data, columns=column_names)
    mapped_louvain_df = pd.DataFrame(columns=column_names)


    for index, row in louvain_df.iterrows():
        mapped_row = []


        for column in louvain_df.columns[1:]:
            community_nodes = row[column]

            if not isinstance(community_nodes, list):
                community_nodes = [community_nodes]

            original_nodes = [mapping_dict.get(node, float('nan')) for node in community_nodes]
            mapped_row.append(original_nodes)

        mapped_louvain_df.loc[index] = [row['Community']] + [item for sublist in mapped_row for item in sublist]
    return mapped_louvain_df