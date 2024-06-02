import numpy as np

def top_k_nearest_neighbors(distance, k=5):

    nearest_neighbors_indices = np.argsort(distance, axis=1)[:, :k]

    nearest_neighbors_distances = np.take_along_axis(distance, nearest_neighbors_indices, axis=1)

    return nearest_neighbors_indices, nearest_neighbors_distances