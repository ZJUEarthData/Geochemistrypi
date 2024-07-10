import numpy as np


# Function to find the top k nearest neighbors for each point
def top_k_nearest_neighbors(distance, k=5):
    # Sort the distances and get indices of the nearest neighbors
    nearest_neighbors_indices = np.argsort(distance, axis=1)[:, :k]

    # Retrieve the distances of the nearest neighbors using the sorted indices
    nearest_neighbors_distances = np.take_along_axis(distance, nearest_neighbors_indices, axis=1)

    # Return the indices and distances of the nearest neighbors
    return nearest_neighbors_indices, nearest_neighbors_distances
