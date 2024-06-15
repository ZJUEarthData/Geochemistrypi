import numpy as np
import scipy


# Calculate the Mahalanobis distance between two single points x and y
def mahalanobis_distance_singal(x, y, inv_cov):
    # Compute the difference between the two points
    x_minus_y = x - y
    # Calculate the Mahalanobis distance using the inverse covariance matrix
    return np.sqrt(np.dot(np.dot(x_minus_y, inv_cov), x_minus_y.T))


# Calculate the Mahalanobis distance between two sets of points
def mahalanobis_distance_calculator(mineral_a, mineral_b):
    # Calculate the inverse covariance matrix of the first set of points
    inv_cov = np.linalg.pinv(np.cov(mineral_a, rowvar=False))
    # Use scipy's cdist function to calculate the distance between each pair of points
    distance = scipy.spatial.distance.cdist(mineral_a, mineral_b, lambda u, v: mahalanobis_distance_singal(u, v, inv_cov))
    return distance


# Calculate the Euclidean distance between two sets of points
def euclidean_distance_calcular(mineral_a, mineral_b):
    # Specify the metric as Euclidean
    metric = "euclidean"
    # Use scipy's cdist function to calculate the distance between each pair of points
    distance = scipy.spatial.distance.cdist(mineral_a, mineral_b, metric)
    return distance


# Compute the distance between two sets of points using either Mahalanobis or Euclidean metric
def compute_distance_between_2(mineral_a, mineral_b, k=1, metric="euclidean"):
    # Select the distance calculation method based on the specified metric
    if metric == "mahalanobis":
        return mahalanobis_distance_calculator(mineral_a, mineral_b)
    elif metric == "euclidean":
        return euclidean_distance_calcular(mineral_a, mineral_b)
    else:
        # Raise an error if an unsupported metric is specified
        raise ValueError("Unsupported distance metric. Supported metrics are 'mahalanobis' and 'euclidean'.")
