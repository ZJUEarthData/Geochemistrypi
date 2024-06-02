import numpy as np
import scipy


def mahalanobis_distance_singal(x, y, inv_cov):
    x_minus_y = x - y

    return np.sqrt(np.dot(np.dot(x_minus_y, inv_cov), x_minus_y.T))


def mahalanobis_distance_calculator(mineral_a, mineral_b):
    inv_cov = np.linalg.pinv(np.cov(mineral_a, rowvar=False))
    distance = scipy.spatial.distance.cdist(mineral_a, mineral_b, lambda u, v: mahalanobis_distance_singal(u, v, inv_cov))
    return distance


def euclidean_distance_calcular(mineral_a, mineral_b):
    metric = "euclidean"
    distance = scipy.spatial.distance.cdist(mineral_a, mineral_b, metric)
    return distance


def compute_distance_between_2(mineral_a, mineral_b, k=1, metric="euclidean"):
    if metric == "mahalanobis":
        return mahalanobis_distance_calculator(mineral_a, mineral_b)
    elif metric == "euclidean":
        return euclidean_distance_calcular(mineral_a, mineral_b)
    else:
        raise ValueError("Unsupported distance metric. Supported metrics are 'mahalanobis' and 'euclidean'.")
