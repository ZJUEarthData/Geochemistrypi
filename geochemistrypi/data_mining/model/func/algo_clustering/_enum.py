from enum import Enum


class ClusteringCommonFunction(Enum):
    CLUSTER_CENTERS = "Cluster Centers"
    CLUSTER_LABELS = "Cluster Labels"
    MODEL_PERSISTENCE = "Model Persistence"
    MODEL_SCORE = "Model Score"


class KMeansSpecialFunction(Enum):
    INERTIA_SCORE = "Inertia Score"


class MeanShiftSpecialFunction(Enum):
    NUM_CLUSTERS = "Num Clusters"
