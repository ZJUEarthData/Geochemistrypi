from enum import Enum


class ClusteringCommonFunction(Enum):
    CLUSTER_CENTERS = "Cluster Centers"
    CLUSTER_LABELS = "Cluster Labels"
    MODEL_PERSISTENCE = "Model Persistence"
    MODEL_SCORE = "Model Score"
    CLUSTER_TWO_DIMENSIONAL_DIAGRAM = "Cluster Two-Dimensional Diagram"
    CLUSTER_THREE_DIMENSIONAL_DIAGRAM = "Cluster Three-Dimensional Diagram"
    SILHOUETTE_DIAGRAM = "Silhouette Diagram"
    SILHOUETTE_VALUE_DIAGRAM = "Silhouette value Diagram"
    SILHOUETTE_VALUE = "Silhouette Score vs K"


class KMeansSpecialFunction(Enum):
    INERTIA_SCORE = "Inertia Score"
    SILHOUETTE_VALUE = "Silhouette Score vs K"


class MeanShiftSpecialFunction(Enum):
    NUM_CLUSTERS = "Num of Clusters"


class AgglomeraSpecialFunction(Enum):
    SILHOUETTE_VALUE = "Silhouette Score vs K"
