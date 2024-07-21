from enum import Enum


class ModeOption(Enum):
    REGRESSION = "Regression"
    CLASSIFICATION = "Classification"
    CLUSTERING = "Clustering"
    DIMENSIONAL_REDUCTION = "Dimensional Reduction"
    ANOMALY_DETECTION = "Anomaly Detection"


class ModeOptionWithMissingValues(Enum):
    REGRESSION = "Regression"
    CLASSIFICATION = "Classification"
    CLUSTERING = "Clustering"
