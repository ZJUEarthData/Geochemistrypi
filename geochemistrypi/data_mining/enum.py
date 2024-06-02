from enum import Enum


class ModeOption(Enum):
    REGRESSION = "Regression"
    CLASSIFICATION = "Classification"
    CLUSTERING = "Clustering"
    DIMENSIONAL_REDUCTION = "Dimensional Reduction"
    ABNORMAL_DETECTION = "Abnormal Detection"


class ModeOptionWithMissingValues(Enum):
    REGRESSION = "Regression"
    CLASSIFICATION = "Classification"
    CLUSTERING = "Clustering"
