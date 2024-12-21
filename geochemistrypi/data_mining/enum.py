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


class DataSource(Enum):
    BUILT_IN = "Built-in"
    DESKTOP = "Desktop"
    ANY_PATH = "Any Path"
