from enum import Enum


class DecompositionCommonFunction(Enum):
    MODEL_PERSISTENCE = "Model Persistence"


class PCASpecialFunction(Enum):
    PRINCIPAL_COMPONENTS = "Principal Components"
    EXPLAINED_VARIANCE_RATIO = "Explained Variance Ratio"
    COMPOSITIONAL_BI_PLOT = "Compositional Bi-plot"
    COMPOSITIONAL_TRI_PLOT = "Compositional Tri-plot"
