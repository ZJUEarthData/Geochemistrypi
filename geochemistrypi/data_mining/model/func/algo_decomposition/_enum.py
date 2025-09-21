from enum import Enum


class DecompositionCommonFunction(Enum):
    MODEL_PERSISTENCE = "Model Persistence"
    DECOMPOSITION_TWO_DIMENSIONAL = "Decomposition Two-Dimensional Diagram"
    DECOMPOSITION_HEATMAP = "Decomposition Heatmap"
    DIMENSIONALITY_REDUCTION_CONTOUR_PLOT = "Dimensionality Reduction Contour Plot"


class PCASpecialFunction(Enum):
    PRINCIPAL_COMPONENTS = "Principal Components"
    EXPLAINED_VARIANCE_RATIO = "Explained Variance Ratio"
    COMPOSITIONAL_BI_PLOT = "Compositional Bi-plot"
    COMPOSITIONAL_TRI_PLOT = "Compositional Tri-plot"
