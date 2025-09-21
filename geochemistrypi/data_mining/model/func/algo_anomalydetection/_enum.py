from enum import Enum


class AnormalyDetectionCommonFunction(Enum):
    PLOT_SCATTER_2D = "Anomaly Detection Two-Dimensional Diagram"
    PLOT_SCATTER_3D = "Anomaly Detection Three-Dimensional Diagram"
    DENSITY_ESTIMATION = "Anomaly Detection Density Estimation"


class LocalOutlierFactorSpecialFunction(Enum):
    PLOT_LOF_SCORE = "Lof Score Diagram"
