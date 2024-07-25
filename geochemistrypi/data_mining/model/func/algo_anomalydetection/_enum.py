from enum import Enum


class AnormalyDetectionCommonFunction(Enum):
    PLOT_SCATTER_2D = "scatter 2d"
    PLOT_SCATTER_3D = "scatter 3d"
    DENSITY_ESTIMATION = "density estimation"


class LocalOutlierFactorSpecialFunction(Enum):
    PLOT_LOF_SCORE = "plot lof score"
