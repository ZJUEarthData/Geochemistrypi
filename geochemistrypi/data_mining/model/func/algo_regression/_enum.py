from enum import Enum


class RegressionCommonFunction(Enum):
    PREDICTED_VS_ACTUAL_DIAGRAM = "Predicted vs. Actual Diagram"
    RESIDUALS_DIAGRAM = "Residuals Diagram"
    MODEL_SCORE = "Model Score"
    CROSS_VALIDATION = "Cross Validation"


class MLPSpecialFunction(Enum):
    LOSS_CURVE_DIAGRAM = "Loss Curve Diagram"
