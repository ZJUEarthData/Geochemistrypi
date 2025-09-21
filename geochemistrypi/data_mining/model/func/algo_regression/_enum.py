from enum import Enum


class RegressionCommonFunction(Enum):
    MODEL_SCORE = "Model Score"
    CROSS_VALIDATION = "Cross Validation"
    MODEL_PREDICTION = "Model Prediction"
    MODEL_PERSISTENCE = "Model Persistence"
    PREDICTED_VS_ACTUAL_DIAGRAM = "Predicted vs. Actual Diagram"
    RESIDUALS_DIAGRAM = "Residuals Diagram"
    PERMUTATION_IMPORTANC_DIAGRAM = "Permutation Importance Diagram"


class XGBoostSpecialFunction(Enum):
    FEATURE_IMPORTANCE_DIAGRAM = "Feature Importance Diagram"


class DecisionTreeSpecialFunction(Enum):
    FEATURE_IMPORTANCE_DIAGRAM = "Feature Importance Diagram"
    SINGLE_TREE_DIAGRAM = "Single Tree Diagram"


class MLPSpecialFunction(Enum):
    LOSS_CURVE_DIAGRAM = "Loss Curve Diagram"


class ClassicalLinearSpecialFunction(Enum):
    LINEAR_REGRESSION_FORMULA = "Linear Regression Formula"
    TWO_DIMENSIONAL_SCATTER_DIAGRAM = "2D Scatter Diagram"
    THREE_DIMENSIONAL_SCATTER_DIAGRAM = "3D Scatter Diagram"
    TWO_DIMENSIONAL_LINE_DIAGRAM = "2D Line Diagram"
    THREE_DIMENSIONAL_SURFACE_DIAGRAM = "3D Surface Diagram"


class LassoSpecialFunction(Enum):
    LASSO_REGRESSION_FORMULA = "Lasso Regression Formula"
    TWO_DIMENSIONAL_SCATTER_DIAGRAM = "2D Scatter Diagram"
    THREE_DIMENSIONAL_SCATTER_DIAGRAM = "3D Scatter Diagram"
    TWO_DIMENSIONAL_LINE_DIAGRAM = "2D Line Diagram"
    THREE_DIMENSIONAL_SURFACE_DIAGRAM = "3D Surface Diagram"


class ElasticNetSpecialFunction(Enum):
    ELASTIC_NET_FORMULA = "Elastic Net Formula"
    TWO_DIMENSIONAL_SCATTER_DIAGRAM = "2D Scatter Diagram"
    THREE_DIMENSIONAL_SCATTER_DIAGRAM = "3D Scatter Diagram"
    TWO_DIMENSIONAL_LINE_DIAGRAM = "2D Line Diagram"
    THREE_DIMENSIONAL_SURFACE_DIAGRAM = "3D Surface Diagram"


class SGDSpecialFunction(Enum):
    SGD_REGRESSION_FORMULA = "SGD Regression Formula"
    TWO_DIMENSIONAL_SCATTER_DIAGRAM = "2D Scatter Diagram"
    THREE_DIMENSIONAL_SCATTER_DIAGRAM = "3D Scatter Diagram"
    TWO_DIMENSIONAL_LINE_DIAGRAM = "2D Line Diagram"
    THREE_DIMENSIONAL_SURFACE_DIAGRAM = "3D Surface Diagram"


class RidgeSpecialFunction(Enum):
    RIDGE_REGRESSION_FORMULA = "Ridge Regression Formula"
    TWO_DIMENSIONAL_SCATTER_DIAGRAM = "2D Scatter Diagram"
    THREE_DIMENSIONAL_SCATTER_DIAGRAM = "3D Scatter Diagram"
    TWO_DIMENSIONAL_LINE_DIAGRAM = "2D Line Diagram"
    THREE_DIMENSIONAL_SURFACE_DIAGRAM = "3D Surface Diagram"


class ExtraTreesSpecialFunction(Enum):
    FEATURE_IMPORTANCE_DIAGRAM = "Feature Importance Diagram"
    SINGLE_TREE_DIAGRAM = "Single Tree Diagram"


class RandomForestSpecialFunction(Enum):
    FEATURE_IMPORTANCE_DIAGRAM = "Feature Importance Diagram"
    SINGLE_TREE_DIAGRAM = "Single Tree Diagram"


class GradientBoostingSpecialFunction(Enum):
    FEATURE_IMPORTANCE_DIAGRAM = "Feature Importance Diagram"
    SINGLE_TREE_DIAGRAM = "Single Tree Diagram"


class PolynomialSpecialFunction(Enum):
    POLYNOMIAL_REGRESSION_FORMULA = "Polynomial Regression Formula"
