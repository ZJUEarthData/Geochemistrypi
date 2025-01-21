from enum import Enum


class ClassificationCommonFunction(Enum):
    MODEL_SCORE = "Model Score"
    CLASSIFICATION_REPORT = "Classification Report"
    CONFUSION_MATRIX = "Confusion Matrix"
    CROSS_VALIDATION = "Cross Validation"
    MODEL_PREDICTION = "Model Prediction"
    MODEL_PERSISTENCE = "Model Persistence"
    PRECISION_RECALL_CURVE = "Precision-Recall Curve"
    PRECISION_RECALL_THRESHOLD_DIAGRAM = "Precision-Recall vs. Threshold Diagram"
    ROC_CURVE = "ROC Curve"
    TWO_DIMENSIONAL_DECISION_BOUNDARY_DIAGRAM = "Two-dimensional Decision Boundary Diagram"
    PERMUTATION_IMPORTANCE_DIAGRAM = "Permutation Importance Diagram"


class DecisionTreeSpecialFunction(Enum):
    FEATURE_IMPORTANCE_DIAGRAM = "Feature Importance Diagram"
    SINGLE_TREE_DIAGRAM = "Single Tree Diagram"


class RandomForestSpecialFunction(Enum):
    FEATURE_IMPORTANCE_DIAGRAM = "Feature Importance Diagram"
    SINGLE_TREE_DIAGRAM = "Single Tree Diagram"


class XGBoostSpecialFunction(Enum):
    FEATURE_IMPORTANCE_DIAGRAM = "Feature Importance Diagram"


class LogisticRegressionSpecialFunction(Enum):
    LOGISTIC_REGRESSION_FORMULA = "Logistic Regression Formula"
    FEATURE_IMPORTANCE_DIAGRAM = "Feature Importance Diagram"


class MLPSpecialFunction(Enum):
    LOSS_CURVE_DIAGRAM = "Loss Curve Diagram"


class ExtraTreesSpecialFunction(Enum):
    FEATURE_IMPORTANCE_DIAGRAM = "Feature Importance Diagram"
    SINGLE_TREE_DIAGRAM = "Single Tree Diagram"


class GradientBoostingSpecialFunction(Enum):
    FEATURE_IMPORTANCE_DIAGRAM = "Feature Importance Diagram"
    SINGLE_TREE_DIAGRAM = "Single Tree Diagram"


class AdaBoostSpecialFunction(Enum):
    FEATURE_IMPORTANCE_DIAGRAM = "Feature Importance Diagram"
    SINGLE_TREE_DIAGRAM = "Single Tree Diagram"


class SGDSpecialFunction(Enum):
    SGD_CLASSIFICATION_FORMULA = "SGD Classification Formula"
