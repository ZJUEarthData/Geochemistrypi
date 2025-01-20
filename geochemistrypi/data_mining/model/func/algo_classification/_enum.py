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
    FEATURE_IMPORTANCE = "Feature Importance"
    TREE_DIAGRAM = "Tree Diagram"


class RandomForestSpecialFunction(Enum):
    FEATURE_IMPORTANCE = "Feature Importance"
    TREE_DIAGRAM = "Tree Diagram"


class XGBoostSpecialFunction(Enum):
    FEATURE_IMPORTANCE = "Feature Importance"


class LogisticRegressionSpecialFunction(Enum):
    FEATURE_IMPORTANCE = "Feature Importance"


class MLPSpecialFunction(Enum):
    LOSS_CURVE_DIAGRAM = "Loss Curve Diagram"


class ExtraTreesSpecialFunction(Enum):
    FEATURE_IMPORTANCE = "Feature Importance"
    TREE_DIAGRAM = "Tree Diagram"


class GradientBoostingSpecialFunction(Enum):
    FEATURE_IMPORTANCE = "Feature Importance"
    TREE_DIAGRAM = "Tree Diagram"


class AdaboostSpecialFunction(Enum):
    SPECIAL_FUNCTION = ["Feature Importance Diagram", "Single Tree Diagram"]
