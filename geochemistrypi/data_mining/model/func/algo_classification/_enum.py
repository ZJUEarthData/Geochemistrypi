from enum import Enum


class ClassificationCommonFunction(Enum):
    MODEL_SCORE = "Model Score"
    CONFUSION_MATRIX = "Confusion Matrix"
    CROSS_VALIDATION = "Cross Validation"
    MODEL_PREDICTION = "Model Prediction"
    MODEL_PERSISTENCE = "Model Persistence"
    PRECISION_RECALL_CURVE = "Precision-Recall Curve"
    PRECISION_RECALL_THRESHOLD_DIAGRAM = "Precision-Recall vs. Threshold Diagram"
    ROC_CURVE = "ROC Curve"
    TWO_DIMENSIONAL_DECISION_BOUNDARY_DIAGRAM = "Two-dimensional Decision Boundary Diagram"
    PERMUTATION_IMPORTANCE_DIAGRAM = "Permutation Importance Diagram"
