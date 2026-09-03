"""Versioned classification choices exposed by the GeochemistryPi 0.8.2 CLI."""

from typing import Final

MODEL_ORDER: Final[tuple[str, ...]] = (
    "logistic_regression",
    "support_vector_machine",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "xgboost",
    "multi_layer_perceptron",
    "gradient_boosting",
    "k_nearest_neighbors",
    "stochastic_gradient_descent",
    "adaboost",
)

MODEL_DISPLAY_NAMES: Final[dict[str, str]] = {
    "logistic_regression": "Logistic Regression",
    "support_vector_machine": "Support Vector Machine",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "extra_trees": "Extra-Trees",
    "xgboost": "XGBoost",
    "multi_layer_perceptron": "Multi-layer Perceptron",
    "gradient_boosting": "Gradient Boosting",
    "k_nearest_neighbors": "K-Nearest Neighbors",
    "stochastic_gradient_descent": "Stochastic Gradient Descent",
    "adaboost": "AdaBoost",
}

MODEL_NUMBERS: Final[dict[str, int]] = {name: index for index, name in enumerate(MODEL_ORDER, start=1)}

# The public CLI limits an unprocessed-missing-value classification run to this
# model. This is a CLI contract, not an MCP-side scientific decision.
MODELS_SUPPORTING_MISSING_VALUES: Final[tuple[str, ...]] = ("xgboost",)

SCALING_METHODS: Final[tuple[str, ...]] = (
    "none",
    "min_max",
    "standardization",
    "mean_normalization",
)
LABEL_STRATEGIES: Final[tuple[str, ...]] = (
    "encode_original",
    "map",
    "interval",
    "quantile",
)
FEATURE_SELECTION_METHODS: Final[tuple[str, ...]] = (
    "none",
    "generic_univariate",
    "select_k_best",
)
TUNING_MODES: Final[tuple[str, ...]] = ("manual", "automl")
MISSING_VALUE_METHODS: Final[tuple[str, ...]] = (
    "error",
    "keep",
    "drop_rows",
    "impute",
)

UNSUPPORTED_INTERACTIONS: Final[tuple[str, ...]] = (
    "sample_balancing: GeochemistryPi 0.8.2 contains a helper, but its public data-mining workflow does not call it",
    "previous_experiment: MCP runs use explicit new experiment and run names so results cannot attach to an ambiguous prior run",
)
