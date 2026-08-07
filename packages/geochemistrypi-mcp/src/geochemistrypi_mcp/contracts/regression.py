"""Versioned regression choices exposed by the GeochemistryPi 0.8.1 CLI."""

from typing import Final

MODEL_ORDER: Final[tuple[str, ...]] = (
    "linear_regression",
    "polynomial_regression",
    "k_nearest_neighbors",
    "support_vector_machine",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "xgboost",
    "multi_layer_perceptron",
    "lasso_regression",
    "elastic_net",
    "stochastic_gradient_descent",
    "bayesian_ridge",
    "ridge_regression",
)

MODEL_DISPLAY_NAMES: Final[dict[str, str]] = {
    "linear_regression": "Linear Regression",
    "polynomial_regression": "Polynomial Regression",
    "k_nearest_neighbors": "K-Nearest Neighbors",
    "support_vector_machine": "Support Vector Machine",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "extra_trees": "Extra-Trees",
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
    "multi_layer_perceptron": "Multi-layer Perceptron",
    "lasso_regression": "Lasso Regression",
    "elastic_net": "Elastic Net",
    "stochastic_gradient_descent": "SGD Regression",
    "bayesian_ridge": "BayesianRidge Regression",
    "ridge_regression": "Ridge Regression",
}

MODEL_NUMBERS: Final[dict[str, int]] = {name: index for index, name in enumerate(MODEL_ORDER, start=1)}

# These two branches deliberately skip the CLI AutoML prompt.
MODELS_WITHOUT_AUTOML: Final[tuple[str, ...]] = (
    "linear_regression",
    "polynomial_regression",
)

# The public CLI restricts an unprocessed-missing-value regression run to XGBoost.
MODELS_SUPPORTING_MISSING_VALUES: Final[tuple[str, ...]] = ("xgboost",)

MODELS_WITH_INTERACTIVE_PLOT_SELECTION: Final[tuple[str, ...]] = (
    "linear_regression",
    "lasso_regression",
    "elastic_net",
    "stochastic_gradient_descent",
    "ridge_regression",
)

UNSUPPORTED_INTERACTIONS: Final[tuple[str, ...]] = (
    "regression.multiple_targets: the CLI contains partial multi-target support, but PR5 exposes one numeric target per validated run",
    "regression.previous_experiment: MCP runs use explicit new experiment and run names so results cannot attach to an ambiguous prior run",
    "regression.automl.linear_regression: the public CLI does not offer AutoML for Linear Regression",
    "regression.automl.polynomial_regression: the public CLI does not offer AutoML for Polynomial Regression",
)
