"""Modern, non-interactive model registry for the v0.8 Online API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.base import RegressorMixin
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures


@dataclass(frozen=True)
class RegressionModelDefinition:
    name: str
    display_name: str
    description: str
    factory: Callable[[], RegressorMixin | Pipeline]


REGRESSION_MODELS: dict[str, RegressionModelDefinition] = {
    definition.name: definition
    for definition in (
        RegressionModelDefinition(
            name="linear_regression",
            display_name="Linear Regression",
            description="Ordinary least-squares linear regression.",
            factory=LinearRegression,
        ),
        RegressionModelDefinition(
            name="polynomial_regression",
            display_name="Polynomial Regression",
            description="Second-order polynomial features followed by linear regression.",
            factory=lambda: make_pipeline(
                PolynomialFeatures(degree=2, include_bias=False),
                LinearRegression(),
            ),
        ),
        RegressionModelDefinition(
            name="lasso_regression",
            display_name="Lasso Regression",
            description="L1-regularized linear regression for sparse coefficients.",
            factory=lambda: Lasso(alpha=1.0, max_iter=10_000),
        ),
        RegressionModelDefinition(
            name="elastic_net",
            display_name="Elastic Net",
            description="Combined L1/L2-regularized linear regression.",
            factory=lambda: ElasticNet(
                alpha=1.0,
                l1_ratio=0.5,
                max_iter=10_000,
                random_state=42,
            ),
        ),
        RegressionModelDefinition(
            name="bayesian_ridge_regression",
            display_name="Bayesian Ridge Regression",
            description="Bayesian linear regression with automatic regularization.",
            factory=BayesianRidge,
        ),
        RegressionModelDefinition(
            name="ridge_regression",
            display_name="Ridge Regression",
            description="L2-regularized linear regression.",
            factory=lambda: Ridge(alpha=1.0),
        ),
    )
}


def get_regression_model(name: str) -> RegressionModelDefinition:
    try:
        return REGRESSION_MODELS[name]
    except KeyError as exc:
        choices = ", ".join(REGRESSION_MODELS)
        raise ValueError(
            f"Unknown regression model '{name}'. Choose one of: {choices}"
        ) from exc


def extract_linear_parameters(
    fitted_model: RegressorMixin | Pipeline,
    feature_names: list[str],
) -> tuple[float, list[str], list[float]]:
    """Return the fitted intercept and coefficient names for supported models."""
    estimator = fitted_model
    coefficient_names = feature_names
    if isinstance(fitted_model, Pipeline):
        polynomial = fitted_model.named_steps["polynomialfeatures"]
        estimator = fitted_model.named_steps["linearregression"]
        coefficient_names = [
            str(name)
            for name in polynomial.get_feature_names_out(feature_names)
        ]

    intercept = float(estimator.intercept_)
    coefficients = [float(value) for value in estimator.coef_]
    return intercept, coefficient_names, coefficients


__all__ = [
    "REGRESSION_MODELS",
    "RegressionModelDefinition",
    "extract_linear_parameters",
    "get_regression_model",
]
