"""Modern, non-interactive model registry for the v0.8 Online API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    ClusterMixin,
    RegressorMixin,
)
from sklearn.cluster import (
    DBSCAN,
    OPTICS,
    AffinityPropagation,
    AgglomerativeClustering,
    KMeans,
    MeanShift,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesRegressor,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.decomposition import PCA
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.manifold import MDS, TSNE
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    LocalOutlierFactor,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier, XGBRegressor


class LabelEncodedXGBClassifier(ClassifierMixin, BaseEstimator):
    """XGBoost classifier that accepts and restores arbitrary class labels."""

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 4,
        subsample: float = 0.8,
        colsample_bytree: float = 1.0,
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, features: Any, target: Any) -> "LabelEncodedXGBClassifier":
        self.label_encoder_ = LabelEncoder()
        encoded_target = self.label_encoder_.fit_transform(np.asarray(target))
        self.classes_ = self.label_encoder_.classes_
        class_count = len(self.classes_)
        parameters: dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "tree_method": "hist",
            "verbosity": 0,
        }
        if class_count > 2:
            parameters.update(
                objective="multi:softprob",
                num_class=class_count,
                eval_metric="mlogloss",
            )
        else:
            parameters.update(
                objective="binary:logistic",
                eval_metric="logloss",
            )
        self.model_ = XGBClassifier(**parameters)
        self.model_.fit(features, encoded_target)
        return self

    def predict(self, features: Any) -> np.ndarray:
        check_is_fitted(self, ("model_", "label_encoder_"))
        encoded = np.asarray(self.model_.predict(features), dtype=int)
        return self.label_encoder_.inverse_transform(encoded)


@dataclass(frozen=True)
class RegressionModelDefinition:
    name: str
    display_name: str
    description: str
    factory: Callable[[], RegressorMixin | Pipeline]


@dataclass(frozen=True)
class ClassificationModelDefinition:
    name: str
    display_name: str
    description: str
    factory: Callable[[], ClassifierMixin | Pipeline]


@dataclass(frozen=True)
class HyperparameterDefinition:
    """A safe, UI-facing subset of an estimator's configurable parameters."""

    name: str
    display_name: str
    description: str
    value_type: str
    default: Any
    estimator_parameter: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    options: tuple[Any, ...] = ()


@dataclass(frozen=True)
class ClusteringModelDefinition:
    name: str
    display_name: str
    description: str
    uses_cluster_count: bool
    factory: Callable[[int], ClusterMixin]


@dataclass(frozen=True)
class DimensionalityReductionModelDefinition:
    name: str
    display_name: str
    description: str
    max_rows: int | None
    factory: Callable[[int, int], Any]


@dataclass(frozen=True)
class AnomalyDetectionModelDefinition:
    name: str
    display_name: str
    description: str
    factory: Callable[[int, str | float], Any]


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
        RegressionModelDefinition(
            name="decision_tree",
            display_name="Decision Tree",
            description="Nonlinear regression using a reproducible decision tree.",
            factory=lambda: DecisionTreeRegressor(random_state=42),
        ),
        RegressionModelDefinition(
            name="extra_trees",
            display_name="Extra-Trees",
            description="Regression using 200 extremely randomized decision trees.",
            factory=lambda: ExtraTreesRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=1,
            ),
        ),
        RegressionModelDefinition(
            name="gradient_boosting",
            display_name="Gradient Boosting",
            description="Sequential gradient-boosted decision-tree regression.",
            factory=lambda: GradientBoostingRegressor(random_state=42),
        ),
        RegressionModelDefinition(
            name="k_nearest_neighbors",
            display_name="K-Nearest Neighbors",
            description="Standardized regression using the five nearest samples.",
            factory=lambda: make_pipeline(
                StandardScaler(),
                KNeighborsRegressor(n_neighbors=5),
            ),
        ),
        RegressionModelDefinition(
            name="multi_layer_perceptron",
            display_name="Multi-layer Perceptron",
            description="Standardized neural-network regression with one hidden layer.",
            factory=lambda: make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(100,),
                    max_iter=2_000,
                    random_state=42,
                ),
            ),
        ),
        RegressionModelDefinition(
            name="random_forest",
            display_name="Random Forest",
            description="Ensemble regression using 200 randomized decision trees.",
            factory=lambda: RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=1,
            ),
        ),
        RegressionModelDefinition(
            name="stochastic_gradient_descent",
            display_name="Stochastic Gradient Descent",
            description="Standardized linear regression optimized by SGD.",
            factory=lambda: make_pipeline(
                StandardScaler(),
                SGDRegressor(
                    max_iter=2_000,
                    tol=1e-3,
                    random_state=42,
                ),
            ),
        ),
        RegressionModelDefinition(
            name="support_vector_machine",
            display_name="Support Vector Machine",
            description="Standardized nonlinear regression with an RBF kernel.",
            factory=lambda: make_pipeline(StandardScaler(), SVR(kernel="rbf")),
        ),
        RegressionModelDefinition(
            name="xgboost",
            display_name="XGBoost",
            description="Gradient-boosted tree regression using the XGBoost engine.",
            factory=lambda: XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=1.0,
                objective="reg:squarederror",
                eval_metric="rmse",
                random_state=42,
                n_jobs=1,
                tree_method="hist",
                verbosity=0,
            ),
        ),
    )
}


CLASSIFICATION_MODELS: dict[str, ClassificationModelDefinition] = {
    definition.name: definition
    for definition in (
        ClassificationModelDefinition(
            name="logistic_regression",
            display_name="Logistic Regression",
            description="Standardized logistic classification with L2 regularization.",
            factory=lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2_000, random_state=42),
            ),
        ),
        ClassificationModelDefinition(
            name="support_vector_machine",
            display_name="Support Vector Machine",
            description="Standardized nonlinear classification with an RBF kernel.",
            factory=lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf")),
        ),
        ClassificationModelDefinition(
            name="decision_tree",
            display_name="Decision Tree",
            description="Decision-tree classification with a reproducible random seed.",
            factory=lambda: DecisionTreeClassifier(random_state=42),
        ),
        ClassificationModelDefinition(
            name="random_forest",
            display_name="Random Forest",
            description="Ensemble classification using 200 randomized decision trees.",
            factory=lambda: RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        ClassificationModelDefinition(
            name="extra_trees",
            display_name="Extra-Trees",
            description="Extremely randomized tree ensemble classification.",
            factory=lambda: ExtraTreesClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        ClassificationModelDefinition(
            name="multi_layer_perceptron",
            display_name="Multi-layer Perceptron",
            description="Standardized neural-network classification with one hidden layer.",
            factory=lambda: make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(100,),
                    max_iter=2_000,
                    random_state=42,
                ),
            ),
        ),
        ClassificationModelDefinition(
            name="gradient_boosting",
            display_name="Gradient Boosting",
            description="Sequential gradient-boosted decision-tree classification.",
            factory=lambda: GradientBoostingClassifier(random_state=42),
        ),
        ClassificationModelDefinition(
            name="k_nearest_neighbors",
            display_name="K-Nearest Neighbors",
            description="Standardized classification using the five nearest samples.",
            factory=lambda: make_pipeline(
                StandardScaler(),
                KNeighborsClassifier(n_neighbors=5),
            ),
        ),
        ClassificationModelDefinition(
            name="stochastic_gradient_descent",
            display_name="Stochastic Gradient Descent",
            description="Standardized linear classification optimized by SGD.",
            factory=lambda: make_pipeline(
                StandardScaler(),
                SGDClassifier(
                    loss="log_loss",
                    max_iter=2_000,
                    tol=1e-3,
                    random_state=42,
                ),
            ),
        ),
        ClassificationModelDefinition(
            name="adaboost",
            display_name="AdaBoost",
            description="Adaptive boosting classification using 100 estimators.",
            factory=lambda: AdaBoostClassifier(
                n_estimators=100,
                random_state=42,
            ),
        ),
        ClassificationModelDefinition(
            name="xgboost",
            display_name="XGBoost",
            description=(
                "Gradient-boosted tree classification with automatic label encoding."
            ),
            factory=LabelEncodedXGBClassifier,
        ),
    )
}


def _parameter(
    name: str,
    display_name: str,
    description: str,
    value_type: str,
    default: Any,
    *,
    estimator_parameter: str | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
    step: float | None = None,
    options: tuple[Any, ...] = (),
) -> HyperparameterDefinition:
    return HyperparameterDefinition(
        name=name,
        display_name=display_name,
        description=description,
        value_type=value_type,
        default=default,
        estimator_parameter=estimator_parameter,
        minimum=minimum,
        maximum=maximum,
        step=step,
        options=options,
    )


REGRESSION_HYPERPARAMETERS: dict[str, tuple[HyperparameterDefinition, ...]] = {
    "linear_regression": (
        _parameter("fit_intercept", "Fit intercept", "Estimate a constant intercept.", "boolean", True),
    ),
    "polynomial_regression": (
        _parameter("degree", "Polynomial degree", "Maximum polynomial feature degree.", "integer", 2, estimator_parameter="polynomialfeatures__degree", minimum=1, maximum=5, step=1),
        _parameter("fit_intercept", "Fit intercept", "Estimate a constant intercept.", "boolean", True, estimator_parameter="linearregression__fit_intercept"),
    ),
    "lasso_regression": (
        _parameter("alpha", "Alpha", "L1 regularization strength.", "number", 1.0, minimum=0.000001, maximum=1000, step=0.01),
        _parameter("max_iter", "Maximum iterations", "Optimizer iteration limit.", "integer", 10000, minimum=100, maximum=50000, step=100),
    ),
    "elastic_net": (
        _parameter("alpha", "Alpha", "Overall regularization strength.", "number", 1.0, minimum=0.000001, maximum=1000, step=0.01),
        _parameter("l1_ratio", "L1 ratio", "Mix of L1 (1) and L2 (0) penalties.", "number", 0.5, minimum=0, maximum=1, step=0.05),
    ),
    "bayesian_ridge_regression": (
        _parameter("tol", "Tolerance", "Stopping tolerance for convergence.", "number", 0.001, minimum=0.00000001, maximum=0.1, step=0.0001),
        _parameter("fit_intercept", "Fit intercept", "Estimate a constant intercept.", "boolean", True),
    ),
    "ridge_regression": (
        _parameter("alpha", "Alpha", "L2 regularization strength.", "number", 1.0, minimum=0, maximum=1000, step=0.01),
        _parameter("fit_intercept", "Fit intercept", "Estimate a constant intercept.", "boolean", True),
    ),
    "decision_tree": (
        _parameter("min_samples_split", "Minimum split samples", "Minimum rows required to split a node.", "integer", 2, minimum=2, maximum=100, step=1),
        _parameter("min_samples_leaf", "Minimum leaf samples", "Minimum rows retained in each leaf.", "integer", 1, minimum=1, maximum=100, step=1),
    ),
    "extra_trees": (
        _parameter("n_estimators", "Number of trees", "Number of randomized trees.", "integer", 200, minimum=10, maximum=1000, step=10),
        _parameter("min_samples_leaf", "Minimum leaf samples", "Minimum rows retained in each leaf.", "integer", 1, minimum=1, maximum=100, step=1),
        _parameter("max_features", "Features per split", "Feature subset evaluated at each split.", "select", 1.0, options=(1.0, "sqrt", "log2")),
    ),
    "gradient_boosting": (
        _parameter("n_estimators", "Boosting stages", "Number of sequential boosting stages.", "integer", 100, minimum=10, maximum=1000, step=10),
        _parameter("learning_rate", "Learning rate", "Contribution of each boosting stage.", "number", 0.1, minimum=0.001, maximum=1, step=0.01),
        _parameter("max_depth", "Tree depth", "Maximum depth of each weak learner.", "integer", 3, minimum=1, maximum=20, step=1),
    ),
    "k_nearest_neighbors": (
        _parameter("n_neighbors", "Neighbors", "Number of nearby samples used for prediction.", "integer", 5, estimator_parameter="kneighborsregressor__n_neighbors", minimum=1, maximum=100, step=1),
        _parameter("weights", "Neighbor weighting", "Weight all neighbors equally or by distance.", "select", "uniform", estimator_parameter="kneighborsregressor__weights", options=("uniform", "distance")),
        _parameter("p", "Distance power", "1 uses Manhattan distance; 2 uses Euclidean distance.", "select", 2, estimator_parameter="kneighborsregressor__p", options=(1, 2)),
    ),
    "multi_layer_perceptron": (
        _parameter("alpha", "Alpha", "L2 regularization strength.", "number", 0.0001, estimator_parameter="mlpregressor__alpha", minimum=0.00000001, maximum=10, step=0.0001),
        _parameter("learning_rate_init", "Initial learning rate", "Initial optimizer step size.", "number", 0.001, estimator_parameter="mlpregressor__learning_rate_init", minimum=0.000001, maximum=1, step=0.0001),
        _parameter("max_iter", "Maximum iterations", "Optimizer iteration limit.", "integer", 2000, estimator_parameter="mlpregressor__max_iter", minimum=100, maximum=10000, step=100),
    ),
    "random_forest": (
        _parameter("n_estimators", "Number of trees", "Number of trees in the forest.", "integer", 200, minimum=10, maximum=1000, step=10),
        _parameter("min_samples_leaf", "Minimum leaf samples", "Minimum rows retained in each leaf.", "integer", 1, minimum=1, maximum=100, step=1),
        _parameter("max_features", "Features per split", "Feature subset evaluated at each split.", "select", 1.0, options=(1.0, "sqrt", "log2")),
    ),
    "stochastic_gradient_descent": (
        _parameter("alpha", "Alpha", "Regularization strength.", "number", 0.0001, estimator_parameter="sgdregressor__alpha", minimum=0.00000001, maximum=10, step=0.0001),
        _parameter("penalty", "Penalty", "Regularization penalty.", "select", "l2", estimator_parameter="sgdregressor__penalty", options=("l2", "l1", "elasticnet")),
        _parameter("max_iter", "Maximum iterations", "Optimizer iteration limit.", "integer", 2000, estimator_parameter="sgdregressor__max_iter", minimum=100, maximum=10000, step=100),
    ),
    "support_vector_machine": (
        _parameter("C", "C", "Penalty applied to prediction errors.", "number", 1.0, estimator_parameter="svr__C", minimum=0.0001, maximum=10000, step=0.1),
        _parameter("epsilon", "Epsilon", "Width of the insensitive loss tube.", "number", 0.1, estimator_parameter="svr__epsilon", minimum=0, maximum=10, step=0.01),
        _parameter("kernel", "Kernel", "Kernel used to model nonlinear relationships.", "select", "rbf", estimator_parameter="svr__kernel", options=("rbf", "linear", "poly", "sigmoid")),
    ),
    "xgboost": (
        _parameter("n_estimators", "Boosting rounds", "Number of boosted trees.", "integer", 100, minimum=10, maximum=1000, step=10),
        _parameter("learning_rate", "Learning rate", "Contribution of each boosted tree.", "number", 0.1, minimum=0.001, maximum=1, step=0.01),
        _parameter("max_depth", "Tree depth", "Maximum depth of each boosted tree.", "integer", 4, minimum=1, maximum=20, step=1),
        _parameter("subsample", "Row subsample", "Fraction of rows used for each boosted tree.", "number", 0.8, minimum=0.1, maximum=1, step=0.05),
    ),
}


CLASSIFICATION_HYPERPARAMETERS: dict[str, tuple[HyperparameterDefinition, ...]] = {
    "logistic_regression": (
        _parameter("C", "C", "Inverse regularization strength.", "number", 1.0, estimator_parameter="logisticregression__C", minimum=0.0001, maximum=10000, step=0.1),
        _parameter("max_iter", "Maximum iterations", "Optimizer iteration limit.", "integer", 2000, estimator_parameter="logisticregression__max_iter", minimum=100, maximum=10000, step=100),
    ),
    "support_vector_machine": (
        _parameter("C", "C", "Penalty applied to classification errors.", "number", 1.0, estimator_parameter="svc__C", minimum=0.0001, maximum=10000, step=0.1),
        _parameter("kernel", "Kernel", "Kernel used to form the class boundary.", "select", "rbf", estimator_parameter="svc__kernel", options=("rbf", "linear", "poly", "sigmoid")),
        _parameter("gamma", "Gamma", "Kernel coefficient strategy.", "select", "scale", estimator_parameter="svc__gamma", options=("scale", "auto")),
    ),
    "decision_tree": REGRESSION_HYPERPARAMETERS["decision_tree"],
    "random_forest": (
        _parameter("n_estimators", "Number of trees", "Number of trees in the forest.", "integer", 200, minimum=10, maximum=1000, step=10),
        _parameter("min_samples_leaf", "Minimum leaf samples", "Minimum rows retained in each leaf.", "integer", 1, minimum=1, maximum=100, step=1),
        _parameter("max_features", "Features per split", "Feature subset evaluated at each split.", "select", "sqrt", options=("sqrt", "log2", 1.0)),
    ),
    "extra_trees": (
        _parameter("n_estimators", "Number of trees", "Number of randomized trees.", "integer", 200, minimum=10, maximum=1000, step=10),
        _parameter("min_samples_leaf", "Minimum leaf samples", "Minimum rows retained in each leaf.", "integer", 1, minimum=1, maximum=100, step=1),
        _parameter("max_features", "Features per split", "Feature subset evaluated at each split.", "select", "sqrt", options=("sqrt", "log2", 1.0)),
    ),
    "multi_layer_perceptron": (
        _parameter("alpha", "Alpha", "L2 regularization strength.", "number", 0.0001, estimator_parameter="mlpclassifier__alpha", minimum=0.00000001, maximum=10, step=0.0001),
        _parameter("learning_rate_init", "Initial learning rate", "Initial optimizer step size.", "number", 0.001, estimator_parameter="mlpclassifier__learning_rate_init", minimum=0.000001, maximum=1, step=0.0001),
        _parameter("max_iter", "Maximum iterations", "Optimizer iteration limit.", "integer", 2000, estimator_parameter="mlpclassifier__max_iter", minimum=100, maximum=10000, step=100),
    ),
    "gradient_boosting": REGRESSION_HYPERPARAMETERS["gradient_boosting"],
    "k_nearest_neighbors": (
        _parameter("n_neighbors", "Neighbors", "Number of nearby samples used for prediction.", "integer", 5, estimator_parameter="kneighborsclassifier__n_neighbors", minimum=1, maximum=100, step=1),
        _parameter("weights", "Neighbor weighting", "Weight all neighbors equally or by distance.", "select", "uniform", estimator_parameter="kneighborsclassifier__weights", options=("uniform", "distance")),
        _parameter("p", "Distance power", "1 uses Manhattan distance; 2 uses Euclidean distance.", "select", 2, estimator_parameter="kneighborsclassifier__p", options=(1, 2)),
    ),
    "stochastic_gradient_descent": (
        _parameter("alpha", "Alpha", "Regularization strength.", "number", 0.0001, estimator_parameter="sgdclassifier__alpha", minimum=0.00000001, maximum=10, step=0.0001),
        _parameter("penalty", "Penalty", "Regularization penalty.", "select", "l2", estimator_parameter="sgdclassifier__penalty", options=("l2", "l1", "elasticnet")),
        _parameter("max_iter", "Maximum iterations", "Optimizer iteration limit.", "integer", 2000, estimator_parameter="sgdclassifier__max_iter", minimum=100, maximum=10000, step=100),
    ),
    "adaboost": (
        _parameter("n_estimators", "Boosting stages", "Maximum number of estimators.", "integer", 100, minimum=10, maximum=1000, step=10),
        _parameter("learning_rate", "Learning rate", "Contribution of each estimator.", "number", 1.0, minimum=0.001, maximum=10, step=0.05),
    ),
    "xgboost": REGRESSION_HYPERPARAMETERS["xgboost"],
}


def get_hyperparameters(task_type: str, model_name: str) -> tuple[HyperparameterDefinition, ...]:
    registry = (
        REGRESSION_HYPERPARAMETERS
        if task_type == "regression"
        else CLASSIFICATION_HYPERPARAMETERS
    )
    return registry.get(model_name, ())


def configure_model(
    task_type: str,
    definition: RegressionModelDefinition | ClassificationModelDefinition,
    supplied: dict[str, Any] | None = None,
) -> RegressorMixin | ClassifierMixin | Pipeline:
    """Build an estimator after validating the public hyperparameter whitelist."""
    supplied = supplied or {}
    if not isinstance(supplied, dict):
        raise ValueError("Hyperparameters must be a JSON object")
    parameter_definitions = {
        item.name: item for item in get_hyperparameters(task_type, definition.name)
    }
    unknown = sorted(set(supplied) - set(parameter_definitions))
    if unknown:
        raise ValueError(
            f"Unsupported hyperparameter(s) for {definition.display_name}: "
            + ", ".join(unknown)
        )

    estimator = definition.factory()
    estimator_parameters: dict[str, Any] = {}
    for name, raw_value in supplied.items():
        item = parameter_definitions[name]
        if item.value_type == "boolean":
            if not isinstance(raw_value, bool):
                raise ValueError(f"Hyperparameter '{name}' must be true or false")
            value = raw_value
        elif item.value_type == "integer":
            if isinstance(raw_value, bool) or not isinstance(raw_value, int):
                raise ValueError(f"Hyperparameter '{name}' must be an integer")
            value = raw_value
        elif item.value_type == "number":
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError(f"Hyperparameter '{name}' must be a number")
            value = float(raw_value)
        elif item.value_type == "select":
            if raw_value not in item.options:
                raise ValueError(
                    f"Hyperparameter '{name}' must be one of: "
                    + ", ".join(str(option) for option in item.options)
                )
            value = raw_value
        else:
            raise ValueError(f"Unsupported hyperparameter type: {item.value_type}")

        if item.minimum is not None and value < item.minimum:
            raise ValueError(f"Hyperparameter '{name}' must be at least {item.minimum}")
        if item.maximum is not None and value > item.maximum:
            raise ValueError(f"Hyperparameter '{name}' must be at most {item.maximum}")
        estimator_parameters[item.estimator_parameter or item.name] = value

    if estimator_parameters:
        estimator.set_params(**estimator_parameters)
    return estimator


CLUSTERING_MODELS: dict[str, ClusteringModelDefinition] = {
    definition.name: definition
    for definition in (
        ClusteringModelDefinition(
            name="kmeans",
            display_name="K-Means",
            description="Centroid-based clustering with a user-selected number of clusters.",
            uses_cluster_count=True,
            factory=lambda cluster_count: KMeans(
                n_clusters=cluster_count,
                random_state=42,
                n_init=10,
            ),
        ),
        ClusteringModelDefinition(
            name="dbscan",
            display_name="DBSCAN",
            description="Density-based clustering with automatic noise detection.",
            uses_cluster_count=False,
            factory=lambda _cluster_count: DBSCAN(eps=0.3, min_samples=3),
        ),
        ClusteringModelDefinition(
            name="agglomerative",
            display_name="Agglomerative Clustering",
            description="Hierarchical clustering with a user-selected number of clusters.",
            uses_cluster_count=True,
            factory=lambda cluster_count: AgglomerativeClustering(
                n_clusters=cluster_count
            ),
        ),
        ClusteringModelDefinition(
            name="affinity_propagation",
            display_name="Affinity Propagation",
            description="Exemplar-based clustering that estimates cluster count automatically.",
            uses_cluster_count=False,
            factory=lambda _cluster_count: AffinityPropagation(
                damping=0.85,
                random_state=42,
            ),
        ),
        ClusteringModelDefinition(
            name="mean_shift",
            display_name="Mean Shift",
            description="Mode-seeking clustering that estimates cluster count automatically.",
            uses_cluster_count=False,
            factory=lambda _cluster_count: MeanShift(),
        ),
        ClusteringModelDefinition(
            name="optics",
            display_name="OPTICS",
            description="Density-ordering clustering with automatic noise detection.",
            uses_cluster_count=False,
            factory=lambda _cluster_count: OPTICS(
                min_samples=3,
                xi=0.05,
                min_cluster_size=0.1,
            ),
        ),
    )
}


DIMENSIONALITY_REDUCTION_MODELS: dict[
    str, DimensionalityReductionModelDefinition
] = {
    definition.name: definition
    for definition in (
        DimensionalityReductionModelDefinition(
            name="pca",
            display_name="PCA",
            description=(
                "Linear principal component analysis with explained-variance "
                "diagnostics."
            ),
            max_rows=None,
            factory=lambda component_count, _sample_count: PCA(
                n_components=component_count,
                random_state=42,
            ),
        ),
        DimensionalityReductionModelDefinition(
            name="tsne",
            display_name="T-SNE",
            description=(
                "Nonlinear neighborhood embedding with an automatically bounded "
                "perplexity."
            ),
            max_rows=5_000,
            factory=lambda component_count, sample_count: TSNE(
                n_components=component_count,
                perplexity=min(30.0, max(2.0, (sample_count - 1) / 3.0)),
                learning_rate="auto",
                init="pca",
                random_state=42,
                method="barnes_hut",
            ),
        ),
        DimensionalityReductionModelDefinition(
            name="mds",
            display_name="MDS",
            description=(
                "Metric multidimensional scaling that preserves pairwise distances."
            ),
            max_rows=2_000,
            factory=lambda component_count, _sample_count: MDS(
                n_components=component_count,
                metric=True,
                n_init=4,
                max_iter=300,
                eps=1e-3,
                n_jobs=None,
                random_state=42,
                dissimilarity="euclidean",
            ),
        ),
    )
}


ANOMALY_DETECTION_MODELS: dict[str, AnomalyDetectionModelDefinition] = {
    definition.name: definition
    for definition in (
        AnomalyDetectionModelDefinition(
            name="isolation_forest",
            display_name="Isolation Forest",
            description=(
                "Tree-based global anomaly detection with an automatic or "
                "user-specified contamination threshold."
            ),
            factory=lambda _sample_count, contamination: IsolationForest(
                n_estimators=100,
                max_samples="auto",
                contamination=contamination,
                n_jobs=-1,
                random_state=42,
            ),
        ),
        AnomalyDetectionModelDefinition(
            name="local_outlier_factor",
            display_name="Local Outlier Factor",
            description=(
                "Local-density anomaly detection using up to 20 nearest neighbors."
            ),
            factory=lambda sample_count, contamination: LocalOutlierFactor(
                n_neighbors=min(20, sample_count - 1),
                contamination=contamination,
                novelty=False,
                n_jobs=-1,
            ),
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


def get_classification_model(name: str) -> ClassificationModelDefinition:
    try:
        return CLASSIFICATION_MODELS[name]
    except KeyError as exc:
        choices = ", ".join(CLASSIFICATION_MODELS)
        raise ValueError(
            f"Unknown classification model '{name}'. Choose one of: {choices}"
        ) from exc


def get_clustering_model(name: str) -> ClusteringModelDefinition:
    try:
        return CLUSTERING_MODELS[name]
    except KeyError as exc:
        choices = ", ".join(CLUSTERING_MODELS)
        raise ValueError(
            f"Unknown clustering model '{name}'. Choose one of: {choices}"
        ) from exc


def get_dimensionality_reduction_model(
    name: str,
) -> DimensionalityReductionModelDefinition:
    try:
        return DIMENSIONALITY_REDUCTION_MODELS[name]
    except KeyError as exc:
        choices = ", ".join(DIMENSIONALITY_REDUCTION_MODELS)
        raise ValueError(
            f"Unknown dimensionality reduction model '{name}'. "
            f"Choose one of: {choices}"
        ) from exc


def get_anomaly_detection_model(name: str) -> AnomalyDetectionModelDefinition:
    try:
        return ANOMALY_DETECTION_MODELS[name]
    except KeyError as exc:
        choices = ", ".join(ANOMALY_DETECTION_MODELS)
        raise ValueError(
            f"Unknown anomaly detection model '{name}'. Choose one of: {choices}"
        ) from exc


def extract_linear_parameters(
    fitted_model: RegressorMixin | Pipeline,
    feature_names: list[str],
) -> tuple[float, list[str], list[float]] | None:
    """Return fitted linear parameters, or ``None`` for nonlinear estimators."""
    estimator = fitted_model
    coefficient_names = feature_names
    if isinstance(fitted_model, Pipeline):
        estimator = fitted_model.named_steps.get("model", fitted_model)
    if isinstance(estimator, Pipeline) and "polynomialfeatures" in estimator.named_steps:
        polynomial = estimator.named_steps["polynomialfeatures"]
        estimator = estimator.named_steps["linearregression"]
        coefficient_names = [
            str(name)
            for name in polynomial.get_feature_names_out(feature_names)
        ]

    if not hasattr(estimator, "intercept_") or not hasattr(estimator, "coef_"):
        return None
    intercept = float(estimator.intercept_)
    coefficients = [float(value) for value in np.ravel(estimator.coef_)]
    return intercept, coefficient_names, coefficients


__all__ = [
    "ANOMALY_DETECTION_MODELS",
    "AnomalyDetectionModelDefinition",
    "CLASSIFICATION_MODELS",
    "ClassificationModelDefinition",
    "CLASSIFICATION_HYPERPARAMETERS",
    "CLUSTERING_MODELS",
    "ClusteringModelDefinition",
    "DIMENSIONALITY_REDUCTION_MODELS",
    "DimensionalityReductionModelDefinition",
    "REGRESSION_MODELS",
    "REGRESSION_HYPERPARAMETERS",
    "RegressionModelDefinition",
    "HyperparameterDefinition",
    "configure_model",
    "extract_linear_parameters",
    "get_anomaly_detection_model",
    "get_classification_model",
    "get_clustering_model",
    "get_dimensionality_reduction_model",
    "get_hyperparameters",
    "get_regression_model",
]
