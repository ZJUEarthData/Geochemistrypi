"""Modern, non-interactive model registry for the v0.8 Online API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sklearn.base import ClassifierMixin, ClusterMixin, RegressorMixin
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
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
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
)
from sklearn.manifold import MDS, TSNE
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


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
    factory: Callable[[int], Any]


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
    )
}


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
                "Tree-based global anomaly detection using the v0.8 default "
                "automatic contamination threshold."
            ),
            factory=lambda _sample_count: IsolationForest(
                n_estimators=100,
                max_samples="auto",
                contamination="auto",
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
            factory=lambda sample_count: LocalOutlierFactor(
                n_neighbors=min(20, sample_count - 1),
                contamination="auto",
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
    "ANOMALY_DETECTION_MODELS",
    "AnomalyDetectionModelDefinition",
    "CLASSIFICATION_MODELS",
    "ClassificationModelDefinition",
    "CLUSTERING_MODELS",
    "ClusteringModelDefinition",
    "DIMENSIONALITY_REDUCTION_MODELS",
    "DimensionalityReductionModelDefinition",
    "REGRESSION_MODELS",
    "RegressionModelDefinition",
    "extract_linear_parameters",
    "get_anomaly_detection_model",
    "get_classification_model",
    "get_clustering_model",
    "get_dimensionality_reduction_model",
    "get_regression_model",
]
