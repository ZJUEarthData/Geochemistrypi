"""Canonical MCP projection of the public CLI scientific-config registry."""

from typing import Final

from .anomaly_detection import MODEL_ORDER as ANOMALY_DETECTION_MODEL_ORDER
from .classification import MODEL_ORDER as CLASSIFICATION_MODEL_ORDER
from .clustering import MODEL_ORDER as CLUSTERING_MODEL_ORDER
from .decomposition import MODEL_ORDER as DECOMPOSITION_MODEL_ORDER
from .regression import MODEL_ORDER as REGRESSION_MODEL_ORDER

SCIENTIFIC_CONFIG_CONTRACT_VERSION: Final[int] = 4

PUBLIC_MANUAL_METHODS_BY_TASK: Final[dict[str, tuple[str, ...]]] = {
    "classification": CLASSIFICATION_MODEL_ORDER,
    "regression": REGRESSION_MODEL_ORDER,
    "clustering": CLUSTERING_MODEL_ORDER,
    "decomposition": DECOMPOSITION_MODEL_ORDER,
    "anomaly_detection": ANOMALY_DETECTION_MODEL_ORDER,
}

# These are machine identities, not presentation labels.  The order follows
# each public CLI menu while the membership mirrors
# ``geochemistrypi.scientific_execution._WORKFLOW_METHODS`` exactly.
SCIENTIFIC_EXECUTION_METHODS_BY_TASK: Final[dict[str, tuple[str, ...]]] = {
    "classification": CLASSIFICATION_MODEL_ORDER,
    "regression": (
        "decision_tree",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "xgboost",
        "multi_layer_perceptron",
        "lasso_regression",
        "elastic_net",
        "stochastic_gradient_descent",
    ),
    "clustering": (
        "kmeans",
        "affinity_propagation",
    ),
    "decomposition": (
        "pca",
        "tsne",
        "mds",
    ),
    "anomaly_detection": (
        "isolation_forest",
        "local_outlier_factor",
    ),
}

SCIENTIFIC_EXECUTION_METHOD_COUNT: Final[int] = sum(len(methods) for methods in SCIENTIFIC_EXECUTION_METHODS_BY_TASK.values())

LEGACY_METHODS_WITHOUT_V4_ATTESTATION_BY_TASK: Final[dict[str, tuple[str, ...]]] = {
    task: tuple(method for method in methods if method not in SCIENTIFIC_EXECUTION_METHODS_BY_TASK[task]) for task, methods in PUBLIC_MANUAL_METHODS_BY_TASK.items()
}

PUBLIC_MANUAL_METHOD_COUNT: Final[int] = sum(len(methods) for methods in PUBLIC_MANUAL_METHODS_BY_TASK.values())
LEGACY_METHOD_WITHOUT_V4_ATTESTATION_COUNT: Final[int] = sum(len(methods) for methods in LEGACY_METHODS_WITHOUT_V4_ATTESTATION_BY_TASK.values())

_PUBLIC_METHOD_IDENTITIES = frozenset((task, method) for task, methods in PUBLIC_MANUAL_METHODS_BY_TASK.items() for method in methods)
_V4_METHOD_IDENTITIES = frozenset((task, method) for task, methods in SCIENTIFIC_EXECUTION_METHODS_BY_TASK.items() for method in methods)
_LEGACY_METHOD_IDENTITIES = frozenset((task, method) for task, methods in LEGACY_METHODS_WITHOUT_V4_ATTESTATION_BY_TASK.items() for method in methods)
if PUBLIC_MANUAL_METHOD_COUNT != 36 or SCIENTIFIC_EXECUTION_METHOD_COUNT != 27 or LEGACY_METHOD_WITHOUT_V4_ATTESTATION_COUNT != 9:
    raise RuntimeError("The public manual/v4/legacy scientific boundary must remain exactly 36/27/9.")
if _V4_METHOD_IDENTITIES & _LEGACY_METHOD_IDENTITIES:
    raise RuntimeError("v4-attested and legacy scientific method identities must be disjoint.")
if _PUBLIC_METHOD_IDENTITIES != _V4_METHOD_IDENTITIES | _LEGACY_METHOD_IDENTITIES:
    raise RuntimeError("v4-attested and legacy identities must partition the public manual method registry.")
