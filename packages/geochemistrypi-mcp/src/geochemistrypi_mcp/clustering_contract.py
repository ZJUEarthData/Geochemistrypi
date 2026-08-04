"""Versioned clustering choices exposed by the GeochemistryPi 0.8.0 CLI."""

from typing import Final

MODEL_ORDER: Final[tuple[str, ...]] = (
    "kmeans",
    "dbscan",
    "agglomerative",
    "affinity_propagation",
    "mean_shift",
)

MODEL_DISPLAY_NAMES: Final[dict[str, str]] = {
    "kmeans": "KMeans",
    "dbscan": "DBSCAN",
    "agglomerative": "Agglomerative",
    "affinity_propagation": "AffinityPropagation",
    "mean_shift": "MeanShift",
}

MODEL_NUMBERS: Final[dict[str, int]] = {
    name: index for index, name in enumerate(MODEL_ORDER, start=1)
}

SCALING_METHODS: Final[tuple[str, ...]] = (
    "none",
    "min_max",
    "standardization",
    "mean_normalization",
)

MISSING_VALUE_METHODS: Final[tuple[str, ...]] = (
    "error",
    "drop_rows",
    "impute",
)

UNSUPPORTED_INTERACTIONS: Final[tuple[str, ...]] = (
    "clustering.application_data: the public CLI does not perform model inference for unsupervised tasks",
    "clustering.target_column: clustering operates on features without a supervised target",
    "clustering.automl: the public CLI does not expose AutoML for clustering",
    "clustering.keep_missing_values: GeochemistryPi 0.8.0 exposes no clustering model when missing values remain unprocessed",
    "clustering.optics: an internal implementation exists but OPTICS is not present in the public 0.8.0 CLI model menu",
    "clustering.previous_experiment: MCP runs use explicit new experiment and run names so results cannot attach to an ambiguous prior run",
)
