"""Versioned decomposition choices exposed by the GeochemistryPi 0.8.0 CLI."""

from typing import Final

MODEL_ORDER: Final[tuple[str, ...]] = ("pca", "tsne", "mds")

MODEL_DISPLAY_NAMES: Final[dict[str, str]] = {
    "pca": "PCA",
    "tsne": "T-SNE",
    "mds": "MDS",
}

MODEL_NUMBERS: Final[dict[str, int]] = {name: index for index, name in enumerate(MODEL_ORDER, start=1)}

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
    "decomposition.application_data: the public CLI does not perform inference for decomposition tasks",
    "decomposition.target_column: decomposition operates on features without a supervised target",
    "decomposition.automl: the public CLI does not expose AutoML for decomposition",
    "decomposition.keep_missing_values: GeochemistryPi 0.8.0 exposes no decomposition model when missing values remain unprocessed",
    "decomposition.feature_selection: the public decomposition branch does not offer supervised feature selection",
    "decomposition.previous_experiment: MCP runs use explicit new experiment and run names",
)
