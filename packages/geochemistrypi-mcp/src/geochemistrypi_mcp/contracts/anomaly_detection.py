"""Versioned anomaly-detection choices exposed by GeochemistryPi 0.8.1."""

from typing import Final

MODEL_ORDER: Final[tuple[str, ...]] = (
    "isolation_forest",
    "local_outlier_factor",
)

MODEL_DISPLAY_NAMES: Final[dict[str, str]] = {
    "isolation_forest": "Isolation Forest",
    "local_outlier_factor": "Local Outlier Factor",
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
    "anomaly_detection.application_data: the public CLI does not perform inference for anomaly-detection tasks",
    "anomaly_detection.target_column: anomaly detection operates on features without a supervised target",
    "anomaly_detection.automl: the public CLI does not expose AutoML for anomaly detection",
    "anomaly_detection.keep_missing_values: GeochemistryPi 0.8.1 exposes no anomaly-detection model when missing values remain unprocessed",
    "anomaly_detection.feature_selection: the public anomaly-detection branch does not offer supervised feature selection",
    "anomaly_detection.previous_experiment: MCP runs use explicit new experiment and run names",
)
