"""Versioned inference support for the ZhangZhou et al. (2024) hybrid SCSS model."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = (
    "Pressure",
    "T",
    "SiO2",
    "TiO2",
    "Al2O3",
    "FeO",
    "MgO",
    "CaO",
    "NiO",
    "Na2O",
    "K2O",
    "H2O",
    "Fe",
    "Ni+Cu+Co",
    "S",
    "O",
)
MODEL_VERSION = "zhangzhou2024-hybrid-rf-v1"
TEMPERATURE_COEFFICIENT = 551.22
PRESSURE_COEFFICIENT = -121.83
MODEL_PATH = Path(__file__).with_name("models") / f"{MODEL_VERSION}.joblib"


@lru_cache(maxsize=1)
def _load_model_artifact() -> dict:
    """Load the trusted model artifact bundled with Geochemistryπ."""

    import joblib

    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Hybrid model artifact is missing: {MODEL_PATH}. "
            "Run scripts/build_hybrid_model.py to rebuild it."
        )
    artifact = joblib.load(MODEL_PATH)
    if artifact.get("model_version") != MODEL_VERSION:
        raise RuntimeError("Hybrid model artifact version does not match the runtime")
    if tuple(artifact.get("feature_columns", ())) != FEATURE_COLUMNS:
        raise RuntimeError("Hybrid model artifact feature columns do not match the runtime")
    return artifact


def predict_hybrid_scss(
    features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return RF baseline, P-T correction factor, and hybrid SCSS in ppm."""

    artifact = _load_model_artifact()
    values = features.loc[:, list(FEATURE_COLUMNS)].astype(float)
    scaled_values = artifact["scaler"].transform(values)
    rf_prediction = np.asarray(
        artifact["model"].predict(scaled_values),
        dtype=float,
    )
    pressure = features["Pressure"].to_numpy(dtype=float)
    temperature = features["T"].to_numpy(dtype=float)
    correction = np.exp(
        TEMPERATURE_COEFFICIENT / temperature
        + PRESSURE_COEFFICIENT * pressure / temperature
    )
    prediction = rf_prediction * correction
    if (
        not np.isfinite(rf_prediction).all()
        or not np.isfinite(correction).all()
        or not np.isfinite(prediction).all()
        or (rf_prediction <= 0).any()
        or (correction <= 0).any()
        or (prediction <= 0).any()
    ):
        raise ValueError("Hybrid parameters produce a non-finite SCSS result")
    return rf_prediction, correction, prediction
