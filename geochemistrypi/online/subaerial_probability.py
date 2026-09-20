"""Versioned surrogate model for subaerial-basalt probability prediction.

The bundled Liu et al. (2024) probabilities are used as the regression target.
This is intentionally described as a surrogate and is not presented as the
authors' original trained model.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit


MODEL_VERSION = "liu-2024-surrogate-hgbr-v1"
MODEL_DISPLAY_NAME = "Liu 2024 surrogate (HistGradientBoosting)"
TARGET_COLUMN = "Estimated Proportion of Subaerial Basalts"
MIN_FEATURES_PER_ROW = 12
RANDOM_STATE = 2025
FEATURE_COLUMNS = (
    "SIO2",
    "TIO2",
    "AL2O3",
    "FEOT",
    "MNO",
    "MGO",
    "CAO",
    "NA2O",
    "K2O",
    "P2O5",
    "SC",
    "V",
    "CR",
    "CO",
    "NI",
    "CU",
    "ZN",
    "GA",
    "RB",
    "SR",
    "Y",
    "ZR",
    "NB",
    "CS",
    "BA",
    "LA",
    "CE",
    "PR",
    "ND",
    "SM",
    "EU",
    "GD",
    "TB",
    "DY",
    "HO",
    "ER",
    "TM",
    "YB",
    "LU",
    "HF",
    "TA",
    "PB",
    "TH",
    "U",
)

TRAINING_DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data_mining"
    / "data"
    / "dataset"
    / "Data_Time_Series.xlsx"
)


@dataclass(frozen=True)
class SurrogateMetrics:
    validation_rows: int
    mean_absolute_error: float
    root_mean_squared_error: float
    r2: float


@dataclass(frozen=True)
class SurrogateBundle:
    model: HistGradientBoostingRegressor
    metrics: SurrogateMetrics
    training_rows: int
    training_sha256: str


@dataclass(frozen=True)
class ProbabilityPrediction:
    probabilities: pd.Series
    available_feature_count: pd.Series
    recognized_features: tuple[str, ...]
    bundle: SurrogateBundle


def _new_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=260,
        max_leaf_nodes=31,
        learning_rate=0.06,
        l2_regularization=0.2,
        random_state=RANDOM_STATE,
    )


def _numeric_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    numeric = pd.DataFrame(index=dataframe.index)
    for feature in FEATURE_COLUMNS:
        values = dataframe[feature] if feature in dataframe.columns else np.nan
        numeric[feature] = pd.to_numeric(values, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan)


@lru_cache(maxsize=1)
def load_subaerial_surrogate() -> SurrogateBundle:
    if not TRAINING_DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Surrogate training dataset is missing: {TRAINING_DATA_PATH}"
        )

    required = {*FEATURE_COLUMNS, TARGET_COLUMN, "SAMPLE_ID"}
    training = pd.read_excel(
        TRAINING_DATA_PATH,
        usecols=lambda column: column in required,
    )
    features = _numeric_features(training)
    target = pd.to_numeric(training[TARGET_COLUMN], errors="coerce")
    valid_target = target.notna() & np.isfinite(target)
    features = features.loc[valid_target]
    target = target.loc[valid_target]
    groups = training.loc[valid_target, "SAMPLE_ID"].astype(str)

    train_index, validation_index = next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=RANDOM_STATE,
        ).split(features, target, groups)
    )
    validation_model = _new_model()
    validation_model.fit(features.iloc[train_index], target.iloc[train_index])
    validation_prediction = np.clip(
        validation_model.predict(features.iloc[validation_index]),
        0.0,
        1.0,
    )
    validation_target = target.iloc[validation_index]
    metrics = SurrogateMetrics(
        validation_rows=int(validation_index.size),
        mean_absolute_error=float(
            mean_absolute_error(validation_target, validation_prediction)
        ),
        root_mean_squared_error=float(
            mean_squared_error(validation_target, validation_prediction) ** 0.5
        ),
        r2=float(r2_score(validation_target, validation_prediction)),
    )

    final_model = _new_model()
    final_model.fit(features, target)
    return SurrogateBundle(
        model=final_model,
        metrics=metrics,
        training_rows=int(features.shape[0]),
        training_sha256=sha256(TRAINING_DATA_PATH.read_bytes()).hexdigest(),
    )


def predict_subaerial_probability(
    dataframe: pd.DataFrame,
) -> ProbabilityPrediction:
    recognized = tuple(
        feature for feature in FEATURE_COLUMNS if feature in dataframe.columns
    )
    if len(recognized) < MIN_FEATURES_PER_ROW:
        raise ValueError(
            "Probability prediction requires at least "
            f"{MIN_FEATURES_PER_ROW} recognized geochemical columns; "
            f"found {len(recognized)}"
        )

    features = _numeric_features(dataframe)
    available = features.notna().sum(axis=1).astype(int)
    eligible = available >= MIN_FEATURES_PER_ROW
    probabilities = pd.Series(np.nan, index=dataframe.index, dtype=float)
    bundle = load_subaerial_surrogate()
    if eligible.any():
        probabilities.loc[eligible] = np.clip(
            bundle.model.predict(features.loc[eligible]),
            0.0,
            1.0,
        )
    return ProbabilityPrediction(
        probabilities=probabilities,
        available_feature_count=available,
        recognized_features=recognized,
        bundle=bundle,
    )


__all__ = [
    "FEATURE_COLUMNS",
    "MIN_FEATURES_PER_ROW",
    "MODEL_DISPLAY_NAME",
    "MODEL_VERSION",
    "ProbabilityPrediction",
    "RANDOM_STATE",
    "SurrogateBundle",
    "predict_subaerial_probability",
]
