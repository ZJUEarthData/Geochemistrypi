"""Build the versioned Random Forest artifact used by the Online hybrid model."""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = (
    REPOSITORY_ROOT
    / "geochemistrypi"
    / "chemical_modeling"
    / "data"
    / "solubility"
    / "Dataset #4.xlsx"
)
MODEL_VERSION = "zhangzhou2024-hybrid-rf-v1"
MODEL_PATH = (
    REPOSITORY_ROOT
    / "geochemistrypi"
    / "chemical_modeling"
    / "model"
    / "func"
    / "algo_solubility"
    / "models"
    / f"{MODEL_VERSION}.joblib"
)
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


def build() -> Path:
    dataframe = pd.read_excel(DATASET_PATH).rename(
        columns={"SCSS Measurement": "SCSS"}
    )
    dataframe = dataframe.fillna(0)
    missing = [
        column
        for column in (*FEATURE_COLUMNS, "SCSS")
        if column not in dataframe.columns
    ]
    if missing:
        raise ValueError(f"Training dataset is missing columns: {missing}")

    train_indices, _ = train_test_split(
        dataframe.index,
        test_size=0.30,
        random_state=100,
    )
    training_features = dataframe.loc[train_indices, list(FEATURE_COLUMNS)]
    training_target = dataframe.loc[train_indices, "SCSS"]
    scaler = StandardScaler().fit(training_features)

    # These were the best RF hyperparameters among the three candidates sampled
    # by the authors' published RandomizedSearchCV configuration. The original
    # notebook left the estimator RNG unset; random_state=20 makes the Online
    # artifact reproducible.
    model = RandomForestRegressor(
        bootstrap=True,
        max_depth=30,
        max_features=1.0,
        min_samples_leaf=1,
        min_samples_split=9,
        n_estimators=200,
        random_state=20,
        n_jobs=-1,
    )
    model.fit(scaler.transform(training_features), training_target)

    artifact = {
        "model_version": MODEL_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "model": model,
        "scaler": scaler,
        "training_rows": int(len(training_features)),
        "training_dataset_rows": int(len(dataframe)),
        "training_dataset_sha256": hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest(),
        "split_test_size": 0.30,
        "split_random_state": 100,
        "model_random_state": 20,
        "sklearn_version": sklearn.__version__,
        "source_doi": "10.1016/j.gca.2023.11.029",
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH, compress=3)
    return MODEL_PATH


if __name__ == "__main__":
    print(build())
