import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from geochemistrypi.data_mining.data.data_readiness import data_split
from geochemistrypi.data_mining.data.inference import PipelineConstrutor
from geochemistrypi.data_mining.data.preprocessing import fit_supervised_preprocessor

HERE = Path(__file__).parent


def _load_reference():
    data = pd.read_csv(HERE / "data" / "supervised_preprocessing_reference.csv")
    golden = json.loads((HERE / "golden" / "supervised_preprocessing_v1.json").read_text(encoding="utf-8"))
    return data, golden


@pytest.mark.scientific
def test_supervised_preprocessing_is_fitted_on_training_rows_only():
    data, golden = _load_reference()
    expected = golden["expected"]
    tolerances = golden["tolerances"]
    X = data[["signal", "test_decoy", "imputation_probe"]]
    y = data[["target"]]
    names = data[["sample_id"]]

    split = data_split(X, y, names, test_size=0.2, stratify=y["target"])
    fitted = fit_supervised_preprocessor(
        split["X Train"],
        split["Y Train"],
        task="classification",
        imputation_method="Mean Value",
        scaling_method="Standardization",
        selection_method="Select K Best",
        features_to_retain=1,
    )
    transformed_test = fitted.transform(split["X Test"])

    assert split["Name Train"]["sample_id"].tolist() == expected["train_sample_ids"]
    assert split["Name Test"]["sample_id"].tolist() == expected["test_sample_ids"]
    assert fitted.feature_names == expected["selected_features"]
    assert transformed_test.columns.tolist() == expected["selected_features"]
    assert transformed_test.index.equals(split["X Test"].index)

    imputer = fitted.pipeline.named_steps["imputer"]
    scaler = fitted.pipeline.named_steps["scaler"]
    selector = fitted.pipeline.named_steps["selector"]
    np.testing.assert_allclose(
        imputer.statistics_,
        expected["imputer_statistics"],
        rtol=tolerances["relative"],
        atol=tolerances["absolute"],
    )
    np.testing.assert_allclose(
        scaler.mean_,
        expected["scaler_means"],
        rtol=tolerances["relative"],
        atol=tolerances["absolute"],
    )
    np.testing.assert_allclose(
        scaler.scale_,
        expected["scaler_scales"],
        rtol=tolerances["relative"],
        atol=tolerances["absolute"],
    )
    np.testing.assert_allclose(
        selector.scores_,
        expected["training_feature_scores"],
        rtol=tolerances["relative"],
        atol=tolerances["absolute"],
    )
    np.testing.assert_allclose(
        transformed_test["signal"],
        expected["transformed_test_signal"],
        rtol=tolerances["relative"],
        atol=tolerances["absolute"],
    )


@pytest.mark.scientific
def test_reference_dataset_would_detect_full_data_feature_selection_leakage():
    data, golden = _load_reference()
    X = data[["signal", "test_decoy", "imputation_probe"]]
    y = data["target"]
    leaky_pipeline = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(), SelectKBest(score_func=f_classif, k=1))

    leaky_pipeline.fit(X, y)

    selector = leaky_pipeline.named_steps["selectkbest"]
    selected_features = list(selector.get_feature_names_out(X.columns))
    assert selected_features == golden["expected"]["full_data_leaky_selected_features"]


@pytest.mark.scientific
def test_saved_transform_configuration_reproduces_training_fitted_preprocessing():
    data, _ = _load_reference()
    X = data[["signal", "test_decoy", "imputation_probe"]]
    y = data[["target"]]
    split = data_split(X, y, data[["sample_id"]], test_size=0.2, stratify=y["target"])
    fitted = fit_supervised_preprocessor(
        split["X Train"],
        split["Y Train"],
        task="classification",
        imputation_method="Mean Value",
        scaling_method="Standardization",
        selection_method="Select K Best",
        features_to_retain=1,
    )
    rebuilt = PipelineConstrutor().chain(fitted.transformer_config)
    rebuilt.fit(split["X Train"], split["Y Train"].iloc[:, 0])

    np.testing.assert_allclose(
        rebuilt.transform(split["X Test"]),
        fitted.transform(split["X Test"]).to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.scientific
def test_supervised_preprocessor_rejects_inference_schema_drift():
    data, _ = _load_reference()
    X = data[["signal", "test_decoy", "imputation_probe"]]
    y = data[["target"]]
    split = data_split(X, y, data[["sample_id"]], test_size=0.2, stratify=y["target"])
    fitted = fit_supervised_preprocessor(
        split["X Train"],
        split["Y Train"],
        task="classification",
        scaling_method="Standardization",
    )

    with pytest.raises(ValueError, match="same order"):
        fitted.transform(split["X Test"][["test_decoy", "signal", "imputation_probe"]])
