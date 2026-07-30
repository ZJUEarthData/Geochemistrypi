import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geochemistrypi.data_mining.constants import FEATURE_SCALING_STRATEGY
from geochemistrypi.data_mining.data.data_readiness import data_split
from geochemistrypi.data_mining.data.preprocessing import feature_scaler

HERE = Path(__file__).parent


@pytest.mark.characterization
def test_legacy_scaler_uses_the_complete_dataset_before_splitting():
    data = pd.read_csv(HERE / "data" / "legacy_full_data_scaling.csv")
    golden = json.loads((HERE / "golden" / "legacy_full_data_scaling_v1.json").read_text(encoding="utf-8"))
    expected = golden["expected"]
    tolerances = golden["tolerances"]

    X = data[["feature"]]
    y = data[["target"]]
    names = data[["sample_id"]]
    _, transformed = feature_scaler(X, FEATURE_SCALING_STRATEGY, 1)
    transformed_frame = pd.DataFrame(transformed, index=X.index, columns=X.columns)
    split = data_split(transformed_frame, y, names, test_size=0.25, stratify=y["target"])

    np.testing.assert_allclose(
        X["feature"].mean(),
        expected["full_data_mean"],
        rtol=tolerances["relative"],
        atol=tolerances["absolute"],
    )
    np.testing.assert_allclose(
        X["feature"].std(ddof=0),
        expected["full_data_standard_deviation"],
        rtol=tolerances["relative"],
        atol=tolerances["absolute"],
    )
    assert split["Name Train"]["sample_id"].tolist() == expected["train_sample_ids"]
    assert split["Name Test"]["sample_id"].tolist() == expected["test_sample_ids"]
    np.testing.assert_allclose(
        split["X Train"]["feature"].mean(),
        expected["transformed_train_mean"],
        rtol=tolerances["relative"],
        atol=tolerances["absolute"],
    )
    np.testing.assert_allclose(
        split["X Test"]["feature"].mean(),
        expected["transformed_test_mean"],
        rtol=tolerances["relative"],
        atol=tolerances["absolute"],
    )
