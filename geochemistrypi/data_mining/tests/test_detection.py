from unittest.mock import patch

import pandas as pd

from geochemistrypi.data_mining.model.detection import IsolationForestAnomalyDetection
from geochemistrypi.data_mining.model.func.algo_anomalydetection._common import density_estimation
from geochemistrypi.data_mining.model.func.algo_anomalydetection._iforest import isolation_forest_manual_hyper_parameters
from geochemistrypi.data_mining.process.detect import AnomalyDetectionModelSelection


def test_isolation_forest_uses_auto_max_samples_when_bootstrap_is_disabled() -> None:
    with (
        patch(
            "geochemistrypi.data_mining.model.func.algo_anomalydetection._iforest.num_input",
            side_effect=[100, 3],
        ),
        patch(
            "geochemistrypi.data_mining.model.func.algo_anomalydetection._iforest.float_input",
            return_value=0.2,
        ),
        patch(
            "geochemistrypi.data_mining.model.func.algo_anomalydetection._iforest.bool_input",
            return_value=False,
        ),
    ):
        settings = isolation_forest_manual_hyper_parameters()

    assert settings["max_samples"] == "auto"


def test_anomaly_detection_keeps_prediction_labels_for_downstream_plots() -> None:
    features = pd.DataFrame(
        {
            "SiO2": [49.0, 50.0, 51.0, 70.0, 48.5, 50.5],
            "MgO": [8.0, 7.5, 7.0, 0.2, 8.5, 7.2],
        }
    )
    names = pd.Series([f"sample-{index}" for index in features.index], name="SampleID")
    settings = {
        "n_estimators": 20,
        "contamination": 0.2,
        "max_features": 2,
        "bootstrap": False,
        "max_samples": "auto",
    }

    with (
        patch.object(
            IsolationForestAnomalyDetection,
            "manual_hyper_parameters",
            return_value=settings,
        ),
        patch.object(IsolationForestAnomalyDetection, "show_info"),
        patch.object(IsolationForestAnomalyDetection, "save_hyper_parameters"),
        patch.object(IsolationForestAnomalyDetection, "common_components"),
        patch.object(IsolationForestAnomalyDetection, "special_components"),
        patch.object(IsolationForestAnomalyDetection, "data_save"),
        patch.object(IsolationForestAnomalyDetection, "model_save"),
    ):
        selection = AnomalyDetectionModelSelection("Isolation Forest")
        selection.activate(
            features,
            None,
            features,
            None,
            None,
            None,
            None,
            None,
            names,
        )

    labels = selection.ad_workflow.anomaly_detection_result
    assert isinstance(labels, pd.Series)
    assert labels.name == "is_abnormal"
    assert labels.index.equals(features.index)
    assert set(labels.unique()) <= {-1, 1}


def test_density_estimation_splits_sklearn_inlier_and_outlier_labels() -> None:
    features = pd.DataFrame(
        {
            "SiO2": [49.0, 50.0, 70.0, 48.5],
            "MgO": [8.0, 7.5, 0.2, 8.5],
        }
    )
    labels = pd.Series([1, 1, -1, 1], name="is_abnormal")

    with (
        patch("seaborn.kdeplot") as kdeplot,
        patch("matplotlib.pyplot.figure"),
        patch("matplotlib.pyplot.title"),
        patch("matplotlib.pyplot.xlabel"),
        patch("matplotlib.pyplot.ylabel"),
        patch("matplotlib.pyplot.legend"),
    ):
        density_estimation(features, labels, "Isolation Forest")

    pd.testing.assert_frame_equal(kdeplot.call_args_list[0].kwargs["data"], features.loc[[0, 1, 3]])
    pd.testing.assert_frame_equal(kdeplot.call_args_list[1].kwargs["data"], features.loc[[2]])
