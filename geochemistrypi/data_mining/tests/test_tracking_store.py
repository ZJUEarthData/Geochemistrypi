from pathlib import Path

import mlflow
import pytest

from geochemistrypi.tracking import (
    TrackingStoreError,
    get_experiment,
    list_experiments,
    require_experiment,
)


def test_persistent_tracking_store_lists_and_reads_bounded_runs(tmp_path: Path) -> None:
    tracking_root = tmp_path / "tracking"
    mlflow.set_tracking_uri(tracking_root.as_uri())
    experiment_id = mlflow.create_experiment("Persistent Experiment")
    with mlflow.start_run(experiment_id=experiment_id, run_name="First Run"):
        mlflow.log_metric("accuracy", 0.75)
        mlflow.log_param("model", "demo")

    listing = list_experiments(tracking_root, maximum_experiments=10)
    assert experiment_id in [item["experiment_id"] for item in listing["experiments"]]

    detail = get_experiment(tracking_root, experiment_id, maximum_runs=1)
    assert detail["experiment"]["name"] == "Persistent Experiment"
    assert detail["runs"][0]["run_name"] == "First Run"
    assert detail["runs"][0]["metrics"]["accuracy"] == 0.75
    assert detail["runs"][0]["params"]["model"] == "demo"


def test_tracking_store_requires_stable_id_and_matching_output_name(tmp_path: Path) -> None:
    tracking_root = tmp_path / "tracking"
    mlflow.set_tracking_uri(tracking_root.as_uri())
    experiment_id = mlflow.create_experiment("Exact Name")

    assert require_experiment(tracking_root, experiment_id, "Exact Name")["experiment_id"] == experiment_id
    with pytest.raises(TrackingStoreError, match="not requested experiment_name"):
        require_experiment(tracking_root, experiment_id, "Ambiguous Name")
    with pytest.raises(TrackingStoreError, match="experiment_id"):
        get_experiment(tracking_root, "../unsafe")
