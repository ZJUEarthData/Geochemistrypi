from pathlib import Path

import mlflow
import pytest

from geochemistrypi.data_mining.utils.mlflow_utils import set_active_experiment
from geochemistrypi.tracking import TrackingStoreError, get_experiment, list_experiments, require_experiment


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


def test_active_experiment_is_inherited_by_nested_automl_runs_in_a_new_store(tmp_path: Path) -> None:
    tracking_root = tmp_path / "tracking"
    tracking_root.mkdir()
    previous_tracking_uri = mlflow.get_tracking_uri()
    from mlflow.tracking import fluent

    previous_active_experiment_id = fluent._active_experiment_id
    try:
        mlflow.set_tracking_uri(tracking_root.as_uri())
        experiment_id = mlflow.create_experiment("AutoML Parent Experiment")
        experiment = set_active_experiment(experiment_id)

        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="Parent"):
            with mlflow.start_run(nested=True, run_name="FLAML Trial") as nested_run:
                assert nested_run.info.experiment_id == experiment_id
    finally:
        while mlflow.active_run() is not None:
            mlflow.end_run()
        fluent._active_experiment_id = previous_active_experiment_id
        mlflow.set_tracking_uri(previous_tracking_uri)
