import json
import subprocess
import sys
from pathlib import Path

import pytest
from geochemistrypi_mcp.experiments import ExperimentManager, ExperimentStoreError
from geochemistrypi_mcp.schemas import GetExperimentRequest, ListExperimentsRequest
from geochemistrypi_mcp.schemas import ClassificationRequest
from geochemistrypi_mcp.runs import RunManager
from geochemistrypi_mcp.settings import McpSettings


def _settings(tmp_path: Path) -> McpSettings:
    scripts = tmp_path / "cli" / "Scripts"
    scripts.mkdir(parents=True)
    (scripts / "geochemistrypi.exe").write_bytes(b"")
    (scripts / "python.exe").write_bytes(b"")
    return McpSettings(
        runs_root=tmp_path / "runs",
        cli_executable=scripts / "geochemistrypi.exe",
        tracking_root=tmp_path / "tracking",
    )


def test_experiment_manager_uses_bounded_core_json_bridge(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr(McpSettings, "require_supported_cli", lambda self: (self.cli_executable, "0.8.0"))
    commands = []

    def run(command, **kwargs):
        commands.append(tuple(command))
        if "list" in command:
            value = {
                "schema_version": 1,
                "tracking_root": str(settings.tracking_root),
                "experiment_count": 1,
                "experiments": [
                    {
                        "experiment_id": "7",
                        "name": "Persistent",
                        "lifecycle_stage": "active",
                        "artifact_location": "file:///artifacts",
                        "tags": {},
                    }
                ],
            }
        else:
            value = {
                "schema_version": 1,
                "tracking_root": str(settings.tracking_root),
                "experiment": {
                    "experiment_id": "7",
                    "name": "Persistent",
                    "lifecycle_stage": "active",
                    "artifact_location": "file:///artifacts",
                    "tags": {},
                },
                "run_count": 0,
                "runs": [],
            }
        return subprocess.CompletedProcess(command, 0, json.dumps(value), "")

    monkeypatch.setattr("geochemistrypi_mcp.experiments.subprocess.run", run)
    manager = ExperimentManager(settings)

    assert manager.list(ListExperimentsRequest(maximum_experiments=4)).experiment_count == 1
    assert manager.get(GetExperimentRequest(experiment_id="7", maximum_runs=0)).experiment.name == "Persistent"
    assert "--maximum-experiments" in commands[0]
    assert "--maximum-runs" in commands[1]
    assert all("--tracking-root" in command for command in commands)

    with pytest.raises(ExperimentStoreError, match="set experiment_name"):
        manager.require_matching_name("7", "Wrong Name")


def test_experiment_manager_sanitizes_cli_failures(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr(McpSettings, "require_supported_cli", lambda self: (self.cli_executable, "0.8.0"))
    monkeypatch.setattr(
        "geochemistrypi_mcp.experiments.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 2, "", json.dumps({"error": "Unknown experiment ID"}) + "\n"
        ),
    )
    with pytest.raises(ExperimentStoreError, match="Unknown experiment ID"):
        ExperimentManager(settings).get(GetExperimentRequest(experiment_id="404"))


def test_existing_experiment_name_mismatch_fails_before_dataset_or_cli_execution(tmp_path: Path) -> None:
    class MismatchStore:
        def require_matching_name(self, experiment_id: str, expected_name: str):
            raise ExperimentStoreError("stable ID belongs to a different experiment")

    settings = McpSettings(runs_root=tmp_path / "runs", cli_executable=Path(sys.executable))
    manager = RunManager(
        settings,
        cli_resolver=lambda: (Path(sys.executable), "0.8.0"),
        experiment_manager=MismatchStore(),
    )
    request = ClassificationRequest(
        training_dataset_path=tmp_path / "not-read.csv",
        experiment_name="Wrong",
        existing_experiment_id="7",
        run_name="Never Started",
        identifier_column="SampleID",
        feature_columns=("SIO2",),
        target_column="Label",
    )
    try:
        with pytest.raises(ExperimentStoreError, match="different experiment"):
            manager.start(request)
        assert not list(settings.runs_root.glob("run-*"))
    finally:
        manager.close()
