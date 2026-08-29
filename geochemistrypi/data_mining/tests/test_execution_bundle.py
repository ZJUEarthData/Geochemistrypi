import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import geochemistrypi.cli as cli
from geochemistrypi.execution_bundle import ExecutionBundle, ExecutionBundleError


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(tmp_path: Path) -> Path:
    data = tmp_path / "prepared.csv"
    data.write_text("SampleID,x\nA,1\n", encoding="utf-8")
    plan = tmp_path / "automation-plan.json"
    plan.write_text('{"schema_version":1}', encoding="utf-8")
    science = tmp_path / "scientific-execution.json"
    science.write_text('{"schema_version":3}', encoding="utf-8")
    bundle = tmp_path / "cli-execution-bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "plan_name": "generic-classification-v1",
                "data_source": "ANY_PATH",
                "training_data": {"path": data.name, "sha256": _sha256(data)},
                "application_data": None,
                "automation_plan": {"path": plan.name, "sha256": _sha256(plan)},
                "scientific_config": {"path": science.name, "sha256": _sha256(science)},
                "world_map_config": "",
                "tracking_root": "",
                "existing_experiment_id": "",
            }
        ),
        encoding="utf-8",
    )
    return bundle


def test_execution_bundle_binds_every_input_and_accepts_matching_relocation(
    tmp_path: Path,
) -> None:
    bundle_path = _bundle(tmp_path)
    loaded = ExecutionBundle.load(bundle_path)
    replacement = tmp_path / "relocated.csv"
    replacement.write_bytes(loaded.training_data.path.read_bytes())

    relocated = ExecutionBundle.load(bundle_path, training_override=replacement)

    assert relocated.plan_name == "generic-classification-v1"
    assert relocated.training_data.path == replacement.resolve()
    assert len(relocated.source_sha256) == 64


def test_execution_bundle_rejects_changed_data(tmp_path: Path) -> None:
    bundle_path = _bundle(tmp_path)
    (tmp_path / "prepared.csv").write_text("SampleID,x\nA,2\n", encoding="utf-8")

    with pytest.raises(ExecutionBundleError, match="recorded SHA-256"):
        ExecutionBundle.load(bundle_path)


def test_replay_command_uses_the_existing_cli_pipeline(
    monkeypatch, tmp_path: Path
) -> None:
    bundle_path = _bundle(tmp_path)
    observed = {}

    def capture(**kwargs):
        observed.update(kwargs)

    monkeypatch.setattr(cli, "_run_cli_pipeline", capture)
    result = CliRunner().invoke(
        cli.app,
        ["replay", "--bundle", str(bundle_path), "--output", str(tmp_path / "events")],
    )

    assert result.exit_code == 0, result.output
    assert observed["data_source_name"] == "ANY_PATH"
    assert observed["training_data_path"] == str((tmp_path / "prepared.csv").resolve())
    assert observed["automation_plan"] == str(
        (tmp_path / "automation-plan.json").resolve()
    )
    assert observed["scientific_config"] == str(
        (tmp_path / "scientific-execution.json").resolve()
    )
    assert observed["automation_events"] == str(
        (tmp_path / "events" / "automation-events.json").resolve()
    )
