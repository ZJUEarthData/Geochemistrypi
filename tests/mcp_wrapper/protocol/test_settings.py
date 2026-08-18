import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from geochemistrypi_mcp.config.constants import CLI_VERSION
from geochemistrypi_mcp.config.settings import McpSettings, SettingsError, default_app_root


@pytest.mark.skipif(os.name != "nt", reason="Windows environment layout regression")
@pytest.mark.parametrize("layout", ("venv", "conda"))
def test_supported_cli_version_uses_the_interpreter_that_owns_the_windows_launcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
) -> None:
    environment_root = tmp_path / layout
    scripts = environment_root / "Scripts"
    scripts.mkdir(parents=True)
    executable = scripts / "geochemistrypi.exe"
    executable.touch()
    interpreter = scripts / "python.exe" if layout == "venv" else environment_root / "python.exe"
    interpreter.touch()
    observed_command: list[str] = []

    def fake_run(command, **_kwargs):
        observed_command.extend(command)
        return SimpleNamespace(stdout=f"{CLI_VERSION}\n")

    monkeypatch.setattr("geochemistrypi_mcp.config.settings.subprocess.run", fake_run)
    settings = McpSettings(runs_root=tmp_path / "runs", cli_executable=executable)

    assert settings.require_supported_cli() == (executable, CLI_VERSION)
    assert observed_command[0] == str(interpreter)


def test_zero_argument_settings_load_persisted_private_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings_file = tmp_path / "settings.json"
    cli = tmp_path / "cli" / "geochemistrypi.exe"
    runs = tmp_path / "runs"
    settings_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cli_executable": str(cli),
                "runs_root": str(runs),
                "maximum_dataset_bytes": 4096,
                "maximum_pending_runs": 3,
                "maximum_process_seconds": 1200,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GEOCHEMISTRYPI_MCP_SETTINGS_FILE", str(settings_file))
    monkeypatch.delenv("GEOCHEMISTRYPI_CLI_EXECUTABLE", raising=False)
    monkeypatch.delenv("GEOCHEMISTRYPI_MCP_RUNS_ROOT", raising=False)
    monkeypatch.delenv("GEOCHEMISTRYPI_MCP_MAX_DATASET_BYTES", raising=False)
    monkeypatch.delenv("GEOCHEMISTRYPI_MCP_MAX_PENDING_RUNS", raising=False)
    monkeypatch.delenv("GEOCHEMISTRYPI_MCP_MAX_PROCESS_SECONDS", raising=False)

    settings = McpSettings.from_environment()

    assert settings.cli_executable == cli.resolve()
    assert settings.runs_root == runs.resolve()
    assert settings.tracking_root == (tmp_path / "tracking").resolve()
    assert settings.service_state_root == (tmp_path / "service-state").resolve()
    assert settings.maximum_dataset_bytes == 4096
    assert settings.maximum_pending_runs == 3
    assert settings.maximum_process_seconds == 1200


def test_development_app_root_override_must_be_absolute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured = tmp_path / "isolated-application"
    monkeypatch.setenv("GEOCHEMISTRYPI_MCP_APP_ROOT", str(configured))
    assert default_app_root() == configured.resolve()

    monkeypatch.setenv("GEOCHEMISTRYPI_MCP_APP_ROOT", "relative-root")
    with pytest.raises(SettingsError, match="must be an absolute path"):
        default_app_root()


def test_custom_runs_root_keeps_all_managed_state_in_the_same_private_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    runs = tmp_path / "isolated" / "runs"
    monkeypatch.setenv("GEOCHEMISTRYPI_MCP_SETTINGS_FILE", str(settings_file))
    monkeypatch.setenv("GEOCHEMISTRYPI_MCP_RUNS_ROOT", str(runs))
    monkeypatch.delenv("GEOCHEMISTRYPI_MCP_TRACKING_ROOT", raising=False)
    monkeypatch.delenv("GEOCHEMISTRYPI_MCP_SERVICE_STATE_ROOT", raising=False)

    settings = McpSettings.from_environment()

    assert settings.runs_root == runs.resolve()
    assert settings.tracking_root == (runs.parent / "tracking").resolve()
    assert settings.service_state_root == (runs.parent / "service-state").resolve()


def test_legacy_schema_one_settings_receive_safe_release_limit_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    monkeypatch.setenv("GEOCHEMISTRYPI_MCP_SETTINGS_FILE", str(settings_file))
    monkeypatch.delenv("GEOCHEMISTRYPI_MCP_MAX_PENDING_RUNS", raising=False)
    monkeypatch.delenv("GEOCHEMISTRYPI_MCP_MAX_PROCESS_SECONDS", raising=False)

    settings = McpSettings.from_environment()

    assert settings.maximum_pending_runs == 8
    assert settings.maximum_process_seconds == 900


def test_schema_two_loads_all_persisted_resource_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "maximum_dataset_bytes": 2048,
                "maximum_columns": 32,
                "maximum_artifact_references": 17,
                "concurrency": 3,
                "maximum_pending_runs": 5,
                "maximum_process_seconds": 45,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GEOCHEMISTRYPI_MCP_SETTINGS_FILE", str(settings_file))
    monkeypatch.delenv("GEOCHEMISTRYPI_MCP_MAX_DATASET_BYTES", raising=False)
    monkeypatch.delenv("GEOCHEMISTRYPI_MCP_MAX_PENDING_RUNS", raising=False)
    monkeypatch.delenv("GEOCHEMISTRYPI_MCP_MAX_PROCESS_SECONDS", raising=False)

    settings = McpSettings.from_environment()

    assert settings.maximum_dataset_bytes == 2048
    assert settings.maximum_columns == 32
    assert settings.maximum_artifact_references == 17
    assert settings.concurrency == 3
    assert settings.maximum_pending_runs == 5
    assert settings.maximum_process_seconds == 45


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("GEOCHEMISTRYPI_MCP_MAX_PENDING_RUNS", "0"),
        ("GEOCHEMISTRYPI_MCP_MAX_PROCESS_SECONDS", "not-an-integer"),
    ),
)
def test_release_limit_environment_overrides_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    monkeypatch.setenv("GEOCHEMISTRYPI_MCP_SETTINGS_FILE", str(settings_file))
    monkeypatch.setenv(name, value)

    with pytest.raises(SettingsError, match=name):
        McpSettings.from_environment()


def test_persisted_settings_reject_unknown_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(json.dumps({"schema_version": 1, "unexpected": True}), encoding="utf-8")
    monkeypatch.setenv("GEOCHEMISTRYPI_MCP_SETTINGS_FILE", str(settings_file))

    with pytest.raises(SettingsError, match="Unknown"):
        McpSettings.from_environment()


def test_pending_run_limit_cannot_be_smaller_than_concurrency(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="cannot be smaller"):
        McpSettings(
            runs_root=tmp_path / "runs",
            cli_executable=None,
            concurrency=2,
            maximum_pending_runs=1,
        )
