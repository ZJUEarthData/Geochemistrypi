import json
import os
import subprocess
from pathlib import Path

import geochemistrypi_mcp.lifecycle.doctor as doctor_module
import pytest
from geochemistrypi_mcp.config.constants import CLI_VERSION, ISOLATED_CLI_ENVIRONMENT_VARIABLES, SERVER_VERSION
from geochemistrypi_mcp.lifecycle.doctor import run_doctor
from geochemistrypi_mcp.lifecycle.setup import SetupPaths


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("placeholder", encoding="utf-8")


def _prepared_paths(tmp_path: Path) -> SetupPaths:
    paths = SetupPaths(tmp_path / "application")
    for path in (paths.mcp_python, paths.server_command, paths.cli_python, paths.cli_command):
        _touch(path)
    paths.runs_root.mkdir(parents=True)
    paths.settings_file.parent.mkdir(parents=True)
    paths.settings_file.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "cli_executable": str(paths.cli_command),
                "runs_root": str(paths.runs_root),
                "tracking_root": str(paths.tracking_root),
                "service_state_root": str(paths.service_state_root),
                "maximum_dataset_bytes": 1024,
                "maximum_columns": 256,
                "maximum_artifact_references": 200,
                "concurrency": 1,
                "maximum_pending_runs": 8,
                "maximum_process_seconds": 900,
            }
        ),
        encoding="utf-8",
    )
    paths.manifest_file.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "server_version": SERVER_VERSION,
                "cli_version": CLI_VERSION,
                "compatibility_policy_version": 2,
                "mcp_python_requires": ">=3.10,<4",
                "cli_python_requires": ">=3.9,<3.10",
                "mcp_sdk_requires": "==2.0.0",
                "server_command": str(paths.server_command),
                "runs_root": str(paths.runs_root),
                "tracking_root": str(paths.tracking_root),
                "service_state_root": str(paths.service_state_root),
                "installation_source": "source",
                "source_fingerprint": "a" * 64,
                "runtime_inventory": {
                    "mcp": {"count": 10, "sha256": "1" * 64},
                    "cli": {"count": 20, "sha256": "2" * 64},
                },
                "rollback_available": False,
                "registered_clients": ["standard"],
            }
        ),
        encoding="utf-8",
    )
    return paths


def _inventory(_):
    return {
        "mcp": {"count": 10, "sha256": "1" * 64},
        "cli": {"count": 20, "sha256": "2" * 64},
    }


def test_doctor_checks_both_runtimes_storage_and_real_protocol_boundary(tmp_path: Path) -> None:
    paths = _prepared_paths(tmp_path)

    def runner(command):
        command = tuple(command)
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, f"Geochemistry Pi {CLI_VERSION}\n", "")
        if "scientific-runtime-ready" in command[-1]:
            return subprocess.CompletedProcess(command, 0, "scientific-runtime-ready\n", "")
        package = "geochemistrypi-mcp" if str(paths.mcp_python) == command[0] else "geochemistrypi"
        python = [3, 11, 9] if package == "geochemistrypi-mcp" else [3, 9, 19]
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps({"python": python, "package": SERVER_VERSION if package.endswith("-mcp") else CLI_VERSION}),
            "",
        )

    report = run_doctor(
        paths,
        runner=runner,
        protocol_probe=lambda _: (True, "13 tools discovered"),
        inventory_probe=_inventory,
    )

    assert report.healthy is True
    assert report.summary == "Doctor: healthy (10/10 checks passed)."
    assert {check.name for check in report.checks} == {
        "settings",
        "install-manifest",
        "release-bundle",
        "runtime-inventory",
        "managed-storage",
        "mcp-runtime",
        "cli-runtime",
        "cli-scientific-runtime",
        "cli-command",
        "mcp-protocol",
    }


def test_doctor_reports_version_and_protocol_failures_without_crashing(tmp_path: Path) -> None:
    paths = _prepared_paths(tmp_path)

    def runner(command):
        command = tuple(command)
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, "Geochemistry Pi 0.7.0\n", "")
        if "scientific-runtime-ready" in command[-1]:
            return subprocess.CompletedProcess(command, 1, "", "libomp is unavailable")
        package = "geochemistrypi-mcp" if str(paths.mcp_python) == command[0] else "geochemistrypi"
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps({"python": [3, 11, 0], "package": "9.9.9" if package.endswith("-mcp") else "0.7.0"}),
            "",
        )

    report = run_doctor(
        paths,
        runner=runner,
        protocol_probe=lambda _: (False, "server did not initialize"),
        inventory_probe=_inventory,
    )

    assert report.healthy is False
    failed = {check.name: check.detail for check in report.checks if not check.healthy}
    assert "does not match" in failed["mcp-runtime"]
    assert "must use Python 3.9" in failed["cli-runtime"]
    assert failed["cli-scientific-runtime"] == "libomp is unavailable"
    assert failed["mcp-protocol"] == "server did not initialize"


def test_doctor_rejects_a_stale_release_compatibility_manifest(tmp_path: Path) -> None:
    paths = _prepared_paths(tmp_path)
    manifest = json.loads(paths.manifest_file.read_text(encoding="utf-8"))
    manifest["compatibility_policy_version"] = 0
    paths.manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    report = run_doctor(
        paths,
        runner=lambda command: subprocess.CompletedProcess(command, 1, "", "not reached"),
        protocol_probe=lambda _: (False, "not reached"),
        inventory_probe=_inventory,
    )

    manifest_check = next(check for check in report.checks if check.name == "install-manifest")
    assert manifest_check.healthy is False
    assert "does not match" in manifest_check.detail


def test_doctor_detects_runtime_inventory_drift(tmp_path: Path) -> None:
    paths = _prepared_paths(tmp_path)
    changed = {
        "mcp": {"count": 11, "sha256": "3" * 64},
        "cli": {"count": 20, "sha256": "2" * 64},
    }

    report = run_doctor(
        paths,
        runner=lambda command: subprocess.CompletedProcess(command, 1, "", "not reached"),
        protocol_probe=lambda _: (False, "not reached"),
        inventory_probe=lambda _: changed,
    )

    check = next(item for item in report.checks if item.name == "runtime-inventory")
    assert check.healthy is False
    assert "changed after setup" in check.detail


def test_doctor_rejects_release_files_for_a_source_install(tmp_path: Path) -> None:
    paths = _prepared_paths(tmp_path)
    paths.release_root.mkdir()

    report = run_doctor(
        paths,
        runner=lambda command: subprocess.CompletedProcess(command, 1, "", "not reached"),
        protocol_probe=lambda _: (False, "not reached"),
        inventory_probe=_inventory,
    )

    check = next(item for item in report.checks if item.name == "release-bundle")
    assert check.healthy is False
    assert "must not expose" in check.detail


def test_default_doctor_runner_removes_foreign_python_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_environment = {}
    for name in ISOLATED_CLI_ENVIRONMENT_VARIABLES:
        monkeypatch.setenv(name, f"foreign-{name}")

    def fake_run(command, **kwargs):
        observed_environment.update(kwargs["env"])
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(doctor_module.subprocess, "run", fake_run)

    doctor_module._default_runner(("placeholder",))

    assert set(ISOLATED_CLI_ENVIRONMENT_VARIABLES).isdisjoint(observed_environment)
    if os.environ.get("PATH"):
        assert observed_environment["PATH"] == os.environ["PATH"]
