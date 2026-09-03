import importlib.util
import os
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PREFLIGHT_PATH = REPOSITORY_ROOT / "packages" / "geochemistrypi-mcp" / "tools" / "release_preflight.py"


def _preflight_module():
    spec = importlib.util.spec_from_file_location("geochemistrypi_release_preflight", PREFLIGHT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parity_preflight_never_uses_existing_user_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    preflight = _preflight_module()
    existing = tmp_path / "existing-user-tracking"
    monkeypatch.setenv("GEOCHEMISTRYPI_MCP_TRACKING_ROOT", str(existing))
    observed = []

    def runner(command, *, cwd=preflight.REPOSITORY_ROOT, environment=None):
        observed.append((tuple(command), cwd, dict(environment)))

    monkeypatch.setattr(preflight, "_run", runner)
    bin_directory = "Scripts" if os.name == "nt" else "bin"
    executable = "python.exe" if os.name == "nt" else "python"
    mcp_python = tmp_path / "mcp-installed" / bin_directory / executable
    cli_command = tmp_path / "cli-installed" / bin_directory / ("geochemistrypi.exe" if os.name == "nt" else "geochemistrypi")

    preflight._parity_tests(mcp_python, cli_command, full=False)

    assert len(observed) == 1
    environment = observed[0][2]
    parity_root = tmp_path / "parity user state 用户数据"
    assert environment["GEOCHEMISTRYPI_MCP_APP_ROOT"] == str(parity_root)
    assert environment["GEOCHEMISTRYPI_MCP_RUNS_ROOT"] == str(parity_root / "runs")
    assert environment["GEOCHEMISTRYPI_MCP_TRACKING_ROOT"] == str(parity_root / "tracking")
    assert environment["GEOCHEMISTRYPI_MCP_SERVICE_STATE_ROOT"] == str(parity_root / "service-state")
    assert environment["GEOCHEMISTRYPI_MCP_SETTINGS_FILE"] == str(parity_root / "config" / "settings.json")
    assert environment["GEOCHEMISTRYPI_MCP_TRACKING_ROOT"] != str(existing)


def test_every_real_parity_process_owns_an_explicit_temporary_tracking_root() -> None:
    representative = (REPOSITORY_ROOT / "tests" / "mcp_wrapper" / "parity" / "test_mcp_cli_parity.py").read_text(encoding="utf-8")
    full_matrix = (REPOSITORY_ROOT / "tests" / "mcp_wrapper" / "parity" / "test_full_parity_matrix.py").read_text(encoding="utf-8")

    assert representative.count("StdioServerParameters(") == representative.count("env=_stdio_environment(")
    assert representative.count(".compile(") == representative.count("plan = _with_tracking_root(")
    assert 'public_command=(*plan.public_command, "--tracking-root"' in full_matrix


def test_default_preflight_runs_every_full_parity_shard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    preflight = _preflight_module()
    observed = []

    def runner(command, *, cwd=preflight.REPOSITORY_ROOT, environment=None):
        observed.append((tuple(command), cwd, dict(environment)))

    monkeypatch.setattr(preflight, "_run", runner)
    bin_directory = "Scripts" if os.name == "nt" else "bin"
    executable = "python.exe" if os.name == "nt" else "python"
    mcp_python = tmp_path / "mcp-installed" / bin_directory / executable
    cli_command = tmp_path / "cli-installed" / bin_directory / ("geochemistrypi.exe" if os.name == "nt" else "geochemistrypi")

    preflight._parity_tests(mcp_python, cli_command, full=True)

    assert len(observed) == 1 + len(preflight.FULL_PARITY_SHARDS)
    assert [item[2]["GEOCHEMISTRYPI_PARITY_SHARD"] for item in observed[1:]] == list(preflight.FULL_PARITY_SHARDS)
    assert all(item[2]["GEOCHEMISTRYPI_FULL_PARITY"] == "1" for item in observed[1:])
    assert all("--basetemp" in item[0] for item in observed[1:])


def test_candidate_cli_environment_installs_test_dependencies_from_the_built_wheel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = _preflight_module()
    observed = []

    def create_environment(root: Path, _version: str) -> Path:
        return root / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")

    def runner(command, *, cwd=preflight.REPOSITORY_ROOT, environment=None):
        observed.append((tuple(command), cwd, environment))

    cli_wheel = tmp_path / "geochemistrypi-0.8.2-py3-none-any.whl"
    mcp_wheel = tmp_path / "geochemistrypi_mcp-0.2.2-py3-none-any.whl"
    cli_command = tmp_path / "geochemistrypi"
    cli_command.touch()
    monkeypatch.setattr(preflight, "_create_environment", create_environment)
    monkeypatch.setattr(preflight, "_run", runner)
    monkeypatch.setattr(preflight, "_venv_command", lambda _root, _name: cli_command)

    preflight._install_candidate_environments(tmp_path, cli_wheel, mcp_wheel)

    assert observed[0][0][-1] == f"geochemistrypi[test] @ {cli_wheel.as_uri()}"
