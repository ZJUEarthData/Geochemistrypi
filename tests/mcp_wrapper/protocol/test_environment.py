import json
import subprocess
from pathlib import Path

import geochemistrypi_mcp.runtime.environment as environment_module


class _UnixVenvInterpreter:
    """Simulate a venv interpreter symlink without requiring symlink privileges."""

    def __init__(self, venv_path: Path, resolved_path: Path) -> None:
        self.venv_path = venv_path
        self.resolved_path = resolved_path

    def __str__(self) -> str:
        return str(self.venv_path)

    def resolve(self) -> Path:
        return self.resolved_path


def test_environment_inspection_executes_the_unresolved_venv_interpreter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cli_executable = tmp_path / "venv" / "bin" / "geochemistrypi"
    cli_executable.parent.mkdir(parents=True)
    cli_executable.write_text("launcher", encoding="utf-8")
    venv_python = cli_executable.parent / "python"
    base_python = tmp_path / "base" / "python3.9"
    interpreter = _UnixVenvInterpreter(venv_python, base_python)
    captured_command: tuple[str, ...] | None = None

    def run(command, **_kwargs):
        nonlocal captured_command
        captured_command = tuple(command)
        observed = {
            "python_version": "3.9.25",
            "python_implementation": "CPython",
            "platform": "Linux-test",
            "dependencies": {"geochemistrypi": environment_module.CLI_VERSION},
            "geochemistrypi_payload": {"file_count": 1, "sha256": "b" * 64},
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(observed), "")

    monkeypatch.setattr(environment_module, "resolve_cli_interpreter", lambda _: interpreter)
    monkeypatch.setattr(environment_module.subprocess, "run", run)
    monkeypatch.setattr(environment_module, "sha256_file", lambda _: "a" * 64)

    snapshot = environment_module.inspect_cli_environment(cli_executable)

    assert captured_command is not None
    assert captured_command[0] == str(venv_python)
    assert captured_command[0] != str(base_python)
    assert snapshot.record["python"]["executable"] == str(venv_python)
