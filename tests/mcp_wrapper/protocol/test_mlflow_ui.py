import json
import socket
import subprocess
import sys
from pathlib import Path

import psutil
import pytest
from geochemistrypi_mcp.api.schemas import StartMlflowUiRequest
from geochemistrypi_mcp.config.settings import McpSettings
from geochemistrypi_mcp.tracking.ui import MlflowUiError, MlflowUiManager


def _settings(tmp_path: Path) -> McpSettings:
    return McpSettings(
        runs_root=tmp_path / "runs",
        cli_executable=None,
        tracking_root=tmp_path / "tracking",
        service_state_root=tmp_path / "state",
    )


def _free_port() -> int:
    with socket.socket() as stream:
        stream.bind(("127.0.0.1", 0))
        return int(stream.getsockname()[1])


def _http_command(_: str, port: int) -> tuple[str, ...]:
    return (sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1")


def test_managed_ui_is_explicit_persistent_and_stops_verified_process(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    first_manager = MlflowUiManager(settings, launch_command=_http_command)
    assert first_manager.status().state == "stopped"

    started = first_manager.start(StartMlflowUiRequest(port=_free_port()))
    try:
        assert started.state == "running"
        assert started.url == f"http://127.0.0.1:{started.port}"
        recovered_manager = MlflowUiManager(settings, launch_command=_http_command)
        assert recovered_manager.status().state == "running"
        assert recovered_manager.stop().state == "stopped"
        assert recovered_manager.status().state == "stopped"
    finally:
        if started.pid and psutil.pid_exists(started.pid):
            psutil.Process(started.pid).kill()


def test_managed_ui_never_kills_unrelated_process_with_reused_metadata(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    manager = MlflowUiManager(settings, launch_command=_http_command)
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        owned = psutil.Process(process.pid)
        manager.state_path.parent.mkdir(parents=True)
        manager.state_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "pid": process.pid,
                    "process_create_time": owned.create_time(),
                    "host": "127.0.0.1",
                    "port": 5000,
                    "tracking_root": str(settings.tracking_root),
                    "tracking_uri": settings.tracking_root.as_uri(),
                    "started_at": "2026-01-01T00:00:00+00:00",
                    "command": [sys.executable, "-m", "mlflow", "ui"],
                }
            ),
            encoding="utf-8",
        )
        assert manager.status().state == "ownership_mismatch"
        with pytest.raises(MlflowUiError, match="will not be stopped"):
            manager.stop()
        assert process.poll() is None
    finally:
        process.kill()
        process.wait(timeout=5)


def test_managed_ui_recovers_stale_state_and_rejects_port_conflict(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    manager = MlflowUiManager(settings, launch_command=_http_command)
    manager.state_path.parent.mkdir(parents=True)
    manager.state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pid": 2147483647,
                "process_create_time": 1.0,
                "host": "127.0.0.1",
                "port": 5000,
                "tracking_root": str(settings.tracking_root),
                "tracking_uri": settings.tracking_root.as_uri(),
                "started_at": "2026-01-01T00:00:00+00:00",
                "command": [sys.executable, "-m", "mlflow", "ui"],
            }
        ),
        encoding="utf-8",
    )
    assert "Recovered stale" in manager.status().message
    assert not manager.state_path.exists()

    with socket.socket() as stream:
        stream.bind(("127.0.0.1", 0))
        port = int(stream.getsockname()[1])
        with pytest.raises(MlflowUiError, match="already in use"):
            manager.start(StartMlflowUiRequest(port=port))


def test_managed_ui_stops_new_process_if_ownership_state_cannot_be_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = MlflowUiManager(_settings(tmp_path), launch_command=_http_command)
    launched_pids: list[int] = []

    def fail_write(_path: Path, value: dict[str, object]) -> None:
        launched_pids.append(int(value["pid"]))
        raise OSError("simulated disk failure")

    monkeypatch.setattr("geochemistrypi_mcp.tracking.ui._atomic_write_json", fail_write)

    with pytest.raises(MlflowUiError, match="newly launched process was stopped"):
        manager.start(StartMlflowUiRequest(port=_free_port()))

    assert len(launched_pids) == 1
    assert not psutil.pid_exists(launched_pids[0])
    assert not manager.state_path.exists()
