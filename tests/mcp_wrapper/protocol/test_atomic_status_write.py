import json
import threading
import time
from pathlib import Path

import pytest
from geochemistrypi_mcp.runtime import runs


def _temporary_status_files(status_path: Path) -> list[Path]:
    return list(status_path.parent.glob(f".{status_path.name}.*.tmp"))


def test_atomic_status_write_normal_path_is_unchanged(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"

    runs._atomic_write_json(status_path, {"state": "queued"})

    assert json.loads(status_path.read_text(encoding="utf-8")) == {"state": "queued"}
    assert _temporary_status_files(status_path) == []


@pytest.mark.parametrize("failure_count", [1, 3])
def test_atomic_status_write_retries_bounded_permission_errors_then_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_count: int,
) -> None:
    status_path = tmp_path / "status.json"
    runs._atomic_write_json(status_path, {"state": "queued"})
    original_replace = runs.os.replace
    attempts = 0
    sleeps: list[float] = []

    def flaky_replace(source: Path, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts <= failure_count:
            raise PermissionError(13, "simulated transient status lock", str(destination))
        original_replace(source, destination)

    monkeypatch.setattr(runs.os, "replace", flaky_replace)
    monkeypatch.setattr(time, "sleep", sleeps.append)

    runs._atomic_write_json(status_path, {"state": "running"})

    assert attempts == failure_count + 1
    assert sleeps == list(runs._ATOMIC_REPLACE_RETRY_DELAYS_SECONDS[:failure_count])
    assert json.loads(status_path.read_text(encoding="utf-8")) == {"state": "running"}
    assert _temporary_status_files(status_path) == []


def test_atomic_status_write_exhaustion_preserves_old_status_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status_path = tmp_path / "status.json"
    old_status = {"state": "queued", "run_id": "run-old"}
    runs._atomic_write_json(status_path, old_status)
    attempts = 0
    sleeps: list[float] = []

    def locked_replace(source: Path, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        raise PermissionError(13, "simulated persistent status lock", str(destination))

    monkeypatch.setattr(runs.os, "replace", locked_replace)
    monkeypatch.setattr(time, "sleep", sleeps.append)

    with pytest.raises(PermissionError, match="Atomic metadata replacement failed after 4 attempts"):
        runs._atomic_write_json(status_path, {"state": "running"})

    assert attempts == 4
    assert sleeps == list(runs._ATOMIC_REPLACE_RETRY_DELAYS_SECONDS)
    assert json.loads(status_path.read_text(encoding="utf-8")) == old_status
    assert _temporary_status_files(status_path) == []


def test_atomic_status_write_does_not_retry_other_os_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status_path = tmp_path / "status.json"
    old_status = {"state": "queued"}
    runs._atomic_write_json(status_path, old_status)
    attempts = 0
    sleeps: list[float] = []

    def failing_replace(source: Path, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        raise OSError("simulated non-permission failure")

    monkeypatch.setattr(runs.os, "replace", failing_replace)
    monkeypatch.setattr(time, "sleep", sleeps.append)

    with pytest.raises(OSError, match="simulated non-permission failure"):
        runs._atomic_write_json(status_path, {"state": "running"})

    assert attempts == 1
    assert sleeps == []
    assert json.loads(status_path.read_text(encoding="utf-8")) == old_status
    assert _temporary_status_files(status_path) == []


def test_atomic_status_write_recovers_from_status_polling_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status_path = tmp_path / "status.json"
    runs._atomic_write_json(status_path, {"state": "queued"})
    original_replace = runs.os.replace
    replace_started = threading.Event()
    poll_finished = threading.Event()
    observed: list[dict[str, str]] = []
    attempts = 0

    def poll_status() -> None:
        assert replace_started.wait(timeout=1)
        observed.append(runs._read_json(status_path))
        poll_finished.set()

    def polling_race_replace(source: Path, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            replace_started.set()
            assert poll_finished.wait(timeout=1)
            raise PermissionError(13, "simulated polling handle", str(destination))
        original_replace(source, destination)

    poller = threading.Thread(target=poll_status)
    poller.start()
    monkeypatch.setattr(runs.os, "replace", polling_race_replace)
    monkeypatch.setattr(time, "sleep", lambda _: None)

    runs._atomic_write_json(status_path, {"state": "running"})
    poller.join(timeout=1)

    assert not poller.is_alive()
    assert observed == [{"state": "queued"}]
    assert attempts == 2
    assert runs._read_json(status_path) == {"state": "running"}
    assert _temporary_status_files(status_path) == []
