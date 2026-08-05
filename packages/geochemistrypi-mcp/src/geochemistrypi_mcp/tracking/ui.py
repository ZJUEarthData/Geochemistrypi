"""Explicit, local-only lifecycle management for the MLflow UI process."""

import json
import os
import secrets
import socket
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import psutil

from ..api.schemas import MlflowUiStatusResponse, StartMlflowUiRequest
from ..config.constants import ISOLATED_CLI_ENVIRONMENT_VARIABLES
from ..config.settings import McpSettings, resolve_cli_interpreter

_HOST = "127.0.0.1"
_OWNERSHIP_ENVIRONMENT_VARIABLE = "GEOCHEMISTRYPI_MCP_MLFLOW_UI_INSTANCE_ID"
_INSTANCE_ID_LENGTH = 64
_STATE_FIELDS_V1 = {
    "schema_version",
    "pid",
    "process_create_time",
    "host",
    "port",
    "tracking_root",
    "tracking_uri",
    "started_at",
    "command",
}
_STATE_FIELDS_BY_SCHEMA = {
    1: _STATE_FIELDS_V1,
    2: _STATE_FIELDS_V1 | {"instance_id"},
}


class MlflowUiError(ValueError):
    """Raised when a managed UI operation cannot be completed safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _commands_match(actual: list[str], recorded: list[str]) -> bool:
    """Compare commands while allowing equivalent executable path spellings."""
    if len(actual) != len(recorded) or actual[1:] != recorded[1:]:
        return False
    if not actual or actual[0] == recorded[0]:
        return bool(actual)
    try:
        return os.path.samefile(actual[0], recorded[0])
    except (OSError, ValueError):
        return False


class MlflowUiManager:
    """Manage one persistent UI process while refusing ambiguous ownership."""

    def __init__(
        self,
        settings: McpSettings,
        launch_command: Callable[[str, int], tuple[str, ...]] | None = None,
    ) -> None:
        self.settings = settings
        if settings.service_state_root is None or settings.tracking_root is None:
            raise MlflowUiError("The installer-owned MLflow service paths are not configured.")
        self.state_path = settings.service_state_root / "mlflow-ui.json"
        self.stdout_path = settings.service_state_root / "mlflow-ui.stdout.log"
        self.stderr_path = settings.service_state_root / "mlflow-ui.stderr.log"
        self.launch_command = launch_command
        self._lock = threading.RLock()

    @property
    def tracking_root(self) -> Path:
        value = self.settings.tracking_root
        if value is None:
            raise MlflowUiError("The installer-owned MLflow tracking root is not configured.")
        return value

    def _read_state(self) -> dict[str, Any] | None:
        if not self.state_path.is_file():
            return None
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise MlflowUiError(f"Managed MLflow UI state is corrupt: {self.state_path}. " "Do not stop a process by PID until this file is repaired or removed manually.") from exc
        schema_version = value.get("schema_version") if isinstance(value, dict) else None
        expected_fields = _STATE_FIELDS_BY_SCHEMA.get(schema_version)
        if expected_fields is None or set(value) != expected_fields:
            raise MlflowUiError("Managed MLflow UI state has unknown or missing fields; process ownership cannot be verified.")
        return value

    @staticmethod
    def _port_is_available(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
            stream.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                stream.bind((_HOST, port))
            except OSError:
                return False
        return True

    @staticmethod
    def _port_is_accepting(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
            stream.settimeout(0.2)
            return stream.connect_ex((_HOST, port)) == 0

    def _owned_process(self, state: dict[str, Any]) -> psutil.Process:
        schema_version = state.get("schema_version")
        try:
            process = psutil.Process(int(state["pid"]))
            if abs(process.create_time() - float(state["process_create_time"])) > 0.01:
                raise MlflowUiError("The recorded MLflow UI PID now belongs to a different process; it will not be stopped.")
            command = process.cmdline() if schema_version == 1 else None
            environment = process.environ() if schema_version == 2 else None
        except psutil.NoSuchProcess as exc:
            raise exc
        except (psutil.AccessDenied, ValueError, TypeError) as exc:
            raise MlflowUiError("The MLflow UI process identity cannot be inspected; it will not be stopped.") from exc
        recorded_command = state.get("command")
        if not isinstance(recorded_command, list) or not recorded_command or not all(isinstance(part, str) and part for part in recorded_command):
            raise MlflowUiError("The recorded MLflow UI command identity is invalid; it will not be stopped.")
        if schema_version == 1 and not _commands_match(command or [], recorded_command):
            raise MlflowUiError("The recorded PID command no longer matches the managed MLflow UI; it will not be stopped.")
        if schema_version == 2:
            instance_id = state.get("instance_id")
            if not isinstance(instance_id, str) or len(instance_id) != _INSTANCE_ID_LENGTH or any(character not in "0123456789abcdef" for character in instance_id):
                raise MlflowUiError("The recorded MLflow UI launch identity is invalid; it will not be stopped.")
            observed_instance_id = (environment or {}).get(_OWNERSHIP_ENVIRONMENT_VARIABLE, "")
            if not secrets.compare_digest(observed_instance_id, instance_id):
                raise MlflowUiError("The recorded PID does not carry the managed MLflow UI launch identity; it will not be stopped.")
        if str(self.tracking_root) != state["tracking_root"]:
            raise MlflowUiError("The recorded MLflow UI uses a different tracking root; it will not be stopped.")
        return process

    @staticmethod
    def _terminate_unrecorded_process(process: subprocess.Popen[bytes]) -> None:
        """Best-effort cleanup when ownership state cannot be persisted."""
        try:
            owned = psutil.Process(process.pid)
            descendants = owned.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            descendants = []
            owned = None
        for child in reversed(descendants):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        if owned is not None:
            try:
                owned.terminate()
            except psutil.NoSuchProcess:
                pass
        else:
            try:
                process.terminate()
            except OSError:
                return
        _, alive = psutil.wait_procs(
            [*descendants, *([owned] if owned is not None else [])],
            timeout=2,
        )
        for remaining in alive:
            try:
                remaining.kill()
            except psutil.NoSuchProcess:
                pass
        if owned is None:
            try:
                process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                except OSError:
                    pass

    def _response(self, state: str, message: str, value: dict[str, Any] | None = None) -> MlflowUiStatusResponse:
        return MlflowUiStatusResponse(
            state=state,
            port=int(value["port"]) if value else None,
            url=f"http://{_HOST}:{value['port']}" if value else None,
            pid=int(value["pid"]) if value else None,
            started_at=str(value["started_at"]) if value else None,
            tracking_root=str(self.tracking_root),
            message=message,
        )

    def status(self) -> MlflowUiStatusResponse:
        with self._lock:
            state = self._read_state()
            if state is None:
                return self._response("stopped", "The managed MLflow UI is not running.")
            try:
                self._owned_process(state)
            except psutil.NoSuchProcess:
                self.state_path.unlink(missing_ok=True)
                return self._response(
                    "stopped",
                    "Recovered stale MLflow UI state after the recorded process exited.",
                )
            except MlflowUiError as exc:
                return self._response("ownership_mismatch", str(exc), state)
            process_state = "running" if self._port_is_accepting(int(state["port"])) else "starting"
            return self._response(
                process_state,
                "The managed MLflow UI is available locally." if process_state == "running" else "The managed MLflow UI process is starting but is not accepting connections yet.",
                state,
            )

    def start(self, request: StartMlflowUiRequest) -> MlflowUiStatusResponse:
        with self._lock:
            current = self.status()
            if current.state in {"running", "starting"}:
                if current.port != request.port:
                    raise MlflowUiError(f"The managed MLflow UI already owns port {current.port}; stop it before selecting port {request.port}.")
                return current
            if current.state == "ownership_mismatch":
                raise MlflowUiError(current.message)
            if not self._port_is_available(request.port):
                raise MlflowUiError(f"Local port {request.port} is already in use. Choose another port; no existing process was modified.")
            self.tracking_root.mkdir(parents=True, exist_ok=True)
            if self.settings.service_state_root is None:
                raise MlflowUiError("The installer-owned service state root is not configured.")
            self.settings.service_state_root.mkdir(parents=True, exist_ok=True)
            uri = self.tracking_root.as_uri()
            if self.launch_command is None:
                executable, _ = self.settings.require_supported_cli()
                interpreter = resolve_cli_interpreter(executable)
                command = (
                    str(interpreter),
                    "-m",
                    "mlflow",
                    "ui",
                    "--backend-store-uri",
                    uri,
                    "--host",
                    _HOST,
                    "--port",
                    str(request.port),
                )
            else:
                command = self.launch_command(uri, request.port)
                if not command or not all(isinstance(part, str) and part for part in command):
                    raise MlflowUiError("The managed MLflow UI launch command is invalid.")
            environment = os.environ.copy()
            for name in ISOLATED_CLI_ENVIRONMENT_VARIABLES:
                environment.pop(name, None)
            instance_id = secrets.token_hex(_INSTANCE_ID_LENGTH // 2)
            environment[_OWNERSHIP_ENVIRONMENT_VARIABLE] = instance_id
            creationflags = 0
            if os.name == "nt":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            with self.stdout_path.open("ab") as stdout, self.stderr_path.open("ab") as stderr:
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=self.tracking_root,
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout,
                        stderr=stderr,
                        creationflags=creationflags,
                    )
                except OSError as exc:
                    raise MlflowUiError("Cannot start MLflow UI in the configured CLI environment.") from exc
            try:
                launched_process = psutil.Process(process.pid)
                create_time = launched_process.create_time()
                observed_command = launched_process.cmdline()
                if not observed_command or not all(isinstance(part, str) and part for part in observed_command):
                    raise ValueError("empty process command")
            except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError) as exc:
                self._terminate_unrecorded_process(process)
                raise MlflowUiError("MLflow UI exited before its process identity could be recorded.") from exc
            state = {
                "schema_version": 2,
                "pid": process.pid,
                "process_create_time": create_time,
                "host": _HOST,
                "port": request.port,
                "tracking_root": str(self.tracking_root),
                "tracking_uri": uri,
                "started_at": _utc_now(),
                "command": observed_command,
                "instance_id": instance_id,
            }
            try:
                _atomic_write_json(self.state_path, state)
            except OSError as exc:
                self._terminate_unrecorded_process(process)
                raise MlflowUiError("MLflow UI ownership state could not be stored, so the newly launched process was stopped.") from exc
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    self.state_path.unlink(missing_ok=True)
                    raise MlflowUiError(f"MLflow UI exited with code {process.returncode}; inspect {self.stderr_path}.")
                if self._port_is_accepting(request.port):
                    return self._response("running", "The managed MLflow UI is available locally.", state)
                time.sleep(0.1)
            return self._response(
                "starting",
                "The managed MLflow UI process started but is not accepting connections yet; call mlflow_ui_status.",
                state,
            )

    def stop(self) -> MlflowUiStatusResponse:
        with self._lock:
            state = self._read_state()
            if state is None:
                return self._response("stopped", "The managed MLflow UI was already stopped.")
            try:
                process = self._owned_process(state)
            except psutil.NoSuchProcess:
                self.state_path.unlink(missing_ok=True)
                return self._response("stopped", "Recovered stale state; the managed MLflow UI was already stopped.")
            descendants = process.children(recursive=True)
            for owned in reversed(descendants):
                try:
                    owned.terminate()
                except psutil.NoSuchProcess:
                    pass
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                pass
            _, alive = psutil.wait_procs([*descendants, process], timeout=5)
            for owned in alive:
                try:
                    owned.kill()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(alive, timeout=5)
            if alive:
                raise MlflowUiError("A verified MLflow UI process did not stop; ownership state was preserved.")
            self.state_path.unlink(missing_ok=True)
            return self._response("stopped", "The verified managed MLflow UI process tree was stopped.")
