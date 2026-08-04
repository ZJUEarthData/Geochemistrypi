"""Internal settings for the local MCP wrapper."""

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .constants import ISOLATED_CLI_ENVIRONMENT_VARIABLES, SUPPORTED_CLI_VERSIONS

CLI_EXECUTABLE_ENV = "GEOCHEMISTRYPI_CLI_EXECUTABLE"
APP_ROOT_ENV = "GEOCHEMISTRYPI_MCP_APP_ROOT"
RUNS_ROOT_ENV = "GEOCHEMISTRYPI_MCP_RUNS_ROOT"
TRACKING_ROOT_ENV = "GEOCHEMISTRYPI_MCP_TRACKING_ROOT"
SERVICE_STATE_ROOT_ENV = "GEOCHEMISTRYPI_MCP_SERVICE_STATE_ROOT"
MAX_DATASET_BYTES_ENV = "GEOCHEMISTRYPI_MCP_MAX_DATASET_BYTES"
MAX_PENDING_RUNS_ENV = "GEOCHEMISTRYPI_MCP_MAX_PENDING_RUNS"
MAX_PROCESS_SECONDS_ENV = "GEOCHEMISTRYPI_MCP_MAX_PROCESS_SECONDS"
SETTINGS_FILE_ENV = "GEOCHEMISTRYPI_MCP_SETTINGS_FILE"
SETTINGS_SCHEMA_VERSION = 2
LEGACY_SETTINGS_SCHEMA_VERSIONS = (1,)
DEFAULT_MAXIMUM_DATASET_BYTES = 512 * 1024 * 1024
DEFAULT_MAXIMUM_COLUMNS = 256
DEFAULT_MAXIMUM_ARTIFACT_REFERENCES = 200
DEFAULT_CONCURRENCY = 1
DEFAULT_MAXIMUM_PENDING_RUNS = 8
DEFAULT_MAXIMUM_PROCESS_SECONDS = 900


class SettingsError(RuntimeError):
    """Raised when local wrapper configuration is unsafe or incomplete."""


def default_app_root() -> Path:
    """Return the platform-native root for private environments and state."""
    configured = os.environ.get(APP_ROOT_ENV)
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            raise SettingsError(f"{APP_ROOT_ENV} must be an absolute path.")
        return path.resolve()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return (Path(base) / "GeochemistryPi MCP").resolve()
    xdg_state = os.environ.get("XDG_STATE_HOME")
    if xdg_state:
        return (Path(xdg_state) / "geochemistrypi-mcp").resolve()
    return (Path.home() / ".local" / "state" / "geochemistrypi-mcp").resolve()


def default_settings_path() -> Path:
    """Return the zero-argument server's persisted settings path."""
    configured = os.environ.get(SETTINGS_FILE_ENV)
    path = Path(configured).expanduser() if configured else default_app_root() / "config" / "settings.json"
    if not path.is_absolute():
        raise SettingsError(f"{SETTINGS_FILE_ENV} must be an absolute path.")
    return path.resolve()


def _default_state_root() -> Path:
    """Keep the existing runs-root helper for development compatibility."""
    return default_app_root() / "runs"


def _absolute_directory(value: str | None, default: Path) -> Path:
    path = Path(value).expanduser() if value else default
    if not path.is_absolute():
        raise SettingsError(f"{RUNS_ROOT_ENV} must be an absolute path.")
    return path.resolve()


def _configured_directory(value: object | None, default: Path, environment_name: str) -> Path:
    path = Path(str(value)).expanduser() if value else default
    if not path.is_absolute():
        raise SettingsError(f"{environment_name} must be an absolute path.")
    return path.resolve()


def _positive_integer(value: str | None, environment_name: str, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise SettingsError(f"{environment_name} must be an integer.") from exc
    if parsed < 1:
        raise SettingsError(f"{environment_name} must be positive.")
    return parsed


def _maximum_dataset_bytes(value: str | None) -> int:
    return _positive_integer(value, MAX_DATASET_BYTES_ENV, DEFAULT_MAXIMUM_DATASET_BYTES)


def _load_persisted_settings(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SettingsError(f"Cannot read GeochemistryPi MCP settings: {path}") from exc
    supported_schemas = (*LEGACY_SETTINGS_SCHEMA_VERSIONS, SETTINGS_SCHEMA_VERSION)
    if not isinstance(value, dict) or value.get("schema_version") not in supported_schemas:
        raise SettingsError(f"Unsupported GeochemistryPi MCP settings schema in {path}.")
    allowed = {
        "schema_version",
        "cli_executable",
        "runs_root",
        "tracking_root",
        "service_state_root",
        "maximum_dataset_bytes",
        "maximum_columns",
        "maximum_artifact_references",
        "concurrency",
        "maximum_pending_runs",
        "maximum_process_seconds",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise SettingsError(f"Unknown GeochemistryPi MCP settings fields in {path}: {unknown}")
    return value


def _cli_interpreter_candidates(executable: Path) -> tuple[Path, ...]:
    """Return interpreter locations used by supported environment layouts."""
    if os.name == "nt":
        # pip/venv/uv place both launchers in Scripts; Conda keeps python.exe
        # one directory above Scripts.
        return (executable.parent / "python.exe", executable.parent.parent / "python.exe")
    return (executable.parent / "python",)


def resolve_cli_interpreter(executable: Path) -> Path:
    """Resolve the Python interpreter that owns a public CLI launcher."""
    candidates = _cli_interpreter_candidates(executable)
    interpreter = next((candidate for candidate in candidates if candidate.is_file()), None)
    if interpreter is None:
        checked = ", ".join(str(candidate) for candidate in candidates)
        raise SettingsError(f"Cannot locate the Python interpreter that owns the configured CLI. Checked: {checked}")
    return interpreter


@dataclass(frozen=True)
class McpSettings:
    """Resolved process settings that are never accepted as MCP tool arguments."""

    runs_root: Path
    cli_executable: Path | None
    tracking_root: Path | None = None
    service_state_root: Path | None = None
    maximum_dataset_bytes: int = DEFAULT_MAXIMUM_DATASET_BYTES
    maximum_columns: int = DEFAULT_MAXIMUM_COLUMNS
    maximum_artifact_references: int = DEFAULT_MAXIMUM_ARTIFACT_REFERENCES
    concurrency: int = DEFAULT_CONCURRENCY
    maximum_pending_runs: int = DEFAULT_MAXIMUM_PENDING_RUNS
    maximum_process_seconds: int = DEFAULT_MAXIMUM_PROCESS_SECONDS

    def __post_init__(self) -> None:
        if not self.runs_root.is_absolute():
            raise SettingsError("The MCP runs root must be absolute.")
        object.__setattr__(
            self,
            "tracking_root",
            (self.tracking_root or self.runs_root.parent / "tracking").resolve(),
        )
        object.__setattr__(
            self,
            "service_state_root",
            (self.service_state_root or self.runs_root.parent / "service-state").resolve(),
        )
        limits = (
            self.maximum_dataset_bytes,
            self.maximum_columns,
            self.maximum_artifact_references,
            self.concurrency,
            self.maximum_pending_runs,
            self.maximum_process_seconds,
        )
        if any(value < 1 for value in limits):
            raise SettingsError("Dataset, artifact, column, concurrency, queue, and process-time limits must be positive.")
        if self.maximum_pending_runs < self.concurrency:
            raise SettingsError("The pending-run limit cannot be smaller than the concurrency limit.")

    @classmethod
    def from_environment(cls) -> "McpSettings":
        """Load explicit development overrides or persisted zero-argument settings."""
        persisted = _load_persisted_settings(default_settings_path())
        configured_cli = os.environ.get(CLI_EXECUTABLE_ENV) or persisted.get("cli_executable")
        discovered_cli = configured_cli or shutil.which("geochemistrypi")
        cli_executable = Path(discovered_cli).expanduser().resolve() if discovered_cli else None
        configured_runs_root = os.environ.get(RUNS_ROOT_ENV) or persisted.get("runs_root")
        configured_tracking_root = os.environ.get(TRACKING_ROOT_ENV) or persisted.get("tracking_root")
        configured_service_state_root = os.environ.get(SERVICE_STATE_ROOT_ENV) or persisted.get("service_state_root")
        configured_maximum_bytes = os.environ.get(MAX_DATASET_BYTES_ENV)
        if configured_maximum_bytes is None and persisted.get("maximum_dataset_bytes") is not None:
            configured_maximum_bytes = str(persisted["maximum_dataset_bytes"])
        configured_pending_runs = os.environ.get(MAX_PENDING_RUNS_ENV)
        if configured_pending_runs is None and persisted.get("maximum_pending_runs") is not None:
            configured_pending_runs = str(persisted["maximum_pending_runs"])
        configured_process_seconds = os.environ.get(MAX_PROCESS_SECONDS_ENV)
        if configured_process_seconds is None and persisted.get("maximum_process_seconds") is not None:
            configured_process_seconds = str(persisted["maximum_process_seconds"])
        runs_root = _absolute_directory(
            str(configured_runs_root) if configured_runs_root else None,
            _default_state_root(),
        )
        return cls(
            runs_root=runs_root,
            cli_executable=cli_executable,
            tracking_root=_configured_directory(
                configured_tracking_root,
                runs_root.parent / "tracking",
                TRACKING_ROOT_ENV,
            ),
            service_state_root=_configured_directory(
                configured_service_state_root,
                runs_root.parent / "service-state",
                SERVICE_STATE_ROOT_ENV,
            ),
            maximum_dataset_bytes=_maximum_dataset_bytes(configured_maximum_bytes),
            maximum_columns=_positive_integer(
                str(persisted["maximum_columns"])
                if persisted.get("maximum_columns") is not None
                else None,
                "persisted maximum_columns",
                DEFAULT_MAXIMUM_COLUMNS,
            ),
            maximum_artifact_references=_positive_integer(
                str(persisted["maximum_artifact_references"])
                if persisted.get("maximum_artifact_references") is not None
                else None,
                "persisted maximum_artifact_references",
                DEFAULT_MAXIMUM_ARTIFACT_REFERENCES,
            ),
            concurrency=_positive_integer(
                str(persisted["concurrency"])
                if persisted.get("concurrency") is not None
                else None,
                "persisted concurrency",
                DEFAULT_CONCURRENCY,
            ),
            maximum_pending_runs=_positive_integer(
                configured_pending_runs,
                MAX_PENDING_RUNS_ENV,
                DEFAULT_MAXIMUM_PENDING_RUNS,
            ),
            maximum_process_seconds=_positive_integer(
                configured_process_seconds,
                MAX_PROCESS_SECONDS_ENV,
                DEFAULT_MAXIMUM_PROCESS_SECONDS,
            ),
        )

    def require_cli_executable(self) -> Path:
        """Return a verified public CLI command or an actionable setup error."""
        if self.cli_executable is None:
            raise SettingsError("GeochemistryPi MCP setup is incomplete. Run 'geochemistrypi-mcp-setup install', or set " f"{CLI_EXECUTABLE_ENV} only for development.")
        if not self.cli_executable.is_absolute() or not self.cli_executable.is_file():
            raise SettingsError(f"Configured GeochemistryPi CLI executable does not exist: {self.cli_executable}")
        return self.cli_executable

    def require_supported_cli(self) -> tuple[Path, str]:
        """Verify the installed distribution version in the CLI's own environment."""
        executable = self.require_cli_executable()
        interpreter = resolve_cli_interpreter(executable)
        command = (
            str(interpreter),
            "-c",
            "from importlib.metadata import version; print(version('geochemistrypi'))",
        )
        process_environment = os.environ.copy()
        for inherited_name in ISOLATED_CLI_ENVIRONMENT_VARIABLES:
            process_environment.pop(inherited_name, None)
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                env=process_environment,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise SettingsError("Cannot verify the installed GeochemistryPi CLI version in its own environment.") from exc
        installed_version = completed.stdout.strip()
        if installed_version not in SUPPORTED_CLI_VERSIONS:
            supported = ", ".join(SUPPORTED_CLI_VERSIONS)
            raise SettingsError(f"Unsupported GeochemistryPi CLI version {installed_version or '<unknown>'}; supported version: {supported}.")
        return executable, installed_version
