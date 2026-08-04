"""One-action local setup, repair, registration, and uninstall workflows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .client_config import STANDARD_CLIENT, SUPPORTED_CLIENTS, ClientAdapter, ClientConfigError, ClientLocations, RegistrationResult, client_adapters
from .constants import CLI_PYTHON_REQUIRES, COMPATIBILITY_POLICY_VERSION, ISOLATED_CLI_ENVIRONMENT_VARIABLES, MCP_PYTHON_REQUIRES, MCP_SDK_REQUIRES, SERVER_VERSION
from .release import ReleaseBundle, ReleaseError, verify_release_bundle
from .settings import (
    DEFAULT_CONCURRENCY,
    DEFAULT_MAXIMUM_ARTIFACT_REFERENCES,
    DEFAULT_MAXIMUM_COLUMNS,
    DEFAULT_MAXIMUM_DATASET_BYTES,
    DEFAULT_MAXIMUM_PENDING_RUNS,
    DEFAULT_MAXIMUM_PROCESS_SECONDS,
    SETTINGS_SCHEMA_VERSION,
    McpSettings,
    SettingsError,
    default_app_root,
)

MCP_PYTHON_VERSION = "3.11"
CLI_PYTHON_VERSION = "3.9"
SETUP_UV_VERSION = "0.11.7"
MANIFEST_SCHEMA_VERSION = 2
LEGACY_MANIFEST_SCHEMA_VERSIONS = (1,)

CLIENT_EXECUTABLE_HINTS: dict[str, tuple[str, ...]] = {
    "codex": ("codex",),
    "claude-code": ("claude",),
    "cursor": ("cursor",),
    "vscode": ("code", "code-insiders"),
    "gemini-cli": ("gemini",),
    "windsurf": ("windsurf",),
    "cline": ("cline",),
    "zed": ("zed",),
    "continue": ("cn",),
    "kiro": ("kiro",),
    "opencode": ("opencode",),
}


class SetupError(RuntimeError):
    """Raised when local installation cannot complete safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _environment_python(environment: Path) -> Path:
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _environment_command(environment: Path, command: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    directory = "Scripts" if os.name == "nt" else "bin"
    return environment / directory / f"{command}{suffix}"


def _rmtree_onerror(_, __, exc_info) -> None:
    """Ignore only files that disappear during a Windows recursive removal."""
    error = exc_info[1]
    if isinstance(error, FileNotFoundError):
        return
    raise error


def _remove_tree(path: Path) -> None:
    """Remove a private environment while preserving meaningful failures."""
    target = path
    if os.name == "nt":
        resolved = str(path.resolve())
        if not resolved.startswith("\\\\?\\"):
            resolved = f"\\\\?\\UNC\\{resolved[2:]}" if resolved.startswith("\\\\") else f"\\\\?\\{resolved}"
        target = Path(resolved)
    shutil.rmtree(target, onerror=_rmtree_onerror)


@dataclass(frozen=True)
class SetupPaths:
    """All installer-owned paths under one platform-native application root."""

    app_root: Path

    @classmethod
    def default(cls) -> "SetupPaths":
        return cls(default_app_root())

    def __post_init__(self) -> None:
        if not self.app_root.is_absolute():
            raise SetupError("The GeochemistryPi MCP application root must be absolute.")

    @property
    def environments_root(self) -> Path:
        return self.app_root / "environments"

    @property
    def mcp_environment(self) -> Path:
        return self.environments_root / "mcp"

    @property
    def cli_environment(self) -> Path:
        return self.environments_root / "cli"

    @property
    def mcp_python(self) -> Path:
        return _environment_python(self.mcp_environment)

    @property
    def cli_python(self) -> Path:
        return _environment_python(self.cli_environment)

    @property
    def server_command(self) -> Path:
        return _environment_command(self.mcp_environment, "geochemistrypi-mcp")

    @property
    def cli_command(self) -> Path:
        return _environment_command(self.cli_environment, "geochemistrypi")

    @property
    def settings_file(self) -> Path:
        return self.app_root / "config" / "settings.json"

    @property
    def manifest_file(self) -> Path:
        return self.app_root / "config" / "install-manifest.json"

    @property
    def standard_client_config(self) -> Path:
        return self.app_root / "config" / "mcp.json"

    @property
    def runs_root(self) -> Path:
        return self.app_root / "runs"

    @property
    def tracking_root(self) -> Path:
        return self.app_root / "tracking"

    @property
    def service_state_root(self) -> Path:
        return self.app_root / "service-state"

    @property
    def release_root(self) -> Path:
        return self.app_root / "release"

    @property
    def release_manifest_file(self) -> Path:
        return self.release_root / "release-manifest.json"

    @property
    def rollback_root(self) -> Path:
        return self.app_root / "rollback"

    @property
    def rollback_environments(self) -> Path:
        return self.rollback_root / "environments"

    @property
    def rollback_release_root(self) -> Path:
        return self.rollback_root / "release"

    @property
    def rollback_settings_file(self) -> Path:
        return self.rollback_root / "settings.json"

    @property
    def rollback_manifest_file(self) -> Path:
        return self.rollback_root / "install-manifest.json"

    @property
    def rollback_metadata_file(self) -> Path:
        return self.rollback_root / "rollback-metadata.json"


@dataclass(frozen=True)
class SourceLayout:
    """Local repository sources used before public package publication."""

    repository_root: Path
    mcp_package_root: Path

    @classmethod
    def discover(cls, start: Path | None = None) -> "SourceLayout":
        seeds = [Path(start or Path.cwd()).resolve(), Path(__file__).resolve()]
        checked: list[Path] = []
        for seed in seeds:
            candidates = (seed, *seed.parents) if seed.is_dir() else (seed.parent, *seed.parents)
            for candidate in candidates:
                if candidate in checked:
                    continue
                checked.append(candidate)
                mcp_root = candidate / "packages" / "geochemistrypi-mcp"
                if (candidate / "pyproject.toml").is_file() and (candidate / "geochemistrypi").is_dir() and (mcp_root / "pyproject.toml").is_file():
                    return cls(candidate, mcp_root)
        locations = ", ".join(str(path) for path in checked[:8])
        raise SetupError(
            "Development source setup must run from a GeochemistryPi repository "
            "clone so both local packages can be installed. Production users "
            "should pass --bundle with a verified release bundle. "
            f"Checked: {locations}"
        )


@dataclass(frozen=True)
class SetupResult:
    """Completed setup state returned to the administrative CLI."""

    action: str
    server_command: Path | None
    runs_root: Path
    clients: tuple[RegistrationResult, ...]
    doctor_healthy: bool
    doctor_summary: str


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
EnvironmentValidator = Callable[[SetupPaths], tuple[str, str]]
DoctorRunner = Callable[[SetupPaths], tuple[bool, str]]
RuntimeInventoryProvider = Callable[[SetupPaths], Mapping[str, object]]


def _default_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    process_environment = os.environ.copy()
    for inherited_name in ISOLATED_CLI_ENVIRONMENT_VARIABLES:
        process_environment.pop(inherited_name, None)
    return subprocess.run(command, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800, env=process_environment)


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_bytes(path: Path, value: bytes, mode: int | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SetupError(f"Cannot parse installer state: {path}") from exc
    if not isinstance(value, dict):
        raise SetupError(f"Installer state must contain a JSON object: {path}")
    return value


def _source_fingerprint(sources: SourceLayout) -> str:
    digest = hashlib.sha256()
    included = [sources.repository_root / "pyproject.toml", sources.mcp_package_root / "pyproject.toml"]
    included.extend(sorted((sources.repository_root / "geochemistrypi").rglob("*.py")))
    included.extend(sorted((sources.mcp_package_root / "src" / "geochemistrypi_mcp").glob("*.py")))
    for path in included:
        digest.update(path.relative_to(sources.repository_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


class SetupManager:
    """Prepare two private environments and register one stable stdio command."""

    def __init__(
        self,
        paths: SetupPaths | None = None,
        sources: SourceLayout | None = None,
        locations: ClientLocations | None = None,
        runner: CommandRunner = _default_runner,
        validator: EnvironmentValidator | None = None,
        doctor: DoctorRunner | None = None,
        runtime_inventory: RuntimeInventoryProvider | None = None,
        bundle: ReleaseBundle | None = None,
        executable_lookup: Callable[[str], str | None] = shutil.which,
    ) -> None:
        self.paths = paths or SetupPaths.default()
        self.sources = sources
        self.locations = locations or ClientLocations.from_environment()
        self.runner = runner
        self.validator = validator or self._validate_environments
        self.doctor = doctor or self._run_doctor
        self.runtime_inventory = runtime_inventory or self._runtime_inventory
        self.bundle = bundle
        self.executable_lookup = executable_lookup

    def _stop_managed_mlflow_ui(self) -> None:
        """Stop only a verified owned UI before replacing its CLI environment."""
        from .tracking_ui import MlflowUiError, MlflowUiManager

        settings = McpSettings(
            runs_root=self.paths.runs_root,
            cli_executable=self.paths.cli_command if self.paths.cli_command.is_file() else None,
            tracking_root=self.paths.tracking_root,
            service_state_root=self.paths.service_state_root,
        )
        try:
            manager = MlflowUiManager(settings)
            status = manager.status()
            if status.state == "ownership_mismatch":
                raise SetupError(
                    "Cannot replace the CLI environment because managed MLflow UI ownership is ambiguous: "
                    f"{status.message}"
                )
            if status.state in {"running", "starting"}:
                manager.stop()
        except MlflowUiError as exc:
            raise SetupError(f"Cannot safely stop the managed MLflow UI: {exc}") from exc

    def _run_checked(self, command: Sequence[str], purpose: str) -> subprocess.CompletedProcess[str]:
        completed = self.runner(tuple(str(part) for part in command))
        if completed.returncode != 0:
            message = " ".join((completed.stderr or completed.stdout).split())[-1000:]
            raise SetupError(f"{purpose} failed: {message or 'command returned a non-zero exit code'}")
        return completed

    def _uv_executable(self) -> Path:
        discovered = self.executable_lookup("uv")
        if not discovered:
            raise SetupError("The local development installer requires uv so it can prepare Python 3.11 and 3.9 without asking for interpreter paths.")
        executable = Path(discovered).resolve()
        version = self._run_checked(
            (str(executable), "--version"),
            "uv version check",
        ).stdout.strip()
        if version.split()[:2] != ["uv", SETUP_UV_VERSION]:
            raise SetupError(
                f"GeochemistryPi MCP setup requires uv {SETUP_UV_VERSION}; "
                f"found {version or '<unknown>'}."
            )
        return executable

    def _prepare_environments(
        self,
        source: SourceLayout | ReleaseBundle,
        clear: bool,
    ) -> None:
        uv = self._uv_executable()
        self.paths.environments_root.mkdir(parents=True, exist_ok=True)
        if isinstance(source, ReleaseBundle):
            mcp_package = source.mcp_wheel
            cli_package = source.cli_wheel
        else:
            mcp_package = source.mcp_package_root
            cli_package = source.repository_root
        for version, environment, python, package, purpose in (
            (MCP_PYTHON_VERSION, self.paths.mcp_environment, self.paths.mcp_python, mcp_package, "MCP environment preparation"),
            (CLI_PYTHON_VERSION, self.paths.cli_environment, self.paths.cli_python, cli_package, "CLI environment preparation"),
        ):
            command = [str(uv), "venv", "--python", version, "--seed"]
            if clear and environment.exists():
                command.append("--clear")
            command.append(str(environment))
            self._run_checked(command, purpose)
            self._run_checked((str(uv), "pip", "install", "--python", str(python), str(package)), f"{purpose} package installation")

    def _validate_environments(self, paths: SetupPaths) -> tuple[str, str]:
        if not paths.server_command.is_file():
            raise SetupError(f"The private MCP command was not installed: {paths.server_command}")
        if not paths.cli_command.is_file():
            raise SetupError(f"The private GeochemistryPi CLI command was not installed: {paths.cli_command}")
        wrapper = self._run_checked(
            (str(paths.mcp_python), "-c", "from importlib.metadata import version; print(version('geochemistrypi-mcp'))"),
            "MCP version handshake",
        ).stdout.strip()
        if wrapper != SERVER_VERSION:
            raise SetupError(f"Installed MCP version {wrapper or '<unknown>'} does not match expected version {SERVER_VERSION}.")
        try:
            _, cli_version = McpSettings(paths.runs_root, paths.cli_command).require_supported_cli()
        except SettingsError as exc:
            raise SetupError(str(exc)) from exc
        return wrapper, cli_version

    def _run_doctor(self, paths: SetupPaths) -> tuple[bool, str]:
        from .doctor import run_doctor

        report = run_doctor(paths)
        return report.healthy, report.summary

    def _runtime_inventory(self, paths: SetupPaths) -> Mapping[str, object]:
        script = (
            "import hashlib,json; from importlib.metadata import distributions; "
            "items=sorted((d.metadata['Name'].lower().replace('_','-').replace('.','-'),d.version) "
            "for d in distributions() if d.metadata.get('Name')); "
            "raw=json.dumps(items,separators=(',',':')).encode(); "
            "print(json.dumps({'count':len(items),'sha256':hashlib.sha256(raw).hexdigest()}))"
        )
        values: dict[str, object] = {}
        for name, interpreter in (
            ("mcp", paths.mcp_python),
            ("cli", paths.cli_python),
        ):
            completed = self._run_checked(
                (str(interpreter), "-c", script),
                f"{name.upper()} runtime inventory",
            )
            try:
                value = json.loads(completed.stdout)
            except json.JSONDecodeError as exc:
                raise SetupError(f"{name.upper()} runtime inventory returned invalid JSON.") from exc
            if (
                not isinstance(value, dict)
                or type(value.get("count")) is not int
                or value["count"] < 1
                or not isinstance(value.get("sha256"), str)
                or len(value["sha256"]) != 64
            ):
                raise SetupError(f"{name.upper()} runtime inventory is incomplete.")
            values[name] = value
        return values

    def _current_manifest(self) -> dict[str, object]:
        manifest = _load_json_object(self.paths.manifest_file)
        supported_schemas = (*LEGACY_MANIFEST_SCHEMA_VERSIONS, MANIFEST_SCHEMA_VERSION)
        if manifest and manifest.get("schema_version") not in supported_schemas:
            raise SetupError(f"Unsupported installer manifest schema in {self.paths.manifest_file}.")
        return manifest

    def _runtime_ready(
        self,
        manifest: Mapping[str, object],
        fingerprint: str,
        installation_source: str,
    ) -> bool:
        checks = (
            manifest.get("schema_version") == MANIFEST_SCHEMA_VERSION,
            manifest.get("server_version") == SERVER_VERSION,
            manifest.get("compatibility_policy_version") == COMPATIBILITY_POLICY_VERSION,
            manifest.get("source_fingerprint") == fingerprint,
            manifest.get("installation_source") == installation_source,
            self.paths.server_command.is_file(),
            self.paths.cli_command.is_file(),
            self.paths.settings_file.is_file(),
        )
        return all(checks)

    def _activate_release_bundle(self, bundle: ReleaseBundle | None) -> None:
        if bundle is None:
            if self.paths.release_root.exists():
                _remove_tree(self.paths.release_root)
            return
        self.paths.app_root.mkdir(parents=True, exist_ok=True)
        staging = self.paths.app_root / f".release-{uuid.uuid4().hex}"
        staging.mkdir()
        try:
            for source in bundle.files:
                shutil.copy2(source, staging / source.name)
            if self.paths.release_root.exists():
                _remove_tree(self.paths.release_root)
            os.replace(staging, self.paths.release_root)
        finally:
            if staging.exists():
                _remove_tree(staging)

    def _assert_owned_path(self, path: Path) -> None:
        try:
            path.resolve().relative_to(self.paths.app_root.resolve())
        except ValueError as exc:
            raise SetupError(
                f"Refusing to modify a path outside the application root: {path}"
            ) from exc

    def _begin_runtime_transaction(self, *, persistent: bool) -> Path:
        """Move the current runtime aside before any destructive replacement."""
        self.paths.app_root.mkdir(parents=True, exist_ok=True)
        prefix = ".upgrade-rollback" if persistent else ".setup-recovery"
        backup = self.paths.app_root / f"{prefix}-{uuid.uuid4().hex}"
        self._assert_owned_path(backup)
        if backup.exists():
            _remove_tree(backup)
        backup.mkdir(parents=True)
        metadata = {
            "schema_version": 1,
            "created_at": _utc_now(),
            "persistent": persistent,
            "environments_present": self.paths.environments_root.is_dir(),
            "release_present": self.paths.release_root.is_dir(),
            "settings_present": self.paths.settings_file.is_file(),
            "manifest_present": self.paths.manifest_file.is_file(),
        }
        if persistent and not metadata["environments_present"]:
            _remove_tree(backup)
            raise SetupError("Upgrade requires an existing private runtime to preserve for rollback.")
        _atomic_write_json(backup / "rollback-metadata.json", metadata)
        try:
            if metadata["environments_present"]:
                os.replace(self.paths.environments_root, backup / "environments")
            if metadata["release_present"]:
                os.replace(self.paths.release_root, backup / "release")
            if metadata["settings_present"]:
                shutil.copy2(self.paths.settings_file, backup / "settings.json")
            if metadata["manifest_present"]:
                shutil.copy2(
                    self.paths.manifest_file,
                    backup / "install-manifest.json",
                )
        except OSError as exc:
            self._restore_runtime_transaction(backup)
            raise SetupError(f"Cannot preserve the current runtime before replacement: {exc}") from exc
        return backup

    def _restore_runtime_transaction(self, backup: Path) -> None:
        """Restore paths moved by a failed install, repair, or upgrade."""
        self._assert_owned_path(backup)
        metadata_path = backup / "rollback-metadata.json"
        metadata = _load_json_object(metadata_path) if metadata_path.is_file() else {}
        for present_key, saved, active in (
            (
                "environments_present",
                backup / "environments",
                self.paths.environments_root,
            ),
            ("release_present", backup / "release", self.paths.release_root),
        ):
            if saved.is_dir():
                if active.exists():
                    _remove_tree(active)
                os.replace(saved, active)
            elif metadata.get(present_key) is False and active.exists():
                _remove_tree(active)
        for present_key, saved, active in (
            ("settings_present", backup / "settings.json", self.paths.settings_file),
            ("manifest_present", backup / "install-manifest.json", self.paths.manifest_file),
        ):
            if saved.is_file():
                active.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(saved, active)
            elif metadata.get(present_key) is False and active.exists():
                active.unlink()
        if backup.exists():
            _remove_tree(backup)

    def _commit_runtime_transaction(
        self,
        backup: Path,
        *,
        persistent: bool,
    ) -> Path | None:
        self._assert_owned_path(backup)
        if persistent:
            retired = self.paths.app_root / f".rollback-retired-{uuid.uuid4().hex}"
            self._assert_owned_path(retired)
            if self.paths.rollback_root.exists():
                os.replace(self.paths.rollback_root, retired)
            try:
                os.replace(backup, self.paths.rollback_root)
            except OSError:
                if retired.exists():
                    os.replace(retired, self.paths.rollback_root)
                raise
            if retired.exists():
                _remove_tree(retired)
            return self.paths.rollback_root
        if backup.exists():
            _remove_tree(backup)
        return None

    def _rollback_snapshot_available(self) -> bool:
        metadata = _load_json_object(self.paths.rollback_metadata_file)
        if not metadata:
            return False
        return all(
            (
                metadata.get("schema_version") == 1,
                metadata.get("persistent") is True,
                metadata.get("environments_present") is True,
                self.paths.rollback_environments.is_dir(),
                not metadata.get("settings_present")
                or self.paths.rollback_settings_file.is_file(),
                not metadata.get("manifest_present")
                or self.paths.rollback_manifest_file.is_file(),
            )
        )

    def _executable_map(self) -> dict[str, Path | None]:
        value = self.executable_lookup("claude")
        return {"claude": Path(value).resolve() if value else None}

    def _auto_clients(self, adapters: Mapping[str, ClientAdapter]) -> tuple[str, ...]:
        selected = [STANDARD_CLIENT]
        for name in SUPPORTED_CLIENTS:
            if name == STANDARD_CLIENT or name not in adapters:
                continue
            adapter = adapters[name]
            detected_by_path = any(path.exists() for path in adapter.detection_paths)
            detected_by_command = any(self.executable_lookup(command) for command in CLIENT_EXECUTABLE_HINTS.get(name, ()))
            if detected_by_path or detected_by_command:
                selected.append(name)
        return tuple(dict.fromkeys(selected))

    def _resolve_clients(self, requested: Sequence[str], adapters: Mapping[str, ClientAdapter]) -> tuple[str, ...]:
        normalized = tuple(requested or ("auto",))
        if "auto" in normalized:
            if len(normalized) != 1:
                raise SetupError("Client selection 'auto' cannot be combined with explicit client names.")
            return self._auto_clients(adapters)
        if "all" in normalized:
            if len(normalized) != 1:
                raise SetupError("Client selection 'all' cannot be combined with explicit client names.")
            selected = tuple(name for name in SUPPORTED_CLIENTS if name in adapters)
        else:
            unknown = sorted(set(normalized) - set(SUPPORTED_CLIENTS))
            if unknown:
                raise SetupError(f"Unknown MCP clients: {unknown}")
            selected = tuple(dict.fromkeys((STANDARD_CLIENT, *normalized)))
        unavailable = sorted(name for name in selected if name not in adapters)
        if unavailable:
            raise SetupError(f"Requested clients are unavailable on this machine: {unavailable}")
        return selected

    def _register_clients(self, clients: Sequence[str], replace: bool) -> tuple[RegistrationResult, ...]:
        executable_map = self._executable_map()
        adapters = client_adapters(self.paths.standard_client_config, self.locations, executable_map, self.runner)
        selected = self._resolve_clients(clients, adapters)
        results = []
        ordered = tuple(name for name in selected if name != "claude-code")
        if "claude-code" in selected:
            ordered = (*ordered, "claude-code")
        for name in ordered:
            adapter = adapters[name]
            results.append(adapter.register(self.paths.server_command, replace=replace))
        return tuple(results)

    def _selected_clients(self, clients: Sequence[str]) -> tuple[str, ...]:
        adapters = client_adapters(
            self.paths.standard_client_config,
            self.locations,
            self._executable_map(),
            self.runner,
        )
        return self._resolve_clients(clients, adapters)

    def _snapshot_client_files(
        self,
        clients: Sequence[str],
    ) -> dict[Path, tuple[bool, bytes, int | None]]:
        adapters = client_adapters(
            self.paths.standard_client_config,
            self.locations,
            self._executable_map(),
            self.runner,
        )
        snapshots: dict[Path, tuple[bool, bytes, int | None]] = {}
        for name in clients:
            adapter = adapters[name]
            path = getattr(adapter, "path", None)
            if not isinstance(path, Path) or path in snapshots:
                continue
            try:
                snapshots[path] = (
                    path.is_file(),
                    path.read_bytes() if path.is_file() else b"",
                    path.stat().st_mode if path.is_file() else None,
                )
            except OSError as exc:
                raise SetupError(
                    f"Cannot snapshot {name} configuration before registration: {path}"
                ) from exc
        return snapshots

    def _restore_client_files(
        self,
        snapshots: Mapping[Path, tuple[bool, bytes, int | None]],
    ) -> None:
        for path, (existed, content, mode) in snapshots.items():
            if existed:
                _atomic_write_bytes(path, content, mode)
            elif path.exists():
                path.unlink()

    def install(
        self,
        clients: Sequence[str] = ("auto",),
        repair: bool = False,
        upgrade: bool = False,
    ) -> SetupResult:
        """Install, repair, or transactionally upgrade the private runtime."""
        if repair and upgrade:
            raise SetupError("Repair and upgrade are separate lifecycle actions.")
        if upgrade and self.bundle is None:
            raise SetupError("Upgrade requires one verified release bundle.")
        manifest = self._current_manifest()
        if self.bundle is not None:
            source: SourceLayout | ReleaseBundle = self.bundle
        elif (
            manifest.get("installation_source") == "release-bundle"
            and self.paths.release_manifest_file.is_file()
        ):
            try:
                active_bundle = verify_release_bundle(
                    self.paths.release_root,
                    require_signatures=False,
                )
            except ReleaseError as exc:
                raise SetupError(
                    "The installed release bundle is damaged; provide the original "
                    f"verified bundle to repair it: {exc}"
                ) from exc
            source = replace(
                active_bundle,
                signatures_verified=manifest.get("signatures_verified") is True,
            )
        else:
            source = self.sources or SourceLayout.discover()
        if isinstance(source, ReleaseBundle):
            fingerprint = source.fingerprint
            installation_source = "release-bundle"
        else:
            fingerprint = _source_fingerprint(source)
            installation_source = "source"
        if upgrade and not manifest:
            raise SetupError("Upgrade requires an existing installation manifest.")
        if upgrade and (
            manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
            or not isinstance(manifest.get("source_fingerprint"), str)
            or not isinstance(manifest.get("installation_source"), str)
            or not self._runtime_ready(
                manifest,
                str(manifest.get("source_fingerprint", "")),
                str(manifest.get("installation_source", "")),
            )
        ):
            raise SetupError(
                "Upgrade requires a healthy current-schema installation; run "
                "repair before selecting a new release bundle."
            )
        if upgrade:
            current_healthy, current_summary = self.doctor(self.paths)
            if not current_healthy:
                raise SetupError(
                    "Upgrade preflight doctor failed; repair the current runtime "
                    f"before upgrading: {current_summary}"
                )
        if upgrade and manifest.get("source_fingerprint") == fingerprint:
            raise SetupError("The selected release bundle is already installed; no upgrade was performed.")
        if (
            manifest
            and isinstance(source, ReleaseBundle)
            and manifest.get("source_fingerprint") != fingerprint
            and not upgrade
        ):
            raise SetupError(
                "A different release bundle must use the upgrade action so the "
                "current runtime is retained for rollback."
            )
        prior_clients = tuple(str(value) for value in manifest.get("registered_clients", ()))
        requested_clients = (
            prior_clients or (STANDARD_CLIENT,)
            if upgrade
            else clients
        )
        refresh_required = repair or upgrade or not self._runtime_ready(
            manifest,
            fingerprint,
            installation_source,
        )
        transaction: Path | None = None
        selected_clients = self._selected_clients(requested_clients)
        registered_clients = tuple(dict.fromkeys((*prior_clients, *selected_clients)))
        registrations: tuple[RegistrationResult, ...] = ()
        try:
            if refresh_required:
                self._stop_managed_mlflow_ui()
                transaction = self._begin_runtime_transaction(persistent=upgrade)
                if (
                    isinstance(source, ReleaseBundle)
                    and source.directory == self.paths.release_root.resolve()
                ):
                    preserved_release = transaction / "release"
                    source = replace(
                        source,
                        directory=preserved_release,
                        manifest_path=preserved_release / source.manifest_path.name,
                        cli_wheel=preserved_release / source.cli_wheel.name,
                        mcp_wheel=preserved_release / source.mcp_wheel.name,
                    )
                self._prepare_environments(source, clear=False)
            wrapper_version, cli_version = self.validator(self.paths)
            inventory = self.runtime_inventory(self.paths)
            self.paths.runs_root.mkdir(parents=True, exist_ok=True)
            self.paths.tracking_root.mkdir(parents=True, exist_ok=True)
            self.paths.service_state_root.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(
                self.paths.settings_file,
                {
                    "schema_version": SETTINGS_SCHEMA_VERSION,
                    "cli_executable": str(self.paths.cli_command),
                    "runs_root": str(self.paths.runs_root),
                    "tracking_root": str(self.paths.tracking_root),
                    "service_state_root": str(self.paths.service_state_root),
                    "maximum_dataset_bytes": DEFAULT_MAXIMUM_DATASET_BYTES,
                    "maximum_columns": DEFAULT_MAXIMUM_COLUMNS,
                    "maximum_artifact_references": DEFAULT_MAXIMUM_ARTIFACT_REFERENCES,
                    "concurrency": DEFAULT_CONCURRENCY,
                    "maximum_pending_runs": DEFAULT_MAXIMUM_PENDING_RUNS,
                    "maximum_process_seconds": DEFAULT_MAXIMUM_PROCESS_SECONDS,
                },
            )
            self._activate_release_bundle(
                source if isinstance(source, ReleaseBundle) else None
            )
            manifest_value: dict[str, object] = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "installed_at": manifest.get("installed_at", _utc_now()),
                "updated_at": _utc_now(),
                "server_version": wrapper_version,
                "cli_version": cli_version,
                "compatibility_policy_version": COMPATIBILITY_POLICY_VERSION,
                "mcp_python_requires": MCP_PYTHON_REQUIRES,
                "cli_python_requires": CLI_PYTHON_REQUIRES,
                "mcp_sdk_requires": MCP_SDK_REQUIRES,
                "server_command": str(self.paths.server_command),
                "runs_root": str(self.paths.runs_root),
                "tracking_root": str(self.paths.tracking_root),
                "service_state_root": str(self.paths.service_state_root),
                "installation_source": installation_source,
                "source_fingerprint": fingerprint,
                "runtime_inventory": inventory,
                "release_manifest_sha256": source.manifest_sha256
                if isinstance(source, ReleaseBundle)
                else None,
                "release_id": source.release_id
                if isinstance(source, ReleaseBundle)
                else None,
                "release_tag": source.release_tag
                if isinstance(source, ReleaseBundle)
                else None,
                "release_artifacts": list(source.manifest["artifacts"])
                if isinstance(source, ReleaseBundle)
                else [],
                "signatures_verified": source.signatures_verified
                if isinstance(source, ReleaseBundle)
                else False,
                "signature_policy": "verified"
                if isinstance(source, ReleaseBundle) and source.signatures_verified
                else (
                    "explicit-development-override"
                    if isinstance(source, ReleaseBundle)
                    else "not-applicable"
                ),
                "rollback_available": (
                    False if upgrade else self._rollback_snapshot_available()
                ),
                "registered_clients": list(registered_clients),
            }
            _atomic_write_json(self.paths.manifest_file, manifest_value)
            doctor_healthy, doctor_summary = self.doctor(self.paths)
            if not doctor_healthy:
                raise SetupError(
                    f"Installation completed but doctor failed: {doctor_summary}"
                )
        except Exception as exc:
            if transaction is not None and transaction.exists():
                self._restore_runtime_transaction(transaction)
            raise
        if transaction is not None:
            try:
                committed = self._commit_runtime_transaction(
                    transaction,
                    persistent=upgrade,
                )
            except Exception as exc:
                recovery = (
                    transaction
                    if transaction.exists()
                    else self.paths.rollback_root
                )
                if recovery.exists():
                    self._restore_runtime_transaction(recovery)
                raise SetupError(
                    "The replacement runtime passed validation but its rollback "
                    "snapshot could not be finalized; the prior runtime was restored."
                ) from exc
            if committed is not None:
                manifest_value["rollback_available"] = True
                manifest_value["updated_at"] = _utc_now()
                _atomic_write_json(self.paths.manifest_file, manifest_value)
                doctor_healthy, doctor_summary = self.doctor(self.paths)
                if not doctor_healthy:
                    self._restore_runtime_transaction(committed)
                    raise SetupError(
                        "Upgrade finalized but the rollback-aware doctor check "
                        f"failed; the prior runtime was restored: {doctor_summary}"
                    )
        client_snapshots = self._snapshot_client_files(selected_clients)
        try:
            registrations = self._register_clients(
                selected_clients,
                replace=repair or upgrade,
            )
        except Exception as exc:
            self._restore_client_files(client_snapshots)
            if upgrade and self.paths.rollback_root.exists():
                self._restore_runtime_transaction(self.paths.rollback_root)
            raise SetupError(
                "Client registration failed after the private runtime passed Doctor. "
                "Client files were restored; rerun setup after resolving the reported "
                f"client issue: {exc}"
            ) from exc
        action = "upgrade" if upgrade else ("repair" if repair else "install")
        return SetupResult(
            action,
            self.paths.server_command,
            self.paths.runs_root,
            registrations,
            True,
            doctor_summary,
        )

    def rollback(self) -> SetupResult:
        """Restore the one retained pre-upgrade runtime as an atomic lifecycle action."""
        if not self._rollback_snapshot_available():
            raise SetupError(
                "No complete rollback snapshot is available. Rollback is created "
                "only after a successful upgrade."
            )
        rollback_manifest = _load_json_object(self.paths.rollback_manifest_file)
        if rollback_manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise SetupError(
                "The rollback snapshot uses an unsupported installer schema and "
                "cannot be restored safely."
            )
        self._stop_managed_mlflow_ui()
        current_backup = self._begin_runtime_transaction(persistent=False)
        registrations: tuple[RegistrationResult, ...] = ()
        client_snapshots: dict[Path, tuple[bool, bytes, int | None]] = {}
        try:
            os.replace(
                self.paths.rollback_environments,
                self.paths.environments_root,
            )
            if self.paths.rollback_release_root.is_dir():
                os.replace(
                    self.paths.rollback_release_root,
                    self.paths.release_root,
                )
            shutil.copy2(
                self.paths.rollback_settings_file,
                self.paths.settings_file,
            )
            shutil.copy2(
                self.paths.rollback_manifest_file,
                self.paths.manifest_file,
            )
            wrapper_version, cli_version = self.validator(self.paths)
            inventory = self.runtime_inventory(self.paths)
            restored_manifest = _load_json_object(self.paths.manifest_file)
            restored_manifest.update(
                {
                    "updated_at": _utc_now(),
                    "server_version": wrapper_version,
                    "cli_version": cli_version,
                    "runtime_inventory": inventory,
                    "rollback_available": False,
                }
            )
            registered = tuple(
                str(value)
                for value in restored_manifest.get("registered_clients", ())
            )
            _atomic_write_json(self.paths.manifest_file, restored_manifest)
            doctor_healthy, doctor_summary = self.doctor(self.paths)
            if not doctor_healthy:
                raise SetupError(f"Rollback doctor failed: {doctor_summary}")
            selected_clients = self._selected_clients(
                registered or (STANDARD_CLIENT,)
            )
            client_snapshots = self._snapshot_client_files(selected_clients)
            registrations = self._register_clients(
                selected_clients,
                replace=True,
            )
        except Exception:
            if client_snapshots:
                self._restore_client_files(client_snapshots)
            self.paths.rollback_root.mkdir(parents=True, exist_ok=True)
            for active, saved in (
                (self.paths.environments_root, self.paths.rollback_environments),
                (self.paths.release_root, self.paths.rollback_release_root),
            ):
                if active.exists():
                    if saved.exists():
                        _remove_tree(saved)
                    os.replace(active, saved)
            self._restore_runtime_transaction(current_backup)
            raise
        self._commit_runtime_transaction(current_backup, persistent=False)
        if self.paths.rollback_root.exists():
            _remove_tree(self.paths.rollback_root)
        return SetupResult(
            "rollback",
            self.paths.server_command,
            self.paths.runs_root,
            registrations,
            True,
            doctor_summary,
        )

    def uninstall(self) -> SetupResult:
        """Remove installer-owned runtimes and client entries while preserving runs."""
        self._stop_managed_mlflow_ui()
        manifest = self._current_manifest()
        registered = tuple(str(value) for value in manifest.get("registered_clients", ()))
        executable_map = self._executable_map()
        adapters = client_adapters(self.paths.standard_client_config, self.locations, executable_map, self.runner)
        results = []
        for name in tuple(dict.fromkeys((STANDARD_CLIENT, *registered))):
            adapter = adapters.get(name)
            if adapter is not None:
                results.append(adapter.unregister(self.paths.server_command))
        for owned_root in (
            self.paths.environments_root,
            self.paths.release_root,
            self.paths.rollback_root,
        ):
            self._assert_owned_path(owned_root)
            if owned_root.exists():
                _remove_tree(owned_root)
        for path in (self.paths.settings_file, self.paths.manifest_file):
            if path.exists():
                path.unlink()
        return SetupResult("uninstall", None, self.paths.runs_root, tuple(results), True, "Private runtimes and owned client entries were removed; run and MLflow tracking data were preserved.")

    def standard_config(self) -> str:
        if not self.paths.standard_client_config.is_file():
            raise SetupError("Standard MCP fallback configuration does not exist; run install first.")
        return self.paths.standard_client_config.read_text(encoding="utf-8")


def _add_bundle_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--bundle",
        type=Path,
        help="Verified release-bundle directory. Omit only for repository development setup or repair of an installed bundle.",
    )
    command.add_argument(
        "--allow-unsigned-bundle",
        action="store_true",
        help="Explicitly allow an unsigned local release candidate; never use for a public release.",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geochemistrypi-mcp-setup",
        description="Prepare and register the local GeochemistryPi MCP runtime.",
    )
    subcommands = parser.add_subparsers(dest="action", required=True)
    for action in ("install", "repair"):
        command = subcommands.add_parser(action, help=f"{action.title()} the private runtime and client registration.")
        command.add_argument(
            "--client",
            action="append",
            choices=("auto", "all", *SUPPORTED_CLIENTS),
            help="Register an explicit client; repeat for several clients. Defaults to safe auto-detection plus a standard JSON fallback.",
        )
        _add_bundle_arguments(command)
    upgrade = subcommands.add_parser(
        "upgrade",
        help="Upgrade transactionally from a verified bundle and retain one rollback snapshot.",
    )
    upgrade.add_argument("--bundle", type=Path, required=True)
    upgrade.add_argument(
        "--allow-unsigned-bundle",
        action="store_true",
        help="Explicitly allow an unsigned local release candidate; never use for a public release.",
    )
    subcommands.add_parser(
        "rollback",
        help="Restore the one pre-upgrade runtime retained after a successful upgrade.",
    )
    subcommands.add_parser("uninstall", help="Remove private runtimes and owned client entries while preserving run data.")
    subcommands.add_parser("print-config", help="Print the generated standard mcpServers JSON fallback.")
    return parser


def _running_inside_private_mcp(paths: SetupPaths) -> bool:
    if os.name != "nt":
        return False
    try:
        Path(sys.executable).resolve().relative_to(paths.mcp_environment.resolve())
    except ValueError:
        return False
    return True


def _windows_external_bootstrap_guidance(
    paths: SetupPaths,
    arguments: Sequence[str],
) -> str:
    """Return a copyable command that cannot hold private-environment DLL locks."""
    action = arguments[0] if arguments else "<action>"
    if action == "upgrade":
        return (
            "Bootstrap upgrade from the new signed MCP wheel with the documented "
            "uvx --from command."
        )
    try:
        active_bundle = verify_release_bundle(
            paths.release_root,
            require_signatures=False,
        )
    except ReleaseError:
        return (
            "Rerun the same action from the repository development bootstrap, "
            "because this source installation has no retained release wheel."
        )
    requirement = (
        "geochemistrypi-mcp[release] @ "
        f"{active_bundle.mcp_wheel.resolve().as_uri()}"
    )
    rendered_arguments = subprocess.list2cmdline(tuple(arguments))
    return (
        f'uv run --isolated --no-project --python {MCP_PYTHON_VERSION} '
        f'--with "{requirement}" geochemistrypi-mcp-setup '
        f"{rendered_arguments}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the administrative lifecycle command without starting MCP stdio."""
    raw_arguments = tuple(argv) if argv is not None else tuple(sys.argv[1:])
    arguments = _parser().parse_args(raw_arguments)
    try:
        default_paths = SetupPaths.default()
        if (
            arguments.action != "print-config"
            and _running_inside_private_mcp(default_paths)
        ):
            guidance = _windows_external_bootstrap_guidance(
                default_paths, raw_arguments
            )
            raise SetupError(
                "Windows cannot safely replace or remove the private environment "
                "from a process running inside it. No files were changed. "
                f"{guidance}"
            )
        bundle_path = getattr(arguments, "bundle", None)
        allow_unsigned = getattr(arguments, "allow_unsigned_bundle", False)
        if allow_unsigned and bundle_path is None:
            raise SetupError("--allow-unsigned-bundle requires --bundle.")
        bundle = (
            verify_release_bundle(
                bundle_path,
                require_signatures=not allow_unsigned,
            )
            if bundle_path is not None
            else None
        )
        manager = SetupManager(paths=default_paths, bundle=bundle)
        if arguments.action in {"install", "repair", "upgrade"}:
            result = manager.install(
                getattr(arguments, "client", None) or ("auto",),
                repair=arguments.action == "repair",
                upgrade=arguments.action == "upgrade",
            )
            print(f"GeochemistryPi MCP {result.action} succeeded.")
            print(f"Server command: {result.server_command}")
            print(f"Runs directory: {result.runs_root}")
            print(f"Registered clients: {', '.join(item.client for item in result.clients)}")
            print(result.doctor_summary)
        elif arguments.action == "rollback":
            result = manager.rollback()
            print("GeochemistryPi MCP rollback succeeded.")
            print(f"Server command: {result.server_command}")
            print(result.doctor_summary)
        elif arguments.action == "uninstall":
            result = manager.uninstall()
            print(result.doctor_summary)
        else:
            print(manager.standard_config(), end="")
    except (ClientConfigError, ReleaseError, SetupError, SettingsError) as exc:
        print(f"GeochemistryPi MCP setup failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
