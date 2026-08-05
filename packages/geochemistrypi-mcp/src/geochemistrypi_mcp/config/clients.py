"""Client-neutral MCP registration with atomic, recoverable config updates."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, MutableSequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

import tomlkit
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

SERVER_ID = "geochemistrypi"
STANDARD_CLIENT = "standard"
SUPPORTED_CLIENTS = (
    STANDARD_CLIENT,
    "codex",
    "claude-desktop",
    "claude-code",
    "cursor",
    "vscode",
    "gemini-cli",
    "windsurf",
    "cline",
    "roo-code",
    "zed",
    "continue",
    "kiro",
    "opencode",
)


class ClientConfigError(RuntimeError):
    """Raised when a client configuration cannot be updated safely."""


@dataclass(frozen=True)
class ClientLocations:
    """Platform locations used by client adapters and deterministic tests."""

    home: Path
    appdata: Path | None
    xdg_config_home: Path | None
    platform: str
    system: str | None = None

    @classmethod
    def from_environment(cls) -> "ClientLocations":
        return cls(
            home=Path.home().resolve(),
            appdata=Path(os.environ["APPDATA"]).resolve() if os.environ.get("APPDATA") else None,
            xdg_config_home=Path(os.environ["XDG_CONFIG_HOME"]).resolve() if os.environ.get("XDG_CONFIG_HOME") else None,
            platform=os.name,
            system=sys_platform(),
        )


@dataclass(frozen=True)
class RegistrationResult:
    """One client registration outcome."""

    client: str
    target: str
    changed: bool


class ClientAdapter(Protocol):
    """Common interface implemented by every supported client registration."""

    name: str
    detection_paths: tuple[Path, ...]

    def register(self, command: Path, replace: bool = False) -> RegistrationResult:
        ...

    def unregister(self, command: Path | None = None) -> RegistrationResult:
        ...


def backup_path(path: Path) -> Path:
    """Return the single recoverable backup path owned by this installer."""
    return path.with_name(f"{path.name}.geochemistrypi.bak")


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _create_backup_once(path: Path) -> None:
    destination = backup_path(path)
    if not path.exists() or destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(handle)
    temporary_path = Path(temporary_name)
    try:
        shutil.copy2(path, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _server_definition(command: Path, include_type: bool = False) -> dict[str, object]:
    definition: dict[str, object] = {"command": str(command), "args": []}
    if include_type:
        definition = {"type": "stdio", **definition}
    return definition


DefinitionFactory = Callable[[Path], dict[str, object]]


class JsonClientAdapter:
    """Update one JSON MCP client while preserving unrelated keys."""

    def __init__(
        self,
        name: str,
        path: Path,
        root_key: str | Sequence[str],
        include_type: bool = False,
        definition_factory: DefinitionFactory | None = None,
        detection_paths: Sequence[Path] | None = None,
    ) -> None:
        self.name = name
        self.path = path
        self.root_path = (root_key,) if isinstance(root_key, str) else tuple(root_key)
        if not self.root_path:
            raise ValueError("A JSON client root path is required.")
        self.definition_factory = definition_factory or (lambda command: _server_definition(command, include_type))
        self.detection_paths = tuple(detection_paths or (path.parent,))

    @property
    def root_key(self) -> str:
        """Retain the original single-key attribute for compatible callers."""
        return self.root_path[0]

    def _load(self) -> dict[str, object]:
        if not self.path.exists():
            return {}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ClientConfigError(f"Cannot parse {self.name} configuration: {self.path}") from exc
        if not isinstance(value, dict):
            raise ClientConfigError(f"{self.name} configuration must contain a JSON object: {self.path}")
        return value

    def _servers(self, document: dict[str, object], create: bool) -> dict[str, object] | None:
        current = document
        for key in self.root_path:
            value = current.get(key)
            if value is None:
                if not create:
                    return None
                value = {}
                current[key] = value
            if not isinstance(value, dict):
                dotted = ".".join(self.root_path)
                raise ClientConfigError(f"{self.name} configuration key {dotted!r} must contain an object.")
            current = value
        return current

    def register(self, command: Path, replace: bool = False) -> RegistrationResult:
        document = self._load()
        servers = self._servers(document, create=True)
        assert servers is not None
        expected = self.definition_factory(command)
        existing = servers.get(SERVER_ID)
        if existing == expected:
            return RegistrationResult(self.name, str(self.path), False)
        if existing is not None and not replace:
            raise ClientConfigError(f"{self.name} already defines {SERVER_ID!r} with a different command; " "run repair to replace only that entry.")
        _create_backup_once(self.path)
        servers[SERVER_ID] = expected
        _atomic_write(self.path, json.dumps(document, indent=2, ensure_ascii=False) + "\n")
        return RegistrationResult(self.name, str(self.path), True)

    def unregister(self, command: Path | None = None) -> RegistrationResult:
        document = self._load()
        servers = self._servers(document, create=False)
        if servers is None or SERVER_ID not in servers:
            return RegistrationResult(self.name, str(self.path), False)
        if command is not None and servers[SERVER_ID] != self.definition_factory(command):
            return RegistrationResult(self.name, str(self.path), False)
        _create_backup_once(self.path)
        del servers[SERVER_ID]
        _atomic_write(self.path, json.dumps(document, indent=2, ensure_ascii=False) + "\n")
        return RegistrationResult(self.name, str(self.path), True)


class OpenCodeClientAdapter(JsonClientAdapter):
    """Write OpenCode JSON without silently destroying a JSONC configuration."""

    def __init__(self, path: Path, detection_paths: Sequence[Path]) -> None:
        super().__init__(
            "opencode",
            path,
            ("mcp", "servers"),
            definition_factory=lambda command: {"type": "local", "command": [str(command)]},
            detection_paths=detection_paths,
        )
        self.jsonc_path = path.with_suffix(".jsonc")

    def register(self, command: Path, replace: bool = False) -> RegistrationResult:
        if self.jsonc_path.exists() and not self.path.exists():
            raise ClientConfigError(
                f"OpenCode uses JSONC at {self.jsonc_path}; setup will not rewrite comments. " "Use OpenCode's MCP settings to add the server manually, or create an OpenCode JSON config."
            )
        return super().register(command, replace)

    def unregister(self, command: Path | None = None) -> RegistrationResult:
        if not self.path.exists():
            return RegistrationResult(self.name, str(self.path), False)
        return super().unregister(command)


class CodexClientAdapter:
    """Update Codex's shared config.toml without disturbing other settings."""

    name = "codex"

    def __init__(self, path: Path) -> None:
        self.path = path
        self.detection_paths = (path.parent,)

    def _load(self):
        if not self.path.exists():
            return tomlkit.document()
        try:
            return tomlkit.parse(self.path.read_text(encoding="utf-8"))
        except (OSError, tomlkit.exceptions.ParseError) as exc:
            raise ClientConfigError(f"Cannot parse Codex configuration: {self.path}") from exc

    @staticmethod
    def _definition(command: Path) -> dict[str, object]:
        return {"command": str(command), "args": [], "startup_timeout_sec": 30}

    def register(self, command: Path, replace: bool = False) -> RegistrationResult:
        document = self._load()
        servers = document.get("mcp_servers")
        if servers is None:
            servers = tomlkit.table()
            document["mcp_servers"] = servers
        if not isinstance(servers, Mapping):
            raise ClientConfigError("Codex configuration key 'mcp_servers' must contain a TOML table.")
        expected = self._definition(command)
        existing = servers.get(SERVER_ID)
        if existing is not None and dict(existing) == expected:
            return RegistrationResult(self.name, str(self.path), False)
        if existing is not None and not replace:
            raise ClientConfigError(f"Codex already defines {SERVER_ID!r} with a different command; " "run repair to replace only that table.")
        _create_backup_once(self.path)
        server = tomlkit.table()
        for key, value in expected.items():
            server[key] = value
        servers[SERVER_ID] = server
        _atomic_write(self.path, tomlkit.dumps(document))
        return RegistrationResult(self.name, str(self.path), True)

    def unregister(self, command: Path | None = None) -> RegistrationResult:
        document = self._load()
        servers = document.get("mcp_servers")
        if not isinstance(servers, Mapping) or SERVER_ID not in servers:
            return RegistrationResult(self.name, str(self.path), False)
        if command is not None and dict(servers[SERVER_ID]) != self._definition(command):
            return RegistrationResult(self.name, str(self.path), False)
        _create_backup_once(self.path)
        del servers[SERVER_ID]
        _atomic_write(self.path, tomlkit.dumps(document))
        return RegistrationResult(self.name, str(self.path), True)


class ContinueClientAdapter:
    """Round-trip Continue's YAML config while preserving comments and ordering."""

    name = "continue"

    def __init__(self, path: Path) -> None:
        self.path = path
        self.detection_paths = (path.parent,)
        self.yaml = YAML(typ="rt")
        self.yaml.preserve_quotes = True
        self.yaml.indent(mapping=2, sequence=4, offset=2)

    @staticmethod
    def _definition(command: Path) -> dict[str, object]:
        return {"name": SERVER_ID, "command": str(command), "args": []}

    def _load(self):
        if not self.path.exists():
            return {"name": "Local Config", "version": "1.0.0", "schema": "v1"}
        try:
            document = self.yaml.load(self.path.read_text(encoding="utf-8"))
        except (OSError, YAMLError) as exc:
            raise ClientConfigError(f"Cannot parse Continue configuration: {self.path}") from exc
        if document is None:
            return {}
        if not isinstance(document, Mapping):
            raise ClientConfigError(f"Continue configuration must contain a YAML mapping: {self.path}")
        return document

    def _servers(self, document, create: bool):
        servers = document.get("mcpServers")
        if servers is None:
            if not create:
                return None
            servers = []
            document["mcpServers"] = servers
        if not isinstance(servers, MutableSequence):
            raise ClientConfigError("Continue configuration key 'mcpServers' must contain a YAML sequence.")
        return servers

    def _dump(self, document) -> str:
        stream = io.StringIO()
        self.yaml.dump(document, stream)
        return stream.getvalue()

    def _matching_indexes(self, servers) -> list[int]:
        return [index for index, value in enumerate(servers) if isinstance(value, Mapping) and value.get("name") == SERVER_ID]

    def register(self, command: Path, replace: bool = False) -> RegistrationResult:
        document = self._load()
        servers = self._servers(document, create=True)
        matches = self._matching_indexes(servers)
        if len(matches) > 1:
            raise ClientConfigError(f"Continue contains more than one {SERVER_ID!r} MCP server entry.")
        expected = self._definition(command)
        if matches and dict(servers[matches[0]]) == expected:
            return RegistrationResult(self.name, str(self.path), False)
        if matches and not replace:
            raise ClientConfigError(f"Continue already defines {SERVER_ID!r} with a different command; " "run repair to replace only that entry.")
        _create_backup_once(self.path)
        if matches:
            servers[matches[0]] = expected
        else:
            servers.append(expected)
        _atomic_write(self.path, self._dump(document))
        return RegistrationResult(self.name, str(self.path), True)

    def unregister(self, command: Path | None = None) -> RegistrationResult:
        document = self._load()
        servers = self._servers(document, create=False)
        if servers is None:
            return RegistrationResult(self.name, str(self.path), False)
        matches = self._matching_indexes(servers)
        if not matches:
            return RegistrationResult(self.name, str(self.path), False)
        if len(matches) > 1:
            raise ClientConfigError(f"Continue contains more than one {SERVER_ID!r} MCP server entry.")
        index = matches[0]
        if command is not None and dict(servers[index]) != self._definition(command):
            return RegistrationResult(self.name, str(self.path), False)
        _create_backup_once(self.path)
        del servers[index]
        _atomic_write(self.path, self._dump(document))
        return RegistrationResult(self.name, str(self.path), True)


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _default_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)


class ClaudeCodeClientAdapter:
    """Use Claude Code's supported user-scope command instead of private files."""

    name = "claude-code"
    detection_paths: tuple[Path, ...] = ()

    def __init__(self, executable: Path, runner: CommandRunner = _default_runner) -> None:
        self.executable = executable
        self.runner = runner

    def register(self, command: Path, replace: bool = False) -> RegistrationResult:
        existing = self.runner((str(self.executable), "mcp", "get", SERVER_ID))
        if existing.returncode == 0 and str(command) in existing.stdout:
            return RegistrationResult(self.name, f"{self.executable} user scope", False)
        if existing.returncode == 0 and not replace:
            raise ClientConfigError(f"Claude Code already defines {SERVER_ID!r} with a different command; " "run repair to replace only that entry.")
        definition = json.dumps(_server_definition(command, include_type=True), separators=(",", ":"))
        added = self.runner((str(self.executable), "mcp", "add-json", "--scope", "user", SERVER_ID, definition))
        if added.returncode != 0:
            message = " ".join((added.stderr or added.stdout).split())[:500]
            if existing.returncode == 0:
                raise ClientConfigError(
                    "Claude Code could not atomically replace its existing "
                    f"{SERVER_ID!r} user-scope server; the original entry was left "
                    f"in place. Remove it in Claude Code and retry: {message or 'unknown error'}"
                )
            raise ClientConfigError(f"Claude Code registration failed: {message or 'unknown error'}")
        return RegistrationResult(self.name, f"{self.executable} user scope", True)

    def unregister(self, command: Path | None = None) -> RegistrationResult:
        existing = self.runner((str(self.executable), "mcp", "get", SERVER_ID))
        if existing.returncode != 0:
            return RegistrationResult(self.name, f"{self.executable} user scope", False)
        if command is not None and str(command) not in existing.stdout:
            return RegistrationResult(self.name, f"{self.executable} user scope", False)
        removed = self.runner((str(self.executable), "mcp", "remove", "--scope", "user", SERVER_ID))
        if removed.returncode != 0:
            raise ClientConfigError(f"Claude Code could not remove its {SERVER_ID!r} user-scope server.")
        return RegistrationResult(self.name, f"{self.executable} user scope", True)


def _json_adapter(name: str, path: Path, root_key: str, detection_root: Path | None = None, include_type: bool = False) -> JsonClientAdapter:
    return JsonClientAdapter(
        name,
        path,
        root_key,
        include_type=include_type,
        detection_paths=(detection_root or path.parent,),
    )


def client_adapters(
    standard_config: Path,
    locations: ClientLocations,
    executables: Mapping[str, Path | None] | None = None,
    runner: CommandRunner = _default_runner,
) -> dict[str, ClientAdapter]:
    """Build the supported client adapters for the current platform."""
    executable_map = dict(executables or {})
    home = locations.home
    config_home = locations.xdg_config_home or home / ".config"
    system = locations.system or sys_platform()

    adapters: dict[str, ClientAdapter] = {
        STANDARD_CLIENT: _json_adapter(STANDARD_CLIENT, standard_config, "mcpServers", standard_config.parent),
        "codex": CodexClientAdapter(home / ".codex" / "config.toml"),
        "cursor": _json_adapter("cursor", home / ".cursor" / "mcp.json", "mcpServers", home / ".cursor"),
        "gemini-cli": _json_adapter("gemini-cli", home / ".gemini" / "settings.json", "mcpServers", home / ".gemini"),
        "windsurf": _json_adapter(
            "windsurf",
            home / ".codeium" / "windsurf" / "mcp_config.json",
            "mcpServers",
            home / ".codeium" / "windsurf",
        ),
        "cline": _json_adapter(
            "cline",
            home / ".cline" / "data" / "settings" / "cline_mcp_settings.json",
            "mcpServers",
            home / ".cline",
        ),
        "continue": ContinueClientAdapter(home / ".continue" / "config.yaml"),
        "kiro": _json_adapter("kiro", home / ".kiro" / "settings" / "mcp.json", "mcpServers", home / ".kiro"),
        "opencode": OpenCodeClientAdapter(config_home / "opencode" / "opencode.json", (config_home / "opencode",)),
    }

    code_user: Path | None = None
    zed_config: Path | None = None
    if locations.platform == "nt":
        if locations.appdata is not None:
            code_user = locations.appdata / "Code" / "User"
            zed_config = locations.appdata / "Zed" / "settings.json"
            adapters["claude-desktop"] = _json_adapter(
                "claude-desktop",
                locations.appdata / "Claude" / "claude_desktop_config.json",
                "mcpServers",
                locations.appdata / "Claude",
            )
    elif system == "darwin":
        code_user = home / "Library" / "Application Support" / "Code" / "User"
        zed_config = home / "Library" / "Application Support" / "Zed" / "settings.json"
        adapters["claude-desktop"] = _json_adapter(
            "claude-desktop",
            home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",
            "mcpServers",
            home / "Library" / "Application Support" / "Claude",
        )
    else:
        code_user = config_home / "Code" / "User"
        zed_config = config_home / "zed" / "settings.json"

    if code_user is not None:
        adapters["vscode"] = _json_adapter("vscode", code_user / "mcp.json", "servers", code_user, include_type=True)
        roo_root = code_user / "globalStorage" / "rooveterinaryinc.roo-cline"
        adapters["roo-code"] = _json_adapter(
            "roo-code",
            roo_root / "settings" / "mcp_settings.json",
            "mcpServers",
            roo_root,
        )
    if zed_config is not None:
        adapters["zed"] = _json_adapter("zed", zed_config, "context_servers", zed_config.parent)

    claude = executable_map.get("claude")
    if claude is not None:
        adapters["claude-code"] = ClaudeCodeClientAdapter(claude, runner)
    return adapters


def sys_platform() -> str:
    """Small indirection for platform-specific tests."""
    import sys

    return sys.platform
