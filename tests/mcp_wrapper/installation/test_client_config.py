import json
import subprocess
from pathlib import Path

import pytest
import tomlkit
from ruamel.yaml import YAML

from geochemistrypi_mcp.client_config import (
    SERVER_ID,
    SUPPORTED_CLIENTS,
    ClaudeCodeClientAdapter,
    ClientConfigError,
    ClientLocations,
    CodexClientAdapter,
    JsonClientAdapter,
    backup_path,
    client_adapters,
)


def test_json_registration_is_atomic_repeatable_and_preserves_unrelated_state(tmp_path: Path) -> None:
    config = tmp_path / "mcp.json"
    original = {"theme": "dark", "mcpServers": {"other": {"command": "other-server"}}}
    config.write_text(json.dumps(original), encoding="utf-8")
    adapter = JsonClientAdapter("test-client", config, "mcpServers")
    command = tmp_path / "geochemistrypi-mcp.exe"

    first = adapter.register(command)
    second = adapter.register(command)

    assert first.changed is True
    assert second.changed is False
    installed = json.loads(config.read_text(encoding="utf-8"))
    assert installed["theme"] == "dark"
    assert installed["mcpServers"]["other"] == {"command": "other-server"}
    assert installed["mcpServers"][SERVER_ID] == {"command": str(command), "args": []}
    assert json.loads(backup_path(config).read_text(encoding="utf-8")) == original

    installed["mcpServers"][SERVER_ID]["command"] = "stale-command"
    config.write_text(json.dumps(installed), encoding="utf-8")
    with pytest.raises(ClientConfigError, match="run repair"):
        adapter.register(command)
    assert adapter.register(command, replace=True).changed is True
    assert json.loads(backup_path(config).read_text(encoding="utf-8")) == original

    installed["mcpServers"][SERVER_ID]["command"] = "user-replaced-command"
    config.write_text(json.dumps(installed), encoding="utf-8")
    assert adapter.unregister(command).changed is False
    assert SERVER_ID in json.loads(config.read_text(encoding="utf-8"))["mcpServers"]

    assert adapter.unregister().changed is True
    remaining = json.loads(config.read_text(encoding="utf-8"))
    assert remaining == {"theme": "dark", "mcpServers": {"other": {"command": "other-server"}}}


def test_codex_registration_preserves_toml_and_removes_only_owned_table(tmp_path: Path) -> None:
    config = tmp_path / ".codex" / "config.toml"
    config.parent.mkdir()
    original = 'model = "gpt-test"\n\n[mcp_servers.other]\ncommand = "other"\n'
    config.write_text(original, encoding="utf-8")
    adapter = CodexClientAdapter(config)
    command = tmp_path / "geochemistrypi-mcp.exe"

    assert adapter.register(command).changed is True
    assert adapter.register(command).changed is False
    installed = tomlkit.parse(config.read_text(encoding="utf-8"))
    assert installed["model"] == "gpt-test"
    assert installed["mcp_servers"]["other"]["command"] == "other"
    assert installed["mcp_servers"][SERVER_ID]["command"] == str(command)
    assert backup_path(config).read_text(encoding="utf-8") == original

    assert adapter.unregister().changed is True
    remaining = tomlkit.parse(config.read_text(encoding="utf-8"))
    assert remaining["mcp_servers"]["other"]["command"] == "other"
    assert SERVER_ID not in remaining["mcp_servers"]


def test_windows_adapters_use_each_clients_supported_schema(tmp_path: Path) -> None:
    locations = ClientLocations(
        home=tmp_path / "home",
        appdata=tmp_path / "appdata",
        xdg_config_home=None,
        platform="nt",
        system="win32",
    )
    adapters = client_adapters(tmp_path / "standard.json", locations)
    command = tmp_path / "server.exe"

    assert set(adapters) == set(SUPPORTED_CLIENTS) - {"claude-code"}
    for name in adapters:
        adapters[name].register(command)

    standard = json.loads((tmp_path / "standard.json").read_text(encoding="utf-8"))
    claude = json.loads((locations.appdata / "Claude" / "claude_desktop_config.json").read_text(encoding="utf-8"))
    cursor = json.loads((locations.home / ".cursor" / "mcp.json").read_text(encoding="utf-8"))
    vscode = json.loads((locations.appdata / "Code" / "User" / "mcp.json").read_text(encoding="utf-8"))
    gemini = json.loads((locations.home / ".gemini" / "settings.json").read_text(encoding="utf-8"))
    windsurf = json.loads((locations.home / ".codeium" / "windsurf" / "mcp_config.json").read_text(encoding="utf-8"))
    cline = json.loads((locations.home / ".cline" / "data" / "settings" / "cline_mcp_settings.json").read_text(encoding="utf-8"))
    roo = json.loads(
        (
            locations.appdata
            / "Code"
            / "User"
            / "globalStorage"
            / "rooveterinaryinc.roo-cline"
            / "settings"
            / "mcp_settings.json"
        ).read_text(encoding="utf-8")
    )
    zed = json.loads((locations.appdata / "Zed" / "settings.json").read_text(encoding="utf-8"))
    kiro = json.loads((locations.home / ".kiro" / "settings" / "mcp.json").read_text(encoding="utf-8"))
    opencode = json.loads((locations.home / ".config" / "opencode" / "opencode.json").read_text(encoding="utf-8"))
    continue_config = YAML(typ="safe").load((locations.home / ".continue" / "config.yaml").read_text(encoding="utf-8"))

    assert standard["mcpServers"][SERVER_ID] == {"command": str(command), "args": []}
    assert claude["mcpServers"][SERVER_ID] == {"command": str(command), "args": []}
    assert cursor["mcpServers"][SERVER_ID] == {"command": str(command), "args": []}
    assert vscode["servers"][SERVER_ID] == {"type": "stdio", "command": str(command), "args": []}
    for document in (gemini, windsurf, cline, roo, kiro):
        assert document["mcpServers"][SERVER_ID] == {"command": str(command), "args": []}
    assert zed["context_servers"][SERVER_ID] == {"command": str(command), "args": []}
    assert continue_config["mcpServers"] == [{"name": SERVER_ID, "command": str(command), "args": []}]
    assert opencode["mcp"]["servers"][SERVER_ID] == {"type": "local", "command": [str(command)]}


def test_continue_yaml_round_trip_preserves_comments_and_unrelated_servers(tmp_path: Path) -> None:
    path = tmp_path / ".continue" / "config.yaml"
    path.parent.mkdir()
    original = """name: My Config
version: 1.0.0
schema: v1
# keep this comment
mcpServers:
  - name: other
    command: other-server
    args: []
"""
    path.write_text(original, encoding="utf-8")
    adapter = client_adapters(tmp_path / "fallback.json", ClientLocations(tmp_path, None, None, "posix", "linux"))["continue"]
    command = tmp_path / "server"

    assert adapter.register(command).changed is True
    assert adapter.register(command).changed is False
    installed_text = path.read_text(encoding="utf-8")
    installed = YAML(typ="safe").load(installed_text)
    assert "# keep this comment" in installed_text
    assert installed["mcpServers"][0]["name"] == "other"
    assert installed["mcpServers"][1] == {"name": SERVER_ID, "command": str(command), "args": []}
    assert backup_path(path).read_text(encoding="utf-8") == original

    installed["mcpServers"][1]["command"] = "changed-by-user"
    YAML().dump(installed, path)
    assert adapter.unregister(command).changed is False
    with pytest.raises(ClientConfigError, match="run repair"):
        adapter.register(command)
    assert adapter.register(command, replace=True).changed is True
    assert adapter.unregister(command).changed is True


def test_opencode_refuses_to_rewrite_jsonc_and_nested_json_is_repeatable(tmp_path: Path) -> None:
    locations = ClientLocations(tmp_path / "home", None, tmp_path / "xdg", "posix", "linux")
    adapter = client_adapters(tmp_path / "fallback.json", locations)["opencode"]
    adapter.jsonc_path.parent.mkdir(parents=True)
    adapter.jsonc_path.write_text('{\n  // user comment\n  "theme": "dark"\n}\n', encoding="utf-8")

    with pytest.raises(ClientConfigError, match="will not rewrite comments"):
        adapter.register(tmp_path / "server")
    assert "user comment" in adapter.jsonc_path.read_text(encoding="utf-8")

    adapter.jsonc_path.unlink()
    command = tmp_path / "server"
    assert adapter.register(command).changed is True
    assert adapter.register(command).changed is False
    assert adapter.unregister(command).changed is True


def test_macos_and_linux_adapters_use_platform_native_paths(tmp_path: Path) -> None:
    mac = ClientLocations(tmp_path / "mac-home", None, None, "posix", "darwin")
    linux = ClientLocations(tmp_path / "linux-home", None, tmp_path / "linux-xdg", "posix", "linux")
    mac_adapters = client_adapters(tmp_path / "mac-standard.json", mac)
    linux_adapters = client_adapters(tmp_path / "linux-standard.json", linux)

    assert mac_adapters["claude-desktop"].path == mac.home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    assert mac_adapters["vscode"].path == mac.home / "Library" / "Application Support" / "Code" / "User" / "mcp.json"
    assert mac_adapters["zed"].path == mac.home / "Library" / "Application Support" / "Zed" / "settings.json"
    assert "claude-desktop" not in linux_adapters
    assert linux_adapters["vscode"].path == linux.xdg_config_home / "Code" / "User" / "mcp.json"
    assert linux_adapters["zed"].path == linux.xdg_config_home / "zed" / "settings.json"
    assert linux_adapters["opencode"].path == linux.xdg_config_home / "opencode" / "opencode.json"


def test_claude_code_adapter_uses_supported_cli_and_repairs_owned_entry(tmp_path: Path) -> None:
    commands: list[tuple[str, ...]] = []
    exists = False
    registered_command = ""

    def runner(command):
        nonlocal exists, registered_command
        command = tuple(command)
        commands.append(command)
        if command[1:3] == ("mcp", "get"):
            return subprocess.CompletedProcess(command, 0 if exists else 1, registered_command, "")
        if command[1:3] == ("mcp", "remove"):
            exists = False
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[1:3] == ("mcp", "add-json"):
            exists = True
            registered_command = json.loads(command[6])["command"]
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(command)

    adapter = ClaudeCodeClientAdapter(tmp_path / "claude.exe", runner)
    server = tmp_path / "geochemistrypi-mcp.exe"

    assert adapter.register(server).changed is True
    assert adapter.register(server).changed is False
    registered_command = "foreign-server"
    with pytest.raises(ClientConfigError, match="run repair"):
        adapter.register(server)
    assert adapter.register(server, replace=True).changed is True
    assert adapter.unregister(server).changed is True
    add_commands = [command for command in commands if command[1:3] == ("mcp", "add-json")]
    assert len(add_commands) == 2
    assert add_commands[0][3:6] == ("--scope", "user", SERVER_ID)
    definition = json.loads(add_commands[0][6])
    assert definition == {"type": "stdio", "command": str(server), "args": []}


def test_claude_code_failed_repair_never_removes_the_original_entry(
    tmp_path: Path,
) -> None:
    commands: list[tuple[str, ...]] = []

    def runner(command):
        command = tuple(command)
        commands.append(command)
        if command[1:3] == ("mcp", "get"):
            return subprocess.CompletedProcess(command, 0, "foreign-server", "")
        if command[1:3] == ("mcp", "add-json"):
            return subprocess.CompletedProcess(command, 1, "", "already exists")
        raise AssertionError(command)

    adapter = ClaudeCodeClientAdapter(tmp_path / "claude.exe", runner)

    with pytest.raises(ClientConfigError, match="original entry was left in place"):
        adapter.register(tmp_path / "geochemistrypi-mcp.exe", replace=True)

    assert not any(command[1:3] == ("mcp", "remove") for command in commands)
