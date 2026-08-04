import json
import os
import subprocess
from pathlib import Path

import geochemistrypi_mcp.lifecycle.setup as setup_runtime
import pytest
from geochemistrypi_mcp.config.clients import ClientLocations
from geochemistrypi_mcp.config.constants import ISOLATED_CLI_ENVIRONMENT_VARIABLES
from geochemistrypi_mcp.lifecycle.release import ReleaseBundle
from geochemistrypi_mcp.lifecycle.setup import SetupError, SetupManager, SetupPaths, SourceLayout, _remove_tree, _rmtree_onerror


def _source_layout(tmp_path: Path) -> SourceLayout:
    repository = tmp_path / "repository"
    package = repository / "packages" / "geochemistrypi-mcp"
    (repository / "geochemistrypi").mkdir(parents=True)
    (package / "src" / "geochemistrypi_mcp").mkdir(parents=True)
    (repository / "pyproject.toml").write_text("[project]\nname='geochemistrypi'\n", encoding="utf-8")
    (repository / "geochemistrypi" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "pyproject.toml").write_text("[project]\nname='geochemistrypi-mcp'\n", encoding="utf-8")
    (package / "src" / "geochemistrypi_mcp" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    return SourceLayout(repository, package)


def _manager(tmp_path: Path, doctor_healthy: bool = True):
    paths = SetupPaths(tmp_path / "application")
    locations = ClientLocations(home=tmp_path / "home", appdata=tmp_path / "appdata", xdg_config_home=None, platform="nt")
    preparations: list[bool] = []
    manager = SetupManager(
        paths=paths,
        sources=_source_layout(tmp_path),
        locations=locations,
        validator=lambda _: ("0.2.0", "0.8.0"),
        doctor=lambda _: (doctor_healthy, "simulated doctor"),
        runtime_inventory=lambda _: {
            "mcp": {"count": 10, "sha256": "1" * 64},
            "cli": {"count": 20, "sha256": "2" * 64},
        },
        executable_lookup=lambda _: None,
    )

    def prepare(_, clear: bool) -> None:
        preparations.append(clear)
        for path in (paths.mcp_python, paths.server_command, paths.cli_python, paths.cli_command):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("placeholder", encoding="utf-8")

    manager._prepare_environments = prepare
    return manager, paths, preparations


def _release_bundle(tmp_path: Path) -> ReleaseBundle:
    root = tmp_path / "release-bundle"
    root.mkdir()
    manifest_path = root / "release-manifest.json"
    cli_wheel = root / "geochemistrypi-0.8.0-py3-none-any.whl"
    mcp_wheel = root / "geochemistrypi_mcp-0.2.0-py3-none-any.whl"
    cli_wheel.write_text("cli-wheel", encoding="utf-8")
    mcp_wheel.write_text("mcp-wheel", encoding="utf-8")
    manifest = {
        "release_id": "geochemistrypi-0.8.0+mcp-0.2.0",
        "release_tag": "mcp-v0.2.0-cli-v0.8.0",
        "artifacts": [
            {"filename": cli_wheel.name},
            {"filename": mcp_wheel.name},
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return ReleaseBundle(
        directory=root,
        manifest_path=manifest_path,
        manifest=manifest,
        cli_wheel=cli_wheel,
        mcp_wheel=mcp_wheel,
        manifest_sha256="b" * 64,
        signatures_verified=False,
    )


def test_install_is_repeatable_repairable_and_uninstall_preserves_runs(tmp_path: Path) -> None:
    manager, paths, preparations = _manager(tmp_path)

    first = manager.install(("standard",))
    second = manager.install(("standard",))

    assert first.doctor_healthy is True
    assert preparations == [False]
    assert second.clients[0].changed is False
    settings = json.loads(paths.settings_file.read_text(encoding="utf-8"))
    assert settings["cli_executable"] == str(paths.cli_command)
    assert settings["runs_root"] == str(paths.runs_root)
    assert settings["tracking_root"] == str(paths.tracking_root)
    assert settings["service_state_root"] == str(paths.service_state_root)
    assert settings["maximum_pending_runs"] == 8
    assert settings["maximum_process_seconds"] == 900
    assert settings["maximum_columns"] == 256
    assert settings["maximum_artifact_references"] == 200
    assert settings["concurrency"] == 1
    manifest = json.loads(paths.manifest_file.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["compatibility_policy_version"] == 2
    assert manifest["mcp_python_requires"] == ">=3.10,<4"
    assert manifest["cli_python_requires"] == ">=3.9,<3.10"
    assert manifest["mcp_sdk_requires"] == "==2.0.0"
    assert manifest["installation_source"] == "source"
    assert manifest["runtime_inventory"]["mcp"]["sha256"] == "1" * 64
    fallback = json.loads(paths.standard_client_config.read_text(encoding="utf-8"))
    assert fallback["mcpServers"]["geochemistrypi"] == {"command": str(paths.server_command), "args": []}

    run_artifact = paths.runs_root / "run-1" / "result.json"
    run_artifact.parent.mkdir(parents=True)
    run_artifact.write_text("{}", encoding="utf-8")
    tracking_artifact = paths.tracking_root / "experiment" / "meta.yaml"
    tracking_artifact.parent.mkdir(parents=True)
    tracking_artifact.write_text("name: persistent\n", encoding="utf-8")
    repaired = manager.install(("standard",), repair=True)
    assert repaired.action == "repair"
    assert preparations == [False, False]

    uninstalled = manager.uninstall()
    assert uninstalled.action == "uninstall"
    assert not paths.environments_root.exists()
    assert not paths.settings_file.exists()
    assert not paths.manifest_file.exists()
    assert run_artifact.read_text(encoding="utf-8") == "{}"
    assert tracking_artifact.read_text(encoding="utf-8") == "name: persistent\n"
    assert "geochemistrypi" not in json.loads(paths.standard_client_config.read_text(encoding="utf-8"))["mcpServers"]


def test_source_change_refreshes_private_environments(tmp_path: Path) -> None:
    manager, _, preparations = _manager(tmp_path)
    manager.install(("standard",))
    source_file = manager.sources.repository_root / "geochemistrypi" / "module.py"
    source_file.write_text("VALUE = 2\n", encoding="utf-8")

    manager.install(("standard",))

    assert preparations == [False, False]


def test_legacy_manifest_refreshes_runtime_for_current_compatibility_policy(tmp_path: Path) -> None:
    manager, paths, preparations = _manager(tmp_path)
    manager.install(("standard",))
    manifest = json.loads(paths.manifest_file.read_text(encoding="utf-8"))
    manifest.pop("compatibility_policy_version")
    paths.manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    manager.install(("standard",))

    assert preparations == [False, False]


def test_failed_doctor_is_actionable_and_repair_can_recover(tmp_path: Path) -> None:
    manager, paths, _ = _manager(tmp_path, doctor_healthy=False)

    with pytest.raises(SetupError, match="doctor failed"):
        manager.install(("standard",))
    assert not paths.manifest_file.exists()
    assert not paths.environments_root.exists()
    assert not paths.standard_client_config.exists()

    manager.doctor = lambda _: (True, "recovered")
    result = manager.install(("standard",), repair=True)
    assert result.doctor_healthy is True


def test_partial_client_registration_failure_restores_every_client_file(
    tmp_path: Path,
) -> None:
    manager, paths, preparations = _manager(tmp_path)
    standard_original = '{"mcpServers":{"foreign":{"command":"keep"}},"theme":"dark"}\n'
    paths.standard_client_config.parent.mkdir(parents=True)
    paths.standard_client_config.write_text(standard_original, encoding="utf-8")
    cursor_config = manager.locations.home / ".cursor" / "mcp.json"
    cursor_config.parent.mkdir(parents=True)
    cursor_original = '{"mcpServers":{"geochemistrypi":{"command":"foreign"}},"theme":"light"}\n'
    cursor_config.write_text(cursor_original, encoding="utf-8")

    with pytest.raises(SetupError, match="Client registration failed"):
        manager.install(("cursor",))

    assert paths.standard_client_config.read_text(encoding="utf-8") == standard_original
    assert cursor_config.read_text(encoding="utf-8") == cursor_original
    assert paths.manifest_file.is_file()
    assert paths.server_command.is_file()

    cursor_config.write_text('{"mcpServers":{},"theme":"light"}\n', encoding="utf-8")
    recovered = manager.install(("cursor",))
    assert recovered.doctor_healthy is True
    assert preparations == [False]


def test_bundle_upgrade_and_rollback_preserve_user_data(tmp_path: Path) -> None:
    manager, paths, preparations = _manager(tmp_path)
    manager.install(("standard",))
    run_artifact = paths.runs_root / "kept-run" / "result.json"
    run_artifact.parent.mkdir(parents=True)
    run_artifact.write_text('{"kept": true}', encoding="utf-8")
    tracking_artifact = paths.tracking_root / "kept-experiment" / "meta.yaml"
    tracking_artifact.parent.mkdir(parents=True)
    tracking_artifact.write_text("name: kept\n", encoding="utf-8")

    manager.bundle = _release_bundle(tmp_path)
    upgraded = manager.install(upgrade=True)

    assert upgraded.action == "upgrade"
    assert preparations == [False, False]
    upgraded_manifest = json.loads(paths.manifest_file.read_text(encoding="utf-8"))
    assert upgraded_manifest["installation_source"] == "release-bundle"
    assert upgraded_manifest["release_manifest_sha256"] == "b" * 64
    assert upgraded_manifest["signature_policy"] == "explicit-development-override"
    assert upgraded_manifest["rollback_available"] is True
    assert paths.release_manifest_file.is_file()
    assert paths.rollback_environments.is_dir()

    rolled_back = manager.rollback()

    assert rolled_back.action == "rollback"
    restored_manifest = json.loads(paths.manifest_file.read_text(encoding="utf-8"))
    assert restored_manifest["installation_source"] == "source"
    assert restored_manifest["rollback_available"] is False
    assert not paths.release_root.exists()
    assert not paths.rollback_root.exists()
    assert run_artifact.read_text(encoding="utf-8") == '{"kept": true}'
    assert tracking_artifact.read_text(encoding="utf-8") == "name: kept\n"


def test_different_bundle_cannot_bypass_upgrade_and_rollback(tmp_path: Path) -> None:
    manager, _, _ = _manager(tmp_path)
    manager.install(("standard",))
    manager.bundle = _release_bundle(tmp_path)

    with pytest.raises(SetupError, match="must use the upgrade action"):
        manager.install(("standard",), repair=True)


def test_bundle_repair_installs_from_the_preserved_transaction_copy(
    tmp_path: Path,
) -> None:
    manager, paths, _ = _manager(tmp_path)
    original = _release_bundle(tmp_path)
    manager.bundle = original
    manager.install(("standard",))
    active = ReleaseBundle(
        directory=paths.release_root.resolve(),
        manifest_path=paths.release_manifest_file,
        manifest=original.manifest,
        cli_wheel=paths.release_root / original.cli_wheel.name,
        mcp_wheel=paths.release_root / original.mcp_wheel.name,
        manifest_sha256=original.manifest_sha256,
        signatures_verified=False,
    )
    manager.bundle = active
    observed_sources: list[ReleaseBundle] = []

    def prepare(source, clear: bool) -> None:
        assert isinstance(source, ReleaseBundle)
        assert source.cli_wheel.is_file()
        assert source.mcp_wheel.is_file()
        observed_sources.append(source)
        for path in (
            paths.mcp_python,
            paths.server_command,
            paths.cli_python,
            paths.cli_command,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("placeholder", encoding="utf-8")

    manager._prepare_environments = prepare

    repaired = manager.install(("standard",), repair=True)

    assert repaired.action == "repair"
    assert len(observed_sources) == 1
    assert observed_sources[0].directory != paths.release_root
    assert paths.release_manifest_file.is_file()


def test_failed_upgrade_restores_the_current_runtime(tmp_path: Path) -> None:
    manager, paths, _ = _manager(tmp_path)
    manager.install(("standard",))
    original_manifest = paths.manifest_file.read_text(encoding="utf-8")
    outcomes = iter(((True, "preflight healthy"), (False, "new runtime broken")))
    manager.doctor = lambda _: next(outcomes)
    manager.bundle = _release_bundle(tmp_path)

    with pytest.raises(SetupError, match="doctor failed"):
        manager.install(upgrade=True)

    assert paths.manifest_file.read_text(encoding="utf-8") == original_manifest
    assert paths.server_command.is_file()
    assert not paths.release_root.exists()
    assert not paths.rollback_root.exists()


def test_failed_rollback_restores_upgraded_runtime_and_snapshot(tmp_path: Path) -> None:
    manager, paths, _ = _manager(tmp_path)
    manager.install(("standard",))
    manager.bundle = _release_bundle(tmp_path)
    manager.install(upgrade=True)
    upgraded_manifest = paths.manifest_file.read_text(encoding="utf-8")
    manager.doctor = lambda _: (False, "restored runtime broken")

    with pytest.raises(SetupError, match="Rollback doctor failed"):
        manager.rollback()

    assert paths.manifest_file.read_text(encoding="utf-8") == upgraded_manifest
    assert paths.release_manifest_file.is_file()
    assert paths.rollback_environments.is_dir()


def test_all_registers_every_file_based_client_available_on_windows(tmp_path: Path) -> None:
    manager, _, _ = _manager(tmp_path)

    result = manager.install(("all",))

    assert tuple(item.client for item in result.clients) == (
        "standard",
        "codex",
        "claude-desktop",
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


def test_auto_detects_installed_client_roots_and_keeps_standard_fallback(tmp_path: Path) -> None:
    manager, _, _ = _manager(tmp_path)
    locations = manager.locations
    for directory in (
        locations.home / ".codex",
        locations.home / ".gemini",
        locations.home / ".cline",
        locations.home / ".continue",
        locations.home / ".kiro",
        locations.appdata / "Code" / "User",
        locations.appdata / "Code" / "User" / "globalStorage" / "rooveterinaryinc.roo-cline",
        locations.appdata / "Zed",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    result = manager.install(("auto",))

    assert tuple(item.client for item in result.clients) == (
        "standard",
        "codex",
        "vscode",
        "gemini-cli",
        "cline",
        "roo-code",
        "zed",
        "continue",
        "kiro",
    )


def test_recursive_uninstall_ignores_only_a_file_that_already_disappeared() -> None:
    missing = FileNotFoundError(2, "already removed", "vanished")
    _rmtree_onerror(os.unlink, "vanished", (FileNotFoundError, missing, None))

    denied = PermissionError(13, "access denied", "locked")
    with pytest.raises(PermissionError, match="access denied"):
        _rmtree_onerror(os.unlink, "locked", (PermissionError, denied, None))


def test_failed_move_recovery_never_deletes_an_environment_not_in_backup(
    tmp_path: Path,
) -> None:
    manager, paths, _ = _manager(tmp_path)
    active = paths.mcp_environment / "still-active.txt"
    active.parent.mkdir(parents=True)
    active.write_text("keep", encoding="utf-8")
    paths.settings_file.parent.mkdir(parents=True)
    paths.settings_file.write_text("settings", encoding="utf-8")
    paths.manifest_file.write_text("manifest", encoding="utf-8")
    backup = paths.app_root / ".setup-recovery-test"
    backup.mkdir(parents=True)
    (backup / "rollback-metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "persistent": False,
                "environments_present": True,
                "release_present": False,
                "settings_present": True,
                "manifest_present": True,
            }
        ),
        encoding="utf-8",
    )

    manager._restore_runtime_transaction(backup)

    assert active.read_text(encoding="utf-8") == "keep"
    assert paths.settings_file.read_text(encoding="utf-8") == "settings"
    assert paths.manifest_file.read_text(encoding="utf-8") == "manifest"
    assert not backup.exists()


def test_installed_windows_lifecycle_refuses_before_mutating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = SetupPaths(tmp_path / "application")
    monkeypatch.setattr(setup_runtime.SetupPaths, "default", lambda: paths)
    monkeypatch.setattr(setup_runtime, "_running_inside_private_mcp", lambda _: True)
    monkeypatch.setattr(
        setup_runtime,
        "_windows_external_bootstrap_guidance",
        lambda value, arguments: "use external bootstrap",
    )

    with pytest.raises(SystemExit) as exc_info:
        setup_runtime.main(("rollback",))
    assert exc_info.value.code == 1


def test_setup_requires_the_exact_pinned_uv_version(tmp_path: Path) -> None:
    uv = tmp_path / "uv.exe"
    uv.touch()
    manager = SetupManager(
        paths=SetupPaths(tmp_path / "application"),
        runner=lambda command: subprocess.CompletedProcess(
            command,
            0,
            "uv 0.12.0\n",
            "",
        ),
        executable_lookup=lambda _: str(uv),
    )

    with pytest.raises(SetupError, match="requires uv 0.11.7"):
        manager._uv_executable()

    manager.runner = lambda command: subprocess.CompletedProcess(
        command,
        0,
        "uv 0.11.7 (build metadata)\n",
        "",
    )
    assert manager._uv_executable() == uv.resolve()


def test_setup_runner_removes_foreign_python_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    observed_environment = {}
    for name in ISOLATED_CLI_ENVIRONMENT_VARIABLES:
        monkeypatch.setenv(name, f"foreign-{name}")

    def fake_run(command, **kwargs):
        observed_environment.update(kwargs["env"])
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(setup_runtime.subprocess, "run", fake_run)

    setup_runtime._default_runner(("placeholder",))

    assert set(ISOLATED_CLI_ENVIRONMENT_VARIABLES).isdisjoint(observed_environment)
    if os.environ.get("PATH"):
        assert observed_environment["PATH"] == os.environ["PATH"]


@pytest.mark.skipif(os.name != "nt", reason="Windows extended-length path regression")
def test_recursive_uninstall_removes_paths_longer_than_legacy_windows_limit(tmp_path: Path) -> None:
    root = tmp_path / "private-environments"
    deepest = root
    while len(str(deepest / "artifact.txt")) <= 260:
        deepest /= "long-environment-directory"
    extended_deepest = Path(f"\\\\?\\{deepest}")
    extended_deepest.mkdir(parents=True)
    (extended_deepest / "artifact.txt").write_text("test", encoding="utf-8")

    _remove_tree(root)

    assert not root.exists()
