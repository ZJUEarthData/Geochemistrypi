"""End-to-end health checks for a prepared GeochemistryPi MCP runtime."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client

from ..config.constants import CLI_PYTHON_REQUIRES, COMPATIBILITY_POLICY_VERSION, ISOLATED_CLI_ENVIRONMENT_VARIABLES, MCP_PYTHON_REQUIRES, MCP_SDK_REQUIRES, SERVER_VERSION, SUPPORTED_CLI_VERSIONS
from ..config.settings import SETTINGS_FILE_ENV, SETTINGS_SCHEMA_VERSION, resolve_cli_interpreter
from .release import SIGSTORE_BUNDLE_SUFFIX, ReleaseError, verify_release_bundle
from .setup import MANIFEST_SCHEMA_VERSION, SetupPaths

EXPECTED_TOOLS = {
    "cancel_run",
    "get_capabilities",
    "get_run_result",
    "get_run_status",
    "inspect_dataset",
    "list_datasets",
    "list_experiments",
    "get_experiment",
    "start_mlflow_ui",
    "mlflow_ui_status",
    "stop_mlflow_ui",
    "validate_analysis",
    "start_analysis",
}


@dataclass(frozen=True)
class DoctorCheck:
    """One named, user-readable diagnostic result."""

    name: str
    healthy: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "healthy": self.healthy, "detail": self.detail}


@dataclass(frozen=True)
class DoctorReport:
    """Complete health report with a stable machine-readable representation."""

    checks: tuple[DoctorCheck, ...]

    @property
    def healthy(self) -> bool:
        return all(check.healthy for check in self.checks)

    @property
    def summary(self) -> str:
        healthy_count = sum(check.healthy for check in self.checks)
        state = "healthy" if self.healthy else "unhealthy"
        return f"Doctor: {state} ({healthy_count}/{len(self.checks)} checks passed)."

    def to_dict(self) -> dict[str, object]:
        return {
            "healthy": self.healthy,
            "summary": self.summary,
            "checks": [check.to_dict() for check in self.checks],
        }


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
ProtocolProbe = Callable[[SetupPaths], tuple[bool, str]]
RuntimeInventoryProbe = Callable[[SetupPaths], Mapping[str, object]]
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _default_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    process_environment = os.environ.copy()
    for inherited_name in ISOLATED_CLI_ENVIRONMENT_VARIABLES:
        process_environment.pop(inherited_name, None)
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        env=process_environment,
    )


def _command_result(
    runner: CommandRunner,
    command: Sequence[str],
    name: str,
    validator: Callable[[str], str],
) -> DoctorCheck:
    try:
        completed = runner(tuple(str(part) for part in command))
    except (OSError, subprocess.SubprocessError) as exc:
        return DoctorCheck(name, False, f"Could not start process: {exc}")
    if completed.returncode != 0:
        detail = " ".join((completed.stderr or completed.stdout).split())[:500]
        return DoctorCheck(name, False, detail or f"Process exited with {completed.returncode}.")
    try:
        detail = validator(completed.stdout.strip())
    except (ValueError, json.JSONDecodeError) as exc:
        return DoctorCheck(name, False, str(exc))
    return DoctorCheck(name, True, detail)


def _settings_check(paths: SetupPaths) -> DoctorCheck:
    try:
        value = json.loads(paths.settings_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return DoctorCheck("settings", False, f"Missing settings file: {paths.settings_file}")
    except (OSError, json.JSONDecodeError) as exc:
        return DoctorCheck("settings", False, f"Cannot parse settings: {exc}")
    expected = {
        "schema_version": SETTINGS_SCHEMA_VERSION,
        "cli_executable": str(paths.cli_command),
        "runs_root": str(paths.runs_root),
        "tracking_root": str(paths.tracking_root),
        "service_state_root": str(paths.service_state_root),
    }
    if not isinstance(value, dict) or any(value.get(key) != item for key, item in expected.items()):
        return DoctorCheck("settings", False, "Persisted settings do not match the private runtime paths.")
    limits = (
        "maximum_dataset_bytes",
        "maximum_columns",
        "maximum_artifact_references",
        "concurrency",
        "maximum_pending_runs",
        "maximum_process_seconds",
    )
    for name in limits:
        if type(value.get(name)) is not int or value[name] < 1:
            return DoctorCheck("settings", False, f"Persisted setting {name} must be a positive integer.")
    if value["maximum_pending_runs"] < value["concurrency"]:
        return DoctorCheck(
            "settings",
            False,
            "Persisted maximum_pending_runs cannot be smaller than concurrency.",
        )
    rendered_limits = ", ".join(f"{name}={value[name]}" for name in limits)
    return DoctorCheck(
        "settings",
        True,
        f"Loaded schema {SETTINGS_SCHEMA_VERSION}; {rendered_limits}.",
    )


def _manifest_check(paths: SetupPaths) -> DoctorCheck:
    try:
        value = json.loads(paths.manifest_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return DoctorCheck("install-manifest", False, f"Missing install manifest: {paths.manifest_file}")
    except (OSError, json.JSONDecodeError) as exc:
        return DoctorCheck("install-manifest", False, f"Cannot parse install manifest: {exc}")
    expected = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "server_version": SERVER_VERSION,
        "cli_version": SUPPORTED_CLI_VERSIONS[0],
        "compatibility_policy_version": COMPATIBILITY_POLICY_VERSION,
        "mcp_python_requires": MCP_PYTHON_REQUIRES,
        "cli_python_requires": CLI_PYTHON_REQUIRES,
        "mcp_sdk_requires": MCP_SDK_REQUIRES,
    }
    if not isinstance(value, dict) or any(value.get(key) != item for key, item in expected.items()):
        return DoctorCheck("install-manifest", False, "Install manifest does not match the current compatibility policy.")
    expected_paths = {
        "server_command": str(paths.server_command),
        "runs_root": str(paths.runs_root),
        "tracking_root": str(paths.tracking_root),
        "service_state_root": str(paths.service_state_root),
    }
    if any(value.get(key) != item for key, item in expected_paths.items()):
        return DoctorCheck(
            "install-manifest",
            False,
            "Install manifest paths do not match the active private runtime.",
        )
    if value.get("installation_source") not in {"source", "release-bundle"}:
        return DoctorCheck("install-manifest", False, "Installation source is invalid.")
    fingerprint = value.get("source_fingerprint")
    if not isinstance(fingerprint, str) or not _SHA256_PATTERN.fullmatch(fingerprint):
        return DoctorCheck("install-manifest", False, "Installation fingerprint is invalid.")
    inventory = value.get("runtime_inventory")
    if not isinstance(inventory, dict) or set(inventory) != {"mcp", "cli"}:
        return DoctorCheck("install-manifest", False, "Runtime inventory is missing or incomplete.")
    for name in ("mcp", "cli"):
        item = inventory.get(name)
        if not isinstance(item, dict) or type(item.get("count")) is not int or item["count"] < 1 or not isinstance(item.get("sha256"), str) or not _SHA256_PATTERN.fullmatch(item["sha256"]):
            return DoctorCheck(
                "install-manifest",
                False,
                f"Runtime inventory entry {name} is invalid.",
            )
    if type(value.get("rollback_available")) is not bool:
        return DoctorCheck("install-manifest", False, "Rollback availability is not explicit.")
    if value["rollback_available"] and (not paths.rollback_metadata_file.is_file() or not paths.rollback_environments.is_dir()):
        return DoctorCheck(
            "install-manifest",
            False,
            "Install manifest advertises rollback, but its private snapshot is incomplete.",
        )
    clients = value.get("registered_clients")
    if not isinstance(clients, list) or "standard" not in clients:
        return DoctorCheck("install-manifest", False, "Registered client inventory is incomplete.")
    return DoctorCheck(
        "install-manifest",
        True,
        f"CLI {value['cli_version']}, MCP {value['server_version']}, compatibility policy {COMPATIBILITY_POLICY_VERSION}.",
    )


def _release_bundle_check(paths: SetupPaths) -> DoctorCheck:
    try:
        install_manifest = json.loads(paths.manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return DoctorCheck("release-bundle", False, f"Cannot inspect install manifest: {exc}")
    if not isinstance(install_manifest, dict):
        return DoctorCheck("release-bundle", False, "Install manifest must be a JSON object.")
    if install_manifest.get("installation_source") == "source":
        if paths.release_root.exists():
            return DoctorCheck(
                "release-bundle",
                False,
                "A source installation must not expose an active release bundle.",
            )
        return DoctorCheck(
            "release-bundle",
            True,
            "Development source installation; no active release bundle applies.",
        )
    try:
        bundle = verify_release_bundle(paths.release_root, require_signatures=False)
    except ReleaseError as exc:
        return DoctorCheck("release-bundle", False, f"Release bundle integrity failed: {exc}")
    comparisons = (
        install_manifest.get("release_manifest_sha256") == bundle.manifest_sha256,
        install_manifest.get("source_fingerprint") == bundle.fingerprint,
        install_manifest.get("release_id") == bundle.release_id,
        install_manifest.get("release_tag") == bundle.release_tag,
        install_manifest.get("release_artifacts") == bundle.manifest["artifacts"],
    )
    if not all(comparisons):
        return DoctorCheck(
            "release-bundle",
            False,
            "Active bundle hashes or release identity do not match the install manifest.",
        )
    signatures_verified = install_manifest.get("signatures_verified") is True
    expected_policy = "verified" if signatures_verified else "explicit-development-override"
    if install_manifest.get("signature_policy") != expected_policy:
        return DoctorCheck("release-bundle", False, "Release signature policy is inconsistent.")
    if signatures_verified:
        missing = [
            artifact.name + SIGSTORE_BUNDLE_SUFFIX
            for artifact in (bundle.manifest_path, bundle.cli_wheel, bundle.mcp_wheel)
            if not artifact.with_name(artifact.name + SIGSTORE_BUNDLE_SUFFIX).is_file()
        ]
        if missing:
            return DoctorCheck(
                "release-bundle",
                False,
                f"Verified release is missing retained Sigstore bundles: {missing}",
            )
    signature_detail = "verified during activation" if signatures_verified else "explicit local unsigned override"
    return DoctorCheck(
        "release-bundle",
        True,
        f"{bundle.release_id}; wheel SHA-256 hashes match; signatures {signature_detail}.",
    )


def _runs_check(paths: SetupPaths) -> DoctorCheck:
    try:
        for storage in (paths.runs_root, paths.tracking_root, paths.service_state_root):
            storage.mkdir(parents=True, exist_ok=True)
            handle, temporary_name = tempfile.mkstemp(prefix=".doctor-", dir=storage)
            os.close(handle)
            Path(temporary_name).unlink()
    except OSError as exc:
        return DoctorCheck("managed-storage", False, f"Managed storage is not writable: {exc}")
    return DoctorCheck("managed-storage", True, "Run, tracking, and service-state storage are writable.")


def _collect_runtime_inventory(
    paths: SetupPaths,
    runner: CommandRunner,
) -> Mapping[str, object]:
    script = (
        "import hashlib,json; from importlib.metadata import distributions; "
        "items=sorted((d.metadata['Name'].lower().replace('_','-').replace('.','-'),d.version) "
        "for d in distributions() if d.metadata.get('Name')); "
        "raw=json.dumps(items,separators=(',',':')).encode(); "
        "print(json.dumps({'count':len(items),'sha256':hashlib.sha256(raw).hexdigest()}))"
    )
    values: dict[str, object] = {}
    for name, interpreter in (("mcp", paths.mcp_python), ("cli", paths.cli_python)):
        completed = runner((str(interpreter), "-c", script))
        if completed.returncode != 0:
            detail = " ".join((completed.stderr or completed.stdout).split())[-500:]
            raise RuntimeError(f"{name.upper()} inventory failed: {detail}")
        try:
            value = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{name.upper()} inventory returned invalid JSON.") from exc
        if not isinstance(value, dict) or type(value.get("count")) is not int or value["count"] < 1 or not isinstance(value.get("sha256"), str) or not _SHA256_PATTERN.fullmatch(value["sha256"]):
            raise RuntimeError(f"{name.upper()} inventory is incomplete.")
        values[name] = value
    return values


def _runtime_inventory_check(
    paths: SetupPaths,
    probe: RuntimeInventoryProbe,
) -> DoctorCheck:
    try:
        manifest = json.loads(paths.manifest_file.read_text(encoding="utf-8"))
        observed = probe(paths)
    except Exception as exc:  # Doctor reports provider failures instead of crashing.
        return DoctorCheck("runtime-inventory", False, f"Cannot verify runtime inventory: {exc}")
    expected = manifest.get("runtime_inventory") if isinstance(manifest, dict) else None
    if observed != expected:
        return DoctorCheck(
            "runtime-inventory",
            False,
            "Installed distribution inventory changed after setup; run repair from the trusted bundle.",
        )
    if not isinstance(observed, Mapping) or not all(isinstance(observed.get(name), Mapping) for name in ("mcp", "cli")):
        return DoctorCheck("runtime-inventory", False, "Runtime inventory is malformed.")
    mcp = observed["mcp"]
    cli = observed["cli"]
    return DoctorCheck(
        "runtime-inventory",
        True,
        f"MCP distributions={mcp['count']}; CLI distributions={cli['count']}; both inventory hashes match.",
    )


def _validate_mcp_runtime(output: str) -> str:
    try:
        value = json.loads(output)
        version_info = tuple(int(part) for part in value["python"][:2])
        package_version = str(value["package"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("MCP runtime returned an invalid version handshake.") from exc
    if version_info != (3, 11):
        raise ValueError(f"MCP runtime must use Python 3.11, found {value['python']}.")
    if package_version != SERVER_VERSION:
        raise ValueError(f"MCP package version {package_version} does not match {SERVER_VERSION}.")
    return f"Python {value['python']}; geochemistrypi-mcp {package_version}."


def _validate_cli_runtime(output: str) -> str:
    try:
        value = json.loads(output)
        version_info = tuple(int(part) for part in value["python"][:2])
        package_version = str(value["package"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("CLI runtime returned an invalid version handshake.") from exc
    if version_info != (3, 9):
        raise ValueError(f"CLI runtime must use Python 3.9, found {value['python']}.")
    if package_version not in SUPPORTED_CLI_VERSIONS:
        supported = ", ".join(SUPPORTED_CLI_VERSIONS)
        raise ValueError(f"Unsupported GeochemistryPi CLI version {package_version}; supported: {supported}.")
    return f"Python {value['python']}; geochemistrypi {package_version}."


async def _probe_protocol(paths: SetupPaths) -> tuple[bool, str]:
    parameters = StdioServerParameters(
        command=str(paths.server_command),
        args=[],
        env={SETTINGS_FILE_ENV: str(paths.settings_file)},
    )
    try:
        async with Client(stdio_client(parameters)) as client:
            listing = await client.list_tools()
    except Exception as exc:  # MCP transports expose several backend-specific errors.
        return False, f"Protocol startup failed: {exc}"
    discovered = {tool.name for tool in listing.tools}
    if discovered != EXPECTED_TOOLS:
        missing = sorted(EXPECTED_TOOLS - discovered)
        unexpected = sorted(discovered - EXPECTED_TOOLS)
        return False, f"Tool discovery mismatch; missing={missing}, unexpected={unexpected}."
    return True, f"Zero-argument stdio startup exposed {len(discovered)} expected tools."


def _default_protocol_probe(paths: SetupPaths) -> tuple[bool, str]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_probe_protocol(paths))
    return False, "Doctor cannot run its stdio probe inside an active asyncio event loop."


def run_doctor(
    paths: SetupPaths | None = None,
    runner: CommandRunner = _default_runner,
    protocol_probe: ProtocolProbe = _default_protocol_probe,
    inventory_probe: RuntimeInventoryProbe | None = None,
) -> DoctorReport:
    """Check persisted state, both private runtimes, run storage, and MCP stdio."""
    resolved_paths = paths or SetupPaths.default()
    resolved_inventory_probe = inventory_probe or (lambda value: _collect_runtime_inventory(value, runner))
    checks = [
        _settings_check(resolved_paths),
        _manifest_check(resolved_paths),
        _release_bundle_check(resolved_paths),
        _runtime_inventory_check(resolved_paths, resolved_inventory_probe),
        _runs_check(resolved_paths),
    ]

    version_script = "import json,sys; from importlib.metadata import version; " "print(json.dumps({'python': list(sys.version_info[:3]), 'package': version(%r)}))"
    checks.append(
        _command_result(
            runner,
            (str(resolved_paths.mcp_python), "-c", version_script % "geochemistrypi-mcp"),
            "mcp-runtime",
            _validate_mcp_runtime,
        )
    )
    try:
        cli_python = resolve_cli_interpreter(resolved_paths.cli_command)
    except Exception as exc:
        checks.append(DoctorCheck("cli-runtime", False, str(exc)))
    else:
        checks.append(
            _command_result(
                runner,
                (str(cli_python), "-c", version_script % "geochemistrypi"),
                "cli-runtime",
                _validate_cli_runtime,
            )
        )
    checks.append(
        _command_result(
            runner,
            (str(resolved_paths.cli_command), "--version"),
            "cli-command",
            lambda output: output or "CLI version command completed.",
        )
    )
    try:
        protocol_healthy, protocol_detail = protocol_probe(resolved_paths)
    except Exception as exc:  # A doctor report should diagnose rather than crash.
        protocol_healthy, protocol_detail = False, f"Protocol probe failed: {exc}"
    checks.append(DoctorCheck("mcp-protocol", protocol_healthy, protocol_detail))
    return DoctorReport(tuple(checks))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="geochemistrypi-mcp-doctor", description="Diagnose the installed GeochemistryPi MCP runtime.")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable health report.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run diagnostics without ever writing protocol data to stdout."""
    arguments = _parser().parse_args(argv)
    report = run_doctor()
    if arguments.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(report.summary)
        for check in report.checks:
            marker = "PASS" if check.healthy else "FAIL"
            print(f"[{marker}] {check.name}: {check.detail}")
    if not report.healthy:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
