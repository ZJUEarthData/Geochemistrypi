"""Observed identity of the isolated GeochemistryPi CLI environment."""

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution, distributions
from pathlib import Path
from typing import Any

from ..config.constants import CLI_VERSION, ISOLATED_CLI_ENVIRONMENT_VARIABLES, SERVER_VERSION
from ..config.settings import resolve_cli_interpreter
from ..data.inspector import sha256_file

_MAX_DISTRIBUTIONS = 2_000
_INSPECTION_SCRIPT = """
import hashlib
import json
import platform
import sys
from importlib.metadata import distribution, distributions

packages = {}
for installed_distribution in distributions():
    name = str(installed_distribution.metadata.get("Name") or "").strip().lower().replace("_", "-")
    if name:
        packages[name] = str(installed_distribution.version)
package = distribution("geochemistrypi")
payload = []
for relative in package.files or ():
    normalized = str(relative).replace("\\\\", "/")
    if not normalized.startswith("geochemistrypi/"):
        continue
    path = package.locate_file(relative)
    if path.is_file():
        payload.append((normalized, hashlib.sha256(path.read_bytes()).hexdigest()))
payload.sort()
payload_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
print(json.dumps({
    "python_version": platform.python_version(),
    "python_implementation": platform.python_implementation(),
    "platform": platform.platform(),
    "dependencies": dict(sorted(packages.items())),
    "geochemistrypi_payload": {
        "file_count": len(payload),
        "sha256": hashlib.sha256(payload_bytes).hexdigest(),
    },
}, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
""".strip()


class EnvironmentInspectionError(RuntimeError):
    """Raised when the CLI environment cannot be attested safely."""


@dataclass(frozen=True)
class EnvironmentSnapshot:
    """Canonical environment record and its content identity."""

    identity_sha256: str
    record: dict[str, Any]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _local_geochemistrypi_payload() -> dict[str, Any]:
    try:
        package = distribution("geochemistrypi")
    except PackageNotFoundError:
        return {"file_count": 0, "sha256": None}
    payload = []
    for relative in package.files or ():
        normalized = str(relative).replace("\\", "/")
        if not normalized.startswith("geochemistrypi/"):
            continue
        path = Path(package.locate_file(relative))
        if path.is_file():
            payload.append((normalized, sha256_file(path)))
    payload.sort()
    return {
        "file_count": len(payload),
        "sha256": hashlib.sha256(_canonical_json_bytes(payload)).hexdigest(),
    }


def inspect_cli_environment(cli_executable: Path) -> EnvironmentSnapshot:
    """Inspect versions in the interpreter that owns the configured CLI launcher."""
    executable = Path(cli_executable).expanduser().resolve()
    interpreter = resolve_cli_interpreter(executable).resolve()
    if interpreter == Path(sys.executable).resolve():
        packages = {}
        for installed_distribution in distributions():
            name = str(installed_distribution.metadata.get("Name") or "").strip().lower().replace("_", "-")
            if name:
                packages[name] = str(installed_distribution.version)
        observed = {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "dependencies": packages,
            "geochemistrypi_payload": _local_geochemistrypi_payload(),
        }
    else:
        process_environment = os.environ.copy()
        for inherited_name in ISOLATED_CLI_ENVIRONMENT_VARIABLES:
            process_environment.pop(inherited_name, None)
        try:
            completed = subprocess.run(
                (str(interpreter), "-c", _INSPECTION_SCRIPT),
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                env=process_environment,
            )
            observed = json.loads(completed.stdout)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            raise EnvironmentInspectionError("Cannot inspect the isolated GeochemistryPi CLI environment.") from exc
    dependencies = observed.get("dependencies")
    if not isinstance(dependencies, dict) or len(dependencies) > _MAX_DISTRIBUTIONS:
        raise EnvironmentInspectionError("The CLI dependency inventory is absent or exceeds the safety limit.")
    normalized_dependencies: dict[str, str] = {}
    for package, version in dependencies.items():
        if not isinstance(package, str) or not isinstance(version, str):
            raise EnvironmentInspectionError("The CLI dependency inventory contains an invalid entry.")
        normalized_dependencies[package.strip().lower().replace("_", "-")] = version.strip()
    required_text = ("python_version", "python_implementation", "platform")
    if any(not isinstance(observed.get(field), str) or not observed[field].strip() for field in required_text):
        raise EnvironmentInspectionError("The CLI environment identity is incomplete.")
    payload = observed.get("geochemistrypi_payload")
    if (
        not isinstance(payload, dict)
        or type(payload.get("file_count")) is not int
        or payload["file_count"] < 0
        or (payload["file_count"] > 0 and (not isinstance(payload.get("sha256"), str) or len(payload["sha256"]) != 64))
    ):
        raise EnvironmentInspectionError("The GeochemistryPi package payload identity is incomplete.")
    record = {
        "schema_version": 1,
        "cli_executable": {
            "path": str(executable),
            "sha256": sha256_file(executable),
        },
        "python": {
            "executable": str(interpreter),
            "executable_sha256": sha256_file(interpreter),
            "version": observed["python_version"].strip(),
            "implementation": observed["python_implementation"].strip(),
        },
        "geochemistrypi": {
            "version": normalized_dependencies.get("geochemistrypi", CLI_VERSION),
            "package_file_count": payload["file_count"],
            "package_payload_sha256": payload["sha256"],
        },
        "mcp": {"version": SERVER_VERSION},
        "platform": observed["platform"].strip(),
        "runtime": {
            "kind": "isolated_cli_environment",
            "python_implementation": observed["python_implementation"].strip(),
        },
        "dependencies": dict(sorted(normalized_dependencies.items())),
    }
    identity_sha256 = hashlib.sha256(_canonical_json_bytes(record)).hexdigest()
    return EnvironmentSnapshot(identity_sha256=identity_sha256, record=record)
