"""Build and verify versioned GeochemistryPi release bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from email.parser import BytesParser
from pathlib import Path
from typing import Callable, Mapping, Sequence
from zipfile import BadZipFile, ZipFile

from ..config.constants import CLI_PYTHON_REQUIRES, MCP_PYTHON_REQUIRES, MCP_SDK_REQUIRES, PENDING_RELEASE_GATES, PUBLIC_RELEASE_READY, SERVER_VERSION, SUPPORTED_CLI_VERSIONS

RELEASE_MANIFEST_SCHEMA_VERSION = 1
RELEASE_MANIFEST_FILENAME = "release-manifest.json"
SIGSTORE_BUNDLE_SUFFIX = ".sigstore.json"
SIGSTORE_OIDC_ISSUER = "https://token.actions.githubusercontent.com"
RELEASE_WORKFLOW_IDENTITY_PREFIX = "https://github.com/ZJUEarthData/Geochemistrypi/" ".github/workflows/release.yml@refs/tags/"
CLI_VERSION = SUPPORTED_CLI_VERSIONS[0]
EXPECTED_RELEASE_TAG = f"mcp-v{SERVER_VERSION}-cli-v{CLI_VERSION}"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_RELEASE_TAG_PATTERN = re.compile(r"^mcp-v[0-9]+\.[0-9]+\.[0-9]+-cli-v[0-9]+\.[0-9]+\.[0-9]+$")
_EXPECTED_DISTRIBUTIONS = {
    "geochemistrypi": (CLI_VERSION, CLI_PYTHON_REQUIRES),
    "geochemistrypi-mcp": (SERVER_VERSION, MCP_PYTHON_REQUIRES),
}
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "release_id",
    "release_tag",
    "channel",
    "generated_at",
    "source_commit",
    "public_release_ready",
    "pending_release_gates",
    "compatibility",
    "publication",
    "signing",
    "artifacts",
}
_ARTIFACT_FIELDS = {
    "distribution",
    "version",
    "filename",
    "size_bytes",
    "sha256",
    "requires_python",
}


class ReleaseError(RuntimeError):
    """Raised when a release bundle is incomplete, ambiguous, or unsafe."""


def _canonical_distribution(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _requirement_tokens(value: str) -> frozenset[str]:
    return frozenset(part.strip() for part in value.split(",") if part.strip())


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class WheelMetadata:
    """Security-relevant metadata read directly from one wheel archive."""

    distribution: str
    version: str
    requires_python: str
    file_count: int


def inspect_wheel(path: Path) -> WheelMetadata:
    """Read one wheel and reject repository test modules in production output."""
    if path.is_symlink():
        raise ReleaseError(f"Release wheels must not be symbolic links: {path.name}")
    try:
        with ZipFile(path) as archive:
            names = archive.namelist()
            metadata_names = [name for name in names if name.endswith(".dist-info/METADATA") and name.count("/") == 1]
            if len(metadata_names) != 1:
                raise ReleaseError(f"Wheel {path.name} must contain exactly one top-level METADATA file.")
            packaged_tests = [name for name in names if "tests" in Path(name).parts or Path(name).name.startswith("test_")]
            if packaged_tests:
                preview = ", ".join(packaged_tests[:5])
                raise ReleaseError(f"Production wheel {path.name} contains repository tests: {preview}")
            message = BytesParser().parsebytes(archive.read(metadata_names[0]))
    except (BadZipFile, KeyError, OSError) as exc:
        raise ReleaseError(f"Cannot inspect wheel archive {path}: {exc}") from exc
    distribution = _canonical_distribution(str(message.get("Name", "")))
    version = str(message.get("Version", ""))
    requires_python = str(message.get("Requires-Python", ""))
    if not distribution or not version or not requires_python:
        raise ReleaseError(f"Wheel {path.name} is missing Name, Version, or Requires-Python metadata.")
    return WheelMetadata(distribution, version, requires_python, len(names))


def _validate_expected_wheel(path: Path) -> WheelMetadata:
    metadata = inspect_wheel(path)
    expected = _EXPECTED_DISTRIBUTIONS.get(metadata.distribution)
    if expected is None:
        raise ReleaseError(f"Unexpected distribution {metadata.distribution!r} in {path.name}.")
    expected_version, expected_python = expected
    if metadata.version != expected_version:
        raise ReleaseError(f"Wheel {path.name} has version {metadata.version}; expected {expected_version}.")
    if _requirement_tokens(metadata.requires_python) != _requirement_tokens(expected_python):
        raise ReleaseError(f"Wheel {path.name} has Requires-Python {metadata.requires_python!r}; " f"expected {expected_python!r}.")
    return metadata


def _generated_at() -> str:
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch is not None:
        try:
            value = datetime.fromtimestamp(int(epoch), tz=timezone.utc)
        except (ValueError, OSError, OverflowError) as exc:
            raise ReleaseError("SOURCE_DATE_EPOCH must be a valid Unix timestamp.") from exc
    else:
        value = datetime.now(timezone.utc)
    return value.isoformat()


def build_release_manifest(
    distribution_directory: Path,
    *,
    release_tag: str,
    source_commit: str,
    generated_at: str | None = None,
) -> Path:
    """Generate the canonical manifest beside exactly two production wheels."""
    directory = distribution_directory.expanduser().resolve()
    if not directory.is_dir():
        raise ReleaseError(f"Release distribution directory does not exist: {directory}")
    if release_tag != EXPECTED_RELEASE_TAG:
        raise ReleaseError(f"Release tag {release_tag!r} does not match this bundle's exact versions; " f"expected {EXPECTED_RELEASE_TAG!r}.")
    if source_commit != "uncommitted" and not _COMMIT_PATTERN.fullmatch(source_commit):
        raise ReleaseError("source_commit must be a 40-character lowercase Git SHA or 'uncommitted'.")
    wheels = sorted(directory.glob("*.whl"))
    if len(wheels) != len(_EXPECTED_DISTRIBUTIONS):
        raise ReleaseError(f"Expected exactly two production wheels, found {[path.name for path in wheels]}.")
    artifacts: list[dict[str, object]] = []
    observed: set[str] = set()
    for wheel in wheels:
        metadata = _validate_expected_wheel(wheel)
        if metadata.distribution in observed:
            raise ReleaseError(f"Release directory contains more than one {metadata.distribution} wheel.")
        observed.add(metadata.distribution)
        artifacts.append(
            {
                "distribution": metadata.distribution,
                "version": metadata.version,
                "filename": wheel.name,
                "size_bytes": wheel.stat().st_size,
                "sha256": _sha256(wheel),
                "requires_python": metadata.requires_python,
            }
        )
    if observed != set(_EXPECTED_DISTRIBUTIONS):
        raise ReleaseError(f"Release distributions do not match the required pair: {sorted(observed)}.")
    artifacts.sort(key=lambda item: str(item["distribution"]))
    release_id = f"geochemistrypi-{CLI_VERSION}+mcp-{SERVER_VERSION}"
    manifest: dict[str, object] = {
        "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
        "release_id": release_id,
        "release_tag": release_tag,
        "channel": "release-candidate",
        "generated_at": generated_at or _generated_at(),
        "source_commit": source_commit,
        "public_release_ready": PUBLIC_RELEASE_READY,
        "pending_release_gates": list(PENDING_RELEASE_GATES),
        "compatibility": {
            "geochemistrypi_mcp": SERVER_VERSION,
            "geochemistrypi_cli": CLI_VERSION,
            "mcp_python": "3.11",
            "cli_python": "3.9",
            "mcp_python_requires": MCP_PYTHON_REQUIRES,
            "cli_python_requires": CLI_PYTHON_REQUIRES,
            "mcp_sdk_requires": MCP_SDK_REQUIRES,
        },
        "publication": {
            "pypi": "deferred",
            "mcp_registry": "deferred",
            "reason": "Publication requires every versioned release gate to have terminal evidence.",
        },
        "signing": {
            "scheme": "sigstore",
            "oidc_issuer": SIGSTORE_OIDC_ISSUER,
            "certificate_identity": RELEASE_WORKFLOW_IDENTITY_PREFIX + release_tag,
            "bundle_suffix": SIGSTORE_BUNDLE_SUFFIX,
            "required_for_public_release": True,
        },
        "artifacts": artifacts,
    }
    _validate_manifest_shape(manifest)
    output = directory / RELEASE_MANIFEST_FILENAME
    _atomic_write_json(output, manifest)
    return output


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReleaseError(f"Missing release manifest: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"Cannot parse release manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseError("Release manifest must contain one JSON object.")
    return value


def _validate_manifest_shape(value: Mapping[str, object]) -> None:
    if set(value) != _TOP_LEVEL_FIELDS:
        missing = sorted(_TOP_LEVEL_FIELDS - set(value))
        unknown = sorted(set(value) - _TOP_LEVEL_FIELDS)
        raise ReleaseError(f"Release manifest fields are not canonical; missing={missing}, unknown={unknown}.")
    if value.get("schema_version") != RELEASE_MANIFEST_SCHEMA_VERSION:
        raise ReleaseError("Unsupported release manifest schema version.")
    if value.get("release_tag") != EXPECTED_RELEASE_TAG or not _RELEASE_TAG_PATTERN.fullmatch(str(value.get("release_tag", ""))):
        raise ReleaseError("Release manifest tag does not match the exact supported versions.")
    if value.get("release_id") != f"geochemistrypi-{CLI_VERSION}+mcp-{SERVER_VERSION}":
        raise ReleaseError("Release manifest ID does not match the exact supported versions.")
    if value.get("channel") != "release-candidate":
        raise ReleaseError("This installer accepts only the versioned release-candidate channel.")
    generated_at = value.get("generated_at")
    if not isinstance(generated_at, str):
        raise ReleaseError("Release manifest generated_at must be an ISO-8601 timestamp.")
    try:
        timestamp = datetime.fromisoformat(generated_at)
    except ValueError as exc:
        raise ReleaseError("Release manifest generated_at must be an ISO-8601 timestamp.") from exc
    if timestamp.tzinfo is None:
        raise ReleaseError("Release manifest generated_at must include a timezone offset.")
    source_commit = str(value.get("source_commit", ""))
    if source_commit != "uncommitted" and not _COMMIT_PATTERN.fullmatch(source_commit):
        raise ReleaseError("Release manifest source commit is invalid.")
    if value.get("public_release_ready") is not PUBLIC_RELEASE_READY:
        raise ReleaseError("Release readiness does not match the compiled compatibility policy.")
    if value.get("pending_release_gates") != list(PENDING_RELEASE_GATES):
        raise ReleaseError("Release manifest pending gates do not match the compiled policy.")
    compatibility = value.get("compatibility")
    expected_compatibility = {
        "geochemistrypi_mcp": SERVER_VERSION,
        "geochemistrypi_cli": CLI_VERSION,
        "mcp_python": "3.11",
        "cli_python": "3.9",
        "mcp_python_requires": MCP_PYTHON_REQUIRES,
        "cli_python_requires": CLI_PYTHON_REQUIRES,
        "mcp_sdk_requires": MCP_SDK_REQUIRES,
    }
    if compatibility != expected_compatibility:
        raise ReleaseError("Release manifest compatibility values do not match this installer.")
    publication = value.get("publication")
    expected_publication = {
        "pypi": "deferred",
        "mcp_registry": "deferred",
        "reason": "Publication requires every versioned release gate to have terminal evidence.",
    }
    if publication != expected_publication:
        raise ReleaseError("Release publication decision must remain explicitly deferred.")
    signing = value.get("signing")
    expected_identity = RELEASE_WORKFLOW_IDENTITY_PREFIX + EXPECTED_RELEASE_TAG
    expected_signing = {
        "scheme": "sigstore",
        "oidc_issuer": SIGSTORE_OIDC_ISSUER,
        "certificate_identity": expected_identity,
        "bundle_suffix": SIGSTORE_BUNDLE_SUFFIX,
        "required_for_public_release": True,
    }
    if signing != expected_signing:
        raise ReleaseError("Release manifest signing policy is not the trusted workflow policy.")


SignatureRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _default_signature_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )


def _verify_signature(
    artifact: Path,
    *,
    certificate_identity: str,
    runner: SignatureRunner,
) -> None:
    bundle = artifact.with_name(artifact.name + SIGSTORE_BUNDLE_SUFFIX)
    if not bundle.is_file():
        raise ReleaseError(f"Missing Sigstore bundle for {artifact.name}: {bundle.name}")
    if bundle.is_symlink():
        raise ReleaseError(f"Sigstore bundles must not be symbolic links: {bundle.name}")
    command = (
        sys.executable,
        "-m",
        "sigstore",
        "verify",
        "identity",
        str(artifact),
        "--bundle",
        str(bundle),
        "--cert-identity",
        certificate_identity,
        "--cert-oidc-issuer",
        SIGSTORE_OIDC_ISSUER,
    )
    try:
        completed = runner(command)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReleaseError(f"Cannot run Sigstore verification for {artifact.name}: {exc}") from exc
    if completed.returncode != 0:
        detail = " ".join((completed.stderr or completed.stdout).split())[-1000:]
        raise ReleaseError(f"Sigstore verification failed for {artifact.name}: " f"{detail or 'verification returned a non-zero exit code'}")


@dataclass(frozen=True)
class ReleaseBundle:
    """A completely hash-checked and optionally signature-verified bundle."""

    directory: Path
    manifest_path: Path
    manifest: Mapping[str, object]
    cli_wheel: Path
    mcp_wheel: Path
    manifest_sha256: str
    signatures_verified: bool

    @property
    def fingerprint(self) -> str:
        return self.manifest_sha256

    @property
    def release_id(self) -> str:
        return str(self.manifest["release_id"])

    @property
    def release_tag(self) -> str:
        return str(self.manifest["release_tag"])

    @property
    def files(self) -> tuple[Path, ...]:
        values = [self.manifest_path, self.cli_wheel, self.mcp_wheel]
        for artifact in tuple(values):
            signature = artifact.with_name(artifact.name + SIGSTORE_BUNDLE_SUFFIX)
            if signature.is_file():
                values.append(signature)
        return tuple(values)


def verify_release_bundle(
    directory: Path,
    *,
    require_signatures: bool = True,
    signature_runner: SignatureRunner = _default_signature_runner,
) -> ReleaseBundle:
    """Verify manifest policy, exact wheel metadata, hashes, and signatures."""
    root = directory.expanduser().resolve()
    if not root.is_dir():
        raise ReleaseError(f"Release bundle directory does not exist: {root}")
    manifest_path = root / RELEASE_MANIFEST_FILENAME
    if manifest_path.is_symlink():
        raise ReleaseError("Release manifest must not be a symbolic link.")
    manifest = _load_json(manifest_path)
    _validate_manifest_shape(manifest)
    certificate_identity = RELEASE_WORKFLOW_IDENTITY_PREFIX + EXPECTED_RELEASE_TAG
    if require_signatures:
        _verify_signature(
            manifest_path,
            certificate_identity=certificate_identity,
            runner=signature_runner,
        )
    artifact_values = manifest.get("artifacts")
    if not isinstance(artifact_values, list) or len(artifact_values) != 2:
        raise ReleaseError("Release manifest must describe exactly two wheels.")
    wheels: dict[str, Path] = {}
    listed_names: set[str] = set()
    for item in artifact_values:
        if not isinstance(item, dict) or set(item) != _ARTIFACT_FIELDS:
            raise ReleaseError("Every release artifact must use the canonical artifact schema.")
        filename = item.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename or filename in listed_names:
            raise ReleaseError("Release artifact filenames must be unique safe basenames.")
        listed_names.add(filename)
        wheel = root / filename
        if not wheel.is_file():
            raise ReleaseError(f"Release artifact is missing: {filename}")
        if type(item.get("size_bytes")) is not int or item["size_bytes"] != wheel.stat().st_size:
            raise ReleaseError(f"Release artifact size does not match the manifest: {filename}")
        expected_hash = item.get("sha256")
        if not isinstance(expected_hash, str) or not _SHA256_PATTERN.fullmatch(expected_hash):
            raise ReleaseError(f"Release artifact has an invalid SHA-256 value: {filename}")
        if _sha256(wheel) != expected_hash:
            raise ReleaseError(f"Release artifact SHA-256 mismatch: {filename}")
        metadata = _validate_expected_wheel(wheel)
        if any(
            (
                item.get("distribution") != metadata.distribution,
                item.get("version") != metadata.version,
                _requirement_tokens(str(item.get("requires_python", ""))) != _requirement_tokens(metadata.requires_python),
            )
        ):
            raise ReleaseError(f"Release artifact metadata does not match the manifest: {filename}")
        if require_signatures:
            _verify_signature(
                wheel,
                certificate_identity=certificate_identity,
                runner=signature_runner,
            )
        wheels[metadata.distribution] = wheel
    extra_wheels = sorted(path.name for path in root.glob("*.whl") if path.name not in listed_names)
    if extra_wheels:
        raise ReleaseError(f"Release bundle contains unlisted wheel files: {extra_wheels}")
    if set(wheels) != set(_EXPECTED_DISTRIBUTIONS):
        raise ReleaseError("Release bundle does not contain the exact CLI/MCP distribution pair.")
    return ReleaseBundle(
        directory=root,
        manifest_path=manifest_path,
        manifest=manifest,
        cli_wheel=wheels["geochemistrypi"],
        mcp_wheel=wheels["geochemistrypi-mcp"],
        manifest_sha256=_sha256(manifest_path),
        signatures_verified=require_signatures,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geochemistrypi-mcp-release",
        description="Build or verify one exact GeochemistryPi CLI/MCP release bundle.",
    )
    subcommands = parser.add_subparsers(dest="action", required=True)
    build = subcommands.add_parser("build-manifest", help="Create the canonical release manifest beside two wheels.")
    build.add_argument("--dist", type=Path, required=True, help="Directory containing exactly the two production wheels.")
    build.add_argument("--release-tag", default=EXPECTED_RELEASE_TAG)
    build.add_argument("--source-commit", required=True)
    verify = subcommands.add_parser("verify", help="Verify release manifest, wheel hashes, metadata, and signatures.")
    verify.add_argument("--bundle", type=Path, required=True)
    verify.add_argument(
        "--allow-unsigned",
        action="store_true",
        help="Skip Sigstore only for an explicit local release-candidate check.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.action == "build-manifest":
            output = build_release_manifest(
                arguments.dist,
                release_tag=arguments.release_tag,
                source_commit=arguments.source_commit,
            )
            print(f"Created release manifest: {output}")
        else:
            bundle = verify_release_bundle(
                arguments.bundle,
                require_signatures=not arguments.allow_unsigned,
            )
            signature_state = "verified" if bundle.signatures_verified else "explicitly skipped"
            print(f"Verified {bundle.release_id}: two exact wheels, SHA-256 hashes match, " f"signatures {signature_state}.")
    except ReleaseError as exc:
        print(f"GeochemistryPi release verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
