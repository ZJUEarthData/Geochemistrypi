"""Verify that CLI release artifacts come from one clean, versioned source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import subprocess
import sys
import tarfile
import tomllib
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence
from zipfile import BadZipFile, ZipFile

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name


class ArtifactIntegrityError(RuntimeError):
    """Raised when release source, metadata, or bytes are not identical."""


@dataclass(frozen=True)
class ReleaseVersions:
    """Versions that must agree before a release can be built or published."""

    cli: str
    mcp: str
    cli_tag: str
    release_tag: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_utf8(value: bytes, label: str) -> str:
    try:
        return value.decode("utf-8").replace("\r\n", "\n")
    except UnicodeError as exc:
        raise ArtifactIntegrityError(f"{label} must be valid UTF-8: {exc}") from exc


def _same_packaged_content(source: bytes, packaged: bytes) -> bool:
    if source == packaged:
        return True
    try:
        return source.decode("utf-8").replace("\r\n", "\n") == packaged.decode("utf-8").replace("\r\n", "\n")
    except UnicodeError:
        return False


def _project(path: Path) -> Mapping[str, object]:
    try:
        value = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise ArtifactIntegrityError(f"Cannot read project metadata {path}: {exc}") from exc
    project = value.get("project")
    if not isinstance(project, dict):
        raise ArtifactIntegrityError(f"Project metadata is missing [project]: {path}")
    return project


def _module_values(path: Path) -> Mapping[str, object]:
    try:
        return runpy.run_path(str(path))
    except (OSError, SyntaxError) as exc:
        raise ArtifactIntegrityError(f"Cannot read version constants {path}: {exc}") from exc


def release_versions(repository: Path) -> ReleaseVersions:
    """Resolve and cross-check every source-of-truth version declaration."""
    root = repository.expanduser().resolve()
    cli_project = _project(root / "pyproject.toml")
    mcp_project = _project(root / "packages" / "geochemistrypi-mcp" / "pyproject.toml")
    cli_module = _module_values(root / "geochemistrypi" / "_version.py")
    constants = _module_values(
        root
        / "packages"
        / "geochemistrypi-mcp"
        / "src"
        / "geochemistrypi_mcp"
        / "config"
        / "constants.py"
    )
    capability_path = (
        root
        / "packages"
        / "geochemistrypi-mcp"
        / "src"
        / "geochemistrypi_mcp"
        / "contracts"
        / "cli_capability_manifest_v1.json"
    )
    try:
        capability = json.loads(capability_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(f"Cannot read CLI capability manifest: {exc}") from exc

    cli = str(cli_project.get("version", ""))
    mcp = str(mcp_project.get("version", ""))
    cli_values = {
        "root pyproject": cli,
        "geochemistrypi._version": str(cli_module.get("__version__", "")),
        "MCP CLI_VERSION": str(constants.get("CLI_VERSION", "")),
        "MCP capability manifest": str(capability.get("cli_version", "")),
    }
    if not cli or any(value != cli for value in cli_values.values()):
        raise ArtifactIntegrityError(f"CLI version declarations disagree: {cli_values}")
    if mcp != str(constants.get("SERVER_VERSION", "")):
        raise ArtifactIntegrityError(
            "MCP version declarations disagree: "
            f"pyproject={mcp!r}, SERVER_VERSION={constants.get('SERVER_VERSION')!r}"
        )
    supported = constants.get("SUPPORTED_CLI_VERSIONS")
    if supported != (cli,):
        raise ArtifactIntegrityError(
            f"MCP must fail closed to exactly CLI {cli}; found SUPPORTED_CLI_VERSIONS={supported!r}"
        )
    return ReleaseVersions(
        cli=cli,
        mcp=mcp,
        cli_tag=f"v{cli}",
        release_tag=f"mcp-v{mcp}-cli-v{cli}",
    )


def verify_clean_tagged_source(repository: Path, release_tag: str) -> ReleaseVersions:
    """Require clean source with annotated CLI and MCP Tags on the same commit."""
    root = repository.expanduser().resolve()
    versions = release_versions(root)
    if release_tag != versions.release_tag:
        raise ArtifactIntegrityError(
            f"Release Tag {release_tag!r} does not match source versions; expected {versions.release_tag!r}."
        )

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if completed.returncode != 0:
            detail = " ".join((completed.stderr or completed.stdout).split())[-1000:]
            raise ArtifactIntegrityError(f"Git source verification failed: {detail}")
        return completed.stdout.strip()

    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ArtifactIntegrityError(f"Release checkout is not clean:\n{status}")
    head = git("rev-parse", "HEAD")
    for tag in (versions.cli_tag, release_tag):
        tag_type = git("cat-file", "-t", f"refs/tags/{tag}")
        if tag_type != "tag":
            raise ArtifactIntegrityError(
                f"Release Tag {tag} must be annotated; found Git object type {tag_type!r}."
            )
    cli_tagged = git("rev-parse", f"{versions.cli_tag}^{{}}")
    release_tagged = git("rev-parse", f"{release_tag}^{{}}")
    if head != cli_tagged or head != release_tagged:
        raise ArtifactIntegrityError(
            "The CLI and MCP release Tags must resolve to the exact build commit: "
            f"HEAD={head}, {versions.cli_tag}={cli_tagged}, {release_tag}={release_tagged}."
        )
    return versions


def _requirement_key(value: str) -> tuple[object, ...]:
    try:
        requirement = Requirement(value)
    except InvalidRequirement as exc:
        raise ArtifactIntegrityError(f"Invalid dependency requirement {value!r}: {exc}") from exc
    return (
        canonicalize_name(requirement.name),
        tuple(sorted(canonicalize_name(extra) for extra in requirement.extras)),
        str(requirement.specifier),
        requirement.url or "",
        str(requirement.marker) if requirement.marker is not None else "",
    )


def _requirements(values: Iterable[str]) -> tuple[tuple[object, ...], ...]:
    return tuple(sorted(_requirement_key(value) for value in values))


def _runtime_requirements(values: Iterable[str]) -> tuple[tuple[object, ...], ...]:
    runtime: list[str] = []
    for value in values:
        try:
            requirement = Requirement(value)
        except InvalidRequirement as exc:
            raise ArtifactIntegrityError(f"Invalid dependency requirement {value!r}: {exc}") from exc
        marker = str(requirement.marker) if requirement.marker is not None else ""
        if "extra ==" not in marker:
            runtime.append(value)
    return _requirements(runtime)


def _metadata_from_wheel(path: Path) -> tuple[Mapping[str, str], tuple[str, ...], Mapping[str, bytes]]:
    try:
        with ZipFile(path) as archive:
            names = archive.namelist()
            metadata_names = [name for name in names if name.endswith(".dist-info/METADATA") and name.count("/") == 1]
            if len(metadata_names) != 1:
                raise ArtifactIntegrityError(f"CLI wheel must contain one top-level METADATA file: {path.name}")
            message = BytesParser(policy=default).parsebytes(archive.read(metadata_names[0]))
            package_files = {
                name: archive.read(name)
                for name in names
                if name.startswith("geochemistrypi/") and not name.endswith("/")
            }
    except (BadZipFile, KeyError, OSError) as exc:
        raise ArtifactIntegrityError(f"Cannot inspect CLI wheel {path}: {exc}") from exc
    fields = {
        "name": str(message.get("Name", "")),
        "version": str(message.get("Version", "")),
        "requires_python": str(message.get("Requires-Python", "")),
    }
    return fields, tuple(str(value) for value in message.get_all("Requires-Dist", [])), package_files


def _metadata_from_sdist(path: Path) -> tuple[Mapping[str, str], tuple[str, ...], bytes, Mapping[str, bytes]]:
    try:
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
            if any(member.issym() or member.islnk() for member in members):
                raise ArtifactIntegrityError(f"CLI sdist contains symbolic or hard links: {path.name}")
            roots = {PurePosixPath(member.name).parts[0] for member in members if PurePosixPath(member.name).parts}
            if len(roots) != 1:
                raise ArtifactIntegrityError(f"CLI sdist must contain one top-level directory: {sorted(roots)}")
            root = next(iter(roots))

            def read(name: str) -> bytes:
                member = archive.getmember(f"{root}/{name}")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ArtifactIntegrityError(f"CLI sdist member is not a regular file: {name}")
                return stream.read()

            pyproject = read("pyproject.toml")
            message = BytesParser(policy=default).parsebytes(read("PKG-INFO"))
            package_files: dict[str, bytes] = {}
            prefix = f"{root}/geochemistrypi/"
            for member in members:
                if not member.isfile() or not member.name.startswith(prefix):
                    continue
                relative = member.name[len(root) + 1 :]
                stream = archive.extractfile(member)
                if stream is None:
                    raise ArtifactIntegrityError(f"Cannot read CLI sdist member: {member.name}")
                package_files[relative] = stream.read()
    except (tarfile.TarError, KeyError, OSError) as exc:
        raise ArtifactIntegrityError(f"Cannot inspect CLI sdist {path}: {exc}") from exc
    fields = {
        "name": str(message.get("Name", "")),
        "version": str(message.get("Version", "")),
        "requires_python": str(message.get("Requires-Python", "")),
    }
    return fields, tuple(str(value) for value in message.get_all("Requires-Dist", [])), pyproject, package_files


def _verify_metadata(
    label: str,
    fields: Mapping[str, str],
    dependencies: Sequence[str],
    project: Mapping[str, object],
) -> None:
    expected_name = str(project.get("name", ""))
    expected_version = str(project.get("version", ""))
    expected_python = str(project.get("requires-python", ""))
    if canonicalize_name(fields["name"]) != canonicalize_name(expected_name):
        raise ArtifactIntegrityError(f"{label} project name does not match pyproject.toml.")
    if fields["version"] != expected_version:
        raise ArtifactIntegrityError(
            f"{label} version {fields['version']!r} does not match pyproject.toml {expected_version!r}."
        )
    try:
        observed_python = SpecifierSet(fields["requires_python"])
        required_python = SpecifierSet(expected_python)
    except InvalidSpecifier as exc:
        raise ArtifactIntegrityError(f"{label} contains invalid Requires-Python metadata: {exc}") from exc
    if observed_python != required_python:
        raise ArtifactIntegrityError(
            f"{label} Requires-Python {fields['requires_python']!r} does not match {expected_python!r}."
        )
    expected_dependencies = project.get("dependencies")
    if not isinstance(expected_dependencies, list) or not all(isinstance(value, str) for value in expected_dependencies):
        raise ArtifactIntegrityError("CLI pyproject.toml dependencies must be a list of requirement strings.")
    if _runtime_requirements(dependencies) != _requirements(expected_dependencies):
        raise ArtifactIntegrityError(f"{label} dependency metadata does not match pyproject.toml exactly.")


def _verify_packaged_source(label: str, repository: Path, files: Mapping[str, bytes]) -> None:
    if not files:
        raise ArtifactIntegrityError(f"{label} contains no GeochemistryPi package files.")
    for relative, packaged in files.items():
        path = repository / PurePosixPath(relative)
        if not path.is_file():
            raise ArtifactIntegrityError(f"{label} contains source absent from the release checkout: {relative}")
        if not _same_packaged_content(path.read_bytes(), packaged):
            raise ArtifactIntegrityError(f"{label} source does not match the release checkout: {relative}")


def verify_cli_artifacts(
    repository: Path,
    cli_distribution: Path,
    release_bundle: Path,
    release_tag: str,
) -> ReleaseVersions:
    """Verify the exact CLI wheel/sdist and reuse of that wheel by the MCP bundle."""
    root = repository.expanduser().resolve()
    cli_dist = cli_distribution.expanduser().resolve()
    bundle = release_bundle.expanduser().resolve()
    versions = release_versions(root)
    if release_tag != versions.release_tag:
        raise ArtifactIntegrityError(
            f"Release Tag {release_tag!r} does not match source versions; expected {versions.release_tag!r}."
        )
    wheels = sorted(cli_dist.glob("*.whl"))
    sdists = sorted(cli_dist.glob("*.tar.gz"))
    unexpected = sorted(path.name for path in cli_dist.iterdir() if path.is_file() and path not in {*wheels, *sdists})
    if len(wheels) != 1 or len(sdists) != 1 or unexpected:
        raise ArtifactIntegrityError(
            "CLI publication directory must contain exactly one wheel and one sdist; "
            f"wheels={[path.name for path in wheels]}, sdists={[path.name for path in sdists]}, unexpected={unexpected}."
        )
    bundle_wheels = sorted(bundle.glob("geochemistrypi-*.whl"))
    if len(bundle_wheels) != 1:
        raise ArtifactIntegrityError(
            f"MCP release bundle must contain exactly one CLI wheel; found {[path.name for path in bundle_wheels]}."
        )
    if wheels[0].name != bundle_wheels[0].name or _sha256(wheels[0]) != _sha256(bundle_wheels[0]):
        raise ArtifactIntegrityError("MCP bundle CLI wheel is not byte-for-byte identical to the PyPI CLI wheel.")

    project = _project(root / "pyproject.toml")
    wheel_fields, wheel_dependencies, wheel_files = _metadata_from_wheel(wheels[0])
    sdist_fields, sdist_dependencies, packaged_pyproject, sdist_files = _metadata_from_sdist(sdists[0])
    _verify_metadata("CLI wheel", wheel_fields, wheel_dependencies, project)
    _verify_metadata("CLI sdist", sdist_fields, sdist_dependencies, project)
    if _normalized_utf8(packaged_pyproject, "CLI sdist pyproject.toml") != _normalized_utf8(
        (root / "pyproject.toml").read_bytes(),
        "Release checkout pyproject.toml",
    ):
        raise ArtifactIntegrityError("CLI sdist pyproject.toml content does not match the release checkout.")
    _verify_packaged_source("CLI wheel", root, wheel_files)
    _verify_packaged_source("CLI sdist", root, sdist_files)
    return versions


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify clean source and exact GeochemistryPi release artifacts.")
    subcommands = parser.add_subparsers(dest="action", required=True)
    source = subcommands.add_parser("verify-source")
    source.add_argument("--repository", type=Path, required=True)
    source.add_argument("--release-tag", required=True)
    artifacts = subcommands.add_parser("verify-artifacts")
    artifacts.add_argument("--repository", type=Path, required=True)
    artifacts.add_argument("--cli-dist", type=Path, required=True)
    artifacts.add_argument("--release-bundle", type=Path, required=True)
    artifacts.add_argument("--release-tag")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.action == "verify-source":
            versions = verify_clean_tagged_source(arguments.repository, arguments.release_tag)
            print(f"Verified clean tagged source for CLI {versions.cli} and MCP {versions.mcp}.")
        else:
            release_tag = arguments.release_tag or release_versions(arguments.repository).release_tag
            versions = verify_cli_artifacts(
                arguments.repository,
                arguments.cli_dist,
                arguments.release_bundle,
                release_tag,
            )
            print(
                "Verified exact CLI wheel/sdist metadata, source bytes, and MCP bundle reuse for "
                f"CLI {versions.cli} and MCP {versions.mcp}."
            )
    except ArtifactIntegrityError as exc:
        print(f"GeochemistryPi release artifact verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
