import json
import subprocess
import tomllib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from geochemistrypi_mcp.config.constants import CLI_PYTHON_REQUIRES, MCP_PYTHON_REQUIRES, SERVER_VERSION
from geochemistrypi_mcp.lifecycle.release import CLI_VERSION, EXPECTED_RELEASE_TAG, RELEASE_MANIFEST_FILENAME, SIGSTORE_BUNDLE_SUFFIX, ReleaseError, build_release_manifest, verify_release_bundle

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _wheel(
    root: Path,
    distribution: str,
    version: str,
    requires_python: str,
    *,
    packaged_test: bool = False,
) -> Path:
    filename_distribution = distribution.replace("-", "_")
    wheel = root / f"{filename_distribution}-{version}-py3-none-any.whl"
    metadata_root = f"{filename_distribution}-{version}.dist-info"
    with ZipFile(wheel, "w", ZIP_DEFLATED) as archive:
        archive.writestr(
            f"{metadata_root}/METADATA",
            "Metadata-Version: 2.4\n" f"Name: {distribution}\n" f"Version: {version}\n" f"Requires-Python: {requires_python}\n",
        )
        archive.writestr(
            f"{metadata_root}/WHEEL",
            "Wheel-Version: 1.0\nTag: py3-none-any\n",
        )
        archive.writestr(
            f"{filename_distribution}/__init__.py",
            "__all__ = ()\n",
        )
        if packaged_test:
            archive.writestr(
                f"{filename_distribution}/tests/test_leak.py",
                "raise AssertionError\n",
            )
    return wheel


def _bundle(tmp_path: Path) -> Path:
    _wheel(tmp_path, "geochemistrypi", "0.8.0", CLI_PYTHON_REQUIRES)
    _wheel(tmp_path, "geochemistrypi-mcp", SERVER_VERSION, MCP_PYTHON_REQUIRES)
    return build_release_manifest(
        tmp_path,
        release_tag=EXPECTED_RELEASE_TAG,
        source_commit="1" * 40,
        generated_at="2026-08-04T00:00:00+00:00",
    )


def test_release_manifest_records_exact_versions_hashes_and_deferred_publication(
    tmp_path: Path,
) -> None:
    manifest_path = _bundle(tmp_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["release_tag"] == EXPECTED_RELEASE_TAG
    assert manifest["source_commit"] == "1" * 40
    assert manifest["public_release_ready"] is False
    assert manifest["publication"]["pypi"] == "deferred"
    assert manifest["publication"]["mcp_registry"] == "deferred"
    assert [value["distribution"] for value in manifest["artifacts"]] == [
        "geochemistrypi",
        "geochemistrypi-mcp",
    ]
    assert all(len(value["sha256"]) == 64 for value in manifest["artifacts"])

    bundle = verify_release_bundle(tmp_path, require_signatures=False)
    assert bundle.manifest_path.name == RELEASE_MANIFEST_FILENAME
    assert bundle.signatures_verified is False


def test_release_version_has_one_canonical_package_value() -> None:
    project = tomllib.loads((REPOSITORY_ROOT / "packages" / "geochemistrypi-mcp" / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["version"] == SERVER_VERSION
    assert EXPECTED_RELEASE_TAG == f"mcp-v{SERVER_VERSION}-cli-v{CLI_VERSION}"


def test_release_workflow_installs_the_signed_artifact_on_every_supported_os() -> None:
    release_workflow = (REPOSITORY_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    engine_workflow = (REPOSITORY_ROOT / ".github" / "workflows" / "geochemistrypi.yml").read_text(encoding="utf-8")
    signed_job = release_workflow.split("  verify-signed-release:", maxsplit=1)[1]

    assert "os: [ubuntu-latest, windows-latest, macos-15-intel]" in signed_job
    assert "uses: actions/download-artifact@v4" in signed_job
    assert 'run("geochemistrypi-mcp-release", "verify", "--bundle", str(bundle))' in signed_job
    assert '"geochemistrypi-mcp-setup", "install"' in signed_job
    assert 'run("geochemistrypi-mcp-doctor", "--json")' in signed_job
    assert 'run("geochemistrypi-mcp-setup", "uninstall")' in signed_job
    assert "--allow-unsigned" not in signed_job
    assert "--release-tag mcp-v" not in engine_workflow


def test_release_bundle_verifies_every_sigstore_identity_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    manifest_path = _bundle(tmp_path)
    signed_files = [manifest_path, *sorted(tmp_path.glob("*.whl"))]
    for artifact in signed_files:
        artifact.with_name(artifact.name + SIGSTORE_BUNDLE_SUFFIX).write_text(
            "{}\n",
            encoding="utf-8",
        )
    commands: list[tuple[str, ...]] = []

    def signature_runner(command):
        commands.append(tuple(command))
        return subprocess.CompletedProcess(command, 0, "verified", "")

    bundle = verify_release_bundle(
        tmp_path,
        signature_runner=signature_runner,
    )

    assert bundle.signatures_verified is True
    assert len(commands) == 3
    assert all("--offline" in command for command in commands)
    assert all("--cert-identity" in command for command in commands)
    assert all("--cert-oidc-issuer" in command for command in commands)

    bundle.cli_wheel.write_bytes(bundle.cli_wheel.read_bytes() + b"tampered")
    with pytest.raises(ReleaseError, match="size does not match"):
        verify_release_bundle(tmp_path, require_signatures=False)


def test_release_signature_verification_is_offline_and_path_safe(tmp_path: Path) -> None:
    bundle_root = tmp_path / "公众 发布包 with spaces"
    bundle_root.mkdir()
    manifest_path = _bundle(bundle_root)
    signed_files = [manifest_path, *sorted(bundle_root.glob("*.whl"))]
    for artifact in signed_files:
        artifact.with_name(artifact.name + SIGSTORE_BUNDLE_SUFFIX).write_text(
            "{}\n",
            encoding="utf-8",
        )

    observed: list[tuple[str, ...]] = []

    def signature_runner(command):
        command = tuple(command)
        observed.append(command)
        assert "--offline" in command
        assert any("公众 发布包 with spaces" in argument for argument in command)
        return subprocess.CompletedProcess(command, 0, "verified", "")

    bundle = verify_release_bundle(bundle_root, signature_runner=signature_runner)

    assert bundle.signatures_verified is True
    assert len(observed) == 3


def test_release_bundle_fails_closed_for_missing_signatures_and_packaged_tests(
    tmp_path: Path,
) -> None:
    _bundle(tmp_path)
    with pytest.raises(ReleaseError, match="Missing Sigstore bundle"):
        verify_release_bundle(tmp_path)

    unsafe = tmp_path / "unsafe"
    unsafe.mkdir()
    _wheel(
        unsafe,
        "geochemistrypi",
        "0.8.0",
        ">=3.9,<3.10",
        packaged_test=True,
    )
    _wheel(unsafe, "geochemistrypi-mcp", SERVER_VERSION, MCP_PYTHON_REQUIRES)
    with pytest.raises(ReleaseError, match="contains repository tests"):
        build_release_manifest(
            unsafe,
            release_tag=EXPECTED_RELEASE_TAG,
            source_commit="2" * 40,
        )


def test_release_manifest_rejects_wrong_tag_and_ambiguous_wheel_sets(
    tmp_path: Path,
) -> None:
    _wheel(tmp_path, "geochemistrypi", "0.8.0", CLI_PYTHON_REQUIRES)
    _wheel(tmp_path, "geochemistrypi-mcp", SERVER_VERSION, MCP_PYTHON_REQUIRES)
    with pytest.raises(ReleaseError, match="exact versions"):
        build_release_manifest(
            tmp_path,
            release_tag="mcp-v9.9.9-cli-v9.9.9",
            source_commit="3" * 40,
        )

    (tmp_path / "unexpected-1.0-py3-none-any.whl").write_bytes(b"not-a-wheel")
    with pytest.raises(ReleaseError, match="exactly two"):
        build_release_manifest(
            tmp_path,
            release_tag=EXPECTED_RELEASE_TAG,
            source_commit="3" * 40,
        )
