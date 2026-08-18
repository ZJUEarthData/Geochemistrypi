import importlib.util
import io
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPOSITORY_ROOT / "packages" / "geochemistrypi-mcp" / "tools" / "release_artifacts.py"
SPEC = importlib.util.spec_from_file_location("geochemistrypi_release_artifacts", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
release_artifacts = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = release_artifacts
SPEC.loader.exec_module(release_artifacts)

ArtifactIntegrityError = release_artifacts.ArtifactIntegrityError
verify_cli_artifacts = release_artifacts.verify_cli_artifacts
verify_clean_tagged_source = release_artifacts.verify_clean_tagged_source


def _write_project(root: Path) -> tuple[Path, Path, str]:
    cli_version = "0.8.1"
    mcp_version = "0.2.1"
    pyproject = "[project]\n" 'name = "geochemistrypi"\n' f'version = "{cli_version}"\n' 'requires-python = ">=3.9,<3.10"\n' 'dependencies = ["flaml==1.0.14", "ray==2.2.0"]\n'
    (root / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    package = root / "geochemistrypi"
    package.mkdir()
    (package / "__init__.py").write_text("PACKAGE = 'source'\n", encoding="utf-8")
    (package / "_version.py").write_text(f'__version__ = "{cli_version}"\n', encoding="utf-8")

    mcp_root = root / "packages" / "geochemistrypi-mcp"
    mcp_root.mkdir(parents=True)
    (mcp_root / "pyproject.toml").write_text(
        "[project]\n" f'name = "geochemistrypi-mcp"\nversion = "{mcp_version}"\n',
        encoding="utf-8",
    )
    constants = mcp_root / "src" / "geochemistrypi_mcp" / "config"
    constants.mkdir(parents=True)
    (constants / "constants.py").write_text(
        f'CLI_VERSION = "{cli_version}"\n' "SUPPORTED_CLI_VERSIONS = (CLI_VERSION,)\n" f'SERVER_VERSION = "{mcp_version}"\n',
        encoding="utf-8",
    )
    contracts = constants.parent / "contracts"
    contracts.mkdir()
    (contracts / "cli_capability_manifest_v1.json").write_text(
        '{"cli_version": "0.8.1"}\n',
        encoding="utf-8",
    )
    return package, mcp_root, pyproject


def _wheel(root: Path, *, dependency: str = "flaml==1.0.14") -> Path:
    wheel = root / "geochemistrypi-0.8.1-py3-none-any.whl"
    with ZipFile(wheel, "w", ZIP_DEFLATED) as archive:
        archive.writestr(
            "geochemistrypi-0.8.1.dist-info/METADATA",
            "Metadata-Version: 2.4\n"
            "Name: geochemistrypi\n"
            "Version: 0.8.1\n"
            "Requires-Python: >=3.9,<3.10\n"
            f"Requires-Dist: {dependency}\n"
            "Requires-Dist: ray==2.2.0\n"
            "Requires-Dist: pytest; extra == 'test'\n",
        )
        archive.writestr("geochemistrypi/__init__.py", "PACKAGE = 'source'\n")
        archive.writestr("geochemistrypi/_version.py", '__version__ = "0.8.1"\n')
    return wheel


def _add_tar_bytes(archive: tarfile.TarFile, name: str, value: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(value)
    archive.addfile(info, io.BytesIO(value))


def _sdist(root: Path, pyproject: str, *, dependency: str = "flaml==1.0.14") -> Path:
    sdist = root / "geochemistrypi-0.8.1.tar.gz"
    prefix = "geochemistrypi-0.8.1"
    with tarfile.open(sdist, "w:gz") as archive:
        _add_tar_bytes(archive, f"{prefix}/pyproject.toml", pyproject.encode())
        _add_tar_bytes(
            archive,
            f"{prefix}/PKG-INFO",
            (
                "Metadata-Version: 2.4\n"
                "Name: geochemistrypi\n"
                "Version: 0.8.1\n"
                "Requires-Python: >=3.9,<3.10\n"
                f"Requires-Dist: {dependency}\n"
                "Requires-Dist: ray==2.2.0\n"
                "Requires-Dist: pytest; extra == 'test'\n"
            ).encode(),
        )
        _add_tar_bytes(archive, f"{prefix}/geochemistrypi/__init__.py", b"PACKAGE = 'source'\n")
        _add_tar_bytes(archive, f"{prefix}/geochemistrypi/_version.py", b'__version__ = "0.8.1"\n')
    return sdist


def _release_tree(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _, _, pyproject = _write_project(repository)
    cli_dist = tmp_path / "cli-dist"
    bundle = tmp_path / "release-bundle"
    cli_dist.mkdir()
    bundle.mkdir()
    wheel = _wheel(cli_dist)
    _sdist(cli_dist, pyproject)
    shutil.copy2(wheel, bundle / wheel.name)
    return repository, cli_dist, bundle, "mcp-v0.2.1-cli-v0.8.1"


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _commit_repository(repository: Path, message: str) -> str:
    _git(repository, "add", ".")
    _git(
        repository,
        "-c",
        "user.name=Release Test",
        "-c",
        "user.email=release-test@example.invalid",
        "commit",
        "-m",
        message,
    )
    return _git(repository, "rev-parse", "HEAD")


def test_release_artifacts_match_source_metadata_and_reuse_exact_cli_wheel(tmp_path: Path) -> None:
    repository, cli_dist, bundle, release_tag = _release_tree(tmp_path)

    versions = verify_cli_artifacts(repository, cli_dist, bundle, release_tag)

    assert versions.cli == "0.8.1"
    assert versions.mcp == "0.2.1"
    assert versions.cli_tag == "v0.8.1"


def test_release_artifacts_reject_broadened_wheel_dependency_metadata(tmp_path: Path) -> None:
    repository, cli_dist, bundle, release_tag = _release_tree(tmp_path)
    wheel = _wheel(cli_dist, dependency="flaml>=1.0.14")
    shutil.copy2(wheel, bundle / wheel.name)

    with pytest.raises(ArtifactIntegrityError, match="dependency metadata"):
        verify_cli_artifacts(repository, cli_dist, bundle, release_tag)


def test_release_artifacts_reject_sdist_pyproject_not_from_checkout(tmp_path: Path) -> None:
    repository, cli_dist, bundle, release_tag = _release_tree(tmp_path)
    changed = (repository / "pyproject.toml").read_text(encoding="utf-8").replace("flaml==1.0.14", "flaml>=1.0.14")
    _sdist(cli_dist, changed, dependency="flaml>=1.0.14")

    with pytest.raises(ArtifactIntegrityError, match="dependency metadata|pyproject.toml"):
        verify_cli_artifacts(repository, cli_dist, bundle, release_tag)


def test_release_source_requires_annotated_cli_and_mcp_tags_on_same_commit(tmp_path: Path) -> None:
    repository, _, _, release_tag = _release_tree(tmp_path)
    _git(repository, "init")
    commit = _commit_repository(repository, "release source")
    for tag in ("v0.8.1", release_tag):
        _git(
            repository,
            "-c",
            "user.name=Release Test",
            "-c",
            "user.email=release-test@example.invalid",
            "tag",
            "-a",
            tag,
            "-m",
            tag,
        )

    versions = verify_clean_tagged_source(repository, release_tag)

    assert _git(repository, "rev-parse", f"{versions.cli_tag}^{{}}") == commit
    assert _git(repository, "rev-parse", f"{versions.release_tag}^{{}}") == commit


def test_release_workflow_preserves_annotated_tag_objects() -> None:
    workflow = (REPOSITORY_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    checkout_versions = re.findall(r"actions/checkout@v(\d+)\.(\d+)\.(\d+)", workflow)

    assert checkout_versions
    assert all(tuple(int(part) for part in version) >= (6, 0, 2) for version in checkout_versions)

    tagged_checkout = workflow.split("- name: Check out the tagged source", 1)[1].split("- name: Set up release verification Python", 1)[0]
    assert "fetch-depth: 0" in tagged_checkout
    assert "fetch-tags: true" in tagged_checkout
    assert "release_artifacts.py verify-source" in workflow


def test_release_source_rejects_cli_tag_on_a_different_commit(tmp_path: Path) -> None:
    repository, _, _, release_tag = _release_tree(tmp_path)
    _git(repository, "init")
    _commit_repository(repository, "CLI source")
    _git(
        repository,
        "-c",
        "user.name=Release Test",
        "-c",
        "user.email=release-test@example.invalid",
        "tag",
        "-a",
        "v0.8.1",
        "-m",
        "CLI 0.8.1",
    )
    (repository / "README.md").write_text("MCP release\n", encoding="utf-8")
    _commit_repository(repository, "MCP source")
    _git(
        repository,
        "-c",
        "user.name=Release Test",
        "-c",
        "user.email=release-test@example.invalid",
        "tag",
        "-a",
        release_tag,
        "-m",
        release_tag,
    )

    with pytest.raises(ArtifactIntegrityError, match="exact build commit"):
        verify_clean_tagged_source(repository, release_tag)
