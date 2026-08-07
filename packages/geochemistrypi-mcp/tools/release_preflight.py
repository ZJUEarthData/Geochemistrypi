"""Run the complete local release-candidate gate without changing Git state.

This script is intentionally outside the production package.  It exercises the
same built wheels that users receive while keeping every temporary runtime and
artifact outside the repository.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MCP_PROJECT = REPOSITORY_ROOT / "packages" / "geochemistrypi-mcp"
PYTEST_CONFIG = REPOSITORY_ROOT / "tests" / "installed-wheel-pytest.ini"
MCP_TESTS = (
    REPOSITORY_ROOT / "tests" / "mcp_wrapper" / "installation",
    REPOSITORY_ROOT / "tests" / "mcp_wrapper" / "interaction",
    REPOSITORY_ROOT / "tests" / "mcp_wrapper" / "protocol",
)
FULL_PARITY_SHARDS = (
    "classification-manual",
    "classification-automl",
    "regression-manual",
    "regression-automl",
    "unsupervised-manual",
    "aggregates",
    "branches-rendering",
)


def _environment(overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "PYTHONHOME",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "__PYVENV_LAUNCHER__",
        "SQLALCHEMY_DATABASE_URL",
        "GEOCHEMISTRYPI_CLI_EXECUTABLE",
        "GEOCHEMISTRYPI_FULL_PARITY",
        "GEOCHEMISTRYPI_PARITY_SHARD",
        "GEOCHEMISTRYPI_MCP_APP_ROOT",
        "GEOCHEMISTRYPI_MCP_RUNS_ROOT",
        "GEOCHEMISTRYPI_MCP_TRACKING_ROOT",
        "GEOCHEMISTRYPI_MCP_SERVICE_STATE_ROOT",
        "GEOCHEMISTRYPI_MCP_SETTINGS_FILE",
        "GEOCHEMISTRYPI_MCP_MAX_DATASET_BYTES",
        "GEOCHEMISTRYPI_MCP_MAX_PENDING_RUNS",
        "GEOCHEMISTRYPI_MCP_MAX_PROCESS_SECONDS",
    ):
        environment.pop(name, None)
    environment["MPLBACKEND"] = "Agg"
    if overrides:
        environment.update(overrides)
    return environment


def _display(command: Sequence[object]) -> str:
    return subprocess.list2cmdline([str(value) for value in command])


def _run(
    command: Sequence[object],
    *,
    cwd: Path = REPOSITORY_ROOT,
    environment: Mapping[str, str] | None = None,
) -> None:
    values = [str(value) for value in command]
    print(f"\n>>> {_display(values)}", flush=True)
    subprocess.run(values, cwd=cwd, env=dict(environment or _environment()), check=True)


def _venv_python(environment_root: Path) -> Path:
    directory = "Scripts" if os.name == "nt" else "bin"
    executable = "python.exe" if os.name == "nt" else "python"
    return environment_root / directory / executable


def _venv_command(environment_root: Path, name: str) -> Path:
    directory = "Scripts" if os.name == "nt" else "bin"
    suffix = ".exe" if os.name == "nt" else ""
    return environment_root / directory / f"{name}{suffix}"


def _create_environment(root: Path, python_version: str) -> Path:
    _run(("uv", "venv", "--python", python_version, root), cwd=root.parent)
    python = _venv_python(root)
    if not python.is_file():
        raise RuntimeError(f"uv did not create the expected interpreter: {python}")
    return python


def _only_wheel(bundle: Path, pattern: str) -> Path:
    wheels = sorted(bundle.glob(pattern))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected one {pattern} wheel, found: {wheels}")
    return wheels[0].resolve()


def _source_commit() -> str:
    status = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if status.stdout.strip():
        return "uncommitted"
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    ).stdout.strip()
    if len(commit) != 40:
        raise RuntimeError(f"Unexpected Git commit value: {commit!r}")
    return commit


def _quality_gate() -> None:
    _run(
        (
            "uv",
            "run",
            "--isolated",
            "--no-project",
            "--python",
            "3.11",
            "--with",
            "pre-commit==4.3.0",
            "pre-commit",
            "run",
            "--all-files",
        )
    )


def _build_artifacts(cli_dist: Path, bundle: Path) -> tuple[Path, Path]:
    _run(
        (
            "uv",
            "run",
            "--isolated",
            "--no-project",
            "--python",
            "3.9",
            "--with",
            "build==1.3.0",
            "python",
            "-m",
            "build",
            "--sdist",
            "--wheel",
            "--outdir",
            cli_dist,
            REPOSITORY_ROOT,
        )
    )
    cli_wheel = _only_wheel(cli_dist, "geochemistrypi-*.whl")
    shutil.copy2(cli_wheel, bundle / cli_wheel.name)
    _run(
        (
            "uv",
            "run",
            "--isolated",
            "--no-project",
            "--python",
            "3.11",
            "--with",
            "build==1.3.0",
            "python",
            "-m",
            "build",
            "--wheel",
            "--outdir",
            bundle,
            MCP_PROJECT,
        )
    )
    _run(
        (
            "uv",
            "run",
            "--isolated",
            "--no-project",
            "--python",
            "3.11",
            "--with",
            "build==1.3.0",
            "python",
            MCP_PROJECT / "tools" / "release_artifacts.py",
            "verify-artifacts",
            "--repository",
            REPOSITORY_ROOT,
            "--cli-dist",
            cli_dist,
            "--release-bundle",
            bundle,
        )
    )
    return _only_wheel(bundle, "geochemistrypi-*.whl"), _only_wheel(bundle, "geochemistrypi_mcp-*.whl")


def _install_candidate_environments(work: Path, cli_wheel: Path, mcp_wheel: Path) -> tuple[Path, Path, Path]:
    cli_root = work / "cli-installed"
    mcp_root = work / "mcp-installed"
    cli_python = _create_environment(cli_root, "3.9")
    mcp_python = _create_environment(mcp_root, "3.11")
    _run(("uv", "pip", "install", "--python", cli_python, "pytest", cli_wheel))
    mcp_requirement = f"geochemistrypi-mcp[test] @ {mcp_wheel.as_uri()}"
    _run(("uv", "pip", "install", "--python", mcp_python, mcp_requirement))
    cli_command = _venv_command(cli_root, "geochemistrypi")
    if not cli_command.is_file():
        raise RuntimeError(f"Installed CLI entry point is missing: {cli_command}")
    return cli_python, mcp_python, cli_command


def _installed_tests(cli_python: Path, mcp_python: Path) -> None:
    _run(
        (
            cli_python,
            "-m",
            "pytest",
            "-c",
            PYTEST_CONFIG,
            "--import-mode=importlib",
            REPOSITORY_ROOT / "geochemistrypi",
            REPOSITORY_ROOT / "tests" / "cli_contract",
            REPOSITORY_ROOT / "tests" / "test_database_boundary.py",
        ),
        cwd=cli_python.parents[2],
    )
    _run(
        (mcp_python, "-m", "pytest", "-c", PYTEST_CONFIG, *MCP_TESTS),
        cwd=mcp_python.parents[2],
    )


def _parity_tests(mcp_python: Path, cli_command: Path, *, full: bool) -> None:
    parity_root = mcp_python.parents[2] / "parity user state 用户数据"
    base_environment = _environment(
        {
            "GEOCHEMISTRYPI_CLI_EXECUTABLE": str(cli_command),
            "GEOCHEMISTRYPI_MCP_APP_ROOT": str(parity_root),
            "GEOCHEMISTRYPI_MCP_RUNS_ROOT": str(parity_root / "runs"),
            "GEOCHEMISTRYPI_MCP_TRACKING_ROOT": str(parity_root / "tracking"),
            "GEOCHEMISTRYPI_MCP_SERVICE_STATE_ROOT": str(parity_root / "service-state"),
            "GEOCHEMISTRYPI_MCP_SETTINGS_FILE": str(parity_root / "config" / "settings.json"),
        }
    )
    _run(
        (
            mcp_python,
            "-m",
            "pytest",
            "-c",
            PYTEST_CONFIG,
            "-m",
            "mcp_cli_parity",
            REPOSITORY_ROOT / "tests" / "mcp_wrapper" / "parity",
        ),
        cwd=mcp_python.parents[2],
        environment=base_environment,
    )
    if not full:
        return
    # Pytest's default Windows temporary hierarchy can consume enough path
    # budget to make otherwise valid scientific output names exceed the legacy
    # 260-character limit.  Use a short ordinary-user temporary root on every
    # platform so local and CI semantics stay aligned.
    full_root = Path(tempfile.mkdtemp(prefix="gpf-"))
    try:
        for index, shard in enumerate(FULL_PARITY_SHARDS):
            environment = dict(base_environment)
            environment["GEOCHEMISTRYPI_FULL_PARITY"] = "1"
            environment["GEOCHEMISTRYPI_PARITY_SHARD"] = shard
            test_file = "test_mcp_cli_parity.py" if shard == "branches-rendering" else "test_full_parity_matrix.py"
            marker = "mcp_cli_parity" if shard == "branches-rendering" else "mcp_cli_full_parity"
            _run(
                (
                    mcp_python,
                    "-m",
                    "pytest",
                    "-c",
                    PYTEST_CONFIG,
                    "--basetemp",
                    full_root / f"s{index}",
                    "-m",
                    marker,
                    REPOSITORY_ROOT / "tests" / "mcp_wrapper" / "parity" / test_file,
                ),
                cwd=mcp_python.parents[2],
                environment=environment,
            )
    except BaseException:
        print(f"Preserved full-parity workspace for diagnosis: {full_root}", file=sys.stderr)
        raise
    else:
        shutil.rmtree(full_root)


def _build_and_verify_manifest(bundle: Path, mcp_python: Path) -> None:
    _run(
        (
            mcp_python,
            "-m",
            "geochemistrypi_mcp.release",
            "build-manifest",
            "--dist",
            bundle,
            "--source-commit",
            _source_commit(),
        )
    )
    _run(
        (
            mcp_python,
            "-m",
            "geochemistrypi_mcp.release",
            "verify",
            "--bundle",
            bundle,
            "--allow-unsigned",
        )
    )


def _lifecycle_gate(work: Path, bundle: Path, mcp_wheel: Path) -> None:
    app_root = work / "GeochemistryPi public acceptance 用户数据"
    environment = _environment({"GEOCHEMISTRYPI_MCP_APP_ROOT": str(app_root)})
    source = (
        "uv",
        "run",
        "--isolated",
        "--no-project",
        "--python",
        "3.11",
        "--with-editable",
        MCP_PROJECT,
        "geochemistrypi-mcp-setup",
    )
    bundle_requirement = f"geochemistrypi-mcp[release] @ {mcp_wheel.as_uri()}"
    candidate = (
        "uv",
        "run",
        "--isolated",
        "--no-project",
        "--python",
        "3.11",
        "--with",
        bundle_requirement,
    )

    _run((*source, "install", "--client", "standard"), environment=environment)
    _run((*source, "upgrade", "--bundle", bundle, "--allow-unsigned-bundle"), environment=environment)
    _run((*source, "rollback"), environment=environment)
    _run((*source, "uninstall"), environment=environment)

    _run(
        (
            *candidate,
            "geochemistrypi-mcp-setup",
            "install",
            "--bundle",
            bundle,
            "--allow-unsigned-bundle",
            "--client",
            "standard",
        ),
        environment=environment,
    )
    _run((*candidate, "geochemistrypi-mcp-setup", "install", "--client", "standard"), environment=environment)

    run_artifact = app_root / "runs" / "preflight-user-data" / "result.json"
    tracking_artifact = app_root / "tracking" / "preflight-user-data" / "meta.yaml"
    run_artifact.parent.mkdir(parents=True, exist_ok=True)
    tracking_artifact.parent.mkdir(parents=True, exist_ok=True)
    run_artifact.write_text('{"preserved": true}\n', encoding="utf-8")
    tracking_artifact.write_text("name: preserved\n", encoding="utf-8")

    _run((*candidate, "geochemistrypi-mcp-setup", "repair", "--client", "standard"), environment=environment)
    _run((*candidate, "geochemistrypi-mcp-doctor", "--json"), environment=environment)
    _run((*candidate, "geochemistrypi-mcp-setup", "uninstall"), environment=environment)
    if run_artifact.read_text(encoding="utf-8") != '{"preserved": true}\n':
        raise RuntimeError("Uninstall changed user run data.")
    if tracking_artifact.read_text(encoding="utf-8") != "name: preserved\n":
        raise RuntimeError("Uninstall changed user tracking data.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run GeochemistryPi's cross-platform local release preflight.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip the seven slow full-model parity shards during iteration.",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Keep temporary wheels and isolated environments for inspection.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if shutil.which("uv") is None:
        print("Release preflight requires uv 0.11.7 on PATH.", file=sys.stderr)
        return 2
    uv_version = subprocess.run(("uv", "--version"), check=True, capture_output=True, text=True).stdout.strip()
    if uv_version != "uv 0.11.7 (9d177269e 2026-04-15)" and not uv_version.startswith("uv 0.11.7 "):
        print(f"Release preflight requires uv 0.11.7; found {uv_version}.", file=sys.stderr)
        return 2

    work = Path(tempfile.mkdtemp(prefix="geochemistrypi-release-preflight-"))
    try:
        print(f"Preflight workspace: {work}")
        _quality_gate()
        cli_dist = work / "cli distributions PyPI"
        bundle = work / "release bundle 发布候选"
        cli_dist.mkdir()
        bundle.mkdir()
        cli_wheel, mcp_wheel = _build_artifacts(cli_dist, bundle)
        cli_python, mcp_python, cli_command = _install_candidate_environments(work, cli_wheel, mcp_wheel)
        _installed_tests(cli_python, mcp_python)
        _parity_tests(mcp_python, cli_command, full=not arguments.quick)
        _build_and_verify_manifest(bundle, mcp_python)
        _lifecycle_gate(work, bundle, mcp_wheel)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"\nRelease preflight failed: {exc}", file=sys.stderr)
        print(f"Preserved failed workspace for diagnosis: {work}", file=sys.stderr)
        return 1

    print("\nRelease preflight passed: formatting, wheels, installed tests, parity, and lifecycle are healthy.")
    print("A signed Tag workflow must still pass on Windows, Linux, and macOS before publication.")
    if arguments.keep_workdir:
        print(f"Preserved workspace: {work}")
    else:
        shutil.rmtree(work)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
