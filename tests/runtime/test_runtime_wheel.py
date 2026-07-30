import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS_ROOT = REPOSITORY_ROOT / "packages" / "geochemistrypi-contracts"
RUNTIME_ROOT = REPOSITORY_ROOT / "packages" / "geochemistrypi-runtime"


def _run(command: list, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.integration
def test_runtime_wheel_is_clean_and_imports_without_engine(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    for package_root in (CONTRACTS_ROOT, RUNTIME_ROOT):
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                str(package_root),
                "--no-deps",
                "--wheel-dir",
                str(wheelhouse),
            ],
            tmp_path,
        )

    contracts_wheels = list(wheelhouse.glob("geochemistrypi_contracts-*.whl"))
    runtime_wheels = list(wheelhouse.glob("geochemistrypi_runtime-*.whl"))
    assert len(contracts_wheels) == 1
    assert len(runtime_wheels) == 1
    runtime_wheel = runtime_wheels[0]

    with zipfile.ZipFile(runtime_wheel) as archive:
        names = set(archive.namelist())
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        package_metadata = archive.read(metadata_name).decode("utf-8")

    assert not any(name.startswith("tests/") or "/tests/" in name for name in names)
    assert "Requires-Dist: geochemistrypi-contracts==0.1.0" in package_metadata
    assert "Requires-Dist: filelock<4,>=3.13" in package_metadata
    assert not any(name.startswith("geochemistrypi/") for name in names)

    install_root = tmp_path / "installed"
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(install_root),
            str(contracts_wheels[0]),
            str(runtime_wheel),
        ],
        tmp_path,
    )
    smoke_test = (
        "import pathlib, sys; "
        f"root = pathlib.Path({str(install_root)!r}).resolve(); "
        "sys.path.insert(0, str(root)); "
        "import geochemistrypi_runtime as runtime; "
        "module_path = pathlib.Path(runtime.__file__).resolve(); "
        "assert root in module_path.parents, (root, module_path); "
        "assert runtime.__version__ == '0.1.0'; "
        "assert 'geochemistrypi' not in sys.modules; "
        "assert 'pandas' not in sys.modules; "
        "assert 'sklearn' not in sys.modules"
    )
    _run([sys.executable, "-c", smoke_test], tmp_path)
