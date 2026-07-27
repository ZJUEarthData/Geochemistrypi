import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import List

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "packages" / "geochemistrypi-contracts"
EXPECTED_SCHEMAS = {
    "geochemistrypi_contracts/schemas/v1/dataset-ref.schema.json",
    "geochemistrypi_contracts/schemas/v1/classification-experiment-spec.schema.json",
    "geochemistrypi_contracts/schemas/v1/experiment-result.schema.json",
    "geochemistrypi_contracts/schemas/v1/error-response.schema.json",
}


def _run(command: List[str], cwd: Path) -> subprocess.CompletedProcess:
    environment = os.environ.copy()
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return subprocess.run(command, cwd=str(cwd), env=environment, check=True, capture_output=True, text=True)


@pytest.mark.integration
def test_contract_wheel_ships_schemas_and_loads_after_install(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(PACKAGE_ROOT),
            "--no-deps",
            "--wheel-dir",
            str(wheelhouse),
        ],
        tmp_path,
    )

    wheels = list(wheelhouse.glob("geochemistrypi_contracts-*.whl"))
    assert len(wheels) == 1
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = archive.read(metadata_name).decode("utf-8")

    assert EXPECTED_SCHEMAS <= names
    assert not any(name.startswith("tests/") or "/tests/" in name for name in names)
    assert "\nRequires-Dist:" not in metadata

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
            str(wheel),
        ],
        tmp_path,
    )

    smoke_test = (
        "import pathlib, sys; "
        f"install_root = pathlib.Path({str(install_root)!r}).resolve(); "
        "sys.path.insert(0, str(install_root)); "
        "import geochemistrypi_contracts as contracts; "
        "module_path = pathlib.Path(contracts.__file__).resolve(); "
        "assert install_root in module_path.parents, (install_root, module_path); "
        "assert len(contracts.SchemaName) == 4; "
        "assert all(contracts.load_schema(name)['x-contract-version'] == contracts.CONTRACT_VERSION for name in contracts.SchemaName)"
    )
    _run([sys.executable, "-S", "-c", smoke_test], tmp_path)
