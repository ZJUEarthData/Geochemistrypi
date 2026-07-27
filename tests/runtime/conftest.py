"""Shared runtime test fixtures."""

from pathlib import Path

import pytest
from geochemistrypi_contracts import (
    CONTRACT_VERSION,
    ClassificationExperimentSpec,
)


@pytest.fixture
def classification_request() -> ClassificationExperimentSpec:
    return ClassificationExperimentSpec.from_dict(
        {
            "schema_version": CONTRACT_VERSION,
            "dataset": {
                "kind": "local_file",
                "path": "D:/data/geochemistry.csv",
                "format": "csv",
                "id_column": None,
                "snapshot_policy": "copy",
            },
            "target_column": "Deposit_Type",
            "preprocessing": {},
            "split": {},
            "model": {"name": "random_forest"},
            "evaluation": {},
        }
    )


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    return tmp_path / "runs"
