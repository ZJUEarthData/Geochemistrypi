"""Cross-layer baseline for the v0.8.0 core migration."""

from pathlib import Path
import tomllib

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from geochemistrypi._version import __version__
from geochemistrypi.data_mining.constants import CLUSTERING_MODELS, MODE_OPTION
from geochemistrypi.data_mining.process.time_series import compute_subaerial_proportion
from geochemistrypi.online.app import create_app


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_version_has_one_python_source(tmp_path: Path) -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert __version__ == "0.8.0"
    assert "version" in pyproject["project"]["dynamic"]
    assert pyproject["tool"]["hatch"]["version"]["path"] == "geochemistrypi/_version.py"

    app = create_app(tmp_path / "runtime")
    assert app.version == __version__
    health = TestClient(app).get("/api/health")
    assert health.status_code == 200
    assert health.json()["version"] == __version__


def test_v080_new_core_entry_points_are_registered() -> None:
    assert "OPTICS" in CLUSTERING_MODELS
    assert "Time Series" in MODE_OPTION
    assert (PROJECT_ROOT / "geochemistrypi/data_mining/data/dataset/Data_Time_Series.xlsx").is_file()


def test_time_series_core_is_deterministic_and_bounded() -> None:
    dataset = pd.DataFrame(
        {
            "R_AGE": [5.0, 15.0, 25.0, 35.0],
            "R_MAX_AGE": [7.0, 17.0, 27.0, 37.0],
            "SBAP": [0.9, 0.2, 0.8, 0.1],
            "LATITUDE": [0.0, 10.0, -10.0, 20.0],
            "LONGITUDE": [0.0, 20.0, 40.0, 60.0],
        }
    )

    first = compute_subaerial_proportion(dataset, bin_width=10.0, n_iter=8)
    second = compute_subaerial_proportion(dataset, bin_width=10.0, n_iter=8)

    for first_array, second_array in zip(first, second):
        np.testing.assert_allclose(first_array, second_array, equal_nan=True)
    age, mean, uncertainty = first
    assert age.shape == mean.shape == uncertainty.shape
    assert np.nanmin(mean) >= 0.0
    assert np.nanmax(mean) <= 100.0
    assert np.nanmin(uncertainty) >= 0.0
