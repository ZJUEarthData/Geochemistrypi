import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from geochemistrypi.cli import app
from geochemistrypi.data_mining.cli_pipeline import semantic_mode_number
from geochemistrypi.data_mining.constants import (
    MODE_OPTION,
    MODE_OPTION_WITH_MISSING_VALUES,
)
from geochemistrypi.data_mining.process.time_series import (
    TimeSeriesValidationError,
    compute_subaerial_proportion,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "R_AGE": [10.0, 20.0, 35.0, 50.0],
            "R_MAX_AGE": [12.0, 25.0, 40.0, 55.0],
            "SBAP": [0.9, 0.1, 0.8, 0.2],
            "LATITUDE": [-20.0, 5.0, 30.0, 45.0],
            "LONGITUDE": [100.0, 110.0, 120.0, 130.0],
        }
    )


def test_time_series_menu_position_maps_to_stable_option_six() -> None:
    assert semantic_mode_number(6, MODE_OPTION) == 6
    assert semantic_mode_number(4, MODE_OPTION_WITH_MISSING_VALUES) == 6


def test_time_series_is_seeded_without_mutating_global_random_state() -> None:
    np.random.seed(99)
    expected = np.random.random(3)
    np.random.seed(99)

    first = compute_subaerial_proportion(_frame(), 10.0, n_iter=5, seed=7)
    second = compute_subaerial_proportion(_frame(), 10.0, n_iter=5, seed=7)

    assert all(np.array_equal(left, right, equal_nan=True) for left, right in zip(first, second))
    assert np.array_equal(np.random.random(3), expected)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("SBAP", 1.1, "between 0 and 1"),
        ("LATITUDE", 91.0, "between -90 and 90"),
        ("LONGITUDE", -181.0, "between -180 and 180"),
        ("R_AGE", np.nan, "missing or non-finite"),
    ],
)
def test_time_series_rejects_invalid_scientific_values(
    column: str, value: float, message: str
) -> None:
    frame = _frame()
    frame.loc[0, column] = value
    with pytest.raises(TimeSeriesValidationError, match=message):
        compute_subaerial_proportion(frame, 10.0, n_iter=2)


def test_time_series_cli_writes_standard_indexable_outputs(tmp_path: Path) -> None:
    source = tmp_path / "time-series.csv"
    output_root = tmp_path / "outputs"
    _frame().to_csv(source, index=False)

    result = CliRunner().invoke(
        app,
        [
            "time-series",
            "--input",
            str(source),
            "--bin-width",
            "10",
            "--iterations",
            "5",
            "--seed",
            "7",
            "--output-root",
            str(output_root),
            "--experiment-name",
            "Time Series Test",
            "--run-name",
            "Deterministic",
            "--no-fit-curve",
        ],
    )

    assert result.exit_code == 0, result.output
    run = output_root / "Time Series Test" / "Deterministic"
    assert (run / "artifacts" / "data" / "Subaerial Proportion.csv").is_file()
    assert (
        run
        / "artifacts"
        / "image"
        / "model_output"
        / "Subaerial Proportion.pdf"
    ).is_file()
    parameters = json.loads(
        (run / "parameters" / "Time Series Parameters.json").read_text(
            encoding="utf-8"
        )
    )
    assert parameters["random_seed"] == 7
    assert parameters["bootstrap_iterations"] == 5
    assert len(parameters["input_sha256"]) == 64
