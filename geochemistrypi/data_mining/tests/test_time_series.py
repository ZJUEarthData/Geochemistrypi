import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from geochemistrypi.cli import app
from geochemistrypi.data_mining.cli_pipeline import semantic_mode_number
from geochemistrypi.data_mining.constants import MODE_OPTION, MODE_OPTION_WITH_MISSING_VALUES
from geochemistrypi.data_mining.process.time_series import TimeSeriesValidationError, compute_binned_time_series, compute_subaerial_proportion
from geochemistrypi.data_mining.run_time_series import _atomic_json, prepare_time_series_dataframe


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


def _continuous_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AGE": [10.0, 20.0, 115.0, 130.0],
            "MIN AGE": [8.0, 18.0, 110.0, 125.0],
            "MAX AGE": [12.0, 22.0, 120.0, 135.0],
            "MGO": [8.0, 9.0, 6.0, 7.0],
            "SIO2": [43.0, 51.0, 48.0, 60.0],
            "LATITUDE": [-20.0, 5.0, 30.0, 45.0],
            "LONGITUDE": [100.0, 110.0, 120.0, 130.0],
        }
    )


def test_time_series_menu_position_maps_to_stable_option_six() -> None:
    assert semantic_mode_number(6, MODE_OPTION) == 6
    assert semantic_mode_number(4, MODE_OPTION_WITH_MISSING_VALUES) == 6


@pytest.mark.skipif(os.name != "nt", reason="Windows path-length regression")
def test_atomic_json_uses_a_short_temporary_name_in_deep_workspaces(tmp_path: Path) -> None:
    target_parent_length = 215
    padding_length = target_parent_length - len(str(tmp_path)) - 1
    if padding_length < 1:
        pytest.skip("Temporary test root is already too long for the bounded-path scenario.")
    target = tmp_path / ("x" * padding_length) / "Embedding Label Overlay Parameters.json"

    _atomic_json(target, {"schema_version": 1, "status": "ok"})

    assert len(str(target.parent)) == target_parent_length
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "status": "ok",
    }


def test_time_series_is_seeded_without_mutating_global_random_state() -> None:
    np.random.seed(99)
    expected = np.random.random(3)
    np.random.seed(99)

    first = compute_subaerial_proportion(_frame(), 10.0, n_iter=5, seed=7)
    second = compute_subaerial_proportion(_frame(), 10.0, n_iter=5, seed=7)

    assert all(np.array_equal(left, right, equal_nan=True) for left, right in zip(first, second))
    assert np.array_equal(np.random.random(3), expected)


def test_time_series_age_uncertainty_is_unsigned_when_comparison_age_is_smaller() -> None:
    later = _frame()
    earlier = _frame()
    earlier.loc[0, "R_MAX_AGE"] = 8.0

    expected = compute_subaerial_proportion(later, 10.0, n_iter=5, seed=7)
    actual = compute_subaerial_proportion(earlier, 10.0, n_iter=5, seed=7)

    assert all(np.array_equal(left, right, equal_nan=True) for left, right in zip(expected, actual))


def test_time_series_bins_follow_central_age_not_comparison_age() -> None:
    age_x, mean, two_sigma = compute_subaerial_proportion(_frame(), 10.0, n_iter=2, seed=7)

    assert age_x.tolist() == [5.0, 15.0, 25.0, 35.0, 45.0]
    assert mean.shape == two_sigma.shape == age_x.shape


def test_continuous_time_series_is_seeded_and_does_not_mutate_global_random_state() -> None:
    np.random.seed(99)
    expected = np.random.random(3)
    np.random.seed(99)

    first = compute_binned_time_series(
        _continuous_frame(),
        100.0,
        n_iter=8,
        age_col="AGE",
        age_min_col="MIN AGE",
        age_max_col="MAX AGE",
        value_col="MGO",
        seed=2025,
        relative_value_two_sigma=0.04,
    )
    second = compute_binned_time_series(
        _continuous_frame(),
        100.0,
        n_iter=8,
        age_col="AGE",
        age_min_col="MIN AGE",
        age_max_col="MAX AGE",
        value_col="MGO",
        seed=2025,
        relative_value_two_sigma=0.04,
    )

    assert all(np.array_equal(left, right, equal_nan=True) for left, right in zip(first, second))
    assert np.array_equal(np.random.random(3), expected)


def test_time_series_preparation_drops_rows_across_the_full_selected_range() -> None:
    frame = _frame().assign(
        **{
            "ROCK NAME": ["A", "B", "C", "D"],
            "MIN_AGE": [9.0, np.nan, 34.0, 49.0],
            "AGE": [10.0, 20.0, 35.0, 50.0],
            "MAX_AGE": [11.0, 21.0, 36.0, 51.0],
            "R_MIN_AGE": [8.0, 18.0, 33.0, 48.0],
        }
    )
    selected_columns = (
        "LATITUDE",
        "LONGITUDE",
        "MIN_AGE",
        "AGE",
        "MAX_AGE",
        "R_MIN_AGE",
        "R_AGE",
        "R_MAX_AGE",
        "SBAP",
    )

    prepared, metadata = prepare_time_series_dataframe(
        frame,
        identifier_column="ROCK NAME",
        selected_columns=selected_columns,
        missing_value_method="drop_rows",
    )

    assert tuple(prepared.columns) == selected_columns
    assert prepared.shape == (3, 9)
    assert metadata == {
        "identifier_column": "ROCK NAME",
        "selected_columns": list(selected_columns),
        "missing_values": {"method": "drop_rows", "columns": []},
        "feature_engineering": "none",
        "input_row_count": 4,
        "analysis_row_count": 3,
        "dropped_row_count": 1,
    }


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("SBAP", 1.1, "between 0 and 1"),
        ("LATITUDE", 91.0, "between -90 and 90"),
        ("LONGITUDE", -181.0, "between -180 and 180"),
        ("R_AGE", np.nan, "missing or non-finite"),
    ],
)
def test_time_series_rejects_invalid_scientific_values(column: str, value: float, message: str) -> None:
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
    assert (run / "artifacts" / "image" / "model_output" / "Subaerial Proportion.pdf").is_file()
    parameters = json.loads((run / "parameters" / "Time Series Parameters.json").read_text(encoding="utf-8"))
    assert parameters["random_seed"] == 7
    assert parameters["bootstrap_iterations"] == 5
    assert len(parameters["input_sha256"]) == 64


def test_time_series_cli_records_explicit_preprocessing_configuration(tmp_path: Path) -> None:
    source = tmp_path / "time-series-preprocessing.csv"
    output_root = tmp_path / "outputs"
    frame = _frame().assign(
        **{
            "ROCK NAME": ["A", "B", "C", "D"],
            "MIN_AGE": [9.0, np.nan, 34.0, 49.0],
            "AGE": [10.0, 20.0, 35.0, 50.0],
            "MAX_AGE": [11.0, 21.0, 36.0, 51.0],
            "R_MIN_AGE": [8.0, 18.0, 33.0, 48.0],
        }
    )
    frame.to_csv(source, index=False)
    selected_columns = (
        "LATITUDE",
        "LONGITUDE",
        "MIN_AGE",
        "AGE",
        "MAX_AGE",
        "R_MIN_AGE",
        "R_AGE",
        "R_MAX_AGE",
        "SBAP",
    )
    arguments = [
        "time-series",
        "--input",
        str(source),
        "--bin-width",
        "10",
        "--iterations",
        "2",
        "--output-root",
        str(output_root),
        "--experiment-name",
        "Time Series Preparation",
        "--run-name",
        "Drop All Selected",
        "--identifier-column",
        "ROCK NAME",
        "--missing-values",
        "drop_rows",
        "--feature-engineering",
        "none",
    ]
    for column in selected_columns:
        arguments.extend(("--selected-column", column))

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 0, result.output
    parameters = json.loads((output_root / "Time Series Preparation" / "Drop All Selected" / "parameters" / "Time Series Parameters.json").read_text(encoding="utf-8"))
    assert parameters["preprocessing"]["identifier_column"] == "ROCK NAME"
    assert parameters["preprocessing"]["selected_columns"] == list(selected_columns)
    assert parameters["preprocessing"]["missing_values"] == {"method": "drop_rows", "columns": []}
    assert parameters["preprocessing"]["feature_engineering"] == "none"
    assert parameters["preprocessing"]["input_row_count"] == 4
    assert parameters["preprocessing"]["analysis_row_count"] == 3
    assert parameters["preprocessing"]["dropped_row_count"] == 1


def test_continuous_time_series_cli_binds_filter_uncertainty_and_outputs(tmp_path: Path) -> None:
    source = tmp_path / "continuous.csv"
    output_root = tmp_path / "outputs"
    _continuous_frame().to_csv(source, index=False)

    result = CliRunner().invoke(
        app,
        [
            "time-series",
            "--analysis-mode",
            "continuous",
            "--input",
            str(source),
            "--bin-width",
            "100",
            "--iterations",
            "8",
            "--seed",
            "2025",
            "--age-column",
            "AGE",
            "--minimum-age-column",
            "MIN AGE",
            "--maximum-age-column",
            "MAX AGE",
            "--value-column",
            "MGO",
            "--filter-column",
            "SIO2",
            "--filter-minimum",
            "43",
            "--filter-maximum",
            "51",
            "--relative-value-two-sigma",
            "0.04",
            "--compact-y-axis",
            "--no-fit-curve",
            "--output-root",
            str(output_root),
            "--experiment-name",
            "Continuous Test",
            "--run-name",
            "Generic Producer",
        ],
    )

    assert result.exit_code == 0, result.output
    run = output_root / "Continuous Test" / "Generic Producer"
    csv_path = run / "artifacts" / "data" / "Continuous Time Series.csv"
    assert csv_path.is_file()
    assert (run / "artifacts" / "image" / "model_output" / "Continuous Time Series.pdf").is_file()
    assert (run / "artifacts" / "image" / "model_output" / "Continuous Time Series.png").is_file()
    assert list(pd.read_csv(csv_path).columns) == ["age_Ma", "mean_value", "two_sem"]
    parameters = json.loads((run / "parameters" / "Time Series Parameters.json").read_text(encoding="utf-8"))
    assert parameters["analysis_mode"] == "continuous"
    assert parameters["relative_value_two_sigma"] == pytest.approx(0.04)
    assert parameters["filter"] == {"column": "SIO2", "minimum": 43.0, "maximum": 51.0, "inclusive": True}
    assert parameters["preprocessing"]["analysis_row_count"] == 3
