"""Production CLI orchestration for reproducible Time Series analysis."""

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .process.time_series import TIME_SERIES_RANDOM_SEED, compute_binned_time_series, compute_subaerial_proportion, plot_and_save
from .utils.base import copy_files, create_geopi_output_dir

_UNSAFE_OUTPUT_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED = {
    "AUX",
    "CON",
    "NUL",
    "PRN",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def _safe_output_name(value: str, field: str) -> str:
    normalized = value.strip()
    invalid_conditions = (
        not normalized,
        normalized in {".", ".."},
        len(normalized) > 40,
        _UNSAFE_OUTPUT_NAME.search(normalized) is not None,
        normalized.endswith((" ", ".")),
        normalized.split(".", 1)[0].upper() in _WINDOWS_RESERVED,
    )
    if any(invalid_conditions):
        raise ValueError(f"{field} must be a portable non-blank directory name of at most 40 characters.")
    return normalized


def _atomic_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=".geopi-",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(str(temporary), str(path))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_time_series_data(input_path: Path, sheet: str = "0") -> pd.DataFrame:
    """Load only the two public dataset formats without guessing legacy Excel."""
    source = Path(input_path).expanduser()
    if not source.is_absolute():
        source = source.resolve()
    else:
        source = source.resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"Time Series input is not a regular file: {source}")
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix == ".xlsx":
        sheet_name: Any = int(sheet) if str(sheet).isdigit() else sheet
        return pd.read_excel(source, sheet_name=sheet_name)
    raise ValueError("Time Series input must be a .csv or .xlsx file.")


def prepare_time_series_dataframe(
    df: pd.DataFrame,
    *,
    identifier_column: Optional[str] = None,
    selected_columns: Sequence[str] = (),
    missing_value_method: str = "error",
    drop_missing_columns: Sequence[str] = (),
    feature_engineering: str = "none",
    analysis_mode: str = "subaerial_proportion",
    age_col: str = "R_AGE",
    age_min_col: Optional[str] = None,
    age_max_col: str = "R_MAX_AGE",
    probability_col: str = "SBAP",
    value_col: Optional[str] = None,
    latitude_col: str = "LATITUDE",
    longitude_col: str = "LONGITUDE",
    filter_col: Optional[str] = None,
    filter_minimum: Optional[float] = None,
    filter_maximum: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply the noninteractive equivalent of the interactive data-preparation steps."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Time Series input must contain at least one data row.")
    if feature_engineering != "none":
        raise ValueError("Time Series feature_engineering currently supports only 'none'.")
    if missing_value_method not in {"error", "drop_rows"}:
        raise ValueError("Time Series missing values must use 'error' or 'drop_rows'.")
    if analysis_mode not in {"subaerial_proportion", "continuous"}:
        raise ValueError("Time Series analysis_mode must be 'subaerial_proportion' or 'continuous'.")

    if analysis_mode == "continuous":
        if age_min_col is None or value_col is None:
            raise ValueError("Continuous Time Series requires age_min_col and value_col.")
        roles = tuple(
            dict.fromkeys(
                (
                    age_col,
                    age_min_col,
                    age_max_col,
                    value_col,
                    latitude_col,
                    longitude_col,
                    *((filter_col,) if filter_col is not None else ()),
                )
            )
        )
    else:
        roles = (age_col, age_max_col, probability_col, latitude_col, longitude_col)
    normalized_selected = tuple(column.strip() for column in selected_columns) or roles
    if any(not column for column in normalized_selected):
        raise ValueError("Time Series selected columns must be non-blank.")
    if len(set(normalized_selected)) != len(normalized_selected):
        raise ValueError("Time Series selected columns must not contain duplicates.")
    missing_roles = sorted(set(roles) - set(normalized_selected))
    if missing_roles:
        raise ValueError(f"Time Series selected columns are missing required analysis roles: {missing_roles}.")

    normalized_identifier = identifier_column.strip() if identifier_column is not None else None
    if normalized_identifier == "":
        raise ValueError("Time Series identifier column must be non-blank.")
    required = set(normalized_selected)
    if normalized_identifier is not None:
        required.add(normalized_identifier)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Time Series input is missing configured columns: {missing}.")

    normalized_drop_columns = tuple(column.strip() for column in drop_missing_columns)
    if any(not column for column in normalized_drop_columns) or len(set(normalized_drop_columns)) != len(normalized_drop_columns):
        raise ValueError("Time Series drop-missing columns must contain unique, non-blank names.")
    if normalized_drop_columns and missing_value_method != "drop_rows":
        raise ValueError("Time Series drop-missing columns require missing_value_method='drop_rows'.")
    unknown_drop_columns = sorted(set(normalized_drop_columns) - set(normalized_selected))
    if unknown_drop_columns:
        raise ValueError(f"Time Series drop-missing columns were not selected: {unknown_drop_columns}.")

    selected = df.loc[:, list(normalized_selected)].copy()
    input_row_count = int(selected.shape[0])
    if missing_value_method == "error":
        missing_mask = selected.isna().any(axis=1).to_numpy()
        if bool(missing_mask.any()):
            rows = [int(index) + 2 for index in np.flatnonzero(missing_mask)[:10]]
            raise ValueError(f"Time Series selected columns contain missing values at data rows {rows}.")
    else:
        subset = list(normalized_drop_columns or normalized_selected)
        selected = selected.dropna(subset=subset).reset_index(drop=True)
    pre_filter_row_count = int(selected.shape[0])
    has_filter_bounds = filter_minimum is not None or filter_maximum is not None
    if has_filter_bounds and filter_col is None:
        raise ValueError("Time Series filter bounds require filter_col.")
    if filter_minimum is not None and filter_maximum is not None and filter_minimum > filter_maximum:
        raise ValueError("Time Series filter_minimum must not exceed filter_maximum.")
    if filter_col is not None:
        try:
            filter_values = pd.to_numeric(selected[filter_col], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Time Series filter column must contain numeric values: {filter_col!r}.") from exc
        mask = pd.Series(True, index=selected.index)
        if filter_minimum is not None:
            mask &= filter_values >= filter_minimum
        if filter_maximum is not None:
            mask &= filter_values <= filter_maximum
        selected = selected.loc[mask].reset_index(drop=True)
    if selected.empty:
        raise ValueError("Time Series missing-value handling removed every data row.")

    analysis_row_count = int(selected.shape[0])
    metadata = {
        "identifier_column": normalized_identifier,
        "selected_columns": list(normalized_selected),
        "missing_values": {
            "method": missing_value_method,
            "columns": list(normalized_drop_columns),
        },
        "feature_engineering": feature_engineering,
        "input_row_count": input_row_count,
        "analysis_row_count": analysis_row_count,
        "dropped_row_count": input_row_count - analysis_row_count,
    }
    if analysis_mode == "continuous" or filter_col is not None:
        metadata["analysis_mode"] = analysis_mode
        metadata["range_filter"] = (
            {
                "column": filter_col,
                "minimum": filter_minimum,
                "maximum": filter_maximum,
                "inclusive": True,
                "input_row_count": pre_filter_row_count,
                "retained_row_count": analysis_row_count,
            }
            if filter_col is not None
            else None
        )
    return selected, metadata


def run_time_series_dataframe(
    df: pd.DataFrame,
    source_path: Path,
    output_root: Path,
    experiment_name: str,
    run_name: str,
    bin_width: float,
    iterations: int = 100,
    seed: int = TIME_SERIES_RANDOM_SEED,
    analysis_mode: str = "subaerial_proportion",
    age_col: str = "R_AGE",
    age_min_col: Optional[str] = None,
    age_max_col: str = "R_MAX_AGE",
    probability_col: str = "SBAP",
    value_col: Optional[str] = None,
    latitude_col: str = "LATITUDE",
    longitude_col: str = "LONGITUDE",
    relative_value_two_sigma: float = 0.0,
    minimum_samples_per_bin: int = 1,
    filter_col: Optional[str] = None,
    filter_minimum: Optional[float] = None,
    filter_maximum: Optional[float] = None,
    age_unit: str = "Ma",
    fit_curve: bool = True,
    compact_y_axis: bool = False,
    preprocessing: Optional[Dict[str, Any]] = None,
) -> Path:
    """Run the shared validated numerical workflow and write standard outputs."""
    experiment_name = _safe_output_name(experiment_name, "experiment_name")
    run_name = _safe_output_name(run_name, "run_name")
    if age_unit not in {"Ma", "Ga"}:
        raise ValueError("age_unit must be 'Ma' or 'Ga'.")
    if analysis_mode == "continuous":
        if age_min_col is None or value_col is None:
            raise ValueError("Continuous Time Series requires minimum-age and value columns.")
        age_x, mean, two_sigma = compute_binned_time_series(
            df,
            bin_width=bin_width,
            n_iter=iterations,
            age_col=age_col,
            age_min_col=age_min_col,
            age_max_col=age_max_col,
            value_col=value_col,
            lat_col=latitude_col,
            lon_col=longitude_col,
            relative_value_two_sigma=relative_value_two_sigma,
            minimum_samples_per_bin=minimum_samples_per_bin,
            seed=seed,
        )
        output_name = "Continuous Time Series"
    elif analysis_mode == "subaerial_proportion":
        age_x, mean, two_sigma = compute_subaerial_proportion(
            df,
            bin_width=bin_width,
            n_iter=iterations,
            age_col=age_col,
            age_max_col=age_max_col,
            prob_col=probability_col,
            lat_col=latitude_col,
            lon_col=longitude_col,
            seed=seed,
        )
        output_name = "Subaerial Proportion"
    else:
        raise ValueError("Unsupported Time Series analysis_mode.")
    root = Path(output_root).expanduser().resolve()
    create_geopi_output_dir(str(root), experiment_name, run_name)
    output_directory = Path(os.environ["GEOPI_OUTPUT_PATH"]).resolve()
    data_directory = Path(os.environ["GEOPI_OUTPUT_ARTIFACTS_DATA_PATH"])
    image_directory = Path(os.environ["GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH"])
    metrics_directory = Path(os.environ["GEOPI_OUTPUT_METRICS_PATH"])
    parameters_directory = Path(os.environ["GEOPI_OUTPUT_PARAMETERS_PATH"])
    summary_directory = Path(os.environ["GEOPI_OUTPUT_SUMMARY_PATH"])
    plot_and_save(
        age_x,
        mean,
        two_sigma,
        out_dir=str(data_directory),
        out_name=output_name,
        age_unit=age_unit,
        fit_curve=fit_curve,
        csv_out_dir=str(data_directory),
        pdf_out_dir=str(image_directory),
        png_out_dir=str(image_directory) if analysis_mode == "continuous" else None,
        y_label=(f"{value_col} (weighted mean)" if analysis_mode == "continuous" else "Estimated Proportion of Subaerial Basalts (%)"),
        series_label=("Weighted mean" if analysis_mode == "continuous" else "Mean proportion"),
        mean_column=("mean_value" if analysis_mode == "continuous" else "mean_percent"),
        uncertainty_column=("two_sem" if analysis_mode == "continuous" else "two_sigma_percent"),
        compact_y_axis=compact_y_axis,
    )
    finite_bins = np.isfinite(mean)
    _atomic_json(
        metrics_directory / "Time Series Metrics.json",
        {
            "schema_version": 1,
            "analysis_mode": analysis_mode,
            "populated_bins": int(finite_bins.sum()),
            "total_bins": int(mean.size),
            ("mean_of_populated_bin_values" if analysis_mode == "continuous" else "mean_of_populated_bin_percentages"): float(np.mean(mean[finite_bins])),
        },
    )
    source = Path(source_path).expanduser().resolve(strict=True)
    _atomic_json(
        parameters_directory / "Time Series Parameters.json",
        {
            "schema_version": 1,
            "analysis_mode": analysis_mode,
            "scientific_method": ("spatiotemporal_weighted_continuous_bootstrap" if analysis_mode == "continuous" else "subaerial_proportion_bootstrap"),
            "input_path": str(source),
            "input_sha256": _sha256(source),
            "bin_width_ma": float(bin_width),
            "bootstrap_iterations": iterations,
            "random_seed": seed,
            "age_unit": age_unit,
            "fit_curve": fit_curve,
            "compact_y_axis": compact_y_axis,
            "relative_value_two_sigma": relative_value_two_sigma if analysis_mode == "continuous" else None,
            "minimum_samples_per_bin": minimum_samples_per_bin,
            "filter": {
                "column": filter_col,
                "minimum": filter_minimum,
                "maximum": filter_maximum,
                "inclusive": True,
            },
            "columns": {
                "age": age_col,
                "minimum_age": age_min_col,
                "maximum_age": age_max_col,
                "probability": probability_col,
                "value": value_col,
                "latitude": latitude_col,
                "longitude": longitude_col,
            },
            "preprocessing": preprocessing
            or {
                "identifier_column": None,
                "selected_columns": list(df.columns),
                "missing_values": {"method": "already_prepared", "columns": []},
                "feature_engineering": "unspecified",
                "input_row_count": int(df.shape[0]),
                "analysis_row_count": int(df.shape[0]),
                "dropped_row_count": 0,
            },
        },
    )
    copy_files(
        os.environ["GEOPI_OUTPUT_ARTIFACTS_PATH"],
        str(metrics_directory),
        str(parameters_directory),
        str(summary_directory),
    )
    return output_directory


def run_time_series_analysis(
    input_path: Path,
    output_root: Path,
    experiment_name: str,
    run_name: str,
    bin_width: float,
    iterations: int = 100,
    seed: int = TIME_SERIES_RANDOM_SEED,
    sheet: str = "0",
    analysis_mode: str = "subaerial_proportion",
    age_col: str = "R_AGE",
    age_min_col: Optional[str] = None,
    age_max_col: str = "R_MAX_AGE",
    probability_col: str = "SBAP",
    value_col: Optional[str] = None,
    latitude_col: str = "LATITUDE",
    longitude_col: str = "LONGITUDE",
    relative_value_two_sigma: float = 0.0,
    minimum_samples_per_bin: int = 1,
    filter_col: Optional[str] = None,
    filter_minimum: Optional[float] = None,
    filter_maximum: Optional[float] = None,
    age_unit: str = "Ma",
    fit_curve: bool = True,
    compact_y_axis: bool = False,
    identifier_column: Optional[str] = None,
    selected_columns: Sequence[str] = (),
    missing_value_method: str = "error",
    drop_missing_columns: Sequence[str] = (),
    feature_engineering: str = "none",
) -> Path:
    source = Path(input_path).expanduser().resolve(strict=True)
    prepared, preprocessing = prepare_time_series_dataframe(
        load_time_series_data(source, sheet),
        identifier_column=identifier_column,
        selected_columns=selected_columns,
        missing_value_method=missing_value_method,
        drop_missing_columns=drop_missing_columns,
        feature_engineering=feature_engineering,
        analysis_mode=analysis_mode,
        age_col=age_col,
        age_min_col=age_min_col,
        age_max_col=age_max_col,
        probability_col=probability_col,
        value_col=value_col,
        latitude_col=latitude_col,
        longitude_col=longitude_col,
        filter_col=filter_col,
        filter_minimum=filter_minimum,
        filter_maximum=filter_maximum,
    )
    return run_time_series_dataframe(
        df=prepared,
        source_path=source,
        output_root=output_root,
        experiment_name=experiment_name,
        run_name=run_name,
        bin_width=bin_width,
        iterations=iterations,
        seed=seed,
        analysis_mode=analysis_mode,
        age_col=age_col,
        age_min_col=age_min_col,
        age_max_col=age_max_col,
        probability_col=probability_col,
        value_col=value_col,
        latitude_col=latitude_col,
        longitude_col=longitude_col,
        relative_value_two_sigma=relative_value_two_sigma,
        minimum_samples_per_bin=minimum_samples_per_bin,
        filter_col=filter_col,
        filter_minimum=filter_minimum,
        filter_maximum=filter_maximum,
        age_unit=age_unit,
        fit_curve=fit_curve,
        compact_y_axis=compact_y_axis,
        preprocessing=preprocessing,
    )


def main(argv: Optional[list] = None) -> None:
    """Backward-compatible module runner delegating to the production workflow."""
    parser = argparse.ArgumentParser(description="Run a reproducible public Time Series producer")
    parser.add_argument("--input", required=True)
    parser.add_argument("--bin-width", type=float, required=True)
    parser.add_argument("--iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=TIME_SERIES_RANDOM_SEED)
    parser.add_argument("--sheet", default="0")
    parser.add_argument(
        "--analysis-mode",
        choices=("subaerial_proportion", "continuous"),
        default="subaerial_proportion",
    )
    parser.add_argument("--out-dir", default="geopi_output")
    parser.add_argument("--experiment-name", default="Time Series")
    parser.add_argument("--run-name", default="Subaerial Proportion")
    parser.add_argument("--age-col", default="R_AGE")
    parser.add_argument("--age-min-col")
    parser.add_argument("--age-max-col", default="R_MAX_AGE")
    parser.add_argument("--prob-col", default="SBAP")
    parser.add_argument("--value-col")
    parser.add_argument("--lat-col", default="LATITUDE")
    parser.add_argument("--lon-col", default="LONGITUDE")
    parser.add_argument("--age-unit", choices=("Ma", "Ga"), default="Ma")
    arguments = parser.parse_args(argv)
    output = run_time_series_analysis(
        input_path=Path(arguments.input),
        output_root=Path(arguments.out_dir),
        experiment_name=arguments.experiment_name,
        run_name=arguments.run_name,
        bin_width=arguments.bin_width,
        iterations=arguments.iter,
        seed=arguments.seed,
        sheet=arguments.sheet,
        analysis_mode=arguments.analysis_mode,
        age_col=arguments.age_col,
        age_min_col=arguments.age_min_col,
        age_max_col=arguments.age_max_col,
        probability_col=arguments.prob_col,
        value_col=arguments.value_col,
        latitude_col=arguments.lat_col,
        longitude_col=arguments.lon_col,
        age_unit=arguments.age_unit,
    )
    print(f"Saved Time Series outputs to {output}")


if __name__ == "__main__":
    main()
