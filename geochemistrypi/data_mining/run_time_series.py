"""Production CLI orchestration for reproducible Time Series analysis."""

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .process.time_series import TIME_SERIES_RANDOM_SEED, compute_subaerial_proportion, plot_and_save
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
        prefix=f".{path.name}.",
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


def run_time_series_dataframe(
    df: pd.DataFrame,
    source_path: Path,
    output_root: Path,
    experiment_name: str,
    run_name: str,
    bin_width: float,
    iterations: int = 100,
    seed: int = TIME_SERIES_RANDOM_SEED,
    age_col: str = "R_AGE",
    age_max_col: str = "R_MAX_AGE",
    probability_col: str = "SBAP",
    latitude_col: str = "LATITUDE",
    longitude_col: str = "LONGITUDE",
    age_unit: str = "Ma",
    fit_curve: bool = True,
) -> Path:
    """Run the shared validated numerical workflow and write standard outputs."""
    experiment_name = _safe_output_name(experiment_name, "experiment_name")
    run_name = _safe_output_name(run_name, "run_name")
    if age_unit not in {"Ma", "Ga"}:
        raise ValueError("age_unit must be 'Ma' or 'Ga'.")
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
        out_name="Subaerial Proportion",
        age_unit=age_unit,
        fit_curve=fit_curve,
        csv_out_dir=str(data_directory),
        pdf_out_dir=str(image_directory),
    )
    finite_bins = np.isfinite(mean)
    _atomic_json(
        metrics_directory / "Time Series Metrics.json",
        {
            "schema_version": 1,
            "populated_bins": int(finite_bins.sum()),
            "total_bins": int(mean.size),
            "mean_of_populated_bin_percentages": float(np.mean(mean[finite_bins])),
        },
    )
    source = Path(source_path).expanduser().resolve(strict=True)
    _atomic_json(
        parameters_directory / "Time Series Parameters.json",
        {
            "schema_version": 1,
            "input_path": str(source),
            "input_sha256": _sha256(source),
            "bin_width_ma": float(bin_width),
            "bootstrap_iterations": iterations,
            "random_seed": seed,
            "age_unit": age_unit,
            "fit_curve": fit_curve,
            "columns": {
                "age": age_col,
                "maximum_age": age_max_col,
                "probability": probability_col,
                "latitude": latitude_col,
                "longitude": longitude_col,
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
    age_col: str = "R_AGE",
    age_max_col: str = "R_MAX_AGE",
    probability_col: str = "SBAP",
    latitude_col: str = "LATITUDE",
    longitude_col: str = "LONGITUDE",
    age_unit: str = "Ma",
    fit_curve: bool = True,
) -> Path:
    source = Path(input_path).expanduser().resolve(strict=True)
    return run_time_series_dataframe(
        load_time_series_data(source, sheet),
        source,
        output_root,
        experiment_name,
        run_name,
        bin_width,
        iterations,
        seed,
        age_col,
        age_max_col,
        probability_col,
        latitude_col,
        longitude_col,
        age_unit,
        fit_curve,
    )


def main(argv: Optional[list] = None) -> None:
    """Backward-compatible module runner delegating to the production workflow."""
    parser = argparse.ArgumentParser(description="Run reproducible subaerial-proportion Time Series analysis")
    parser.add_argument("--input", required=True)
    parser.add_argument("--bin-width", type=float, required=True)
    parser.add_argument("--iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=TIME_SERIES_RANDOM_SEED)
    parser.add_argument("--sheet", default="0")
    parser.add_argument("--out-dir", default="geopi_output")
    parser.add_argument("--experiment-name", default="Time Series")
    parser.add_argument("--run-name", default="Subaerial Proportion")
    parser.add_argument("--age-col", default="R_AGE")
    parser.add_argument("--age-max-col", default="R_MAX_AGE")
    parser.add_argument("--prob-col", default="SBAP")
    parser.add_argument("--lat-col", default="LATITUDE")
    parser.add_argument("--lon-col", default="LONGITUDE")
    parser.add_argument("--age-unit", choices=("Ma", "Ga"), default="Ma")
    arguments = parser.parse_args(argv)
    output = run_time_series_analysis(
        Path(arguments.input),
        Path(arguments.out_dir),
        arguments.experiment_name,
        arguments.run_name,
        arguments.bin_width,
        arguments.iter,
        arguments.seed,
        arguments.sheet,
        arguments.age_col,
        arguments.age_max_col,
        arguments.prob_col,
        arguments.lat_col,
        arguments.lon_col,
        arguments.age_unit,
    )
    print(f"Saved Time Series outputs to {output}")


if __name__ == "__main__":
    main()
