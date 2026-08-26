"""Join existing embedding coordinates and externally supplied labels by identifier."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg", force=True)

from .run_reference_anomaly_series import _normalized_label, _positive_label_keys  # noqa: E402
from .run_time_series import _atomic_json, _safe_output_name, _sha256, load_time_series_data  # noqa: E402
from .utils.base import copy_files, create_geopi_output_dir  # noqa: E402


def _canonical_sha256(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _identifier_key(value: Any) -> str:
    if pd.isna(value):
        raise ValueError("Identifiers must not contain missing values.")
    key = str(value).strip()
    if not key or "\n" in key or "\r" in key:
        raise ValueError("Identifiers must be non-blank single-line values.")
    return key


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    context: str,
) -> Tuple[str, ...]:
    normalized = tuple(column.strip() for column in columns)
    if any(not column or "\n" in column or "\r" in column for column in normalized):
        raise ValueError(f"{context} columns must be non-blank single-line names.")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{context} columns must have distinct roles.")
    missing = sorted(set(normalized) - set(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing configured columns: {missing}.")
    return normalized


def _prepare_overlay(
    coordinates: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    coordinate_identifier_column: str,
    label_identifier_column: str,
    x_column: str,
    y_column: str,
    label_column: str,
    positive_label_values: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if coordinates.empty or labels.empty:
        raise ValueError("Coordinate and label inputs must each contain at least one row.")
    reserved = {"overlay_identifier", "is_anomaly"}
    if reserved & set(coordinates.columns) or reserved & set(labels.columns):
        raise ValueError("Input columns must not use reserved overlay output names.")
    coordinate_roles = _require_columns(
        coordinates,
        (coordinate_identifier_column, x_column, y_column),
        "Coordinate input",
    )
    label_roles = _require_columns(
        labels,
        (label_identifier_column, label_column),
        "Label input",
    )
    cross_table_conflicts = sorted(
        {
            label_column,
            *((label_identifier_column,) if label_identifier_column != coordinate_identifier_column else ()),
        }
        & {coordinate_identifier_column, x_column, y_column}
    )
    if cross_table_conflicts:
        raise ValueError("Coordinate and label scientific roles must use distinct column names " "except for a shared identifier name: " f"{cross_table_conflicts}.")
    positive_keys = _positive_label_keys(
        positive_label_values,
        "positive_label_values",
    )

    coordinate_table = coordinates.loc[:, list(coordinate_roles)].copy()
    label_table = labels.loc[:, list(label_roles)].copy()
    coordinate_table.insert(
        0,
        "overlay_identifier",
        coordinate_table[coordinate_identifier_column].map(_identifier_key),
    )
    label_table.insert(
        0,
        "overlay_identifier",
        label_table[label_identifier_column].map(_identifier_key),
    )
    if coordinate_table["overlay_identifier"].duplicated().any():
        raise ValueError("Coordinate identifiers must be unique.")
    if label_table["overlay_identifier"].duplicated().any():
        raise ValueError("Label identifiers must be unique.")
    coordinate_keys = set(coordinate_table["overlay_identifier"])
    label_keys = set(label_table["overlay_identifier"])
    if coordinate_keys != label_keys:
        missing_labels = sorted(coordinate_keys - label_keys)[:10]
        missing_coordinates = sorted(label_keys - coordinate_keys)[:10]
        raise ValueError("Coordinate and label identifier sets must match exactly; " f"missing_labels={missing_labels}, missing_coordinates={missing_coordinates}.")
    for column in (x_column, y_column):
        coordinate_table[column] = pd.to_numeric(
            coordinate_table[column],
            errors="coerce",
        )
        if not np.isfinite(coordinate_table[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Embedding coordinate column {column!r} must contain finite numeric values.")
    label_table["is_anomaly"] = label_table[label_column].map(lambda value: _normalized_label(value) in positive_keys)
    joined = coordinate_table.merge(
        label_table,
        on="overlay_identifier",
        how="left",
        validate="one_to_one",
        sort=False,
        suffixes=("_coordinate", "_label"),
    )
    identity = [
        {
            "identifier": str(row["overlay_identifier"]),
            "x": float(row[x_column]),
            "y": float(row[y_column]),
            "label": _normalized_label(row[label_column]),
            "is_anomaly": bool(row["is_anomaly"]),
        }
        for row in joined.to_dict(orient="records")
    ]
    anomaly_count = int(joined["is_anomaly"].sum())
    return joined, {
        "row_count": int(joined.shape[0]),
        "anomaly_count": anomaly_count,
        "non_anomaly_count": int(joined.shape[0] - anomaly_count),
        "ordered_join_identity_sha256": _canonical_sha256(identity),
    }


def _plot_overlay(
    joined: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    png_path: Path,
    pdf_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    ordinary = joined.loc[~joined["is_anomaly"]]
    anomalies = joined.loc[joined["is_anomaly"]]
    axis.scatter(
        ordinary[x_column],
        ordinary[y_column],
        s=18,
        color="#4C78A8",
        alpha=0.72,
        linewidths=0,
        label=f"Non-anomaly (n={ordinary.shape[0]})",
    )
    axis.scatter(
        anomalies[x_column],
        anomalies[y_column],
        s=38,
        facecolors="none",
        edgecolors="#D62728",
        linewidths=1.1,
        label=f"Anomaly (n={anomalies.shape[0]})",
        zorder=3,
    )
    axis.set_xlabel(x_column)
    axis.set_ylabel(y_column)
    axis.set_title("Embedding coordinates with anomaly labels")
    axis.grid(color="#d9d9d9", linewidth=0.45, alpha=0.65)
    axis.legend(frameon=False)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)


def _artifact_entry(path: Path, output_directory: Path, role: str) -> Dict[str, Any]:
    return {
        "role": role,
        "relative_path": path.relative_to(output_directory).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def run_embedding_label_overlay(
    *,
    coordinate_path: Path,
    label_path: Path,
    output_root: Path,
    experiment_name: str,
    run_name: str,
    coordinate_sheet: str,
    label_sheet: str,
    coordinate_identifier_column: str,
    label_identifier_column: str,
    x_column: str,
    y_column: str,
    label_column: str,
    positive_label_values: Sequence[str],
) -> Path:
    """Write a lossless identifier join, counts, figures, and provenance."""

    experiment_name = _safe_output_name(experiment_name, "experiment_name")
    run_name = _safe_output_name(run_name, "run_name")
    coordinate_source = Path(coordinate_path).expanduser().resolve(strict=True)
    label_source = Path(label_path).expanduser().resolve(strict=True)
    joined, counts = _prepare_overlay(
        load_time_series_data(coordinate_source, coordinate_sheet),
        load_time_series_data(label_source, label_sheet),
        coordinate_identifier_column=coordinate_identifier_column,
        label_identifier_column=label_identifier_column,
        x_column=x_column,
        y_column=y_column,
        label_column=label_column,
        positive_label_values=positive_label_values,
    )

    root = Path(output_root).expanduser().resolve()
    create_geopi_output_dir(str(root), experiment_name, run_name)
    output_directory = Path(os.environ["GEOPI_OUTPUT_PATH"]).resolve()
    data_directory = Path(os.environ["GEOPI_OUTPUT_ARTIFACTS_DATA_PATH"]).resolve()
    image_directory = Path(os.environ["GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH"]).resolve()
    metrics_directory = Path(os.environ["GEOPI_OUTPUT_METRICS_PATH"]).resolve()
    parameters_directory = Path(os.environ["GEOPI_OUTPUT_PARAMETERS_PATH"]).resolve()
    summary_directory = Path(os.environ["GEOPI_OUTPUT_SUMMARY_PATH"]).resolve()

    joined_path = data_directory / "Embedding Label Overlay.csv"
    png_path = image_directory / "Embedding Label Overlay.png"
    pdf_path = image_directory / "Embedding Label Overlay.pdf"
    counts_path = metrics_directory / "Embedding Label Overlay Counts.json"
    parameters_path = parameters_directory / "Embedding Label Overlay Parameters.json"
    artifact_index_path = summary_directory / "Embedding Label Overlay Artifact Index.json"
    manifest_path = summary_directory / "Embedding Label Overlay Manifest.json"

    joined.to_csv(joined_path, index=False)
    _plot_overlay(
        joined,
        x_column=x_column,
        y_column=y_column,
        png_path=png_path,
        pdf_path=pdf_path,
    )
    _atomic_json(counts_path, {"schema_version": 1, **counts})
    parameters = {
        "schema_version": 1,
        "workflow": "embedding_label_overlay",
        "join_policy": "exact_identifier_set_one_to_one",
        "coordinate_input": {
            "path": str(coordinate_source),
            "sha256": _sha256(coordinate_source),
            "sheet": coordinate_sheet,
        },
        "label_input": {
            "path": str(label_source),
            "sha256": _sha256(label_source),
            "sheet": label_sheet,
        },
        "columns": {
            "coordinate_identifier": coordinate_identifier_column,
            "label_identifier": label_identifier_column,
            "x": x_column,
            "y": y_column,
            "label": label_column,
        },
        "positive_label_values": list(positive_label_values),
    }
    _atomic_json(parameters_path, parameters)
    primary_artifacts = (
        (joined_path, "embedding_label_overlay.joined_table"),
        (png_path, "embedding_label_overlay.figure.png"),
        (pdf_path, "embedding_label_overlay.figure.pdf"),
        (counts_path, "embedding_label_overlay.counts"),
        (parameters_path, "provenance.parameters"),
    )
    artifact_entries = [_artifact_entry(path, output_directory, role) for path, role in primary_artifacts]
    _atomic_json(
        artifact_index_path,
        {"schema_version": 1, "artifacts": artifact_entries},
    )
    _atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "workflow": "embedding_label_overlay",
            "join_policy": "exact_identifier_set_one_to_one",
            "counts": counts,
            "artifact_index": {
                "relative_path": artifact_index_path.relative_to(output_directory).as_posix(),
                "sha256": _sha256(artifact_index_path),
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
