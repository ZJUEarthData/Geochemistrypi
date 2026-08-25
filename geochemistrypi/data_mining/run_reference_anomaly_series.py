"""Reference-labelled multi-signal time-series production workflow.

This module visualizes externally supplied labels and optional event records.  It
does not fit an anomaly detector and never derives or changes the supplied labels.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg", force=True)

from .run_time_series import (
    _atomic_json,
    _safe_output_name,
    _sha256,
    load_time_series_data,
)
from .utils.base import copy_files, create_geopi_output_dir

_ASSOCIATION_DIRECTIONS = {"before_event", "after_event", "symmetric"}
_COMPARISON_PROVENANCE = {"calculated", "external", "reference"}


def _canonical_sha256(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _normalized_label(value: Any) -> str:
    if pd.isna(value):
        raise ValueError("Label columns must not contain missing values.")
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError("Label columns must contain finite values.")
        if number.is_integer():
            return str(int(number))
        return format(number, ".17g")
    return str(value).strip().casefold()


def _positive_label_keys(values: Sequence[str], field: str) -> frozenset[str]:
    if not values:
        raise ValueError(f"{field} must contain at least one value.")
    keys = []
    for raw in values:
        value = raw.strip()
        if not value or "\n" in value or "\r" in value:
            raise ValueError(f"{field} must contain non-blank single-line values.")
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = value
        if isinstance(decoded, (dict, list)) or decoded is None:
            raise ValueError(f"{field} values must be JSON scalar values or strings.")
        keys.append(_normalized_label(decoded))
    if len(keys) != len(set(keys)):
        raise ValueError(f"{field} must not contain duplicate values after normalization.")
    return frozenset(keys)


def _parse_times(values: pd.Series, field: str) -> pd.Series:
    if values.isna().any():
        raise ValueError(f"{field} must not contain missing values.")
    try:
        parsed = pd.to_datetime(values, errors="raise", utc=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} contains an invalid date/time value.") from exc
    if parsed.isna().any():
        raise ValueError(f"{field} contains an invalid date/time value.")
    return parsed


def _validate_columns(frame: pd.DataFrame, required: Sequence[str], context: str) -> None:
    normalized = tuple(column.strip() for column in required)
    if any(not column or "\n" in column or "\r" in column for column in normalized):
        raise ValueError(f"{context} columns must be non-blank single-line names.")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{context} columns must have distinct roles.")
    missing = sorted(set(normalized) - set(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing configured columns: {missing}.")


def _prepare_observations(
    frame: pd.DataFrame,
    *,
    time_column: str,
    signal_columns: Sequence[str],
    reference_label_column: str,
    reference_positive_values: Sequence[str],
    comparison_label_column: Optional[str],
    comparison_positive_values: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if frame.empty:
        raise ValueError("Observation input must contain at least one row.")
    if "source_row" in frame.columns:
        raise ValueError("Observation input column 'source_row' is reserved for provenance.")
    signals = tuple(column.strip() for column in signal_columns)
    if not signals:
        raise ValueError("signal_columns must contain at least one column.")
    required = (
        time_column,
        *signals,
        reference_label_column,
        *((comparison_label_column,) if comparison_label_column is not None else ()),
    )
    _validate_columns(frame, required, "Observation input")
    reference_keys = _positive_label_keys(
        reference_positive_values,
        "reference_positive_values",
    )
    if comparison_label_column is None and comparison_positive_values:
        raise ValueError("comparison_positive_values require comparison_label_column.")
    comparison_keys = (
        _positive_label_keys(comparison_positive_values, "comparison_positive_values")
        if comparison_label_column is not None
        else frozenset()
    )

    selected = frame.loc[:, list(required)].copy()
    selected.insert(0, "source_row", np.arange(2, selected.shape[0] + 2, dtype=int))
    selected["observation_time_utc"] = _parse_times(selected[time_column], time_column)
    if selected["observation_time_utc"].duplicated().any():
        duplicate_count = int(selected["observation_time_utc"].duplicated(keep=False).sum())
        raise ValueError(
            "Observation time identifiers must be unique; "
            f"{duplicate_count} rows share a configured time value."
        )
    for column in signals:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
        values = selected[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Signal column {column!r} must contain only finite numeric values.")
    selected["reference_is_anomaly"] = selected[reference_label_column].map(
        lambda value: _normalized_label(value) in reference_keys
    )
    if comparison_label_column is not None:
        selected["comparison_is_anomaly"] = selected[comparison_label_column].map(
            lambda value: _normalized_label(value) in comparison_keys
        )
    selected = selected.sort_values(
        ["observation_time_utc", "source_row"],
        kind="stable",
    ).reset_index(drop=True)
    identity_rows = []
    for row in selected.to_dict(orient="records"):
        identity_rows.append(
            {
                "source_row": int(row["source_row"]),
                "time": row["observation_time_utc"].isoformat(),
                "signals": [float(row[column]) for column in signals],
                "reference_label": _normalized_label(row[reference_label_column]),
                **(
                    {
                        "comparison_label": _normalized_label(
                            row[comparison_label_column]
                        )
                    }
                    if comparison_label_column is not None
                    else {}
                ),
            }
        )
    return selected, {
        "source_order_identity_sha256": _canonical_sha256(
            sorted(identity_rows, key=lambda item: item["source_row"])
        ),
        "time_order_identity_sha256": _canonical_sha256(identity_rows),
        "row_count": int(selected.shape[0]),
        "reference_anomaly_count": int(selected["reference_is_anomaly"].sum()),
        "comparison_anomaly_count": (
            int(selected["comparison_is_anomaly"].sum())
            if comparison_label_column is not None
            else None
        ),
    }


def _prepare_events(
    frame: Optional[pd.DataFrame],
    *,
    event_time_column: Optional[str],
    event_identifier_column: Optional[str],
    event_filter_column: Optional[str],
    event_filter_values: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    columns = [
        "event_source_row",
        "event_identifier",
        "event_time_utc",
    ]
    if frame is None:
        return pd.DataFrame(columns=columns), {
            "input_event_count": 0,
            "retained_event_count": 0,
            "event_identity_sha256": None,
        }
    if frame.empty:
        raise ValueError("Event input must contain at least one row.")
    if "event_source_row" in frame.columns:
        raise ValueError("Event input column 'event_source_row' is reserved for provenance.")
    if event_time_column is None:
        raise ValueError("event_time_column is required when an event dataset is supplied.")
    required = (
        event_time_column,
        *((event_identifier_column,) if event_identifier_column is not None else ()),
        *((event_filter_column,) if event_filter_column is not None else ()),
    )
    _validate_columns(frame, required, "Event input")
    if bool(event_filter_column) != bool(event_filter_values):
        raise ValueError(
            "event_filter_column and event_filter_values must be supplied together."
        )
    filter_keys = (
        _positive_label_keys(event_filter_values, "event_filter_values")
        if event_filter_column is not None
        else frozenset()
    )
    events = frame.copy()
    events.insert(0, "event_source_row", np.arange(2, events.shape[0] + 2, dtype=int))
    input_count = int(events.shape[0])
    if event_filter_column is not None:
        events = events.loc[
            events[event_filter_column].map(
                lambda value: _normalized_label(value) in filter_keys
            )
        ].copy()
    if events.empty:
        raise ValueError("Event filtering removed every event row.")
    events["event_time_utc"] = _parse_times(events[event_time_column], event_time_column)
    if event_identifier_column is None:
        events["event_identifier"] = events["event_source_row"].map(
            lambda value: f"source-row-{int(value)}"
        )
    else:
        if events[event_identifier_column].isna().any():
            raise ValueError("Event identifiers must not contain missing values.")
        events["event_identifier"] = events[event_identifier_column].map(str)
        if events["event_identifier"].str.strip().eq("").any():
            raise ValueError("Event identifiers must be non-blank.")
    if events["event_identifier"].duplicated().any():
        raise ValueError("Event identifiers must be unique after filtering.")
    events = events.sort_values(
        ["event_time_utc", "event_source_row"],
        kind="stable",
    ).reset_index(drop=True)
    identity = [
        {
            "source_row": int(row.event_source_row),
            "identifier": str(row.event_identifier),
            "time": row.event_time_utc.isoformat(),
        }
        for row in events.itertuples(index=False)
    ]
    return events, {
        "input_event_count": input_count,
        "retained_event_count": int(events.shape[0]),
        "event_identity_sha256": _canonical_sha256(identity),
    }


def _associate_events(
    events: pd.DataFrame,
    anomaly_times: pd.Series,
    *,
    window_days: Optional[float],
    direction: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if direction not in _ASSOCIATION_DIRECTIONS:
        raise ValueError(
            f"association_direction must be one of {sorted(_ASSOCIATION_DIRECTIONS)}."
        )
    if window_days is not None and (
        not np.isfinite(window_days) or window_days < 0
    ):
        raise ValueError("association_window_days must be a finite non-negative value.")
    anomalies = list(anomaly_times.sort_values())
    records = []
    for event in events.itertuples(index=False):
        candidates = []
        for anomaly_time in anomalies:
            signed_days = (
                event.event_time_utc - anomaly_time
            ).total_seconds() / 86_400.0
            if direction == "before_event" and signed_days < 0:
                continue
            if direction == "after_event" and signed_days > 0:
                continue
            candidates.append((abs(signed_days), signed_days, anomaly_time))
        nearest = min(candidates, default=None, key=lambda item: item[0])
        distance = nearest[0] if nearest is not None else None
        associated = (
            window_days is not None
            and distance is not None
            and distance <= window_days
        )
        records.append(
            {
                "event_source_row": int(event.event_source_row),
                "event_identifier": str(event.event_identifier),
                "event_time_utc": event.event_time_utc.isoformat(),
                "nearest_reference_anomaly_time_utc": (
                    nearest[2].isoformat() if nearest is not None else None
                ),
                "signed_days_event_minus_anomaly": (
                    float(nearest[1]) if nearest is not None else None
                ),
                "absolute_distance_days": (
                    float(distance) if distance is not None else None
                ),
                "within_association_window": bool(associated),
            }
        )
    table = pd.DataFrame.from_records(
        records,
        columns=[
            "event_source_row",
            "event_identifier",
            "event_time_utc",
            "nearest_reference_anomaly_time_utc",
            "signed_days_event_minus_anomaly",
            "absolute_distance_days",
            "within_association_window",
        ],
    )
    associated_count = (
        int(table["within_association_window"].sum()) if not table.empty else 0
    )
    return table, {
        "association_window_days": window_days,
        "association_direction": direction,
        "associated_event_count": associated_count,
        "association_rate": (
            float(associated_count / table.shape[0]) if not table.empty else None
        ),
    }


def _plot_reference_series(
    observations: pd.DataFrame,
    events: pd.DataFrame,
    *,
    signal_columns: Sequence[str],
    comparison_label_column: Optional[str],
    png_path: Path,
    pdf_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    signals = tuple(signal_columns)
    numeric_values = observations.loc[:, list(signals)].to_numpy(dtype=float)
    minimum = float(np.min(numeric_values))
    maximum = float(np.max(numeric_values))
    span = maximum - minimum
    if span == 0:
        span = max(abs(maximum), 1.0)
    reference_level = maximum + span * 0.10
    comparison_level = maximum + span * 0.19
    event_level = maximum + span * (0.28 if comparison_label_column else 0.19)

    figure, axis = plt.subplots(figsize=(12, 5.8), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(signals)))
    for column, color in zip(signals, colors):
        axis.plot(
            observations["observation_time_utc"],
            observations[column],
            linewidth=1.0,
            color=color,
            label=column,
        )
    reference_rows = observations.loc[observations["reference_is_anomaly"]]
    axis.scatter(
        reference_rows["observation_time_utc"],
        np.full(reference_rows.shape[0], reference_level),
        s=34,
        facecolors="none",
        edgecolors="#00cfd5",
        linewidths=1.4,
        label="Reference anomaly",
        zorder=4,
    )
    if comparison_label_column is not None:
        comparison_rows = observations.loc[observations["comparison_is_anomaly"]]
        axis.scatter(
            comparison_rows["observation_time_utc"],
            np.full(comparison_rows.shape[0], comparison_level),
            s=28,
            marker="x",
            color="#f39c12",
            linewidths=1.3,
            label="Comparison anomaly",
            zorder=4,
        )
    if not events.empty:
        axis.scatter(
            events["event_time_utc"],
            np.full(events.shape[0], event_level),
            s=42,
            marker="v",
            color="#e31a1c",
            label="Event",
            zorder=4,
        )
    if not events.empty:
        top_marker_level = event_level
    elif comparison_label_column is not None:
        top_marker_level = comparison_level
    else:
        top_marker_level = reference_level
    upper = top_marker_level + span * 0.08
    axis.set_ylim(minimum - span * 0.08, upper)
    axis.set_xlabel("Observation time (UTC)")
    axis.set_ylabel("Signal value")
    axis.set_title("Reference-labelled anomaly time series")
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.5, alpha=0.7)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=min(4, len(signals) + 2),
        frameon=False,
    )
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


def run_reference_anomaly_series(
    *,
    input_path: Path,
    output_root: Path,
    experiment_name: str,
    run_name: str,
    sheet: str,
    time_column: str,
    signal_columns: Sequence[str],
    reference_label_column: str,
    reference_positive_values: Sequence[str],
    reference_label_provenance: str = "external_reference",
    comparison_label_column: Optional[str] = None,
    comparison_positive_values: Sequence[str] = (),
    comparison_label_provenance: str = "calculated",
    event_path: Optional[Path] = None,
    event_sheet: str = "0",
    event_time_column: Optional[str] = None,
    event_identifier_column: Optional[str] = None,
    event_filter_column: Optional[str] = None,
    event_filter_values: Sequence[str] = (),
    association_window_days: Optional[float] = None,
    association_direction: str = "before_event",
) -> Path:
    """Write native reference-series plots, tables, metrics, and provenance."""

    experiment_name = _safe_output_name(experiment_name, "experiment_name")
    run_name = _safe_output_name(run_name, "run_name")
    provenance = reference_label_provenance.strip()
    if not provenance or "\n" in provenance or "\r" in provenance:
        raise ValueError("reference_label_provenance must be a non-blank single-line value.")
    if comparison_label_provenance not in _COMPARISON_PROVENANCE:
        raise ValueError(
            "comparison_label_provenance must be calculated, external, or reference."
        )
    source = Path(input_path).expanduser().resolve(strict=True)
    observations, observation_identity = _prepare_observations(
        load_time_series_data(source, sheet),
        time_column=time_column,
        signal_columns=signal_columns,
        reference_label_column=reference_label_column,
        reference_positive_values=reference_positive_values,
        comparison_label_column=comparison_label_column,
        comparison_positive_values=comparison_positive_values,
    )

    event_source = (
        Path(event_path).expanduser().resolve(strict=True)
        if event_path is not None
        else None
    )
    event_frame = (
        load_time_series_data(event_source, event_sheet)
        if event_source is not None
        else None
    )
    events, event_identity = _prepare_events(
        event_frame,
        event_time_column=event_time_column,
        event_identifier_column=event_identifier_column,
        event_filter_column=event_filter_column,
        event_filter_values=event_filter_values,
    )
    anomaly_times = observations.loc[
        observations["reference_is_anomaly"], "observation_time_utc"
    ]
    associations, association_metrics = _associate_events(
        events,
        anomaly_times,
        window_days=association_window_days,
        direction=association_direction,
    )

    root = Path(output_root).expanduser().resolve()
    create_geopi_output_dir(str(root), experiment_name, run_name)
    output_directory = Path(os.environ["GEOPI_OUTPUT_PATH"]).resolve()
    data_directory = Path(os.environ["GEOPI_OUTPUT_ARTIFACTS_DATA_PATH"])
    image_directory = Path(os.environ["GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH"])
    metrics_directory = Path(os.environ["GEOPI_OUTPUT_METRICS_PATH"])
    parameters_directory = Path(os.environ["GEOPI_OUTPUT_PARAMETERS_PATH"])
    summary_directory = Path(os.environ["GEOPI_OUTPUT_SUMMARY_PATH"])

    joined_path = data_directory / "Reference Anomaly Time Series.csv"
    association_path = data_directory / "Reference Anomaly Event Associations.csv"
    png_path = image_directory / "Reference Anomaly Time Series.png"
    pdf_path = image_directory / "Reference Anomaly Time Series.pdf"
    metrics_path = metrics_directory / "Reference Anomaly Time Series Metrics.json"
    parameters_path = parameters_directory / "Reference Anomaly Time Series Parameters.json"
    artifact_index_path = summary_directory / "Reference Anomaly Artifact Index.json"
    manifest_path = summary_directory / "Reference Anomaly Time Series Manifest.json"

    csv_observations = observations.copy()
    csv_observations["observation_time_utc"] = csv_observations[
        "observation_time_utc"
    ].map(lambda value: value.isoformat())
    csv_observations.to_csv(joined_path, index=False)
    associations.to_csv(association_path, index=False)
    _plot_reference_series(
        observations,
        events,
        signal_columns=signal_columns,
        comparison_label_column=comparison_label_column,
        png_path=png_path,
        pdf_path=pdf_path,
    )

    metrics = {
        "schema_version": 1,
        "observation_count": observation_identity["row_count"],
        "reference_anomaly_count": observation_identity["reference_anomaly_count"],
        "comparison_anomaly_count": observation_identity["comparison_anomaly_count"],
        "input_event_count": event_identity["input_event_count"],
        "retained_event_count": event_identity["retained_event_count"],
        **association_metrics,
    }
    _atomic_json(metrics_path, metrics)
    parameters = {
        "schema_version": 1,
        "workflow": "reference_anomaly_time_series",
        "observation_input": {
            "path": str(source),
            "sha256": _sha256(source),
            "sheet": sheet,
        },
        "event_input": (
            None
            if event_source is None
            else {
                "path": str(event_source),
                "sha256": _sha256(event_source),
                "sheet": event_sheet,
            }
        ),
        "columns": {
            "time": time_column,
            "signals": list(signal_columns),
            "reference_label": reference_label_column,
            "comparison_label": comparison_label_column,
            "event_time": event_time_column,
            "event_identifier": event_identifier_column,
            "event_filter": event_filter_column,
        },
        "label_semantics": {
            "reference_positive_values": list(reference_positive_values),
            "reference_provenance": provenance,
            "comparison_positive_values": list(comparison_positive_values),
            "comparison_provenance": (
                comparison_label_provenance
                if comparison_label_column is not None
                else None
            ),
        },
        "event_filter_values": list(event_filter_values),
        "association": {
            "window_days": association_window_days,
            "direction": association_direction,
            "affects_labels": False,
        },
    }
    _atomic_json(parameters_path, parameters)

    primary_artifacts = (
        (joined_path, "reference_anomaly.joined_observations"),
        (association_path, "reference_anomaly.event_associations"),
        (png_path, "reference_anomaly.figure.png"),
        (pdf_path, "reference_anomaly.figure.pdf"),
        (metrics_path, "reference_anomaly.metrics"),
        (parameters_path, "provenance.parameters"),
    )
    artifact_entries = [
        _artifact_entry(path, output_directory, role)
        for path, role in primary_artifacts
    ]
    _atomic_json(
        artifact_index_path,
        {
            "schema_version": 1,
            "artifacts": artifact_entries,
        },
    )
    _atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "workflow": "reference_anomaly_time_series",
            "label_provenance": {
                "reference": provenance,
                "comparison": (
                    comparison_label_provenance
                    if comparison_label_column is not None
                    else None
                ),
            },
            "observation_identity": observation_identity,
            "event_identity": event_identity,
            "metrics": metrics,
            "artifact_index": {
                "relative_path": artifact_index_path.relative_to(
                    output_directory
                ).as_posix(),
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
