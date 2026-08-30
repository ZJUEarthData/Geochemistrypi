import json
from pathlib import Path

import pandas as pd
import pytest

from geochemistrypi.data_mining.run_reference_anomaly_series import _associate_events, _prepare_events, _prepare_observations, run_reference_anomaly_series


def _observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "when": ["2024-01-09", "2024-01-01", "2024-01-05"],
            "signal a": [3.0, 1.0, 2.0],
            "signal b": [30.0, 10.0, 20.0],
            "archived label": [0, 1, 1],
            "fresh label": [1, 0, 0],
        }
    )


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event id": ["outside", "included-2", "included-1"],
            "event time": ["2024-01-20", "2024-01-11", "2024-01-06"],
            "use": [False, True, True],
        }
    )


def test_reference_series_preserves_labels_and_limits_window_to_evaluation() -> None:
    observations, identity = _prepare_observations(
        _observations(),
        time_column="when",
        signal_columns=("signal a", "signal b"),
        reference_label_column="archived label",
        reference_positive_values=("1",),
        comparison_label_column="fresh label",
        comparison_positive_values=("1",),
    )
    events, event_identity = _prepare_events(
        _events(),
        event_time_column="event time",
        event_identifier_column="event id",
        event_filter_column="use",
        event_filter_values=("true",),
    )
    associations, metrics = _associate_events(
        events,
        observations.loc[observations["reference_is_anomaly"], "observation_time_utc"],
        window_days=3,
        direction="before_event",
    )

    assert observations["source_row"].tolist() == [3, 4, 2]
    assert observations["reference_is_anomaly"].tolist() == [True, True, False]
    assert observations["comparison_is_anomaly"].tolist() == [False, False, True]
    assert identity["reference_anomaly_count"] == 2
    assert identity["comparison_anomaly_count"] == 1
    assert event_identity["input_event_count"] == 3
    assert event_identity["retained_event_count"] == 2
    assert associations["within_association_window"].tolist() == [True, False]
    assert metrics["associated_event_count"] == 1
    assert metrics["association_direction"] == "before_event"


def test_reference_series_rejects_duplicate_time_identity() -> None:
    frame = _observations()
    frame.loc[1, "when"] = frame.loc[0, "when"]
    with pytest.raises(ValueError, match="time identifiers must be unique"):
        _prepare_observations(
            frame,
            time_column="when",
            signal_columns=("signal a",),
            reference_label_column="archived label",
            reference_positive_values=("1",),
            comparison_label_column=None,
            comparison_positive_values=(),
        )


def test_reference_series_writes_native_evidence_package(tmp_path: Path) -> None:
    observation_path = tmp_path / "observations.csv"
    event_path = tmp_path / "events.csv"
    _observations().to_csv(observation_path, index=False)
    _events().to_csv(event_path, index=False)

    output = run_reference_anomaly_series(
        input_path=observation_path,
        output_root=tmp_path / "output",
        experiment_name="Generic Reference",
        run_name="Overlay",
        sheet="0",
        time_column="when",
        signal_columns=("signal a", "signal b"),
        reference_label_column="archived label",
        reference_positive_values=("1",),
        reference_label_provenance="archived_external",
        comparison_label_column="fresh label",
        comparison_positive_values=("1",),
        comparison_label_provenance="calculated",
        event_path=event_path,
        event_time_column="event time",
        event_identifier_column="event id",
        event_filter_column="use",
        event_filter_values=("true",),
        association_window_days=3,
        association_direction="before_event",
    )

    expected = (
        output / "artifacts" / "data" / "Reference Anomaly Time Series.csv",
        output / "artifacts" / "data" / "Reference Anomaly Event Associations.csv",
        output / "artifacts" / "image" / "model_output" / "Reference Anomaly Time Series.png",
        output / "artifacts" / "image" / "model_output" / "Reference Anomaly Time Series.pdf",
        output / "metrics" / "Reference Anomaly Time Series Metrics.json",
        output / "parameters" / "Reference Anomaly Time Series Parameters.json",
        output / "summary" / "Reference Anomaly Artifact Index.json",
        output / "summary" / "Reference Anomaly Time Series Manifest.json",
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in expected)
    metrics = json.loads(expected[4].read_text(encoding="utf-8"))
    assert metrics["reference_anomaly_count"] == 2
    assert metrics["comparison_anomaly_count"] == 1
    assert metrics["retained_event_count"] == 2
    manifest = json.loads(expected[7].read_text(encoding="utf-8"))
    assert manifest["label_provenance"] == {
        "reference": "archived_external",
        "comparison": "calculated",
    }
    assert manifest["observation_identity"]["source_order_identity_sha256"]
    assert manifest["event_identity"]["event_identity_sha256"]
