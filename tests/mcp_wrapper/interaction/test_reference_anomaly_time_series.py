import csv
from pathlib import Path

import pytest
from geochemistrypi_mcp.api.schemas import TimeSeriesRequest
from geochemistrypi_mcp.planning.interaction_plan import AnalysisPlanCompiler, PlanCompilationError
from geochemistrypi_mcp.planning.scientific_contract import assess_scientific_compatibility, canonical_scientific_contract, planned_artifact_requirements
from pydantic import ValidationError


def _write_observations(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sample time", "signal one", "signal two", "reference label"))
        writer.writerows(
            (
                ("2025-01-01", 1.0, 10.0, 1),
                ("2025-01-03", 2.0, 20.0, 0),
                ("2025-01-06", 3.0, 30.0, 1),
            )
        )


def _write_events(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("event id", "event time", "accepted"))
        writer.writerows(
            (
                ("e1", "2025-01-04", True),
                ("e2", "2025-01-08", False),
            )
        )


def _request(observations: Path, events: Path) -> TimeSeriesRequest:
    return TimeSeriesRequest(
        mode="reference_anomaly_series",
        training_dataset_path=observations,
        experiment_name="Reference Series",
        run_name="Generic Overlay",
        time_column="sample time",
        signal_columns=("signal one", "signal two"),
        reference_label_column="reference label",
        reference_positive_values=("1",),
        reference_label_provenance="archived_external",
        event_dataset_path=events,
        event_time_column="event time",
        event_identifier_column="event id",
        event_filter_column="accepted",
        event_filter_values=("true",),
        association_window_days=5,
        association_direction="before_event",
    )


def test_reference_anomaly_series_compiles_to_existing_analysis_toolchain(
    tmp_path: Path,
) -> None:
    observations = tmp_path / "observations.csv"
    events = tmp_path / "events.csv"
    _write_observations(observations)
    _write_events(events)
    request = _request(observations, events)

    plan = AnalysisPlanCompiler().compile(
        request,
        cli_executable=tmp_path / "geochemistrypi.exe",
    )
    requirements = planned_artifact_requirements(request, plan)
    assessment = assess_scientific_compatibility(request, plan, requirements)
    contract = canonical_scientific_contract(request, plan)

    assert plan.execution_ready is True
    assert plan.workflow_family == "time_series"
    assert plan.workflow_mode == "reference_anomaly_series"
    assert plan.method == "reference_label_event_overlay"
    assert plan.adapter_id == "geochemistrypi-cli.time-series.reference-anomaly-series"
    assert plan.steps == ()
    assert plan.public_command[1] == "reference-anomaly-time-series"
    assert "--association-window-days" in plan.public_command
    assert "--event-filter-column" in plan.public_command
    assert len(plan.expected_output_relative_paths) == 8
    assert assessment.execution_ready is True
    assert assessment.blocking_issues == ()
    assert contract["datasets"]["events"]["path"] == str(events)
    assert contract["column_roles"]["reference_label"] == "reference label"
    assert contract["parameters"]["association_direction"] == "before_event"
    mapped_roles = {mapping.output_role for mapping in plan.artifact_mappings}
    assert {
        "reference_anomaly.joined_observations",
        "reference_anomaly.event_associations",
        "reference_anomaly.figure",
        "reference_anomaly.metrics",
        "provenance.artifact_index",
        "provenance.scientific_manifest",
    } <= mapped_roles


def test_reference_anomaly_series_rejects_incomplete_event_semantics(
    tmp_path: Path,
) -> None:
    observations = tmp_path / "observations.csv"
    events = tmp_path / "events.csv"
    _write_observations(observations)
    _write_events(events)
    with pytest.raises(ValidationError, match="event_dataset_path requires event_time_column"):
        TimeSeriesRequest(
            mode="reference_anomaly_series",
            training_dataset_path=observations,
            experiment_name="Reference Series",
            run_name="Generic Overlay",
            time_column="sample time",
            signal_columns=("signal one",),
            reference_label_column="reference label",
            reference_positive_values=("1",),
            event_dataset_path=events,
        )


def test_reference_anomaly_series_fails_before_cli_on_missing_signal(
    tmp_path: Path,
) -> None:
    observations = tmp_path / "observations.csv"
    events = tmp_path / "events.csv"
    _write_observations(observations)
    _write_events(events)
    request = _request(observations, events).model_copy(update={"signal_columns": ("absent signal",)})
    with pytest.raises(PlanCompilationError, match="absent from the observation dataset"):
        AnalysisPlanCompiler().compile(
            request,
            cli_executable=tmp_path / "geochemistrypi.exe",
        )
