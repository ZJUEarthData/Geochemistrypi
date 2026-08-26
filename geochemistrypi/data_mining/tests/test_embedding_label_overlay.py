import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

import geochemistrypi.cli as cli_module
from geochemistrypi.data_mining.run_embedding_label_overlay import _prepare_overlay, run_embedding_label_overlay


def _coordinates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample id": ["c", "a", "b"],
            "PC 1": [3.0, 1.0, 2.0],
            "PC 2": [30.0, 10.0, 20.0],
        }
    )


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record id": ["b", "c", "a"],
            "anomaly label": [-1, 1, -1],
        }
    )


def test_overlay_joins_by_identifier_without_reordering_coordinates() -> None:
    joined, counts = _prepare_overlay(
        _coordinates(),
        _labels(),
        coordinate_identifier_column="sample id",
        label_identifier_column="record id",
        x_column="PC 1",
        y_column="PC 2",
        label_column="anomaly label",
        positive_label_values=("-1",),
    )

    assert joined["overlay_identifier"].tolist() == ["c", "a", "b"]
    assert joined["is_anomaly"].tolist() == [False, True, True]
    assert counts["row_count"] == 3
    assert counts["anomaly_count"] == 2
    assert counts["non_anomaly_count"] == 1
    assert len(counts["ordered_join_identity_sha256"]) == 64


@pytest.mark.parametrize(
    "coordinates,labels,message",
    (
        (
            pd.concat([_coordinates(), _coordinates().iloc[[0]]], ignore_index=True),
            _labels(),
            "Coordinate identifiers must be unique",
        ),
        (
            _coordinates(),
            _labels().iloc[:2],
            "identifier sets must match exactly",
        ),
    ),
)
def test_overlay_rejects_duplicate_or_incomplete_identifier_pairing(
    coordinates: pd.DataFrame,
    labels: pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _prepare_overlay(
            coordinates,
            labels,
            coordinate_identifier_column="sample id",
            label_identifier_column="record id",
            x_column="PC 1",
            y_column="PC 2",
            label_column="anomaly label",
            positive_label_values=("-1",),
        )


def test_overlay_writes_native_evidence_package(tmp_path: Path) -> None:
    coordinate_path = tmp_path / "coordinates.csv"
    label_path = tmp_path / "labels.csv"
    _coordinates().to_csv(coordinate_path, index=False)
    _labels().to_csv(label_path, index=False)

    output = run_embedding_label_overlay(
        coordinate_path=coordinate_path,
        label_path=label_path,
        output_root=tmp_path / "output",
        experiment_name="Generic Composition",
        run_name="Overlay",
        coordinate_sheet="0",
        label_sheet="0",
        coordinate_identifier_column="sample id",
        label_identifier_column="record id",
        x_column="PC 1",
        y_column="PC 2",
        label_column="anomaly label",
        positive_label_values=("-1",),
    )

    expected = (
        output / "artifacts" / "data" / "Embedding Label Overlay.csv",
        output / "artifacts" / "image" / "model_output" / "Embedding Label Overlay.png",
        output / "artifacts" / "image" / "model_output" / "Embedding Label Overlay.pdf",
        output / "metrics" / "Embedding Label Overlay Counts.json",
        output / "parameters" / "Embedding Label Overlay Parameters.json",
        output / "summary" / "Embedding Label Overlay Artifact Index.json",
        output / "summary" / "Embedding Label Overlay Manifest.json",
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in expected)
    counts = json.loads(expected[3].read_text(encoding="utf-8"))
    assert counts["row_count"] == 3
    assert counts["anomaly_count"] == 2
    manifest = json.loads(expected[6].read_text(encoding="utf-8"))
    assert manifest["join_policy"] == "exact_identifier_set_one_to_one"
    assert len(manifest["artifact_index"]["sha256"]) == 64


def test_overlay_cli_supports_the_standard_noninteractive_automation_contract(tmp_path: Path) -> None:
    coordinate_path = tmp_path / "coordinates.csv"
    label_path = tmp_path / "labels.csv"
    _coordinates().to_csv(coordinate_path, index=False)
    _labels().to_csv(label_path, index=False)
    plan_path = (tmp_path / "automation-plan.json").resolve()
    events_path = (tmp_path / "automation-events.json").resolve()
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "plan_name": "embedding-label-overlay-test",
                "inputs": [],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli_module.app,
        [
            "embedding-label-overlay",
            "--coordinates",
            str(coordinate_path),
            "--labels",
            str(label_path),
            "--coordinate-identifier-column",
            "sample id",
            "--label-identifier-column",
            "record id",
            "--x-column",
            "PC 1",
            "--y-column",
            "PC 2",
            "--label-column",
            "anomaly label",
            "--positive-label-value",
            "-1",
            "--output-root",
            str(tmp_path / "output"),
            "--automation-plan",
            str(plan_path),
            "--automation-events",
            str(events_path),
        ],
    )

    assert result.exit_code == 0, result.output
    events = json.loads(events_path.read_text(encoding="utf-8"))
    assert events["status"] == "completed"
    assert events["completed_input_ids"] == []
    assert events["events"] == []
