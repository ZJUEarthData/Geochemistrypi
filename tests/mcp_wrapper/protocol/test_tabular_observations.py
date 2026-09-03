import hashlib
import json
from pathlib import Path

import pytest
from geochemistrypi_mcp.api.schemas import ArtifactReference
from geochemistrypi_mcp.config.constants import ARTIFACT_INDEX_SCHEMA_VERSION
from geochemistrypi_mcp.runtime import tabular_observations
from geochemistrypi_mcp.runtime.tabular_observations import TabularObservationError, build_required_tabular_observations
from openpyxl import Workbook


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _category(relative_path: str) -> str:
    for part in Path(relative_path).parts:
        if part in {"artifacts", "metrics", "parameters", "summary"}:
            return part
    raise AssertionError(relative_path)


def _reference(
    output: Path,
    relative_path: str,
    *,
    requirement_id: str,
    scientific_type: str,
) -> ArtifactReference:
    path = output / Path(relative_path)
    return ArtifactReference(
        artifact_id=f"artifact-{hashlib.sha256(relative_path.encode()).hexdigest()[:16]}",
        category=_category(relative_path),
        relative_path=relative_path.replace("\\", "/"),
        local_path=str(path),
        size_bytes=path.stat().st_size,
        media_type={
            ".csv": "text/csv",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".json": "application/json",
            ".txt": "application/json",
        }[path.suffix.lower()],
        sha256=_sha256(path),
        requirement_id=requirement_id,
        requirement_ids=(requirement_id,),
        scientific_type=scientific_type,
        metadata={"producer": "geochemistrypi_cli"},
    )


def _index(output: Path, references: list[ArtifactReference]) -> tuple[Path, str]:
    path = output.parent / "artifact-index.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": ARTIFACT_INDEX_SCHEMA_VERSION,
                "run_id": "run-0123456789abcdef",
                "artifacts": [item.model_dump(mode="json") for item in references],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path, _sha256(path)


def _write_xlsx(path: Path, columns: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Observed"
    worksheet.append(columns)
    for row in rows:
        worksheet.append(row)
    workbook.save(path)
    workbook.close()


def _write_csv(path: Path, columns: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(columns)]
    lines.extend(",".join(str(value) for value in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_small_importance_rows_and_large_prediction_shape_are_both_available(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    importance_path = output / "artifacts/image/model_output/Feature Importance Diagram - XGBoost.xlsx"
    prediction_path = output / "artifacts/data/Application Data Predicted.xlsx"
    ranking = [[f"oxide-{index}", 1.0 / index] for index in range(1, 10)]
    _write_xlsx(importance_path, ["feature", "gain"], ranking)
    _write_xlsx(
        prediction_path,
        ["SAMPLE NAME", "Label Predicted"],
        [[f"A-{index}", index % 2] for index in range(1006)],
    )
    references = [
        _reference(
            output,
            "artifacts/image/model_output/Feature Importance Diagram - XGBoost.xlsx",
            requirement_id="classification.feature-importance",
            scientific_type="feature_importance_table",
        ),
        _reference(
            output,
            "artifacts/data/Application Data Predicted.xlsx",
            requirement_id="classification.application-predictions",
            scientific_type="prediction_table",
        ),
    ]
    index_path, index_sha256 = _index(output, references)

    result = build_required_tabular_observations(output, index_path, index_sha256)

    assert result.total_count == 2
    by_name = {Path(item.relative_path).name: item for item in result.observations}
    importance = by_name["Feature Importance Diagram - XGBoost.xlsx"]
    predictions = by_name["Application Data Predicted.xlsx"]
    assert importance.row_count == 9
    assert importance.column_count == 2
    assert importance.columns == ("feature", "gain")
    assert importance.rows_included is True
    assert tuple(row[0] for row in importance.rows) == tuple(f"oxide-{index}" for index in range(1, 10))
    assert predictions.row_count == 1006
    assert predictions.column_count == 2
    assert predictions.rows_included is False
    assert predictions.rows == ()
    assert predictions.rows_omission_reason == "large_table"
    assert result.returned_cell_count <= 512
    assert result.returned_utf8_bytes <= 16 * 1024


def test_csv_xlsx_json_and_json_txt_tables_are_read_without_metric_duplication(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    csv_path = output / "artifacts/data/Subaerial Proportion.csv"
    xlsx_path = output / "artifacts/data/Residuals.xlsx"
    json_path = output / "artifacts/data/Events.json"
    txt_path = output / "artifacts/data/Scores.txt"
    non_table_path = output / "parameters/Parameters.json"
    _write_csv(csv_path, ["Age", "Estimate"], [[0, 0.2], [100, 0.3]])
    _write_xlsx(xlsx_path, ["Observed", "Residual"], [[1, 0.1], [2, -0.1]])
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps([{"event": "E1", "age": 10}, {"event": "E2", "age": 20}]), encoding="utf-8")
    txt_path.write_text(json.dumps({"columns": ["name", "score"], "rows": [["A", 1], ["B", 2]]}), encoding="utf-8")
    non_table_path.parent.mkdir(parents=True, exist_ok=True)
    non_table_path.write_text(json.dumps({"seed": 2025, "mode": "manual"}), encoding="utf-8")
    specifications = (
        ("artifacts/data/Subaerial Proportion.csv", "ts.bins", "time_series_bin_table"),
        ("artifacts/data/Residuals.xlsx", "regression.residuals", "residual_table"),
        ("artifacts/data/Events.json", "timeseries.events", "event_association_table"),
        ("artifacts/data/Scores.txt", "anomaly.scores", "anomaly_scores"),
        ("parameters/Parameters.json", "parameters.record", "parameter_record"),
    )
    references = [
        _reference(
            output,
            relative_path,
            requirement_id=requirement_id,
            scientific_type=scientific_type,
        )
        for relative_path, requirement_id, scientific_type in specifications
    ]
    index_path, index_sha256 = _index(output, references)

    result = build_required_tabular_observations(output, index_path, index_sha256)

    assert result.total_count == 4
    assert {item.format for item in result.observations} == {"csv", "xlsx", "json", "txt"}
    assert all(item.rows_included for item in result.observations)
    assert all(item.row_count == 2 for item in result.observations)


def test_global_cell_budget_never_returns_partial_table_rows(tmp_path: Path) -> None:
    output = tmp_path / "output"
    columns = [f"C{index}" for index in range(10)]
    rows = [[row * 10 + column for column in range(10)] for row in range(30)]
    references = []
    for name in ("A.csv", "B.csv"):
        relative_path = f"artifacts/data/{name}"
        _write_csv(output / relative_path, columns, rows)
        references.append(
            _reference(
                output,
                relative_path,
                requirement_id=f"budget.{name[0].lower()}",
                scientific_type="machine_readable_table",
            )
        )
    index_path, index_sha256 = _index(output, references)

    result = build_required_tabular_observations(output, index_path, index_sha256)

    assert result.total_count == 2
    assert result.observations[0].rows_included is True
    assert len(result.observations[0].rows) == 30
    assert result.observations[1].rows_included is False
    assert result.observations[1].rows == ()
    assert result.observations[1].rows_omission_reason == "total_cell_budget"
    assert result.returned_cell_count == 300


def test_indexed_file_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "output"
    relative_path = "artifacts/data/Observed.csv"
    path = output / relative_path
    _write_csv(path, ["name", "value"], [["A", 1]])
    reference = _reference(
        output,
        relative_path,
        requirement_id="hash.observed",
        scientific_type="machine_readable_table",
    )
    index_path, index_sha256 = _index(output, [reference])
    path.write_text("name,value\nA,2\n", encoding="utf-8")

    with pytest.raises(TabularObservationError, match="SHA-256"):
        build_required_tabular_observations(output, index_path, index_sha256)


def test_flat_summary_copy_is_not_returned_as_a_second_table(tmp_path: Path) -> None:
    output = tmp_path / "output"
    source_relative = "artifacts/data/Observed.xlsx"
    summary_relative = "summary/Observed.xlsx"
    _write_xlsx(output / source_relative, ["name", "value"], [["A", 1]])
    (output / summary_relative).parent.mkdir(parents=True, exist_ok=True)
    (output / summary_relative).write_bytes((output / source_relative).read_bytes())
    references = [
        _reference(
            output,
            source_relative,
            requirement_id="mirror.source",
            scientific_type="machine_readable_table",
        ),
        _reference(
            output,
            summary_relative,
            requirement_id="mirror.summary",
            scientific_type="machine_readable_table",
        ),
    ]
    index_path, index_sha256 = _index(output, references)

    result = build_required_tabular_observations(output, index_path, index_sha256)

    assert result.total_count == 1
    assert result.observations[0].relative_path == source_relative


@pytest.mark.parametrize(
    "workflow,relative_path,scientific_type,suffix,large",
    (
        ("classification", "artifacts/data/Importance.xlsx", "feature_importance_table", ".xlsx", False),
        ("regression", "artifacts/data/Residuals.xlsx", "residual_table", ".xlsx", False),
        ("clustering", "artifacts/data/Cluster Labels.xlsx", "cluster_assignments", ".xlsx", True),
        ("decomposition", "artifacts/data/X Reduced.xlsx", "embedding_coordinates", ".xlsx", True),
        ("anomaly", "artifacts/data/X Abnormal Detection.xlsx", "anomaly_assignments", ".xlsx", True),
        ("time_series", "artifacts/data/Continuous Time Series.csv", "time_series_bin_table", ".csv", False),
    ),
)
def test_six_workflow_table_roles_share_one_truthful_bounded_contract(
    tmp_path: Path,
    workflow: str,
    relative_path: str,
    scientific_type: str,
    suffix: str,
    large: bool,
) -> None:
    output = tmp_path / workflow
    rows = [[f"S-{index}", index] for index in range(600 if large else 3)]
    path = output / relative_path
    if suffix == ".xlsx":
        _write_xlsx(path, ["name", "value"], rows)
    else:
        _write_csv(path, ["name", "value"], rows)
    reference = _reference(
        output,
        relative_path,
        requirement_id=f"workflow.{workflow}",
        scientific_type=scientific_type,
    )
    index_path, index_sha256 = _index(output, [reference])

    result = build_required_tabular_observations(output, index_path, index_sha256)

    assert result.total_count == 1
    observation = result.observations[0]
    assert observation.row_count == (600 if large else 3)
    assert observation.column_count == 2
    assert observation.rows_included is (not large)


def test_run_with_no_table_like_required_output_returns_hash_bound_empty_summary(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    parameter_path = output / "parameters/Parameters.json"
    parameter_path.parent.mkdir(parents=True, exist_ok=True)
    parameter_path.write_text(json.dumps({"seed": 2025}), encoding="utf-8")
    reference = _reference(
        output,
        "parameters/Parameters.json",
        requirement_id="parameters.record",
        scientific_type="parameter_record",
    )
    index_path, index_sha256 = _index(output, [reference])

    result = build_required_tabular_observations(output, index_path, index_sha256)

    assert result.artifact_index_sha256 == index_sha256
    assert result.total_count == 0
    assert result.observations == ()
    assert result.returned_utf8_bytes == 2
    assert result.omitted_artifact_count == 1
    assert result.omission_reason_counts == {"not_tabular": 1}


def test_observation_size_limit_is_a_bounded_omission_not_a_scientific_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    relative_path = "artifacts/data/Large.csv"
    path = output / relative_path
    _write_csv(path, ["name", "value"], [["A", 1], ["B", 2]])
    reference = _reference(
        output,
        relative_path,
        requirement_id="large.table",
        scientific_type="machine_readable_table",
    )
    index_path, index_sha256 = _index(output, [reference])
    monkeypatch.setattr(tabular_observations, "_MAX_TABULAR_FILE_BYTES", 8)

    result = build_required_tabular_observations(output, index_path, index_sha256)

    assert result.total_count == 0
    assert result.observations == ()
    assert result.omitted_artifact_count == 1
    assert result.omission_reason_counts == {"file_size_limit": 1}
    assert result.omissions_sha256 != hashlib.sha256(b"[]").hexdigest()


def test_unreadable_xlsx_is_a_bounded_omission_after_hash_verification(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    relative_path = "artifacts/data/Unreadable.xlsx"
    path = output / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not an XLSX container")
    reference = _reference(
        output,
        relative_path,
        requirement_id="unreadable.table",
        scientific_type="machine_readable_table",
    )
    index_path, index_sha256 = _index(output, [reference])

    result = build_required_tabular_observations(output, index_path, index_sha256)

    assert result.total_count == 0
    assert result.observations == ()
    assert result.omitted_artifact_count == 1
    assert result.omission_reason_counts == {"parse_unavailable": 1}
