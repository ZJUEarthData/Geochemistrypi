import hashlib
import json
from pathlib import Path

from typer.testing import CliRunner

from geochemistrypi.cli import app
from geochemistrypi.data_mining import datasets as dataset_module
from geochemistrypi.data_mining.datasets import dataset_catalog


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_every_declared_builtin_is_listed_without_modification() -> None:
    root = Path(dataset_module.BUILT_IN_DATASET_PATH)
    before = {path.name: (_sha256(path), path.stat().st_mtime_ns) for path in root.iterdir()}

    result = dataset_catalog("builtin")

    after = {path.name: (_sha256(path), path.stat().st_mtime_ns) for path in root.iterdir()}
    assert before == after
    assert len(result["datasets"]) == 8
    assert {item["dataset_id"] for item in result["datasets"]} == {
        "builtin:application_classification",
        "builtin:application_regression",
        "builtin:anomaly_detection",
        "builtin:classification",
        "builtin:clustering",
        "builtin:decomposition",
        "builtin:regression",
        "builtin:time_series",
    }
    assert all(item["sha256"] == _sha256(Path(item["path"])) for item in result["datasets"])
    assert all("branch.world_map" not in item["analysis_blockers"] for item in result["datasets"])


def test_exact_builtin_selector_constructs_only_the_selected_entry(monkeypatch) -> None:
    original_entry = dataset_module._entry
    constructed: list[str] = []

    def recording_entry(path, *args, **kwargs):
        constructed.append(path.name)
        return original_entry(path, *args, **kwargs)

    monkeypatch.setattr(dataset_module, "_entry", recording_entry)

    result = dataset_catalog(
        "builtin",
        dataset_ids=("builtin:classification",),
    )

    assert constructed == ["Data_Classification.xlsx"]
    assert [item["dataset_id"] for item in result["datasets"]] == ["builtin:classification"]


def test_desktop_discovery_is_read_only_non_recursive_and_format_limited(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    desktop_root = home / "Desktop" / "geopi_input"
    desktop_root.mkdir(parents=True)
    (desktop_root / "valid.csv").write_text("id,value\n1,2\n", encoding="utf-8")
    (desktop_root / "unsupported.xls").write_bytes(b"legacy")
    nested = desktop_root / "nested"
    nested.mkdir()
    (nested / "hidden.csv").write_text("id,value\n1,2\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    before = sorted(str(path.relative_to(home)) for path in home.rglob("*"))

    result = dataset_catalog("desktop")

    after = sorted(str(path.relative_to(home)) for path in home.rglob("*"))
    assert before == after
    assert [item["file_name"] for item in result["datasets"]] == ["valid.csv"]


def test_desktop_discovery_does_not_create_a_missing_directory(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "missing-home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    result = dataset_catalog("desktop")

    assert result["datasets"] == []
    assert not home.exists()
    assert "did not create it" in result["warnings"][0]


def test_desktop_discovery_warns_when_a_file_changes_during_read(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    desktop_root = home / "Desktop" / "geopi_input"
    desktop_root.mkdir(parents=True)
    (desktop_root / "changing.csv").write_text("id,value\n1,2\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(
        dataset_module,
        "_entry",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("file changed")),
    )

    result = dataset_catalog("desktop")

    assert result["datasets"] == []
    assert "changed while being read" in result["warnings"][0]


def test_compact_builtin_counts_ignore_formatted_blank_excel_rows() -> None:
    result = dataset_catalog(
        "builtin",
        dataset_ids=("builtin:classification", "builtin:clustering"),
    )
    counts = {entry["dataset_id"]: entry["row_count"] for entry in result["datasets"]}

    assert counts == {
        "builtin:classification": 2011,
        "builtin:clustering": 2011,
    }
    assert all("inspection" not in entry for entry in result["datasets"])


def test_full_classification_inspection_reports_scientific_quality_fields() -> None:
    selected_columns = (
        "SAMPLE NAME",
        "Label",
        "SIO2(WT%)",
        "TIO2(WT%)",
        "AL2O3(WT%)",
        "CR2O3(WT%)",
        "FEOT(WT%)",
        "CAO(WT%)",
        "MGO(WT%)",
        "NA2O(WT%)",
    )
    result = dataset_catalog(
        "builtin",
        dataset_ids=("builtin:classification",),
        detail="full",
        inspection_columns=selected_columns,
    )
    entry = result["datasets"][0]
    inspection = entry["inspection"]

    assert entry["row_count"] == inspection["row_count"] == 2011
    assert inspection["selected_columns"] == list(selected_columns)
    assert inspection["selected_complete_row_count"] == 2011
    assert inspection["selected_rows_with_any_missing"] == 0
    assert inspection["selected_rows_with_any_nonfinite"] == 0
    assert all(column in inspection["columns"] for column in selected_columns)
    assert all(inspection["missing_counts"][column] == 0 for column in selected_columns)
    assert all(inspection["nonfinite_counts"][column] == 0 for column in selected_columns if column != "SAMPLE NAME")
    label_counts = {item["value"]: item["count"] for item in inspection["low_cardinality_value_counts"]["Label"]}
    assert label_counts == {0: 534, 1: 1477}


def test_full_time_series_inspection_reports_selected_missing_row_contract() -> None:
    selected_columns = (
        "LATITUDE",
        "LONGITUDE",
        "MIN_AGE",
        "AGE",
        "MAX_AGE",
        "R_MIN_AGE",
        "R_AGE",
        "R_MAX_AGE",
        "Estimated Proportion of Subaerial Basalts",
    )
    result = dataset_catalog(
        "builtin",
        dataset_ids=("builtin:time_series",),
        detail="full",
        inspection_columns=selected_columns,
    )
    inspection = result["datasets"][0]["inspection"]

    assert inspection["row_count"] == 22640
    assert inspection["selected_rows_with_any_missing"] == 17
    assert inspection["selected_complete_row_count"] == 22623
    assert inspection["selected_rows_with_any_nonfinite"] == 0
    assert inspection["minimums"]["R_AGE"] >= 0
    assert inspection["minimums"]["R_MAX_AGE"] >= 0
    assert -90 <= inspection["minimums"]["LATITUDE"]
    assert inspection["maximums"]["LATITUDE"] <= 90
    assert -180 <= inspection["minimums"]["LONGITUDE"]
    assert inspection["maximums"]["LONGITUDE"] <= 180
    probability = "Estimated Proportion of Subaerial Basalts"
    assert 0 <= inspection["minimums"][probability]
    assert inspection["maximums"][probability] <= 1


def test_full_inspection_combines_missing_and_nonfinite_rows_and_emits_strict_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    desktop_root = home / "Desktop" / "geopi_input"
    desktop_root.mkdir(parents=True)
    dataset = desktop_root / "quality.csv"
    dataset.write_text(
        "A,B\n" "1,2\n" ",inf\n" ",3\n" "4,-inf\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    result = CliRunner().invoke(
        app,
        [
            "datasets",
            "--source",
            "desktop",
            "--file-name",
            dataset.name,
            "--detail",
            "full",
            "--inspect-column",
            "A",
            "--inspect-column",
            "B",
        ],
    )
    assert result.exit_code == 0, result.output

    def reject_nonstandard_constant(value: str) -> None:
        raise ValueError(f"Non-standard JSON constant: {value}")

    parsed = json.loads(result.output, parse_constant=reject_nonstandard_constant)
    inspection = parsed["datasets"][0]["inspection"]
    assert inspection["selected_rows_with_any_missing"] == 2
    assert inspection["selected_rows_with_any_nonfinite"] == 2
    assert inspection["selected_rows_with_any_invalid"] == 3
    assert inspection["selected_complete_row_count"] == 1
    assert inspection["nonfinite_counts"]["B"] == 2
    assert {item["value"] for item in inspection["low_cardinality_value_counts"]["B"]} == {
        2.0,
        3.0,
        "Infinity",
        "-Infinity",
    }


def test_datasets_help_discovers_full_inspection_contract() -> None:
    result = CliRunner().invoke(app, ["datasets", "--help"])

    assert result.exit_code == 0, result.output
    assert "--detail" in result.output
    assert "--inspect-column" in result.output
