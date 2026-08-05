import hashlib
import json
import os
import re
import shutil
import subprocess
import sysconfig
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List

import pandas as pd
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIRECTORY = Path(__file__).with_name("fixtures")
DATASET_PATH = FIXTURE_DIRECTORY / "classification_baseline.csv"
INTERACTION_PATH = FIXTURE_DIRECTORY / "classification_interaction_v1.json"
MANIFEST_PATH = FIXTURE_DIRECTORY / "classification_output_manifest_v1.json"
GOLDEN_PATH = FIXTURE_DIRECTORY / "classification_golden_v1.json"
ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha256(path: Path) -> str:
    """Hash source text independently of checkout newline conventions."""
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _normalize_console_output(value: str) -> str:
    return ANSI_ESCAPE.sub("", value).replace("\r", "").replace("\f", "\n")


def _public_cli_executable() -> str:
    executable_name = "geochemistrypi.exe" if os.name == "nt" else "geochemistrypi"
    current_environment_command = Path(sysconfig.get_path("scripts")) / executable_name
    if current_environment_command.is_file():
        return str(current_environment_command)
    executable = shutil.which("geochemistrypi")
    assert executable is not None, "Install the project wheel or editable package so the public 'geochemistrypi' command is available."
    return executable


def _expected_output_files(manifest: Dict[str, Any]) -> List[str]:
    primary_files = [f"artifacts/{path}" for path in manifest["artifacts"]]
    primary_files.extend(f"metrics/{path}" for path in manifest["metrics"])
    primary_files.extend(f"parameters/{path}" for path in manifest["parameters"])
    summary_names = [PurePosixPath(path).name for path in primary_files]
    assert len(summary_names) == len(set(summary_names)), "The normalized summary manifest contains colliding basenames."
    return sorted(primary_files + [f"summary/{name}" for name in summary_names])


def _assert_prompt_sequence(stdout: str, steps: List[Dict[str, str]]) -> None:
    normalized = _normalize_console_output(stdout)
    cursor = 0
    for step in steps:
        prompt = step["prompt_contains"]
        position = normalized.find(prompt, cursor)
        assert position >= 0, f"CLI prompt step {step['id']!r} was not observed after character {cursor}: {prompt!r}"
        cursor = position + len(prompt)


def _assert_close(actual: float, expected: float, tolerance: float, name: str) -> None:
    assert actual == pytest.approx(expected, abs=tolerance, rel=0), f"Unexpected value for {name}"


def test_classification_fixture_integrity_and_provenance() -> None:
    golden = _load_json(GOLDEN_PATH)
    fixture = golden["fixture"]

    assert _sha256(DATASET_PATH) == fixture["sha256"]
    source_path = REPOSITORY_ROOT / fixture["source_path"]
    assert _sha256(source_path) == fixture["source_sha256"]

    data = pd.read_csv(DATASET_PATH)
    assert len(data) == fixture["rows"]
    assert data["SampleID"].is_unique
    assert {str(label): int(count) for label, count in data["Label"].value_counts().sort_index().items()} == fixture["class_counts"]


def test_public_cli_entry_files_match_recorded_baseline() -> None:
    golden = _load_json(GOLDEN_PATH)

    for relative_path, expected_hash in golden["cli_entry_files"].items():
        actual_hash = _source_sha256(REPOSITORY_ROOT / relative_path)
        assert actual_hash == expected_hash, f"Public CLI entry file changed without an intentional contract-baseline update: {relative_path}"


def test_public_cli_source_hash_ignores_checkout_line_endings(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_bytes(b"first\r\nsecond\r\n")

    assert _source_sha256(source) == hashlib.sha256(b"first\nsecond\n").hexdigest()


def test_direct_classification_cli_contract(tmp_path: Path) -> None:
    interaction = _load_json(INTERACTION_PATH)
    manifest = _load_json(MANIFEST_PATH)
    golden = _load_json(GOLDEN_PATH)
    executable = _public_cli_executable()

    fixture_hash_before = _sha256(DATASET_PATH)
    command = [value.format(fixture=str(DATASET_PATH)) for value in interaction["public_command"]]
    command[0] = executable
    responses = "\n".join(step["response"] for step in interaction["steps"]) + "\n"
    environment = os.environ.copy()
    environment.pop("SQLALCHEMY_DATABASE_URL", None)
    environment.update(
        {
            "COLUMNS": "200",
            "LINES": "60",
            "MPLBACKEND": "Agg",
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
            "TERM": "dumb",
        }
    )

    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        input=responses,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=240,
    )
    combined_output = result.stdout + "\n" + result.stderr
    assert result.returncode == 0, f"Direct public CLI run failed with exit code {result.returncode}:\n{combined_output[-12000:]}"
    assert _sha256(DATASET_PATH) == fixture_hash_before == golden["fixture"]["sha256"]
    _assert_prompt_sequence(combined_output, interaction["steps"])

    run_directory = tmp_path / "geopi_output" / interaction["experiment_name"] / interaction["run_name"]
    assert run_directory.is_dir()
    top_level_directories = sorted(path.name for path in run_directory.iterdir() if path.is_dir())
    assert top_level_directories == sorted(manifest["top_level_directories"])
    for relative_path in manifest["required_nested_directories"]:
        assert (run_directory / relative_path).is_dir(), f"Missing output directory: {relative_path}"

    actual_files = sorted(path.relative_to(run_directory).as_posix() for path in run_directory.rglob("*") if path.is_file())
    expected_files = _expected_output_files(manifest)
    assert len(actual_files) == manifest["expected_total_file_count"]
    assert actual_files == expected_files

    metric_path = run_directory / "metrics" / "Model Score - Logistic Regression.txt"
    model_metrics = _load_json(metric_path)
    metric_tolerance = golden["tolerances"]["model_metric_absolute"]
    for name, expected in golden["model_metrics"].items():
        if isinstance(expected, (int, float)):
            _assert_close(float(model_metrics[name]), float(expected), metric_tolerance, name)
        else:
            assert model_metrics[name] == expected

    cross_validation = _load_json(run_directory / "metrics" / "Cross Validation - Logistic Regression.txt")
    assert cross_validation["K-Fold"] == golden["cross_validation"]["K-Fold"]
    cross_validation_tolerance = golden["tolerances"]["cross_validation_absolute"]
    for metric_name in ("Accuracy", "Precision", "Recall", "F1 Score"):
        for statistic in ("Mean", "Standard Deviation"):
            _assert_close(
                float(cross_validation[metric_name][statistic]),
                float(golden["cross_validation"][metric_name][statistic]),
                cross_validation_tolerance,
                f"cross validation {metric_name} {statistic}",
            )

    hyperparameters = _load_json(run_directory / "parameters" / "Hyper Parameters - Logistic Regression.txt")
    assert hyperparameters == golden["model"]["hyperparameters"]

    actual = pd.read_excel(run_directory / "artifacts" / "data" / "Y Test.xlsx")
    predicted = pd.read_excel(run_directory / "artifacts" / "data" / "Y Test Predict Decoded.xlsx")
    assert actual["SampleID"].tolist() == predicted["SampleID"].tolist()
    prediction_records = [
        {
            "SampleID": sample_id,
            "actual_encoded": int(actual_encoded),
            "predicted_encoded": int(predicted_encoded),
            "predicted_label": int(predicted_label),
        }
        for sample_id, actual_encoded, predicted_encoded, predicted_label in zip(
            actual["SampleID"],
            actual["Label"],
            predicted["Label"],
            predicted["Label_decoded"],
        )
    ]
    assert prediction_records == golden["test_predictions"]
