import json
from pathlib import Path

import pytest
from geochemistrypi_mcp.runtime.artifacts import ArtifactDiscoveryError, discover_artifacts, read_time_series_preprocessing_summary

_LIU_PARAMETERS_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "liu_time_series"


def test_artifact_discovery_indexes_parent_and_all_model_children(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    for directory in ("artifacts", "metrics", "parameters", "summary"):
        (output / directory).mkdir(parents=True)
    (output / "summary" / "Aggregate Model Results.json").write_text("{}", encoding="utf-8")
    for model, score in (("Model A", 0.8), ("Model B", 0.7)):
        for directory in ("artifacts", "metrics", "parameters", "summary"):
            (output / model / directory).mkdir(parents=True)
        (output / model / "artifacts" / "model.joblib").write_bytes(model.encode())
        (output / model / "metrics" / "Score.json").write_text(json.dumps({"score": score}), encoding="utf-8")

    discovered = discover_artifacts(output, maximum_response_references=100)
    relative_paths = {item.relative_path for item in discovered.response_references}

    assert "summary/Aggregate Model Results.json" in relative_paths
    assert "Model A/artifacts/model.joblib" in relative_paths
    assert "Model B/metrics/Score.json" in relative_paths
    assert discovered.reported_metrics == {
        "Model A/metrics/Score.json": {"score": 0.8},
        "Model B/metrics/Score.json": {"score": 0.7},
    }
    assert {item.category for item in discovered.response_references} == {
        "artifacts",
        "metrics",
        "summary",
    }


def test_time_series_preprocessing_summary_uses_indexed_cli_parameters() -> None:
    summary = read_time_series_preprocessing_summary(
        _LIU_PARAMETERS_FIXTURE_ROOT,
        source_row_count=22640,
        indexed_relative_paths=("parameters/Time Series Parameters.json",),
    )

    assert summary.model_dump() == {
        "input_row_count": 22640,
        "analysis_row_count": 22623,
        "dropped_row_count": 17,
    }


@pytest.mark.parametrize(
    ("payload", "source_row_count"),
    [
        ({"preprocessing": {"input_row_count": 4, "analysis_row_count": 5, "dropped_row_count": 0}}, 4),
        ({"preprocessing": {"input_row_count": 4, "analysis_row_count": 3, "dropped_row_count": 0}}, 4),
        ({"preprocessing": {"input_row_count": 4, "analysis_row_count": 3, "dropped_row_count": 1}}, 5),
        ({"preprocessing": {"input_row_count": "4", "analysis_row_count": 3, "dropped_row_count": 1}}, 4),
        ({"preprocessing": {"input_row_count": True, "analysis_row_count": 1, "dropped_row_count": 0}}, 1),
    ],
    ids=("analysis-too-large", "bad-difference", "source-mismatch", "string-count", "boolean-count"),
)
def test_time_series_preprocessing_summary_rejects_inconsistent_or_untyped_counts(
    tmp_path: Path,
    payload: dict,
    source_row_count: int,
) -> None:
    parameters = tmp_path / "parameters" / "Time Series Parameters.json"
    parameters.parent.mkdir()
    parameters.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtifactDiscoveryError):
        read_time_series_preprocessing_summary(
            tmp_path,
            source_row_count=source_row_count,
            indexed_relative_paths=("parameters/Time Series Parameters.json",),
        )


@pytest.mark.parametrize("case", ("missing", "malformed", "not-indexed"))
def test_time_series_preprocessing_summary_fails_closed_when_unavailable(
    tmp_path: Path,
    case: str,
) -> None:
    parameters = tmp_path / "parameters" / "Time Series Parameters.json"
    parameters.parent.mkdir()
    indexed = ("parameters/Time Series Parameters.json",)
    if case == "malformed":
        parameters.write_text("{", encoding="utf-8")
    elif case == "not-indexed":
        parameters.write_text(
            json.dumps(
                {
                    "preprocessing": {
                        "input_row_count": 4,
                        "analysis_row_count": 3,
                        "dropped_row_count": 1,
                    }
                }
            ),
            encoding="utf-8",
        )
        indexed = ()

    with pytest.raises(ArtifactDiscoveryError):
        read_time_series_preprocessing_summary(
            tmp_path,
            source_row_count=4,
            indexed_relative_paths=indexed,
        )
