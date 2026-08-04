import json
from pathlib import Path

from geochemistrypi_mcp.artifacts import discover_artifacts


def test_artifact_discovery_indexes_parent_and_all_model_children(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    for directory in ("artifacts", "metrics", "parameters", "summary"):
        (output / directory).mkdir(parents=True)
    (output / "summary" / "Aggregate Model Results.json").write_text(
        "{}", encoding="utf-8"
    )
    for model, score in (("Model A", 0.8), ("Model B", 0.7)):
        for directory in ("artifacts", "metrics", "parameters", "summary"):
            (output / model / directory).mkdir(parents=True)
        (output / model / "artifacts" / "model.joblib").write_bytes(model.encode())
        (output / model / "metrics" / "Score.json").write_text(
            json.dumps({"score": score}), encoding="utf-8"
        )

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
