import json
from pathlib import Path

from geochemistrypi.data_mining.aggregate import child_result, safe_child_error, write_aggregate_manifest


def test_aggregate_manifest_is_atomic_complete_and_counts_child_artifacts(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "Experiment" / "Run"
    first = parent / "First Model"
    second = parent / "Second Model"
    (first / "metrics").mkdir(parents=True)
    (second / "artifacts").mkdir(parents=True)
    (first / "metrics" / "score.json").write_text("{}", encoding="utf-8")
    (second / "artifacts" / "partial.txt").write_text("partial", encoding="utf-8")
    children = [
        child_result(parent, "First Model", first, "succeeded"),
        child_result(
            parent,
            "Second Model",
            second,
            "failed",
            safe_child_error(RuntimeError("  child\nfailed  ")),
        ),
    ]

    manifest_path = write_aggregate_manifest(
        parent,
        "classification",
        "manual",
        ("First Model", "Second Model"),
        children,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["state"] == "partial_failure"
    assert manifest["succeeded_count"] == 1
    assert manifest["failed_count"] == 1
    assert manifest["children"][0]["artifact_count"] == 1
    assert manifest["children"][1]["error"] == "child failed"
    assert not list(manifest_path.parent.glob("*.tmp"))


def test_aggregate_manifest_reports_complete_only_without_failures(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "Experiment" / "Run"
    child = parent / "Only Model"
    child.mkdir(parents=True)
    manifest_path = write_aggregate_manifest(
        parent,
        "clustering",
        "manual",
        ("Only Model",),
        [child_result(parent, "Only Model", child, "succeeded")],
    )

    assert json.loads(manifest_path.read_text(encoding="utf-8"))["state"] == "complete"
