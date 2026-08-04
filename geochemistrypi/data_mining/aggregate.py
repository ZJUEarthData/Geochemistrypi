"""Durable parent/child result metadata for the public all-models workflow."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

AGGREGATE_MANIFEST_SCHEMA_VERSION = 1


def safe_child_error(error: BaseException) -> str:
    """Bound one child failure without exposing tracebacks or stopping siblings."""
    message = " ".join(str(error).split())
    return (message or type(error).__name__)[:1000]


def child_result(
    parent_directory: Path,
    model: str,
    child_directory: Path,
    state: str,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_parent = Path(parent_directory).resolve()
    resolved_child = Path(child_directory).resolve()
    relative = resolved_child.relative_to(resolved_parent).as_posix()
    artifact_count = (
        sum(1 for path in resolved_child.rglob("*") if path.is_file())
        if resolved_child.is_dir()
        else 0
    )
    return {
        "model": model,
        "state": state,
        "output_relative_path": relative,
        "artifact_count": artifact_count,
        "error": error,
    }


def write_aggregate_manifest(
    parent_directory: Path,
    task: str,
    tuning: str,
    expected_models: Sequence[str],
    children: List[Dict[str, Any]],
) -> Path:
    """Atomically publish one complete or partial aggregate summary."""
    parent = Path(parent_directory).resolve()
    summary = parent / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    succeeded = sum(child["state"] == "succeeded" for child in children)
    failed = sum(child["state"] == "failed" for child in children)
    payload = {
        "schema_version": AGGREGATE_MANIFEST_SCHEMA_VERSION,
        "task": task,
        "selection_mode": "all",
        "tuning": tuning,
        "state": "complete" if failed == 0 else "partial_failure",
        "expected_model_count": len(expected_models),
        "succeeded_count": succeeded,
        "failed_count": failed,
        "children": children,
    }
    destination = summary / "Aggregate Model Results.json"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(summary),
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(str(temporary), str(destination))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination
