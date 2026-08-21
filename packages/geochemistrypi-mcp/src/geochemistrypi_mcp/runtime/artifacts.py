"""Discover original CLI artifacts without loading models or recreating results."""

import hashlib
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..api.schemas import ArtifactReference, PreprocessingSummary

_REQUIRED_OUTPUT_DIRECTORIES = ("artifacts", "metrics", "parameters", "summary")
_MAX_INDEXED_ARTIFACTS = 10_000
_MAX_METRIC_FILES = 20
_MAX_METRIC_FILE_BYTES = 1024 * 1024
_MAX_METRIC_VALUES = 100
_MAX_PARAMETERS_FILE_BYTES = 1024 * 1024
_TIME_SERIES_PARAMETERS_RELATIVE_PATH = "parameters/Time Series Parameters.json"


class ArtifactDiscoveryError(RuntimeError):
    """Raised when the real CLI output contract is absent or unsafe."""


@dataclass(frozen=True)
class ArtifactDiscovery:
    """Bounded response references plus the complete wrapper-owned index."""

    response_references: tuple[ArtifactReference, ...]
    all_index_entries: tuple[dict[str, Any], ...]
    truncated: bool
    reported_metrics: dict[str, Any]


def _artifact_id(relative_path: str) -> str:
    return f"artifact-{hashlib.sha256(relative_path.encode('utf-8')).hexdigest()[:16]}"


def _media_type(path: Path) -> str:
    known = {
        ".csv": "text/csv",
        ".joblib": "application/x-joblib",
        ".json": "application/json",
        ".txt": "application/json",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
    return known.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _bounded_json(value: Any, remaining: list[int]) -> Any:
    if remaining[0] <= 0:
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        remaining[0] -= 1
        if isinstance(value, str) and len(value) > 500:
            return f"{value[:499]}…"
        return value
    if isinstance(value, list):
        return [_bounded_json(item, remaining) for item in value[:50] if remaining[0] > 0]
    if isinstance(value, dict):
        bounded: dict[str, Any] = {}
        for key, item in list(value.items())[:50]:
            if remaining[0] <= 0:
                break
            bounded[str(key)] = _bounded_json(item, remaining)
        return bounded
    remaining[0] -= 1
    return str(value)[:500]


def _reported_metrics(output_directory: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    remaining = [_MAX_METRIC_VALUES]
    metric_paths = []
    for path in output_directory.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".txt"}:
            continue
        parts = path.relative_to(output_directory).parts
        if (parts and parts[0] == "metrics") or (len(parts) >= 2 and parts[1] == "metrics"):
            metric_paths.append(path)
    metric_paths.sort()
    for path in metric_paths[:_MAX_METRIC_FILES]:
        if path.stat().st_size > _MAX_METRIC_FILE_BYTES:
            continue
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        relative = path.relative_to(output_directory).as_posix()
        key = path.name if path.parent == output_directory / "metrics" else relative
        metrics[key] = _bounded_json(parsed, remaining)
        if remaining[0] <= 0:
            break
    return metrics


def read_time_series_preprocessing_summary(
    output_directory: Path,
    *,
    source_row_count: int,
    indexed_relative_paths: tuple[str, ...] | list[str] | set[str],
) -> PreprocessingSummary:
    """Read strict row counts from the indexed original Time Series parameter artifact."""
    normalized_index = {str(path).replace("\\", "/") for path in indexed_relative_paths}
    if _TIME_SERIES_PARAMETERS_RELATIVE_PATH not in normalized_index:
        raise ArtifactDiscoveryError("The original Time Series parameter artifact was not indexed.")
    parameters_path = Path(output_directory).resolve() / Path(_TIME_SERIES_PARAMETERS_RELATIVE_PATH)
    if not parameters_path.is_file():
        raise ArtifactDiscoveryError("The original Time Series parameter artifact is missing.")
    try:
        if parameters_path.stat().st_size > _MAX_PARAMETERS_FILE_BYTES:
            raise ArtifactDiscoveryError("The original Time Series parameter artifact exceeds the safety limit.")
        parsed = json.loads(parameters_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactDiscoveryError("The original Time Series parameter artifact is unavailable or malformed.") from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("preprocessing"), dict):
        raise ArtifactDiscoveryError("The original Time Series parameter artifact has no preprocessing object.")
    preprocessing = parsed["preprocessing"]
    try:
        summary = PreprocessingSummary.model_validate(
            {
                "input_row_count": preprocessing.get("input_row_count"),
                "analysis_row_count": preprocessing.get("analysis_row_count"),
                "dropped_row_count": preprocessing.get("dropped_row_count"),
            }
        )
    except ValidationError as exc:
        raise ArtifactDiscoveryError("The original Time Series preprocessing row counts are invalid.") from exc
    if summary.input_row_count != source_row_count:
        raise ArtifactDiscoveryError("The original Time Series input row count does not match the indexed source rows.")
    return summary


def discover_artifacts(output_directory: Path, maximum_response_references: int) -> ArtifactDiscovery:
    """Index existing files under the four original CLI output directories."""
    output_directory = Path(output_directory).resolve()
    missing = [name for name in _REQUIRED_OUTPUT_DIRECTORIES if not (output_directory / name).is_dir()]
    if missing:
        raise ArtifactDiscoveryError(f"CLI output is missing required directories: {missing}")
    categorized_files = []
    for path in output_directory.rglob("*"):
        if not path.is_file():
            continue
        parts = path.relative_to(output_directory).parts
        if parts and parts[0] in _REQUIRED_OUTPUT_DIRECTORIES:
            category = parts[0]
        elif len(parts) >= 2 and parts[1] in _REQUIRED_OUTPUT_DIRECTORIES:
            category = parts[1]
        else:
            continue
        categorized_files.append((path, category))
    categorized_files.sort(key=lambda item: item[0])
    files = [path for path, _ in categorized_files]
    if len(files) > _MAX_INDEXED_ARTIFACTS:
        raise ArtifactDiscoveryError(f"CLI produced {len(files)} files; the safety limit is {_MAX_INDEXED_ARTIFACTS}.")
    references = []
    index_entries = []
    for path, category in categorized_files:
        relative = path.relative_to(output_directory).as_posix()
        reference = ArtifactReference(
            artifact_id=_artifact_id(relative),
            category=category,
            relative_path=relative,
            local_path=str(path),
            size_bytes=path.stat().st_size,
            media_type=_media_type(path),
        )
        index_entries.append(reference.model_dump(mode="json"))
        if len(references) < maximum_response_references:
            references.append(reference)
    return ArtifactDiscovery(
        response_references=tuple(references),
        all_index_entries=tuple(index_entries),
        truncated=len(files) > len(references),
        reported_metrics=_reported_metrics(output_directory),
    )
