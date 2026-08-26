"""Discover original CLI artifacts without loading models or recreating results."""

import hashlib
import json
import mimetypes
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..api.schemas import ArtifactReference, ArtifactRequirement, PreprocessingSummary
from ..planning.artifact_mapping import AdapterArtifactMapping
from ..planning.scientific_contract import artifact_requirement_matches, describe_scientific_output

_REQUIRED_OUTPUT_DIRECTORIES = ("artifacts", "metrics", "parameters", "summary")
_MAX_INDEXED_ARTIFACTS = 10_000
_MAX_METRIC_FILES = 20
_MAX_METRIC_FILE_BYTES = 1024 * 1024
_MAX_METRIC_VALUES = 100
_MAX_PARAMETERS_FILE_BYTES = 1024 * 1024
_MAX_PNG_PIXELS = 50_000_000
_MAX_PNG_COMPRESSED_BYTES = 64 * 1024 * 1024
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
    requirement_matches: dict[str, tuple[str, ...]]
    missing_requirement_ids: tuple[str, ...]
    requirement_failures: dict[str, str]


def _artifact_id(relative_path: str) -> str:
    return f"artifact-{hashlib.sha256(relative_path.encode('utf-8')).hexdigest()[:16]}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matched_requirement(
    relative_path: str,
    requirements: tuple[ArtifactRequirement, ...],
    workflow_family: str | None,
    artifact_mappings: tuple[AdapterArtifactMapping, ...],
) -> tuple[ArtifactRequirement, ...]:
    descriptors = _produced_descriptors(relative_path, workflow_family, artifact_mappings)
    return tuple(requirement for requirement in requirements if any(artifact_requirement_matches(requirement, descriptor) for descriptor in descriptors))


def _same_output_path(declared: str, produced: str) -> bool:
    declared = declared.replace("\\", "/")
    produced = produced.replace("\\", "/")
    return declared == produced or declared.endswith(f"/{produced}") or produced.endswith(f"/{declared}")


def _produced_descriptors(
    relative_path: str,
    workflow_family: str | None,
    artifact_mappings: tuple[AdapterArtifactMapping, ...],
) -> tuple[dict[str, Any], ...]:
    fallback = describe_scientific_output(relative_path, workflow_family)
    descriptors = []
    for mapping in artifact_mappings:
        if mapping.availability != "available" or mapping.relative_path is None:
            continue
        if not _same_output_path(mapping.relative_path, relative_path):
            continue
        descriptor = dict(fallback)
        descriptor.update(
            {
                "mapping_id": mapping.mapping_id,
                "scientific_type": mapping.scientific_type,
                "output_role": mapping.output_role,
            }
        )
        descriptors.append(descriptor)
    if not descriptors or all((descriptor["scientific_type"], descriptor["output_role"]) != (fallback["scientific_type"], fallback["output_role"]) for descriptor in descriptors):
        descriptors.append(fallback)
    return tuple(descriptors)


def _has_required_json_keys(path: Path, keys: tuple[str, ...]) -> bool:
    if not keys:
        return True
    if path.suffix.lower() not in {".json", ".txt"} or path.stat().st_size > _MAX_PARAMETERS_FILE_BYTES:
        return False
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    for key in keys:
        value = parsed
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return False
            value = value[part]
    return True


def _paeth_predictor(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    left_distance = abs(estimate - left)
    above_distance = abs(estimate - above)
    upper_left_distance = abs(estimate - upper_left)
    if left_distance <= above_distance and left_distance <= upper_left_distance:
        return left
    if above_distance <= upper_left_distance:
        return above
    return upper_left


def _png_has_visible_plot_data(path: Path) -> bool:
    """Reject a valid PNG whose central plotting region contains no visible marks."""

    try:
        encoded = path.read_bytes()
    except OSError:
        return False
    if not encoded.startswith(b"\x89PNG\r\n\x1a\n"):
        return False
    offset = 8
    header = None
    compressed_parts = []
    compressed_size = 0
    while offset + 12 <= len(encoded):
        length = struct.unpack(">I", encoded[offset : offset + 4])[0]
        chunk_end = offset + 12 + length
        if chunk_end > len(encoded):
            return False
        chunk_type = encoded[offset + 4 : offset + 8]
        payload = encoded[offset + 8 : offset + 8 + length]
        if chunk_type == b"IHDR":
            if header is not None or length != 13:
                return False
            header = struct.unpack(">IIBBBBB", payload)
        elif chunk_type == b"IDAT":
            compressed_size += length
            if compressed_size > _MAX_PNG_COMPRESSED_BYTES:
                return False
            compressed_parts.append(payload)
        elif chunk_type == b"IEND":
            break
        offset = chunk_end
    if header is None or not compressed_parts:
        return False
    width, height, bit_depth, color_type, compression, filtering, interlace = header
    channels_by_color_type = {0: 1, 2: 3, 4: 2, 6: 4}
    channels = channels_by_color_type.get(color_type)
    if channels is None or bit_depth != 8 or compression != 0 or filtering != 0 or interlace != 0 or width < 2 or height < 2 or width * height > _MAX_PNG_PIXELS:
        return False
    row_bytes = width * channels
    expected_size = height * (row_bytes + 1)
    try:
        decompressor = zlib.decompressobj()
        decoded = decompressor.decompress(
            b"".join(compressed_parts),
            expected_size + 1,
        )
        decoded += decompressor.flush()
    except zlib.error:
        return False
    if len(decoded) != expected_size or not decompressor.eof:
        return False
    x_start = width * 15 // 100
    x_stop = max(x_start + 1, width * 85 // 100)
    y_start = height * 15 // 100
    y_stop = max(y_start + 1, height * 85 // 100)
    required_visible_pixels = max(4, min(width, height) // 50)
    prior = bytearray(row_bytes)
    cursor = 0
    visible_pixels = 0
    for y_position in range(height):
        filter_type = decoded[cursor]
        cursor += 1
        scanline = bytearray(decoded[cursor : cursor + row_bytes])
        cursor += row_bytes
        if filter_type not in {0, 1, 2, 3, 4}:
            return False
        for index in range(row_bytes):
            left = scanline[index - channels] if index >= channels else 0
            above = prior[index]
            upper_left = prior[index - channels] if index >= channels else 0
            if filter_type == 1:
                scanline[index] = (scanline[index] + left) & 0xFF
            elif filter_type == 2:
                scanline[index] = (scanline[index] + above) & 0xFF
            elif filter_type == 3:
                scanline[index] = (scanline[index] + ((left + above) // 2)) & 0xFF
            elif filter_type == 4:
                scanline[index] = (scanline[index] + _paeth_predictor(left, above, upper_left)) & 0xFF
        if y_start <= y_position < y_stop:
            for x_position in range(x_start, x_stop):
                pixel_offset = x_position * channels
                if color_type in {0, 4}:
                    red = green = blue = scanline[pixel_offset]
                    alpha = scanline[pixel_offset + 1] if color_type == 4 else 255
                else:
                    red, green, blue = scanline[pixel_offset : pixel_offset + 3]
                    alpha = scanline[pixel_offset + 3] if color_type == 6 else 255
                if alpha > 16 and (max(red, green, blue) - min(red, green, blue) >= 24 or max(red, green, blue) < 180):
                    visible_pixels += 1
                    if visible_pixels >= required_visible_pixels:
                        return True
        prior = scanline
    return False


def _requirement_content_failure(
    path: Path,
    requirement: ArtifactRequirement,
) -> str | None:
    if not _has_required_json_keys(path, requirement.required_json_keys):
        return "matched artifact is missing required JSON keys"
    if requirement.scientific_type == "observed_predicted_figure":
        if path.suffix.lower() != ".png" or not _png_has_visible_plot_data(path):
            return "matched figure contains no visible plot data in the central plotting region"
    return None


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


def discover_artifacts(
    output_directory: Path,
    maximum_response_references: int,
    requirements: tuple[ArtifactRequirement, ...] = (),
    workflow_family: str | None = None,
    artifact_mappings: tuple[AdapterArtifactMapping, ...] = (),
) -> ArtifactDiscovery:
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
    requirement_paths: dict[str, list[str]] = {requirement.requirement_id: [] for requirement in requirements}
    requirement_content_hashes: dict[str, set[str]] = {requirement.requirement_id: set() for requirement in requirements}
    requirement_failures: dict[str, str] = {}
    for path, category in categorized_files:
        relative = path.relative_to(output_directory).as_posix()
        content_sha256 = _sha256(path)
        matched_requirements = _matched_requirement(relative, requirements, workflow_family, artifact_mappings)
        content_failures = {requirement.requirement_id: _requirement_content_failure(path, requirement) for requirement in matched_requirements}
        satisfied_requirements = tuple(requirement for requirement in matched_requirements if content_failures[requirement.requirement_id] is None)
        for requirement in matched_requirements:
            if requirement in satisfied_requirements:
                requirement_paths[requirement.requirement_id].append(relative)
                requirement_content_hashes[requirement.requirement_id].add(content_sha256)
            else:
                requirement_failures[requirement.requirement_id] = str(content_failures[requirement.requirement_id])
        descriptors = _produced_descriptors(relative, workflow_family, artifact_mappings)
        descriptor = descriptors[0]
        reference = ArtifactReference(
            artifact_id=_artifact_id(relative),
            category=category,
            relative_path=relative,
            local_path=str(path),
            size_bytes=path.stat().st_size,
            media_type=_media_type(path),
            sha256=content_sha256,
            requirement_id=satisfied_requirements[0].requirement_id if satisfied_requirements else None,
            requirement_ids=tuple(requirement.requirement_id for requirement in satisfied_requirements),
            scientific_type=descriptor["scientific_type"],
            metadata={
                "producer": "geochemistrypi_cli",
                "hash_algorithm": "sha256",
                "output_role": descriptor["output_role"],
                "output_roles": list(dict.fromkeys(item["output_role"] for item in descriptors)),
                "adapter_mapping_ids": [item["mapping_id"] for item in descriptors if item.get("mapping_id") is not None],
            },
        )
        index_entries.append(reference.model_dump(mode="json"))
        if len(references) < maximum_response_references:
            references.append(reference)
    missing_requirement_ids = []
    for requirement in requirements:
        count = len(requirement_content_hashes[requirement.requirement_id])
        minimum_count = getattr(requirement, "minimum_count", getattr(requirement, "count", 1))
        maximum_count = getattr(requirement, "maximum_count", getattr(requirement, "count", None))
        if getattr(requirement, "required", True) and count < minimum_count:
            missing_requirement_ids.append(requirement.requirement_id)
            requirement_failures.setdefault(requirement.requirement_id, f"produced {count} artifact(s), fewer than minimum_count={minimum_count}")
        if maximum_count is not None and count > maximum_count:
            missing_requirement_ids.append(requirement.requirement_id)
            requirement_failures[requirement.requirement_id] = f"produced {count} artifact(s), more than maximum_count={maximum_count}"
    return ArtifactDiscovery(
        response_references=tuple(references),
        all_index_entries=tuple(index_entries),
        truncated=len(files) > len(references),
        reported_metrics=_reported_metrics(output_directory),
        requirement_matches={key: tuple(value) for key, value in requirement_paths.items()},
        missing_requirement_ids=tuple(dict.fromkeys(missing_requirement_ids)),
        requirement_failures=requirement_failures,
    )
