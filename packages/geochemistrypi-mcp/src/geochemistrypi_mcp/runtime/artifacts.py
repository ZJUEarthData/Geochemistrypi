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
from .result_views import partition_artifact_views

_REQUIRED_OUTPUT_DIRECTORIES = ("artifacts", "metrics", "parameters", "summary")
_MAX_INDEXED_ARTIFACTS = 10_000
_MAX_METRIC_FILES = 20
_MAX_METRIC_FILE_BYTES = 1024 * 1024
_MAX_METRIC_VALUES = 100
_MAX_PARAMETERS_FILE_BYTES = 1024 * 1024
_MAX_PNG_PIXELS = 50_000_000
_MAX_PNG_COMPRESSED_BYTES = 64 * 1024 * 1024
_TIME_SERIES_PARAMETERS_RELATIVE_PATH = "parameters/Time Series Parameters.json"
_SHA256_HEX_LENGTH = 64
SCIENTIFIC_ESTIMATOR_IDENTITIES = {
    ("classification", "logistic_regression"): ("sklearn", "LogisticRegression"),
    ("classification", "support_vector_machine"): ("sklearn", "SVC"),
    ("classification", "decision_tree"): ("sklearn", "DecisionTreeClassifier"),
    ("classification", "random_forest"): ("sklearn", "RandomForestClassifier"),
    ("classification", "extra_trees"): ("sklearn", "ExtraTreesClassifier"),
    ("classification", "xgboost"): ("xgboost", "XGBClassifier"),
    ("classification", "multi_layer_perceptron"): ("sklearn", "MLPClassifier"),
    ("classification", "gradient_boosting"): ("sklearn", "GradientBoostingClassifier"),
    ("classification", "k_nearest_neighbors"): ("sklearn", "KNeighborsClassifier"),
    ("classification", "stochastic_gradient_descent"): ("sklearn", "SGDClassifier"),
    ("classification", "adaboost"): ("sklearn", "AdaBoostClassifier"),
    ("regression", "decision_tree"): ("sklearn", "DecisionTreeRegressor"),
    ("regression", "random_forest"): ("sklearn", "RandomForestRegressor"),
    ("regression", "extra_trees"): ("sklearn", "ExtraTreesRegressor"),
    ("regression", "gradient_boosting"): ("sklearn", "GradientBoostingRegressor"),
    ("regression", "xgboost"): ("xgboost", "XGBRegressor"),
    ("regression", "multi_layer_perceptron"): ("sklearn", "MLPRegressor"),
    ("regression", "lasso_regression"): ("sklearn", "Lasso"),
    ("regression", "elastic_net"): ("sklearn", "ElasticNet"),
    ("regression", "stochastic_gradient_descent"): ("sklearn", "SGDRegressor"),
    ("clustering", "kmeans"): ("sklearn", "KMeans"),
    ("clustering", "affinity_propagation"): ("sklearn", "AffinityPropagation"),
    ("embedding", "pca"): ("sklearn", "PCA"),
    ("embedding", "tsne"): ("sklearn", "TSNE"),
    ("embedding", "mds"): ("sklearn", "MDS"),
    ("outlier_detection", "isolation_forest"): ("sklearn", "IsolationForest"),
    ("outlier_detection", "local_outlier_factor"): ("sklearn", "LocalOutlierFactor"),
}


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


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == _SHA256_HEX_LENGTH and all(character in "0123456789abcdef" for character in value)


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _classification_metric_semantics_failure(
    value: Any,
    contract: dict[str, Any],
) -> str | None:
    workflow_mode = contract.get("workflow_mode")
    if workflow_mode != "classification":
        return None if value is None else "non-classification attestation contains classification metric semantics"
    if not isinstance(value, dict):
        return "classification attestation is missing metric semantics"
    fields = {
        "schema_version",
        "requested_average",
        "effective_average",
        "requested_positive_label",
        "aggregate_semantic_positive_label",
        "aggregate_encoded_positive_label",
        "curve_semantic_positive_label",
        "curve_encoded_positive_label",
        "curve_probability_column_index",
        "consumers",
    }
    if set(value) != fields or value["schema_version"] != 2 or isinstance(value["schema_version"], bool):
        return "classification metric semantics have an invalid schema"
    requested_average = contract.get("classification_metric_average")
    requested_positive = contract.get("classification_positive_label")
    if value["requested_average"] != requested_average or _canonical_json(value["requested_positive_label"]) != _canonical_json(requested_positive):
        return "classification metric semantics do not match the requested average and positive label"
    effective_average = value["effective_average"]
    if requested_average == "auto":
        if effective_average not in {"binary", "weighted"}:
            return "classification metric semantics contain an invalid effective average"
    elif effective_average != requested_average:
        return "classification metric semantics contain an invalid effective average"

    aggregate_semantic = value["aggregate_semantic_positive_label"]
    aggregate_encoded = value["aggregate_encoded_positive_label"]
    curve_semantic = value["curve_semantic_positive_label"]
    curve_encoded = value["curve_encoded_positive_label"]
    probability_index = value["curve_probability_column_index"]
    if effective_average == "binary":
        if (
            aggregate_semantic is None
            or isinstance(aggregate_encoded, bool)
            or not isinstance(aggregate_encoded, int)
            or _canonical_json(aggregate_semantic) != _canonical_json(curve_semantic)
            or aggregate_encoded != curve_encoded
            or isinstance(probability_index, bool)
            or not isinstance(probability_index, int)
            or probability_index < 0
        ):
            return "binary classification metric semantics use inconsistent positive classes"
        if requested_positive is not None and _canonical_json(aggregate_semantic) != _canonical_json(requested_positive):
            return "binary classification metric semantics do not consume the requested positive label"
    elif aggregate_semantic is not None or aggregate_encoded is not None:
        return "non-binary aggregate metric semantics contain a positive class"

    curve_values = (curve_semantic, curve_encoded, probability_index)
    curve_is_present = any(item is not None for item in curve_values)
    if curve_is_present and (
        curve_semantic is None
        or isinstance(curve_encoded, bool)
        or not isinstance(curve_encoded, int)
        or isinstance(probability_index, bool)
        or not isinstance(probability_index, int)
        or probability_index < 0
    ):
        return "classification curve metric semantics are incomplete"

    consumers = value["consumers"]
    required_consumers = {"holdout_score", "cross_validation"}
    if curve_is_present:
        required_consumers.update({"precision_recall", "precision_recall_threshold", "roc"})
    if not isinstance(consumers, dict) or not required_consumers.issubset(consumers):
        return "classification metric semantics are missing required consumers"
    for name in ("holdout_score", "cross_validation"):
        consumer = consumers[name]
        if (
            not isinstance(consumer, dict)
            or consumer.get("consumer_kind") != "aggregate_metric"
            or consumer.get("effective_average") != effective_average
            or _canonical_json(consumer.get("aggregate_encoded_positive_label")) != _canonical_json(aggregate_encoded)
        ):
            return f"classification aggregate consumer {name} is inconsistent"
    for name in ("precision_recall", "precision_recall_threshold", "roc"):
        if name not in required_consumers:
            continue
        consumer = consumers[name]
        if (
            not isinstance(consumer, dict)
            or consumer.get("consumer_kind") != "binary_curve"
            or _canonical_json(consumer.get("curve_encoded_positive_label")) != _canonical_json(curve_encoded)
            or _canonical_json(consumer.get("probability_column_index")) != _canonical_json(probability_index)
        ):
            return f"classification curve consumer {name} is inconsistent"
    return None


def _parameter_attestation_failure(
    path: Path,
    *,
    expected_source_sha256: str | None,
    expected_source_contract: dict[str, Any] | None,
) -> str | None:
    """Validate the CLI attestation semantically and bind it to this run's sidecar."""

    if path.suffix.lower() not in {".json", ".txt"} or path.stat().st_size > _MAX_PARAMETERS_FILE_BYTES:
        return "scientific execution attestation is not a bounded JSON artifact"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "scientific execution attestation is not valid JSON"
    if not isinstance(record, dict):
        return "scientific execution attestation must be a JSON object"
    required_fields = {
        "schema_version",
        "contract",
        "effective_model_parameters",
        "verified_parameter_names",
        "estimator_identity",
        "classification_metric_semantics",
        "verification_status",
        "attestation_sha256",
    }
    if set(record) != required_fields:
        return "scientific execution attestation has an invalid field contract"
    if record["schema_version"] != 2 or isinstance(record["schema_version"], bool):
        return "scientific execution attestation schema_version is not 2"
    if record["verification_status"] != "matched":
        return "scientific execution attestation verification_status is not matched"
    attestation_sha256 = record["attestation_sha256"]
    if not _is_sha256(attestation_sha256):
        return "scientific execution attestation has an invalid attestation_sha256"
    unhashed_record = dict(record)
    unhashed_record.pop("attestation_sha256")
    if _canonical_json_sha256(unhashed_record) != attestation_sha256:
        return "scientific execution attestation self-hash does not match its canonical content"

    contract = record["contract"]
    if not isinstance(contract, dict):
        return "scientific execution attestation contract must be a JSON object"
    source_sha256 = contract.get("source_sha256")
    if not _is_sha256(source_sha256):
        return "scientific execution attestation contract.source_sha256 is invalid"
    if not _is_sha256(expected_source_sha256) or expected_source_contract is None:
        return "scientific execution attestation source identity is unavailable"
    if source_sha256 != expected_source_sha256:
        return "scientific execution attestation source hash does not match this run's scientific execution sidecar"
    attested_source_contract = dict(contract)
    attested_source_contract.pop("source_sha256")
    if _canonical_json(attested_source_contract) != _canonical_json(expected_source_contract):
        return "scientific execution attestation contract does not match this run's scientific execution sidecar"

    effective_parameters = record["effective_model_parameters"]
    verified_names = record["verified_parameter_names"]
    if not isinstance(effective_parameters, dict) or not effective_parameters:
        return "scientific execution attestation effective_model_parameters must be a non-empty JSON object"
    model_parameters = contract.get("model_parameters")
    if not isinstance(model_parameters, dict):
        return "scientific execution attestation contract.model_parameters must be a JSON object"
    expected_parameter_names = set(model_parameters)
    for name, expected_value in model_parameters.items():
        if name not in effective_parameters:
            return f"scientific execution attestation is missing effective parameter {name!r}"
        observed_value = effective_parameters[name]
        if contract.get("workflow_mode") == "classification" and contract.get("method") == "xgboost" and name == "objective" and expected_value == "auto":
            if observed_value not in {"binary:logistic", "multi:softprob"}:
                return "scientific execution attestation resolved an invalid XGBoost objective"
        elif _canonical_json(observed_value) != _canonical_json(expected_value):
            return f"scientific execution attestation effective parameter {name!r} does not match its source contract"
    model_seed = contract.get("model_seed")
    if model_seed is not None:
        expected_parameter_names.add("random_state")
        if "random_state" not in effective_parameters or _canonical_json(effective_parameters["random_state"]) != _canonical_json(model_seed):
            return "scientific execution attestation random_state does not match model_seed"
    if contract.get("method") == "local_outlier_factor":
        expected_parameter_names.add("novelty")
        expected_novelty = contract.get("evaluation_mode") == "novelty_detection"
        if effective_parameters.get("novelty") is not expected_novelty:
            return "scientific execution attestation novelty does not match evaluation_mode"
    if not isinstance(verified_names, list) or any(not isinstance(name, str) or not name for name in verified_names) or verified_names != sorted(expected_parameter_names):
        return "scientific execution attestation verified_parameter_names do not exactly cover the source contract"

    identity = record["estimator_identity"]
    if not isinstance(identity, dict) or set(identity) != {"expected", "observed"}:
        return "scientific execution attestation estimator_identity has an invalid field contract"
    expected = identity["expected"]
    observed = identity["observed"]
    if not isinstance(expected, dict) or set(expected) != {"module_root", "class_name"} or not all(isinstance(expected.get(key), str) and expected[key] for key in ("module_root", "class_name")):
        return "scientific execution attestation expected estimator identity is invalid"
    trusted_identity = SCIENTIFIC_ESTIMATOR_IDENTITIES.get((contract.get("workflow_mode"), contract.get("method")))
    if trusted_identity is None:
        return "scientific execution attestation method has no trusted estimator identity"
    if expected != {
        "module_root": trusted_identity[0],
        "class_name": trusted_identity[1],
    }:
        return "scientific execution attestation expected estimator identity does not match the trusted method registry"
    if (
        not isinstance(observed, dict)
        or not {"module", "qualname"}.issubset(observed)
        or set(observed) - {"module", "qualname", "wrapper"}
        or not all(isinstance(observed.get(key), str) and observed[key] for key in ("module", "qualname"))
    ):
        return "scientific execution attestation observed estimator identity is invalid"
    if observed["module"].split(".", 1)[0] != expected["module_root"] or observed["qualname"].rsplit(".", 1)[-1] != expected["class_name"]:
        return "scientific execution attestation expected and observed estimator identities do not match"
    wrapper = observed.get("wrapper")
    if wrapper is not None and (
        not isinstance(wrapper, dict)
        or set(wrapper) != {"module", "qualname", "fitted_estimator_count"}
        or contract.get("workflow_mode") != "regression"
        or not isinstance(wrapper.get("module"), str)
        or not wrapper["module"]
        or not isinstance(wrapper.get("qualname"), str)
        or not wrapper["qualname"]
        or wrapper["module"].split(".", 1)[0] != "sklearn"
        or wrapper["qualname"].rsplit(".", 1)[-1] != "MultiOutputRegressor"
        or isinstance(wrapper.get("fitted_estimator_count"), bool)
        or not isinstance(wrapper.get("fitted_estimator_count"), int)
        or wrapper["fitted_estimator_count"] < 1
    ):
        return "scientific execution attestation multi-output wrapper identity is invalid"
    metric_failure = _classification_metric_semantics_failure(
        record["classification_metric_semantics"],
        contract,
    )
    if metric_failure is not None:
        return metric_failure
    return None


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
    *,
    expected_attestation_source_sha256: str | None = None,
    expected_attestation_source_contract: dict[str, Any] | None = None,
) -> str | None:
    if requirement.scientific_type == "parameter_attestation":
        semantic_failure = _parameter_attestation_failure(
            path,
            expected_source_sha256=expected_attestation_source_sha256,
            expected_source_contract=expected_attestation_source_contract,
        )
        if semantic_failure is not None:
            return semantic_failure
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
    expected_attestation_source_sha256: str | None = None,
    expected_attestation_source_contract: dict[str, Any] | None = None,
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
    requirement_paths: dict[str, list[str]] = {requirement.requirement_id: [] for requirement in requirements}
    requirement_content_hashes: dict[str, set[str]] = {requirement.requirement_id: set() for requirement in requirements}
    requirement_failures: dict[str, str] = {}
    for path, category in categorized_files:
        relative = path.relative_to(output_directory).as_posix()
        content_sha256 = _sha256(path)
        matched_requirements = _matched_requirement(relative, requirements, workflow_family, artifact_mappings)
        content_failures = {
            requirement.requirement_id: _requirement_content_failure(
                path,
                requirement,
                expected_attestation_source_sha256=expected_attestation_source_sha256,
                expected_attestation_source_contract=expected_attestation_source_contract,
            )
            for requirement in matched_requirements
        }
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
        references.append(reference)
    artifact_views = partition_artifact_views(tuple(references))
    mirror_sources = dict(artifact_views.summary_mirror_sources)
    annotated_references = tuple(
        reference.model_copy(
            update={
                "metadata": {
                    **reference.metadata,
                    "summary_mirror": True,
                    "mirror_of_artifact_id": mirror_sources[reference.artifact_id],
                }
            }
        )
        if reference.artifact_id in mirror_sources
        else reference
        for reference in references
    )
    response_references = annotated_references[: max(0, maximum_response_references)]
    index_entries = tuple(reference.model_dump(mode="json") for reference in annotated_references)
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
        response_references=response_references,
        all_index_entries=index_entries,
        truncated=len(files) > len(response_references),
        reported_metrics=_reported_metrics(output_directory),
        requirement_matches={key: tuple(value) for key, value in requirement_paths.items()},
        missing_requirement_ids=tuple(dict.fromkeys(missing_requirement_ids)),
        requirement_failures=requirement_failures,
    )
