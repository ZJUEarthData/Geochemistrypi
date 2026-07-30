"""Dependency-free engine dataclasses for the v1 wire contracts."""

import copy
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Type, TypeVar, Union

from .schema import CONTRACT_VERSION

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MODEL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


class StringEnum(str, Enum):
    """Enum whose values serialize directly to JSON strings."""


class DatasetKind(StringEnum):
    LOCAL_FILE = "local_file"


class DatasetFormat(StringEnum):
    CSV = "csv"
    XLSX = "xlsx"


class SnapshotPolicy(StringEnum):
    COPY = "copy"
    REFERENCE = "reference"


class MissingValueStrategy(StringEnum):
    NONE = "none"
    MEAN = "mean"
    MEDIAN = "median"
    MOST_FREQUENT = "most_frequent"


class ScalingStrategy(StringEnum):
    NONE = "none"
    MIN_MAX = "min_max"
    STANDARD = "standard"
    MEAN_NORMALIZATION = "mean_normalization"


class ClassBalanceStrategy(StringEnum):
    NONE = "none"
    RANDOM_OVER = "random_over"
    RANDOM_UNDER = "random_under"
    OVER_UNDER = "over_under"


class SplitStrategy(StringEnum):
    STRATIFIED_RANDOM = "stratified_random"
    GROUP = "group"


class ModelMode(StringEnum):
    MANUAL = "manual"
    AUTOML = "automl"


class PrimaryMetric(StringEnum):
    ACCURACY = "accuracy"
    BALANCED_ACCURACY = "balanced_accuracy"
    MACRO_F1 = "macro_f1"
    WEIGHTED_F1 = "weighted_f1"


class RunStatus(StringEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ErrorCode(StringEnum):
    UNSUPPORTED_CONTRACT_VERSION = "UNSUPPORTED_CONTRACT_VERSION"
    SCHEMA_VALIDATION_FAILED = "SCHEMA_VALIDATION_FAILED"
    INVALID_DATASET = "INVALID_DATASET"
    DATASET_CHANGED = "DATASET_CHANGED"
    INVALID_TARGET_COLUMN = "INVALID_TARGET_COLUMN"
    INVALID_FEATURE_COLUMNS = "INVALID_FEATURE_COLUMNS"
    INVALID_SPLIT_CONFIGURATION = "INVALID_SPLIT_CONFIGURATION"
    INVALID_MODEL_PARAMETERS = "INVALID_MODEL_PARAMETERS"
    RUN_NOT_FOUND = "RUN_NOT_FOUND"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ErrorStage(StringEnum):
    CONTRACT = "contract"
    VALIDATION = "validation"
    DATASET = "dataset"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    EVALUATION = "evaluation"
    PERSISTENCE = "persistence"
    RUNTIME = "runtime"


EnumType = TypeVar("EnumType", bound=Enum)
LabelValue = Union[str, int, float]


def _prepare_mapping(data: Mapping[str, Any], allowed: Iterable[str], required: Iterable[str], label: str) -> Dict[str, Any]:
    if not isinstance(data, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    result = dict(data)
    allowed_set = set(allowed)
    unknown = sorted(set(result) - allowed_set)
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {unknown}.")
    missing = sorted(set(required) - set(result))
    if missing:
        raise ValueError(f"{label} is missing required fields: {missing}.")
    return result


def _as_enum(enum_type: Type[EnumType], value: Any, field_name: str) -> EnumType:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        valid = ", ".join(repr(item.value) for item in enum_type)
        raise ValueError(f"{field_name} must be one of: {valid}.") from exc


def _nonempty_string(value: Any, field_name: str, max_length: int = 1024) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if len(value) > max_length:
        raise ValueError(f"{field_name} must contain at most {max_length} characters.")
    return value


def _optional_string(value: Any, field_name: str, max_length: int = 1024) -> Optional[str]:
    if value is None:
        return None
    return _nonempty_string(value, field_name, max_length)


def _string_tuple(values: Any, field_name: str, allow_empty: bool = False, max_items: int = 10000, item_max_length: int = 256) -> Tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, (list, tuple)):
        raise TypeError(f"{field_name} must be a list or tuple of strings.")
    result = tuple(_nonempty_string(value, f"{field_name} item", item_max_length) for value in values)
    if not allow_empty and not result:
        raise ValueError(f"{field_name} must not be empty.")
    if len(result) > max_items:
        raise ValueError(f"{field_name} must contain at most {max_items} values.")
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must not contain duplicate values.")
    return result


def _validate_json_value(value: Any, field_name: str, depth: int = 0) -> None:
    if depth > 8:
        raise ValueError(f"{field_name} exceeds the maximum nesting depth of 8.")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} contains a non-finite number.")
        return
    if isinstance(value, (list, tuple)):
        if len(value) > 1000:
            raise ValueError(f"{field_name} arrays must contain at most 1000 values.")
        for index, item in enumerate(value):
            _validate_json_value(item, f"{field_name}[{index}]", depth + 1)
        return
    if isinstance(value, Mapping):
        if len(value) > 100:
            raise ValueError(f"{field_name} objects must contain at most 100 fields.")
        for key, item in value.items():
            _nonempty_string(key, f"{field_name} key", 128)
            _validate_json_value(item, f"{field_name}.{key}", depth + 1)
        return
    raise TypeError(f"{field_name} contains a value that cannot be represented in JSON: {type(value).__name__}.")


def _json_mapping(value: Any, field_name: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object.")
    _validate_json_value(value, field_name)
    return _canonical_json_value(value)


def _canonical_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _canonical_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    return copy.deepcopy(value)


def _sha256(value: Any, field_name: str, nullable: bool = False) -> Optional[str]:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase 64-character SHA-256 digest.")
    return value


def _relative_path(value: Any, field_name: str) -> str:
    path_value = _nonempty_string(value, field_name, 512)
    if "\\" in path_value or re.match(r"^[A-Za-z]:", path_value):
        raise ValueError(f"{field_name} must be a portable relative path using forward slashes.")
    path = PurePosixPath(path_value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field_name} must not be absolute or contain parent traversal.")
    return path.as_posix()


@dataclass
class DatasetReadOptions:
    encoding: Optional[str] = "utf-8"
    delimiter: Optional[str] = ","
    sheet_name: Optional[Union[str, int]] = None
    header_row: Optional[int] = 0
    na_values: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.encoding = _optional_string(self.encoding, "read_options.encoding", 64)
        if self.delimiter is not None and (not isinstance(self.delimiter, str) or len(self.delimiter) != 1):
            raise ValueError("read_options.delimiter must be null or exactly one character.")
        if self.sheet_name is not None:
            if isinstance(self.sheet_name, bool) or not isinstance(self.sheet_name, (str, int)):
                raise TypeError("read_options.sheet_name must be a string, integer, or null.")
            if isinstance(self.sheet_name, str):
                self.sheet_name = _nonempty_string(self.sheet_name, "read_options.sheet_name", 128)
            elif self.sheet_name < 0:
                raise ValueError("read_options.sheet_name must be non-negative when it is an integer.")
        if self.header_row is not None:
            if isinstance(self.header_row, bool) or not isinstance(self.header_row, int) or self.header_row < 0:
                raise ValueError("read_options.header_row must be a non-negative integer or null.")
        self.na_values = _string_tuple(self.na_values, "read_options.na_values", allow_empty=True, max_items=100)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "encoding": self.encoding,
            "delimiter": self.delimiter,
            "sheet_name": self.sheet_name,
            "header_row": self.header_row,
            "na_values": list(self.na_values),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetReadOptions":
        values = _prepare_mapping(data, {"encoding", "delimiter", "sheet_name", "header_row", "na_values"}, set(), "DatasetReadOptions")
        return cls(**values)


@dataclass
class DatasetRef:
    kind: DatasetKind
    path: str
    format: DatasetFormat
    id_column: Optional[str]
    snapshot_policy: SnapshotPolicy
    read_options: DatasetReadOptions = field(default_factory=DatasetReadOptions)
    expected_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        self.kind = _as_enum(DatasetKind, self.kind, "dataset.kind")
        self.path = _nonempty_string(self.path, "dataset.path", 1024)
        self.format = _as_enum(DatasetFormat, self.format, "dataset.format")
        self.id_column = _optional_string(self.id_column, "dataset.id_column", 256)
        self.snapshot_policy = _as_enum(SnapshotPolicy, self.snapshot_policy, "dataset.snapshot_policy")
        if isinstance(self.read_options, Mapping):
            self.read_options = DatasetReadOptions.from_dict(self.read_options)
        if not isinstance(self.read_options, DatasetReadOptions):
            raise TypeError("dataset.read_options must be DatasetReadOptions or a compatible mapping.")
        self.expected_sha256 = _sha256(self.expected_sha256, "dataset.expected_sha256", nullable=True)

        suffix = PurePosixPath(self.path.replace("\\", "/")).suffix.lower()
        expected_suffix = f".{self.format.value}"
        if suffix != expected_suffix:
            raise ValueError(f"dataset.path extension {suffix!r} does not match dataset.format {self.format.value!r}.")
        if self.format is DatasetFormat.CSV and self.read_options.sheet_name is not None:
            raise ValueError("dataset.read_options.sheet_name must be null for CSV datasets.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "path": self.path,
            "format": self.format.value,
            "id_column": self.id_column,
            "read_options": self.read_options.to_dict(),
            "expected_sha256": self.expected_sha256,
            "snapshot_policy": self.snapshot_policy.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetRef":
        values = _prepare_mapping(
            data,
            {"kind", "path", "format", "id_column", "read_options", "expected_sha256", "snapshot_policy"},
            {"kind", "path", "format", "id_column", "snapshot_policy"},
            "DatasetRef",
        )
        if "read_options" in values:
            values["read_options"] = DatasetReadOptions.from_dict(values["read_options"])
        return cls(**values)


@dataclass
class PreprocessingSpec:
    missing_values: MissingValueStrategy = MissingValueStrategy.MEDIAN
    scaling: ScalingStrategy = ScalingStrategy.STANDARD
    class_balance: ClassBalanceStrategy = ClassBalanceStrategy.NONE

    def __post_init__(self) -> None:
        self.missing_values = _as_enum(MissingValueStrategy, self.missing_values, "preprocessing.missing_values")
        self.scaling = _as_enum(ScalingStrategy, self.scaling, "preprocessing.scaling")
        self.class_balance = _as_enum(ClassBalanceStrategy, self.class_balance, "preprocessing.class_balance")

    def to_dict(self) -> Dict[str, str]:
        return {
            "missing_values": self.missing_values.value,
            "scaling": self.scaling.value,
            "class_balance": self.class_balance.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PreprocessingSpec":
        values = _prepare_mapping(data, {"missing_values", "scaling", "class_balance"}, set(), "PreprocessingSpec")
        return cls(**values)


@dataclass
class SplitSpec:
    strategy: SplitStrategy = SplitStrategy.STRATIFIED_RANDOM
    test_size: float = 0.2
    group_column: Optional[str] = None
    random_seed: int = 42

    def __post_init__(self) -> None:
        self.strategy = _as_enum(SplitStrategy, self.strategy, "split.strategy")
        if isinstance(self.test_size, bool) or not isinstance(self.test_size, (int, float)) or not 0 < float(self.test_size) < 1:
            raise ValueError("split.test_size must be a number strictly between 0 and 1.")
        self.test_size = float(self.test_size)
        self.group_column = _optional_string(self.group_column, "split.group_column", 256)
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int) or not 0 <= self.random_seed <= 4294967295:
            raise ValueError("split.random_seed must be an integer between 0 and 4294967295.")
        if self.strategy is SplitStrategy.GROUP and self.group_column is None:
            raise ValueError("split.group_column is required when split.strategy is 'group'.")
        if self.strategy is SplitStrategy.STRATIFIED_RANDOM and self.group_column is not None:
            raise ValueError("split.group_column must be null when split.strategy is 'stratified_random'.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "test_size": self.test_size,
            "group_column": self.group_column,
            "random_seed": self.random_seed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SplitSpec":
        values = _prepare_mapping(data, {"strategy", "test_size", "group_column", "random_seed"}, set(), "SplitSpec")
        return cls(**values)


@dataclass
class ModelSpec:
    name: str
    mode: ModelMode = ModelMode.MANUAL
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = _nonempty_string(self.name, "model.name", 64)
        if not _MODEL_NAME_PATTERN.fullmatch(self.name):
            raise ValueError("model.name must use lowercase letters, digits, and underscores and start with a letter.")
        self.mode = _as_enum(ModelMode, self.mode, "model.mode")
        self.parameters = _json_mapping(self.parameters, "model.parameters")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode.value,
            "parameters": copy.deepcopy(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelSpec":
        values = _prepare_mapping(data, {"name", "mode", "parameters"}, {"name"}, "ModelSpec")
        return cls(**values)


@dataclass
class EvaluationSpec:
    primary_metric: PrimaryMetric = PrimaryMetric.MACRO_F1
    positive_label: Optional[LabelValue] = None
    cross_validation_folds: int = 5

    def __post_init__(self) -> None:
        self.primary_metric = _as_enum(PrimaryMetric, self.primary_metric, "evaluation.primary_metric")
        if self.positive_label is not None:
            if isinstance(self.positive_label, bool) or not isinstance(self.positive_label, (str, int, float)):
                raise TypeError("evaluation.positive_label must be a string, number, or null.")
            if isinstance(self.positive_label, str):
                self.positive_label = _nonempty_string(self.positive_label, "evaluation.positive_label", 256)
            if isinstance(self.positive_label, float) and not math.isfinite(self.positive_label):
                raise ValueError("evaluation.positive_label must be finite.")
        if isinstance(self.cross_validation_folds, bool) or not isinstance(self.cross_validation_folds, int) or not 2 <= self.cross_validation_folds <= 100:
            raise ValueError("evaluation.cross_validation_folds must be an integer between 2 and 100.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_metric": self.primary_metric.value,
            "positive_label": self.positive_label,
            "cross_validation_folds": self.cross_validation_folds,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationSpec":
        values = _prepare_mapping(data, {"primary_metric", "positive_label", "cross_validation_folds"}, set(), "EvaluationSpec")
        return cls(**values)


@dataclass
class ClassificationExperimentSpec:
    schema_version: str
    dataset: DatasetRef
    target_column: str
    preprocessing: PreprocessingSpec
    split: SplitSpec
    model: ModelSpec
    evaluation: EvaluationSpec
    client_request_id: Optional[str] = None
    feature_columns: Optional[Tuple[str, ...]] = None
    group_column: Optional[str] = None

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_VERSION:
            raise ValueError(f"schema_version must be {CONTRACT_VERSION!r}.")
        if isinstance(self.dataset, Mapping):
            self.dataset = DatasetRef.from_dict(self.dataset)
        if not isinstance(self.dataset, DatasetRef):
            raise TypeError("dataset must be DatasetRef or a compatible mapping.")
        self.target_column = _nonempty_string(self.target_column, "target_column", 256)
        if isinstance(self.preprocessing, Mapping):
            self.preprocessing = PreprocessingSpec.from_dict(self.preprocessing)
        if not isinstance(self.preprocessing, PreprocessingSpec):
            raise TypeError("preprocessing must be PreprocessingSpec or a compatible mapping.")
        if isinstance(self.split, Mapping):
            self.split = SplitSpec.from_dict(self.split)
        if not isinstance(self.split, SplitSpec):
            raise TypeError("split must be SplitSpec or a compatible mapping.")
        if isinstance(self.model, Mapping):
            self.model = ModelSpec.from_dict(self.model)
        if not isinstance(self.model, ModelSpec):
            raise TypeError("model must be ModelSpec or a compatible mapping.")
        if isinstance(self.evaluation, Mapping):
            self.evaluation = EvaluationSpec.from_dict(self.evaluation)
        if not isinstance(self.evaluation, EvaluationSpec):
            raise TypeError("evaluation must be EvaluationSpec or a compatible mapping.")

        self.client_request_id = _optional_string(self.client_request_id, "client_request_id", 128)
        if self.feature_columns is not None:
            self.feature_columns = _string_tuple(self.feature_columns, "feature_columns")
            if self.target_column in self.feature_columns:
                raise ValueError("feature_columns must not include target_column.")
        self.group_column = _optional_string(self.group_column, "group_column", 256)
        if self.group_column == self.target_column:
            raise ValueError("group_column must not equal target_column.")
        if self.feature_columns is not None and self.group_column in self.feature_columns:
            raise ValueError("feature_columns must not include group_column.")
        if self.split.strategy is SplitStrategy.GROUP and self.group_column != self.split.group_column:
            raise ValueError("group_column must match split.group_column for group splitting.")
        if self.split.strategy is SplitStrategy.STRATIFIED_RANDOM and self.group_column is not None:
            raise ValueError("group_column must be null for stratified_random splitting.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "client_request_id": self.client_request_id,
            "dataset": self.dataset.to_dict(),
            "target_column": self.target_column,
            "feature_columns": list(self.feature_columns) if self.feature_columns is not None else None,
            "group_column": self.group_column,
            "preprocessing": self.preprocessing.to_dict(),
            "split": self.split.to_dict(),
            "model": self.model.to_dict(),
            "evaluation": self.evaluation.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClassificationExperimentSpec":
        values = _prepare_mapping(
            data,
            {
                "schema_version",
                "client_request_id",
                "dataset",
                "target_column",
                "feature_columns",
                "group_column",
                "preprocessing",
                "split",
                "model",
                "evaluation",
            },
            {"schema_version", "dataset", "target_column", "preprocessing", "split", "model", "evaluation"},
            "ClassificationExperimentSpec",
        )
        values["dataset"] = DatasetRef.from_dict(values["dataset"])
        values["preprocessing"] = PreprocessingSpec.from_dict(values["preprocessing"])
        values["split"] = SplitSpec.from_dict(values["split"])
        values["model"] = ModelSpec.from_dict(values["model"])
        values["evaluation"] = EvaluationSpec.from_dict(values["evaluation"])
        return cls(**values)


@dataclass
class ArtifactRef:
    artifact_id: str
    role: str
    media_type: str
    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        self.artifact_id = _nonempty_string(self.artifact_id, "artifact.artifact_id", 128)
        if not _IDENTIFIER_PATTERN.fullmatch(self.artifact_id):
            raise ValueError("artifact.artifact_id contains unsupported characters.")
        self.role = _nonempty_string(self.role, "artifact.role", 64)
        if not _MODEL_NAME_PATTERN.fullmatch(self.role):
            raise ValueError("artifact.role must use lowercase letters, digits, and underscores.")
        self.media_type = _nonempty_string(self.media_type, "artifact.media_type", 128)
        if not re.fullmatch(r"[^/\s]+/[^/\s]+", self.media_type):
            raise ValueError("artifact.media_type must be an IANA-style media type.")
        self.relative_path = _relative_path(self.relative_path, "artifact.relative_path")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise ValueError("artifact.size_bytes must be a non-negative integer.")
        self.sha256 = _sha256(self.sha256, "artifact.sha256") or ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "role": self.role,
            "media_type": self.media_type,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactRef":
        fields = {"artifact_id", "role", "media_type", "relative_path", "size_bytes", "sha256"}
        values = _prepare_mapping(data, fields, fields, "ArtifactRef")
        return cls(**values)


@dataclass
class ExperimentResult:
    schema_version: str
    run_id: str
    request_hash: str
    status: RunStatus
    metrics: Dict[str, float]
    artifacts: Tuple[ArtifactRef, ...]
    warnings: Tuple[str, ...]
    manifest_path: str
    provenance_path: str

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_VERSION:
            raise ValueError(f"schema_version must be {CONTRACT_VERSION!r}.")
        self.run_id = _nonempty_string(self.run_id, "run_id", 128)
        if not _IDENTIFIER_PATTERN.fullmatch(self.run_id):
            raise ValueError("run_id contains unsupported characters.")
        self.request_hash = _sha256(self.request_hash, "request_hash") or ""
        self.status = _as_enum(RunStatus, self.status, "status")
        if not isinstance(self.metrics, Mapping) or len(self.metrics) > 100:
            raise TypeError("metrics must be an object with at most 100 entries.")
        normalized_metrics: Dict[str, float] = {}
        for name, value in self.metrics.items():
            metric_name = _nonempty_string(name, "metric name", 128)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"Metric {metric_name!r} must be a finite number.")
            normalized_metrics[metric_name] = float(value)
        self.metrics = normalized_metrics

        if not isinstance(self.artifacts, (list, tuple)):
            raise TypeError("artifacts must be a list or tuple.")
        normalized_artifacts: List[ArtifactRef] = []
        for artifact in self.artifacts:
            normalized_artifacts.append(ArtifactRef.from_dict(artifact) if isinstance(artifact, Mapping) else artifact)
        if not all(isinstance(artifact, ArtifactRef) for artifact in normalized_artifacts):
            raise TypeError("artifacts must contain ArtifactRef values.")
        artifact_ids = [artifact.artifact_id for artifact in normalized_artifacts]
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("artifacts must not contain duplicate artifact_id values.")
        self.artifacts = tuple(normalized_artifacts)
        self.warnings = _string_tuple(self.warnings, "warnings", allow_empty=True, max_items=100, item_max_length=1000)
        self.manifest_path = _relative_path(self.manifest_path, "manifest_path")
        self.provenance_path = _relative_path(self.provenance_path, "provenance_path")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "request_hash": self.request_hash,
            "status": self.status.value,
            "metrics": dict(self.metrics),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "warnings": list(self.warnings),
            "manifest_path": self.manifest_path,
            "provenance_path": self.provenance_path,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentResult":
        fields = {
            "schema_version",
            "run_id",
            "request_hash",
            "status",
            "metrics",
            "artifacts",
            "warnings",
            "manifest_path",
            "provenance_path",
        }
        values = _prepare_mapping(data, fields, fields, "ExperimentResult")
        values["artifacts"] = tuple(ArtifactRef.from_dict(item) for item in values["artifacts"])
        return cls(**values)


@dataclass
class ContractError:
    code: ErrorCode
    message: str
    stage: ErrorStage
    run_id: Optional[str]
    retryable: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.code = _as_enum(ErrorCode, self.code, "error.code")
        self.message = _nonempty_string(self.message, "error.message", 1000)
        self.stage = _as_enum(ErrorStage, self.stage, "error.stage")
        self.run_id = _optional_string(self.run_id, "error.run_id", 128)
        if self.run_id is not None and not _IDENTIFIER_PATTERN.fullmatch(self.run_id):
            raise ValueError("error.run_id contains unsupported characters.")
        if not isinstance(self.retryable, bool):
            raise TypeError("error.retryable must be a boolean.")
        self.details = _json_mapping(self.details, "error.details")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "stage": self.stage.value,
            "run_id": self.run_id,
            "retryable": self.retryable,
            "details": copy.deepcopy(self.details),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContractError":
        fields = {"code", "message", "stage", "run_id", "retryable", "details"}
        values = _prepare_mapping(data, fields, {"code", "message", "stage", "run_id", "retryable"}, "ContractError")
        return cls(**values)


@dataclass
class ErrorResponse:
    error: ContractError

    def __post_init__(self) -> None:
        if isinstance(self.error, Mapping):
            self.error = ContractError.from_dict(self.error)
        if not isinstance(self.error, ContractError):
            raise TypeError("error must be ContractError or a compatible mapping.")

    def to_dict(self) -> Dict[str, Any]:
        return {"error": self.error.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ErrorResponse":
        values = _prepare_mapping(data, {"error"}, {"error"}, "ErrorResponse")
        return cls(error=ContractError.from_dict(values["error"]))
