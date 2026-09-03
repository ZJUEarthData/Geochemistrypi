"""Bounded planning views over complete immutable validation receipts."""

import hashlib
import json
from typing import Annotated, Any, Generic, Literal, TypeVar, Union

from pydantic import BaseModel, Field, model_validator

from .schemas import (
    AffineTargetTransformation,
    AnalysisValidationDetailRequest,
    AnalysisValidationResponse,
    ArtifactRequirement,
    DisabledWorldMap,
    EncodeOriginalLabels,
    EngineeredFeature,
    FeatureSelection,
    ImputeMissingValues,
    KeepMissingValues,
    ModelParameterValue,
    RejectMissingValues,
    ScalingMethod,
    StrictModel,
    TimeSeriesArtifactRequirement,
)

_MAX_COMPACT_VALIDATION_JSON_BYTES = 64 * 1024
_COMPACT_VALIDATION_METADATA_RESERVE_BYTES = 8 * 1024
_DEFAULT_SEQUENCE_PREFIX = 16
_DIAGNOSTIC_SEQUENCE_PREFIX = 8
_MAPPING_PREFIX = 16
_MAX_COMPACT_TEXT_JSON_BYTES = 512

_ReceiptValue = TypeVar("_ReceiptValue")


def _json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_value(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _json_size_bytes(value: Any) -> int:
    return len(_canonical_json_bytes(value))


def _bounded_json_text(value: str, maximum_json_bytes: int) -> tuple[str, bool]:
    """Keep a readable prefix whose escaped JSON representation is bounded."""
    if _json_size_bytes(value) <= maximum_json_bytes:
        return value, False
    lower = 0
    upper = len(value)
    while lower < upper:
        middle = (lower + upper + 1) // 2
        if _json_size_bytes(value[:middle]) <= maximum_json_bytes:
            lower = middle
        else:
            upper = middle - 1
    return value[:lower], True


class CompactTextReceipt(StrictModel):
    """One bounded text prefix bound to the complete original string."""

    text: str = Field(max_length=_MAX_COMPACT_TEXT_JSON_BYTES)
    truncated: bool
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_complete_text_hash(self) -> "CompactTextReceipt":
        if not self.truncated and self.sha256 != _canonical_json_sha256(self.text):
            raise ValueError("a complete compact text must match its SHA-256")
        return self


class CompactSequenceReceipt(StrictModel, Generic[_ReceiptValue]):
    """Typed ordered prefix plus the identity of the complete source sequence."""

    prefix: tuple[_ReceiptValue, ...] = ()
    total_count: int = Field(ge=0)
    truncated: bool
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_prefix_count(self) -> "CompactSequenceReceipt[_ReceiptValue]":
        if len(self.prefix) > self.total_count:
            raise ValueError("a compact sequence prefix cannot exceed its complete count")
        nested_truncation = any(isinstance(item, BaseModel) and bool(getattr(item, "truncated", False)) for item in self.prefix)
        if self.truncated != (len(self.prefix) < self.total_count or nested_truncation):
            raise ValueError("compact sequence truncation metadata is inconsistent")
        literal_prefix = all(item is None or isinstance(item, (bool, int, float, str)) for item in self.prefix)
        if not self.truncated and literal_prefix and self.sha256 != _canonical_json_sha256(self.prefix):
            raise ValueError("a complete compact literal sequence must match its SHA-256")
        return self


class CompactMappingEntry(StrictModel, Generic[_ReceiptValue]):
    """One deterministic mapping entry with a bounded, hash-bound key."""

    key: CompactTextReceipt
    value: _ReceiptValue


class CompactMappingReceipt(StrictModel, Generic[_ReceiptValue]):
    """Typed sorted mapping prefix plus the identity of the complete mapping."""

    prefix: tuple[CompactMappingEntry[_ReceiptValue], ...] = ()
    total_count: int = Field(ge=0)
    truncated: bool
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_prefix_count(self) -> "CompactMappingReceipt[_ReceiptValue]":
        if len(self.prefix) > self.total_count:
            raise ValueError("a compact mapping prefix cannot exceed its complete count")
        nested_truncation = any(entry.key.truncated or (isinstance(entry.value, BaseModel) and bool(getattr(entry.value, "truncated", False))) for entry in self.prefix)
        if self.truncated != (len(self.prefix) < self.total_count or nested_truncation):
            raise ValueError("compact mapping truncation metadata is inconsistent")
        return self


class CompactValidationReadiness(StrictModel):
    """Every independent readiness dimension required before execution."""

    valid: Literal[True] = True
    execution_ready: bool
    comparison_ready: bool
    claim_ready: Literal[False] = False
    schema_status: Literal["valid"] = "valid"
    scientific_status: Literal["valid", "requirements_unmet"]
    adapter_status: Literal["available", "unavailable", "requirements_unmet"]
    artifact_status: Literal["planned", "requirements_unmet"]
    environment_status: Literal["READY", "MISMATCH", "UNSPECIFIED"]


class CompactValidationDatasetIdentity(StrictModel):
    """Source and prepared-view identities without paths or preparation records."""

    source: Literal["path", "builtin", "desktop"]
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prepared_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prepared_size_bytes: int | None = Field(None, ge=0)
    source_row_count: int | None = Field(None, ge=0)
    prepared_row_count: int | None = Field(None, ge=0)
    dropped_row_count: int | None = Field(None, ge=0)
    preparation_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    row_identity_scheme: str | None = None
    row_identity_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")


class CompactValidationEventDatasetIdentity(StrictModel):
    """Immutable identity for an optional Time Series event source, without its path."""

    source: Literal["path"] = "path"
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class CompactValidationColumnRoles(StrictModel):
    """Observed columns and every model-facing scientific role."""

    columns: CompactSequenceReceipt[CompactTextReceipt]
    identifier_column: str | None = None
    feature_columns: CompactSequenceReceipt[CompactTextReceipt]
    selected_columns: CompactSequenceReceipt[CompactTextReceipt]
    target_column: str | None = None
    target_columns: CompactSequenceReceipt[CompactTextReceipt]


class CompactArtifactRequirementSummary(StrictModel):
    """Exact native output identity, cardinality, and bounded content contracts."""

    requirement_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]+$", max_length=120)
    scientific_type: str = Field(min_length=1, max_length=120)
    output_role: str | None = Field(None, min_length=1, max_length=120)
    required: bool
    category: Literal["artifacts", "metrics", "parameters", "summary"] | None = None
    # These source values are already short, bounded schema literals.  A
    # sequence-level identity is sufficient; wrapping every MIME type or JSON
    # key in its own text/hash receipt made the compact projection larger than
    # the corresponding full contract.
    media_types: CompactSequenceReceipt[str]
    expected_relative_path: str | None = Field(None, min_length=1, max_length=512)
    path_pattern: str | None = Field(None, min_length=1, max_length=512)
    minimum_count: int = Field(ge=0)
    maximum_count: int | None = Field(None, ge=1)
    required_json_keys: CompactSequenceReceipt[str]


class CompactTypedSemanticLabel(StrictModel):
    """Type-preserving positive-label identity used by the scientific contract."""

    type: Literal["boolean", "integer", "number", "string"]
    value: bool | int | float | str

    @model_validator(mode="after")
    def validate_typed_value(self) -> "CompactTypedSemanticLabel":
        expected = "boolean" if isinstance(self.value, bool) else "integer" if isinstance(self.value, int) else "number" if isinstance(self.value, float) else "string"
        if self.type != expected:
            raise ValueError("semantic label type does not match its value")
        return self


class CompactDropMissingRows(StrictModel):
    """Drop-row decision with a bounded selected-column sequence."""

    method: Literal["drop_rows"] = "drop_rows"
    columns: CompactSequenceReceipt[CompactTextReceipt]


CompactMissingValueHandling = Annotated[
    Union[
        RejectMissingValues,
        KeepMissingValues,
        CompactDropMissingRows,
        ImputeMissingValues,
    ],
    Field(discriminator="method"),
]


class CompactMapLabels(StrictModel):
    strategy: Literal["map"] = "map"
    mapping: CompactMappingReceipt[CompactTextReceipt]


class CompactIntervalLabels(StrictModel):
    strategy: Literal["interval"] = "interval"
    cut_points: CompactSequenceReceipt[float]
    labels: CompactSequenceReceipt[CompactTextReceipt] | None = None


class CompactQuantileLabels(StrictModel):
    strategy: Literal["quantile"] = "quantile"
    number_of_classes: int = Field(ge=2, le=20)
    labels: CompactSequenceReceipt[CompactTextReceipt] | None = None


CompactLabelCustomization = Annotated[
    Union[
        EncodeOriginalLabels,
        CompactMapLabels,
        CompactIntervalLabels,
        CompactQuantileLabels,
    ],
    Field(discriminator="strategy"),
]


class CompactEnabledWorldMap(StrictModel):
    enabled: Literal[True] = True
    longitude_column: str = Field(min_length=1, max_length=128)
    latitude_column: str = Field(min_length=1, max_length=128)
    value_columns: CompactSequenceReceipt[CompactTextReceipt]


CompactWorldMapConfiguration = Annotated[
    Union[DisabledWorldMap, CompactEnabledWorldMap],
    Field(discriminator="enabled"),
]


class CompactValidationEvaluationDecisions(StrictModel):
    """Controller-readable evaluation decisions that materially change results."""

    requested_mode: str = Field(min_length=1, max_length=80)
    effective_mode: str = Field(min_length=1, max_length=80)
    requested_test_ratio: float | None = Field(None, gt=0, lt=1)
    effective_test_ratio: float | None = Field(None, gt=0, lt=1)
    requested_split_strategy: str | None = Field(None, min_length=1, max_length=80)
    effective_split_strategy: str | None = Field(None, min_length=1, max_length=80)
    requested_cross_validation_folds: int | None = Field(None, ge=2, le=100)
    effective_cross_validation_folds: int | None = Field(None, ge=2, le=100)
    requested_metrics: CompactSequenceReceipt[CompactTextReceipt]
    metric_artifact_bindings: CompactMappingReceipt[CompactTextReceipt]
    required_artifact_ids: CompactSequenceReceipt[CompactTextReceipt]
    class_order: CompactSequenceReceipt[CompactTextReceipt]
    requested_confusion_matrix_normalization: str | None = Field(None, min_length=1, max_length=40)
    effective_confusion_matrix_normalization: str | None = Field(None, min_length=1, max_length=40)
    requested_metric_average: str | None = Field(None, min_length=1, max_length=40)
    effective_metric_average: str | None = Field(None, min_length=1, max_length=40)
    requested_positive_label: CompactTypedSemanticLabel | None = None
    effective_positive_label: CompactTypedSemanticLabel | None = None


class CompactValidationPreprocessingDecisions(StrictModel):
    """Exact active preprocessing values, excluding only provenance duplication."""

    missing_values: CompactMissingValueHandling | None = None
    scaling: ScalingMethod | None = None
    feature_selection: FeatureSelection | None = None
    engineered_features: CompactSequenceReceipt[EngineeredFeature]
    label_customization: CompactLabelCustomization | None = None
    world_map: CompactWorldMapConfiguration | None = None
    target_transformations: CompactMappingReceipt[AffineTargetTransformation]
    sample_balancing: Literal["none"] | None = None
    metadata_columns: CompactSequenceReceipt[CompactTextReceipt]
    feature_engineering: Literal["none"] | None = None


class CompactValidationApplicationDecisions(StrictModel):
    """Scientific role and column bindings for the secondary dataset."""

    enabled: bool | None
    role: Literal["inference", "external_evaluation", "artifact_overlay", "none", "not_reported"]
    training_identifier_column: str | None = Field(None, min_length=1, max_length=128)
    secondary_identifier_column: str | None = Field(None, min_length=1, max_length=128)
    target_columns: CompactSequenceReceipt[CompactTextReceipt]
    label_used_as_feature: bool | None


class CompactDecompositionOverlayContract(StrictModel):
    contract_type: Literal["decomposition_embedding_label_overlay"]
    coordinate_sheet: str = Field(min_length=1, max_length=128)
    label_sheet: str = Field(min_length=1, max_length=128)
    coordinate_identifier_column: str = Field(min_length=1, max_length=128)
    label_identifier_column: str = Field(min_length=1, max_length=128)
    label_column: str = Field(min_length=1, max_length=128)
    positive_label_values: CompactSequenceReceipt[CompactTextReceipt]
    join_policy: Literal["exact_identifier_set_one_to_one"]


class _CompactTimeSeriesContractBase(StrictModel):
    identifier_column: str | None = Field(None, min_length=1, max_length=128)
    selected_columns: CompactSequenceReceipt[CompactTextReceipt]
    sheet: str = Field(min_length=1, max_length=128)


class CompactSubaerialProportionContract(_CompactTimeSeriesContractBase):
    contract_type: Literal["time_series_subaerial_proportion"]
    age_column: str = Field(min_length=1, max_length=128)
    maximum_age_column: str = Field(min_length=1, max_length=128)
    probability_column: str = Field(min_length=1, max_length=128)
    latitude_column: str = Field(min_length=1, max_length=128)
    longitude_column: str = Field(min_length=1, max_length=128)


class CompactContinuousTimeSeriesContract(_CompactTimeSeriesContractBase):
    contract_type: Literal["time_series_continuous"]
    age_column: str = Field(min_length=1, max_length=128)
    minimum_age_column: str = Field(min_length=1, max_length=128)
    maximum_age_column: str = Field(min_length=1, max_length=128)
    value_column: str = Field(min_length=1, max_length=128)
    latitude_column: str = Field(min_length=1, max_length=128)
    longitude_column: str = Field(min_length=1, max_length=128)
    filter_column: str | None = Field(None, min_length=1, max_length=128)


class CompactElementMeanTimeSeriesContract(_CompactTimeSeriesContractBase):
    contract_type: Literal["time_series_element_mean"]
    age_column: str = Field(min_length=1, max_length=128)
    element_columns: CompactSequenceReceipt[CompactTextReceipt]
    filter_column: str | None = Field(None, min_length=1, max_length=128)


class CompactReferenceAnomalySeriesContract(_CompactTimeSeriesContractBase):
    contract_type: Literal["time_series_reference_anomaly_series"]
    time_column: str = Field(min_length=1, max_length=128)
    signal_columns: CompactSequenceReceipt[CompactTextReceipt]
    reference_label_column: str = Field(min_length=1, max_length=128)
    reference_positive_values: CompactSequenceReceipt[CompactTextReceipt]
    comparison_label_column: str | None = Field(None, min_length=1, max_length=128)
    comparison_positive_values: CompactSequenceReceipt[CompactTextReceipt]
    event_sheet: str | None = Field(None, min_length=1, max_length=128)
    event_time_column: str | None = Field(None, min_length=1, max_length=128)
    event_identifier_column: str | None = Field(None, min_length=1, max_length=128)
    event_filter_column: str | None = Field(None, min_length=1, max_length=128)
    event_filter_values: CompactSequenceReceipt[CompactTextReceipt]


CompactWorkflowSpecificContract = Annotated[
    Union[
        CompactDecompositionOverlayContract,
        CompactSubaerialProportionContract,
        CompactContinuousTimeSeriesContract,
        CompactElementMeanTimeSeriesContract,
        CompactReferenceAnomalySeriesContract,
    ],
    Field(discriminator="contract_type"),
]


class CompactValidationBindingDecisions(StrictModel):
    """Observed binding modes that prevent prompt-only parameters from masquerading as bound."""

    model: str = Field(min_length=1, max_length=80)
    preprocessing: str = Field(min_length=1, max_length=80)
    scientific_execution_contract_bound: bool | None
    workflow_specific_contract: CompactWorkflowSpecificContract | None = None


class CompactValidationExecutionDecisions(StrictModel):
    """Four bounded decision groups needed to audit what validation will execute."""

    evaluation: CompactValidationEvaluationDecisions
    preprocessing: CompactValidationPreprocessingDecisions
    application: CompactValidationApplicationDecisions
    bindings: CompactValidationBindingDecisions


class CompactValidationEnvironmentIdentity(StrictModel):
    """Environment decision identities without the full installed-package record."""

    status: Literal["READY", "MISMATCH", "UNSPECIFIED"]
    observed_identity_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    requested_identity_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    profile_id: str | None = None
    profile_identity_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")


class CompactValidationExperimentIdentity(StrictModel):
    """MLflow destination identity required to decide whether execution may start."""

    mode: Literal["new", "existing", "not_applicable"]
    name: str
    existing_experiment_id: str | None = None


class CompactAnalysisValidationResponse(StrictModel):
    """Token-bounded validation receipt preserving every start/stop decision field."""

    response_detail: Literal["compact"] = "compact"
    validation_id: str = Field(pattern=r"^val-[0-9a-f]{32}$")
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    contains_truncated_content: bool
    truncated_sections: tuple[str, ...]
    start_relevant_content_complete: bool
    full_detail_request: AnalysisValidationDetailRequest
    canonical_contract_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiled_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_expires_at: str
    readiness: CompactValidationReadiness
    blocking_issues: CompactSequenceReceipt[CompactTextReceipt]
    warnings: CompactSequenceReceipt[CompactTextReceipt]
    task: Literal[
        "classification",
        "regression",
        "clustering",
        "decomposition",
        "anomaly_detection",
        "time_series",
    ]
    workflow_family: Literal[
        "time_series",
        "supervised_learning",
        "dimension_reduction",
        "clustering",
        "anomaly_detection",
        "artifact_composition",
    ]
    workflow_mode: str = Field(min_length=1, max_length=80)
    method: str = Field(min_length=1, max_length=120)
    scientific_contract_id: str = Field(min_length=1, max_length=255)
    adapter_id: str | None = Field(None, max_length=160)
    adapter_version: str | None = Field(None, max_length=40)
    adapter_identity: str | None = Field(None, max_length=220)
    models: tuple[str, ...]
    estimated_model_count: int = Field(ge=1)
    tuning: Literal["manual", "automl", "not_applicable"]
    training: CompactValidationDatasetIdentity
    application: CompactValidationDatasetIdentity | None = None
    event: CompactValidationEventDatasetIdentity | None = None
    column_roles: CompactValidationColumnRoles
    requested_seeds: dict[str, int] = Field(default_factory=dict, max_length=8)
    effective_seeds: dict[str, int] = Field(default_factory=dict, max_length=8)
    execution_decisions: CompactValidationExecutionDecisions
    resolved_model_parameters: dict[str, ModelParameterValue] = Field(default_factory=dict, max_length=64)
    artifact_requirement_count: int = Field(ge=0)
    artifact_requirements: CompactSequenceReceipt[CompactArtifactRequirementSummary]
    environment: CompactValidationEnvironmentIdentity
    experiment: CompactValidationExperimentIdentity
    analysis_process_started: Literal[False] = False

    @model_validator(mode="after")
    def validate_compact_budget_and_counts(self) -> "CompactAnalysisValidationResponse":
        if self.artifact_requirement_count != self.artifact_requirements.total_count:
            raise ValueError("artifact_requirement_count must equal the complete requirement count")
        if self.full_detail_request.validation_id != self.validation_id or self.full_detail_request.request_hash != self.request_hash:
            raise ValueError("full_detail_request must identify this exact validation receipt")
        projected = self.model_dump(mode="json")
        projected.pop("contains_truncated_content")
        projected.pop("truncated_sections")
        projected.pop("start_relevant_content_complete")
        if self.contains_truncated_content != _contains_truncation(projected):
            raise ValueError("contains_truncated_content must describe the compact projection")
        truncated_sections = tuple(_truncated_sections(projected))
        if self.truncated_sections != truncated_sections:
            raise ValueError("truncated_sections must identify every truncated compact receipt")
        expected_start_complete = all(section == "column_roles.columns" for section in truncated_sections)
        if self.start_relevant_content_complete != expected_start_complete:
            raise ValueError("start_relevant_content_complete must ignore only the supplemental " "unselected observed-column inventory")
        if _json_size_bytes(self.model_dump(mode="json")) > _MAX_COMPACT_VALIDATION_JSON_BYTES:
            raise ValueError("compact validation exceeds the 64 KiB structured JSON budget")
        return self


class FullAnalysisValidationDetailResponse(StrictModel):
    """Complete decision sequences recovered from one immutable validation record."""

    response_detail: Literal["full"] = "full"
    validation_id: str = Field(pattern=r"^val-[0-9a-f]{32}$")
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_contract_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiled_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_expires_at: str
    task: Literal[
        "classification",
        "regression",
        "clustering",
        "decomposition",
        "anomaly_detection",
        "time_series",
    ]
    execution_ready: bool
    blocking_issues: tuple[str, ...]
    blocking_issues_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    warnings: tuple[str, ...]
    warnings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_requirement_count: int = Field(ge=0)
    artifact_requirements: tuple[
        ArtifactRequirement | TimeSeriesArtifactRequirement,
        ...,
    ]
    artifact_requirements_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    complete_validation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_process_started: Literal[False] = False

    @model_validator(mode="after")
    def validate_complete_sequences(self) -> "FullAnalysisValidationDetailResponse":
        if self.artifact_requirement_count != len(self.artifact_requirements):
            raise ValueError("artifact_requirement_count must equal the complete requirement count")
        if self.blocking_issues_sha256 != _canonical_json_sha256(self.blocking_issues):
            raise ValueError("blocking_issues_sha256 must identify the complete sequence")
        if self.warnings_sha256 != _canonical_json_sha256(self.warnings):
            raise ValueError("warnings_sha256 must identify the complete sequence")
        if self.artifact_requirements_sha256 != _canonical_json_sha256(self.artifact_requirements):
            raise ValueError("artifact_requirements_sha256 must identify the complete sequence")
        return self


def full_analysis_validation_detail(
    response: AnalysisValidationResponse,
) -> FullAnalysisValidationDetailResponse:
    """Project every potentially truncated validation decision sequence."""
    return FullAnalysisValidationDetailResponse(
        validation_id=response.validation_id,
        request_hash=response.request_hash,
        canonical_contract_hash=response.canonical_contract_hash,
        compiled_plan_hash=response.compiled_plan_hash,
        validation_expires_at=response.validation_expires_at,
        task=response.task,
        execution_ready=response.execution_ready,
        blocking_issues=response.blocking_issues,
        blocking_issues_sha256=_canonical_json_sha256(response.blocking_issues),
        warnings=response.warnings,
        warnings_sha256=_canonical_json_sha256(response.warnings),
        artifact_requirement_count=len(response.artifact_requirements),
        artifact_requirements=response.artifact_requirements,
        artifact_requirements_sha256=_canonical_json_sha256(response.artifact_requirements),
        complete_validation_sha256=_canonical_json_sha256(response),
    )


def _compact_text_receipt(value: str) -> CompactTextReceipt:
    bounded, truncated = _bounded_json_text(str(value), _MAX_COMPACT_TEXT_JSON_BYTES)
    return CompactTextReceipt(
        text=bounded,
        truncated=truncated,
        sha256=_canonical_json_sha256(str(value)),
    )


def _contains_truncation(value: Any) -> bool:
    value = _json_value(value)
    if isinstance(value, dict):
        if value.get("truncated") is True:
            return True
        return any(_contains_truncation(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_truncation(item) for item in value)
    return False


def _truncated_sections(
    value: Any,
    path: tuple[str | int, ...] = (),
) -> list[str]:
    """Return deterministic paths for every outermost truncated receipt."""

    value = _json_value(value)
    if isinstance(value, dict):
        if value.get("truncated") is True:
            rendered = ""
            for part in path:
                rendered += f"[{part}]" if isinstance(part, int) else ("." if rendered else "") + part
            return [rendered]
        paths: list[str] = []
        for key, item in value.items():
            paths.extend(_truncated_sections(item, (*path, key)))
        return paths
    if isinstance(value, list):
        paths = []
        for index, item in enumerate(value):
            paths.extend(_truncated_sections(item, (*path, index)))
        return paths
    return []


def _compact_sequence(
    values: Any,
    *,
    projector: Any = None,
    maximum_prefix: int = _DEFAULT_SEQUENCE_PREFIX,
    identity_values: Any = None,
) -> CompactSequenceReceipt[Any]:
    source = tuple(values or ())
    project = projector or (lambda item: item)
    prefix = tuple(project(item) for item in source[:maximum_prefix])
    return CompactSequenceReceipt(
        prefix=prefix,
        total_count=len(source),
        truncated=len(prefix) < len(source) or any(_contains_truncation(item) for item in prefix),
        sha256=_canonical_json_sha256(list(source) if identity_values is None else list(identity_values)),
    )


def _compact_text_sequence(
    values: Any,
    *,
    maximum_prefix: int = _DEFAULT_SEQUENCE_PREFIX,
) -> CompactSequenceReceipt[CompactTextReceipt]:
    return _compact_sequence(
        values,
        projector=_compact_text_receipt,
        maximum_prefix=maximum_prefix,
    )


def _compact_mapping(
    values: Any,
    *,
    value_projector: Any = None,
    maximum_prefix: int = _MAPPING_PREFIX,
) -> CompactMappingReceipt[Any]:
    source = dict(values or {})
    project = value_projector or (lambda item: item)
    ordered_keys = sorted(source, key=lambda item: _canonical_json_bytes(str(item)))
    prefix = tuple(
        CompactMappingEntry(
            key=_compact_text_receipt(str(key)),
            value=project(source[key]),
        )
        for key in ordered_keys[:maximum_prefix]
    )
    return CompactMappingReceipt(
        prefix=prefix,
        total_count=len(source),
        truncated=(len(prefix) < len(source) or any(_contains_truncation(entry) for entry in prefix)),
        sha256=_canonical_json_sha256(source),
    )


def _compact_text_mapping(values: Any) -> CompactMappingReceipt[CompactTextReceipt]:
    return _compact_mapping(values, value_projector=_compact_text_receipt)


def _bounded_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _preparation_counts(
    preparation: dict[str, Any] | None,
    prepared_row_count: int | None,
) -> tuple[int | None, int | None, int | None]:
    table = preparation.get("table", {}) if isinstance(preparation, dict) else {}
    if not isinstance(table, dict):
        table = {}
    retained = _bounded_int(table.get("source_row_count"))
    prepared = prepared_row_count if prepared_row_count is not None else retained
    source = _bounded_int(table.get("input_row_count"))
    if source is None:
        source = retained if retained is not None else prepared
    dropped = _bounded_int(table.get("filtered_row_count"))
    if dropped is None and source is not None and prepared is not None and source >= prepared:
        dropped = source - prepared
    return source, prepared, dropped


def _preparation_hash(preparation: dict[str, Any] | None) -> str | None:
    value = preparation.get("contract_hash") if isinstance(preparation, dict) else None
    return value if isinstance(value, str) and len(value) == 64 else None


def _environment_profile_value(response: AnalysisValidationResponse, key: str) -> Any:
    requested = response.environment_profile.get("requested", {})
    return requested.get(key) if isinstance(requested, dict) else None


def _execution_decisions_value(response: AnalysisValidationResponse) -> dict[str, Any]:
    decisions = response.execution_decisions
    if (
        isinstance(decisions, dict)
        and isinstance(decisions.get("evaluation"), dict)
        and "effective_mode" in decisions["evaluation"]
        and isinstance(decisions.get("preprocessing"), dict)
        and "missing_values" in decisions["preprocessing"]
        and isinstance(decisions.get("application"), dict)
        and "secondary_identifier_column" in decisions["application"]
        and isinstance(decisions.get("bindings"), dict)
        and "workflow_specific_contract" in decisions["bindings"]
    ):
        return response.execution_decisions
    return {
        "evaluation": {
            "requested_mode": "not_reported",
            "effective_mode": "not_reported",
            "requested_test_ratio": None,
            "effective_test_ratio": None,
            "requested_split_strategy": None,
            "effective_split_strategy": None,
            "requested_cross_validation_folds": None,
            "effective_cross_validation_folds": None,
            "requested_metrics": (),
            "metric_artifact_bindings": {},
            "required_artifact_ids": (),
            "class_order": (),
            "requested_confusion_matrix_normalization": None,
            "effective_confusion_matrix_normalization": None,
            "requested_metric_average": None,
            "effective_metric_average": None,
            "requested_positive_label": None,
            "effective_positive_label": None,
        },
        "preprocessing": {
            "missing_values": None,
            "scaling": None,
            "feature_selection": None,
            "engineered_features": (),
            "label_customization": None,
            "world_map": None,
            "target_transformations": {},
            "sample_balancing": None,
            "metadata_columns": (),
            "feature_engineering": None,
        },
        "application": {
            "enabled": None,
            "role": "not_reported",
            "training_identifier_column": None,
            "secondary_identifier_column": None,
            "target_columns": (),
            "label_used_as_feature": None,
        },
        "bindings": {
            "model": "not_reported",
            "preprocessing": "not_reported",
            "scientific_execution_contract_bound": None,
            "workflow_specific_contract": None,
        },
    }


def _model_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return dict(value) if isinstance(value, dict) else {}


def _compact_missing_values(value: Any) -> CompactMissingValueHandling | None:
    if value is None:
        return None
    data = _model_dict(value)
    method = data.get("method")
    if method == "drop_rows":
        return CompactDropMissingRows(
            columns=_compact_text_sequence(data.get("columns", ())),
        )
    if method == "error":
        return RejectMissingValues.model_validate(data)
    if method == "keep":
        return KeepMissingValues.model_validate(data)
    if method == "impute":
        return ImputeMissingValues.model_validate(data)
    raise ValueError("unknown compact missing-value decision")


def _compact_label_customization(value: Any) -> CompactLabelCustomization | None:
    if value is None:
        return None
    data = _model_dict(value)
    strategy = data.get("strategy")
    if strategy == "encode_original":
        return EncodeOriginalLabels.model_validate(data)
    if strategy == "map":
        return CompactMapLabels(mapping=_compact_text_mapping(data.get("mapping", {})))
    if strategy == "interval":
        labels = data.get("labels")
        return CompactIntervalLabels(
            cut_points=_compact_sequence(data.get("cut_points", ())),
            labels=_compact_text_sequence(labels) if labels is not None else None,
        )
    if strategy == "quantile":
        labels = data.get("labels")
        return CompactQuantileLabels(
            number_of_classes=data["number_of_classes"],
            labels=_compact_text_sequence(labels) if labels is not None else None,
        )
    raise ValueError("unknown compact label-customization decision")


def _compact_world_map(value: Any) -> CompactWorldMapConfiguration | None:
    if value is None:
        return None
    data = _model_dict(value)
    if data.get("enabled") is False:
        return DisabledWorldMap.model_validate(data)
    if data.get("enabled") is True:
        return CompactEnabledWorldMap(
            longitude_column=data["longitude_column"],
            latitude_column=data["latitude_column"],
            value_columns=_compact_text_sequence(data.get("value_columns", ())),
        )
    raise ValueError("unknown compact world-map decision")


def _compact_workflow_contract(value: Any) -> CompactWorkflowSpecificContract | None:
    if value is None:
        return None
    data = _model_dict(value)
    contract_type = data.get("contract_type")
    data["selected_columns"] = _compact_text_sequence(data.get("selected_columns", ()))
    if contract_type == "decomposition_embedding_label_overlay":
        data.pop("selected_columns", None)
        data["positive_label_values"] = _compact_text_sequence(data.get("positive_label_values", ()))
        return CompactDecompositionOverlayContract.model_validate(data)
    if contract_type == "time_series_subaerial_proportion":
        return CompactSubaerialProportionContract.model_validate(data)
    if contract_type == "time_series_continuous":
        return CompactContinuousTimeSeriesContract.model_validate(data)
    if contract_type == "time_series_element_mean":
        data["element_columns"] = _compact_text_sequence(data.get("element_columns", ()))
        return CompactElementMeanTimeSeriesContract.model_validate(data)
    if contract_type == "time_series_reference_anomaly_series":
        for field in (
            "signal_columns",
            "reference_positive_values",
            "comparison_positive_values",
            "event_filter_values",
        ):
            data[field] = _compact_text_sequence(data.get(field, ()))
        return CompactReferenceAnomalySeriesContract.model_validate(data)
    raise ValueError("unknown compact workflow-specific contract")


def _compact_execution_decisions(
    response: AnalysisValidationResponse,
) -> CompactValidationExecutionDecisions:
    source = _execution_decisions_value(response)
    evaluation = dict(source["evaluation"])
    evaluation["requested_metrics"] = _compact_text_sequence(evaluation.get("requested_metrics", ()))
    evaluation["metric_artifact_bindings"] = _compact_text_mapping(evaluation.get("metric_artifact_bindings", {}))
    evaluation["required_artifact_ids"] = _compact_text_sequence(evaluation.get("required_artifact_ids", ()))
    evaluation["class_order"] = _compact_text_sequence(evaluation.get("class_order", ()))

    preprocessing = dict(source["preprocessing"])
    preprocessing["missing_values"] = _compact_missing_values(preprocessing.get("missing_values"))
    preprocessing["engineered_features"] = _compact_sequence(
        preprocessing.get("engineered_features", ()),
        projector=EngineeredFeature.model_validate,
    )
    preprocessing["label_customization"] = _compact_label_customization(preprocessing.get("label_customization"))
    preprocessing["world_map"] = _compact_world_map(preprocessing.get("world_map"))
    preprocessing["target_transformations"] = _compact_mapping(
        preprocessing.get("target_transformations", {}),
        value_projector=AffineTargetTransformation.model_validate,
    )
    preprocessing["metadata_columns"] = _compact_text_sequence(preprocessing.get("metadata_columns", ()))

    application = dict(source["application"])
    application["target_columns"] = _compact_text_sequence(application.get("target_columns", ()))

    bindings = dict(source["bindings"])
    bindings["workflow_specific_contract"] = _compact_workflow_contract(bindings.get("workflow_specific_contract"))
    return CompactValidationExecutionDecisions(
        evaluation=CompactValidationEvaluationDecisions.model_validate(evaluation),
        preprocessing=CompactValidationPreprocessingDecisions.model_validate(preprocessing),
        application=CompactValidationApplicationDecisions.model_validate(application),
        bindings=CompactValidationBindingDecisions.model_validate(bindings),
    )


def _artifact_summary(
    requirement: ArtifactRequirement | TimeSeriesArtifactRequirement,
) -> CompactArtifactRequirementSummary:
    if isinstance(requirement, TimeSeriesArtifactRequirement):
        return CompactArtifactRequirementSummary(
            requirement_id=requirement.requirement_id,
            scientific_type=requirement.scientific_type,
            path_pattern=requirement.path_pattern,
            required=True,
            media_types=_compact_sequence(()),
            minimum_count=requirement.count,
            maximum_count=requirement.count,
            required_json_keys=_compact_sequence(requirement.required_json_keys),
        )
    return CompactArtifactRequirementSummary(
        requirement_id=requirement.requirement_id,
        scientific_type=requirement.scientific_type,
        output_role=requirement.output_role,
        required=requirement.required,
        category=requirement.category,
        media_types=_compact_sequence(requirement.media_types),
        expected_relative_path=requirement.expected_relative_path,
        path_pattern=requirement.path_pattern,
        minimum_count=requirement.minimum_count,
        maximum_count=requirement.maximum_count,
        required_json_keys=_compact_sequence(requirement.required_json_keys),
    )


def _receipt_prefix_paths(
    value: Any,
    path: tuple[str | int, ...] = (),
) -> list[tuple[str | int, ...]]:
    """Find every mutable receipt prefix in a JSON-compatible payload."""
    paths: list[tuple[str | int, ...]] = []
    if isinstance(value, dict):
        if (
            isinstance(value.get("prefix"), list)
            and isinstance(value.get("total_count"), int)
            and isinstance(value.get("truncated"), bool)
            and isinstance(value.get("sha256"), str)
            and value["prefix"]
        ):
            paths.append(path)
        for key, child in value.items():
            paths.extend(_receipt_prefix_paths(child, (*path, key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_receipt_prefix_paths(child, (*path, index)))
    return paths


def _value_at_path(value: Any, path: tuple[str | int, ...]) -> Any:
    current = value
    for part in path:
        current = current[part]
    return current


def _fit_compact_validation_budget(
    payload: dict[str, Any],
    maximum_bytes: int = _MAX_COMPACT_VALIDATION_JSON_BYTES,
) -> dict[str, Any]:
    """Deterministically shrink receipt prefixes until structured JSON is <=64 KiB."""
    while _json_size_bytes(payload) > maximum_bytes:
        candidates: list[tuple[int, int, str, tuple[str | int, ...]]] = []
        for path in _receipt_prefix_paths(payload):
            receipt = _value_at_path(payload, path)
            prefix = receipt["prefix"]
            # Retain one item in each non-empty receipt until every receipt has
            # reached that floor; only then may the final item be removed.
            preservation_phase = 0 if len(prefix) > 1 else 1
            item_size = _json_size_bytes(prefix[-1])
            candidates.append(
                (
                    preservation_phase,
                    -item_size,
                    json.dumps(path, ensure_ascii=True, separators=(",", ":")),
                    path,
                )
            )
        if not candidates:
            raise ValueError("compact validation cannot fit the 64 KiB budget after all " "bounded receipt prefixes were exhausted")
        _, _, _, selected_path = min(candidates)
        selected = _value_at_path(payload, selected_path)
        selected["prefix"].pop()
        selected["truncated"] = True
    return payload


def compact_analysis_validation(
    response: AnalysisValidationResponse,
) -> CompactAnalysisValidationResponse:
    """Project a full successful validation record without changing its immutable receipt."""
    training_source_rows, training_prepared_rows, training_dropped_rows = _preparation_counts(
        response.dataset_preparation,
        response.source_row_count,
    )
    training = CompactValidationDatasetIdentity(
        source=response.training_source,
        source_sha256=response.source_dataset_sha256 or response.training_sha256,
        prepared_sha256=response.training_sha256,
        prepared_size_bytes=response.training_size_bytes,
        source_row_count=training_source_rows,
        prepared_row_count=training_prepared_rows,
        dropped_row_count=training_dropped_rows,
        preparation_sha256=response.dataset_preparation_sha256 or _preparation_hash(response.dataset_preparation),
        row_identity_scheme=response.row_identity_scheme,
        row_identity_sha256=response.row_identity_sha256,
    )
    application = None
    if response.application_source is not None and response.application_sha256 is not None:
        application_source_rows, application_prepared_rows, application_dropped_rows = _preparation_counts(
            response.application_preparation,
            response.application_source_row_count,
        )
        application = CompactValidationDatasetIdentity(
            source=response.application_source,
            source_sha256=response.application_source_sha256 or response.application_sha256,
            prepared_sha256=response.application_sha256,
            source_row_count=application_source_rows,
            prepared_row_count=application_prepared_rows,
            dropped_row_count=application_dropped_rows,
            preparation_sha256=_preparation_hash(response.application_preparation),
            row_identity_sha256=response.application_row_identity_sha256,
        )
    event = None
    if response.event_source_sha256 is not None and response.event_size_bytes is not None:
        event = CompactValidationEventDatasetIdentity(
            source_sha256=response.event_source_sha256,
            size_bytes=response.event_size_bytes,
        )
    requirements = tuple(_artifact_summary(item) for item in response.artifact_requirements)
    requested_identity = _environment_profile_value(response, "expected_identity_sha256")
    profile_id = _environment_profile_value(response, "profile_id")
    compact_payload = dict(
        validation_id=response.validation_id,
        request_hash=response.request_hash,
        contains_truncated_content=False,
        full_detail_request=AnalysisValidationDetailRequest(
            validation_id=response.validation_id,
            request_hash=response.request_hash,
            detail="full",
        ),
        canonical_contract_hash=response.canonical_contract_hash,
        compiled_plan_hash=response.compiled_plan_hash,
        validation_expires_at=response.validation_expires_at,
        readiness=CompactValidationReadiness(
            valid=response.valid,
            execution_ready=response.execution_ready,
            comparison_ready=response.comparison_ready,
            claim_ready=response.claim_ready,
            schema_status=response.schema_status,
            scientific_status=response.scientific_status,
            adapter_status=response.adapter_status,
            artifact_status=response.artifact_status,
            environment_status=response.environment_status,
        ),
        blocking_issues=_compact_text_sequence(
            response.blocking_issues,
            maximum_prefix=_DIAGNOSTIC_SEQUENCE_PREFIX,
        ),
        warnings=_compact_text_sequence(
            response.warnings,
            maximum_prefix=_DIAGNOSTIC_SEQUENCE_PREFIX,
        ),
        task=response.task,
        workflow_family=response.workflow_family,
        workflow_mode=response.workflow_mode,
        method=response.method,
        scientific_contract_id=response.scientific_contract_id,
        adapter_id=response.adapter_id,
        adapter_version=response.adapter_version,
        adapter_identity=response.adapter_identity,
        models=response.models,
        estimated_model_count=response.estimated_model_count,
        tuning=response.tuning,
        training=training,
        application=application,
        event=event,
        column_roles=CompactValidationColumnRoles(
            columns=_compact_text_sequence(response.columns),
            identifier_column=response.identifier_column,
            feature_columns=_compact_text_sequence(response.feature_columns),
            selected_columns=_compact_text_sequence(response.selected_columns),
            target_column=response.target_column,
            target_columns=_compact_text_sequence(response.target_columns),
        ),
        requested_seeds=response.requested_seeds,
        effective_seeds=response.effective_seeds,
        execution_decisions=_compact_execution_decisions(response),
        resolved_model_parameters=response.resolved_model_parameters,
        artifact_requirement_count=len(requirements),
        artifact_requirements=_compact_sequence(
            requirements,
            identity_values=response.artifact_requirements,
        ),
        environment=CompactValidationEnvironmentIdentity(
            status=response.environment_status,
            observed_identity_sha256=response.environment_identity_sha256,
            requested_identity_sha256=(requested_identity if isinstance(requested_identity, str) and len(requested_identity) == 64 else None),
            profile_id=profile_id if isinstance(profile_id, str) else None,
            profile_identity_sha256=response.environment_profile_identity_sha256,
        ),
        experiment=CompactValidationExperimentIdentity(
            mode=response.experiment_mode,
            name=response.experiment_name,
            existing_experiment_id=response.existing_experiment_id,
        ),
        analysis_process_started=response.analysis_process_started,
    )
    bounded_payload = _fit_compact_validation_budget(
        _json_value(compact_payload),
        _MAX_COMPACT_VALIDATION_JSON_BYTES - _COMPACT_VALIDATION_METADATA_RESERVE_BYTES,
    )
    content = {
        key: value
        for key, value in bounded_payload.items()
        if key
        not in {
            "contains_truncated_content",
            "truncated_sections",
            "start_relevant_content_complete",
        }
    }
    truncated_sections = tuple(_truncated_sections(content))
    bounded_payload["contains_truncated_content"] = bool(truncated_sections)
    bounded_payload["truncated_sections"] = truncated_sections
    bounded_payload["start_relevant_content_complete"] = all(section == "column_roles.columns" for section in truncated_sections)
    return CompactAnalysisValidationResponse.model_validate(bounded_payload)
