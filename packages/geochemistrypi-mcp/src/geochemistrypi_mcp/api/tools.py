"""Strict low-level MCP tool definitions and dispatch."""

import hashlib
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable
from copy import deepcopy
from functools import reduce
from operator import or_
from pathlib import Path
from typing import Any

import anyio
from mcp import MCPError
from mcp.server import ServerRequestContext
from mcp.types import INVALID_PARAMS, CallToolRequestParams, CallToolResult, ListToolsResult, PaginatedRequestParams, TextContent, Tool
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from ..config.clients import SUPPORTED_CLIENTS
from ..config.constants import (
    ARTIFACT_INDEX_SCHEMA_VERSION,
    CLI_AUTOMATION_CONTRACT_VERSION,
    CLI_PYTHON_REQUIRES,
    COMPATIBILITY_POLICY_VERSION,
    INTERACTION_PLAN_VERSION,
    MCP_PYTHON_REQUIRES,
    MCP_SDK_REQUIRES,
    PENDING_RELEASE_GATES,
    PUBLIC_RELEASE_READY,
    RELEASE_CHANNEL,
    SERVER_NAME,
    SERVER_VERSION,
    SUPPORTED_CLI_VERSIONS,
    TARGET_OPERATING_SYSTEMS,
)
from ..config.settings import McpSettings, SettingsError
from ..contracts.anomaly_detection import MISSING_VALUE_METHODS as ANOMALY_DETECTION_MISSING_VALUE_METHODS
from ..contracts.anomaly_detection import MODEL_ORDER as ANOMALY_DETECTION_MODEL_ORDER
from ..contracts.anomaly_detection import SCALING_METHODS as ANOMALY_DETECTION_SCALING_METHODS
from ..contracts.anomaly_detection import UNSUPPORTED_INTERACTIONS as ANOMALY_DETECTION_UNSUPPORTED_INTERACTIONS
from ..contracts.classification import FEATURE_SELECTION_METHODS, LABEL_STRATEGIES, MISSING_VALUE_METHODS
from ..contracts.classification import MODEL_ORDER as CLASSIFICATION_MODEL_ORDER
from ..contracts.classification import SCALING_METHODS, TUNING_MODES
from ..contracts.classification import UNSUPPORTED_INTERACTIONS as CLASSIFICATION_UNSUPPORTED_INTERACTIONS
from ..contracts.clustering import MISSING_VALUE_METHODS as CLUSTERING_MISSING_VALUE_METHODS
from ..contracts.clustering import MODEL_ORDER as CLUSTERING_MODEL_ORDER
from ..contracts.clustering import SCALING_METHODS as CLUSTERING_SCALING_METHODS
from ..contracts.clustering import UNSUPPORTED_INTERACTIONS as CLUSTERING_UNSUPPORTED_INTERACTIONS
from ..contracts.decomposition import MISSING_VALUE_METHODS as DECOMPOSITION_MISSING_VALUE_METHODS
from ..contracts.decomposition import MODEL_ORDER as DECOMPOSITION_MODEL_ORDER
from ..contracts.decomposition import SCALING_METHODS as DECOMPOSITION_SCALING_METHODS
from ..contracts.decomposition import UNSUPPORTED_INTERACTIONS as DECOMPOSITION_UNSUPPORTED_INTERACTIONS
from ..contracts.manifest import CapabilityManifestError, known_gap_ids, load_capability_manifest, public_capabilities
from ..contracts.regression import MODEL_ORDER as REGRESSION_MODEL_ORDER
from ..contracts.regression import MODELS_WITHOUT_AUTOML
from ..contracts.regression import UNSUPPORTED_INTERACTIONS as REGRESSION_UNSUPPORTED_INTERACTIONS
from ..contracts.scientific_execution import LEGACY_METHODS_WITHOUT_V4_ATTESTATION_BY_TASK, PUBLIC_MANUAL_METHODS_BY_TASK, SCIENTIFIC_EXECUTION_METHODS_BY_TASK
from ..data.catalog import DatasetCatalogError
from ..data.headers import source_allows_pandas_duplicate_mangling
from ..data.inspector import DatasetInspectionError
from ..data.inspector import inspect_dataset as inspect_local_dataset
from ..data.inspector import snapshot_dataset
from ..data.preparation import DatasetPreparationError, prepare_dataset_view
from ..planning.interaction_plan import PlanCompilationError
from ..runtime.cli_capabilities import probe_cli_capabilities
from ..runtime.cli_driver import CliDriverError
from ..runtime.environment import EnvironmentInspectionError
from ..runtime.runs import InputIntegrityError, RunManager, RunNotFoundError, RunStateError
from ..tracking.experiments import ExperimentManager, ExperimentStoreError
from ..tracking.ui import MlflowUiError, MlflowUiManager
from .directory_views import (
    CompactGetExperimentResponse,
    CompactListDatasetsResponse,
    CompactListExperimentsResponse,
    DirectoryViewError,
    FullGetExperimentResponse,
    FullListDatasetsResponse,
    FullListExperimentsResponse,
    GetExperimentNotModifiedResponse,
    GetExperimentViewRequest,
    ListDatasetsNotModifiedResponse,
    ListDatasetsViewRequest,
    ListExperimentsNotModifiedResponse,
    ListExperimentsViewRequest,
    get_experiment_response_view,
    list_datasets_response_view,
    list_experiments_response_view,
)
from .output_contracts import (
    MAX_PUBLIC_TOOL_ERROR_ACTUAL_VALUE_SUMMARIES,
    MAX_PUBLIC_TOOL_ERROR_LOCATIONS,
    MAX_PUBLIC_TOOL_ERROR_PROBLEM_CHARS,
    MAX_PUBLIC_TOOL_ERROR_ROOT_CAUSES,
    PublicToolErrorResponse,
)
from .response_views import capabilities_response_view, capabilities_sha256, capability_projection_sha256, dataset_inspection_response_view
from .schemas import (
    START_READY_METHODS_BY_TASK,
    AnalysisRequest,
    AnalysisValidationDetailRequest,
    AnomalyDetectionRequest,
    CancelRunResponse,
    CapabilitiesNotModifiedResponse,
    CapabilitiesRequest,
    CapabilitiesResponse,
    ClassificationRequest,
    ClusteringRequest,
    CompactCapabilitiesResponse,
    CompactDatasetInspectionResponse,
    CompactRunArtifactPageResponse,
    CompactRunResultResponse,
    CompatibilityPolicy,
    DatasetInspectionRequest,
    DatasetInspectionResponse,
    DecompositionRequest,
    GetExperimentRequest,
    ListDatasetsRequest,
    ListExperimentsRequest,
    MlflowUiStatusResponse,
    OutputContractSchemaResponse,
    PendingRunResultResponse,
    RegressionRequest,
    RequestSchemaLookupArguments,
    RequestSchemaResolver,
    RequestSchemaResponse,
    ResourceLimits,
    RunLookupRequest,
    RunResultNotModifiedResponse,
    RunResultRequest,
    RunResultResponse,
    RunStatusRequest,
    RunStatusResponse,
    ScientificAttestationCapabilities,
    StartAnalysisByValidationRequest,
    StartAnalysisResponse,
    StartMlflowUiRequest,
    StartReadyCapabilitiesNotModifiedResponse,
    StartReadyCapabilitiesResponse,
    TaskValidationRequestContract,
    TimeSeriesRequest,
    ValidationRequestNavigation,
    _time_series_mode_ownership_json_conditions,
)
from .terminal_receipts import TerminalRunNotModifiedResponse, TerminalRunReceipt, terminal_result_response_view
from .validation_views import CompactAnalysisValidationResponse, FullAnalysisValidationDetailResponse, compact_analysis_validation, full_analysis_validation_detail

LOGGER = logging.getLogger(__name__)
_MAX_MODEL_TEXT = 4000
_MAX_COMPACT_ARTIFACT_REFERENCES = 32
_MAX_COMPACT_DATASET_COLUMNS = 12
_PUBLIC_ERROR_TEXT_PREFIX = "GeochemistryPi request rejected: "
_MAX_PUBLIC_ERROR_MESSAGE_BYTES = _MAX_MODEL_TEXT - len(_PUBLIC_ERROR_TEXT_PREFIX.encode("utf-8"))
_MAX_PUBLIC_ERROR_FIELD_TEXT = 256
_SAFE_FIELD_PART = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")
_WINDOWS_LOCAL_PATH = re.compile(r"(?i)(?:[A-Z]:[\\/]|\\\\)[^\r\n,;]+")
_POSIX_LOCAL_PATH = re.compile(r"(?<![\w:])/(?:[^\s'\",;]+/)*[^\s'\",;]*")
ANALYSIS_SCHEMA_TASK_ENV = "GEOCHEMISTRYPI_MCP_ANALYSIS_SCHEMA_TASK"
_ANALYSIS_REQUEST_MODELS: dict[str, type[BaseModel]] = {
    "classification": ClassificationRequest,
    "regression": RegressionRequest,
    "clustering": ClusteringRequest,
    "decomposition": DecompositionRequest,
    "anomaly_detection": AnomalyDetectionRequest,
    "time_series": TimeSeriesRequest,
}

# Full response schemas remain generated from the strict Pydantic response
# models and are retained in-process for validation/audit.  The MCP tools/list
# payload advertises only a hash-addressed envelope because replaying thirteen
# complete response unions on every model turn adds no information needed to
# construct a tool call.
_FULL_OUTPUT_SCHEMA_REGISTRY: dict[str, dict[str, Any]] = {}


class EmptyRequest(BaseModel):
    """Strict empty arguments for capability discovery."""

    model_config = ConfigDict(extra="forbid")


class ContractLookupError(ValueError):
    """A content-addressed public protocol contract was not found."""


_ToolFunction = Callable[[BaseModel], BaseModel]
_SCHEMA_OMITTED_KEYS = frozenset({"title"})
_SCHEMA_ANNOTATION_KEYS = frozenset(
    {
        "description",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
        "$comment",
    }
)
_SCHEMA_OBJECT_MAP_KEYS = frozenset(
    {
        "properties",
        "$defs",
        "patternProperties",
        "dependentSchemas",
    }
)
_SCHEMA_ARRAY_KEYS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_SCHEMA_CHILD_KEYS = frozenset(
    {
        "items",
        "contains",
        "not",
        "if",
        "then",
        "else",
        "additionalProperties",
        "propertyNames",
        "unevaluatedProperties",
        "unevaluatedItems",
    }
)
_PUBLIC_ERRORS = (
    ContractLookupError,
    CapabilityManifestError,
    CliDriverError,
    DatasetCatalogError,
    DatasetInspectionError,
    DatasetPreparationError,
    PlanCompilationError,
    EnvironmentInspectionError,
    InputIntegrityError,
    RunNotFoundError,
    RunStateError,
    ExperimentStoreError,
    MlflowUiError,
    SettingsError,
    DirectoryViewError,
)


def _advertised_input_schema(value: Any) -> Any:
    """Remove generated titles while retaining client guidance and defaults."""
    if isinstance(value, dict):
        return {key: _advertised_input_schema(child) for key, child in value.items() if key not in _SCHEMA_OMITTED_KEYS}
    if isinstance(value, list):
        return [_advertised_input_schema(child) for child in value]
    return value


def _require_advertised_analysis_task(schema: dict[str, Any]) -> dict[str, Any]:
    """Require explicit task dispatch for new clients without removing legacy runtime support."""
    definitions = schema.get("$defs")
    branches = [definitions[model.__name__] for model in _ANALYSIS_REQUEST_MODELS.values()] if isinstance(definitions, dict) and "oneOf" in schema else [schema]
    for branch in branches:
        if "task" not in branch.get("properties", {}):
            continue
        required = list(branch.get("required", ()))
        if "task" not in required:
            branch["required"] = ["task", *required]
    return schema


def _schema_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _canonical_schema(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _single_required_property(value: Any) -> tuple[str, dict[str, Any]] | None:
    """Recognize a one-field presence-and-value assertion."""
    if not isinstance(value, dict) or set(value) != {"properties", "required"}:
        return None
    properties = value.get("properties")
    required = value.get("required")
    if not isinstance(properties, dict) or not isinstance(required, list) or len(properties) != 1 or len(required) != 1:
        return None
    field = required[0]
    if not isinstance(field, str) or set(properties) != {field} or not isinstance(properties[field], dict):
        return None
    return field, properties[field]


def _negated_schema(value: dict[str, Any]) -> dict[str, Any]:
    """Return a validation-equivalent negation with elementary cancellations."""
    if set(value) == {"not"} and isinstance(value["not"], dict):
        return value["not"]
    if set(value) == {"anyOf"} and isinstance(value["anyOf"], list):
        return {"allOf": [_negated_schema(item) for item in value["anyOf"]]}
    if set(value) == {"allOf"} and isinstance(value["allOf"], list):
        return {"anyOf": [_negated_schema(item) for item in value["allOf"]]}
    return {"not": value}


def _source_xor_branches(value: Any) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Recognize Pydantic's expanded exact-one-of-two-fields contract."""
    if not isinstance(value, dict) or set(value) != {"oneOf"}:
        return None
    branches = value["oneOf"]
    if not isinstance(branches, list) or len(branches) != 2:
        return None
    parsed: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for branch in branches:
        if not isinstance(branch, dict) or set(branch) != {"allOf"}:
            return None
        conditions = branch["allOf"]
        if not isinstance(conditions, list) or len(conditions) != 2:
            return None
        positive = [condition for condition in conditions if _single_required_property(condition) is not None]
        negative = [condition["not"] for condition in conditions if isinstance(condition, dict) and set(condition) == {"not"} and _single_required_property(condition["not"]) is not None]
        if len(positive) != 1 or len(negative) != 1:
            return None
        parsed.append((positive[0], negative[0]))
    if _canonical_schema(parsed[0][0]) != _canonical_schema(parsed[1][1]):
        return None
    if _canonical_schema(parsed[0][1]) != _canonical_schema(parsed[1][0]):
        return None
    return parsed[0][0], parsed[1][0]


def _preserve_expanded_union(path: tuple[str, ...]) -> bool:
    """Keep the few public union shapes that clients inspect directly."""
    return len(path) >= 2 and path[-2:] in {
        ("properties", "bin_width"),
        ("properties", "training_dataset"),
        ("properties", "if_result_sha256"),
    }


def _compact_schema_logic(value: Any, path: tuple[str, ...] = ()) -> Any:
    """Use shorter Draft 2020-12 forms without dropping any contract metadata."""
    if isinstance(value, list):
        return [_compact_schema_logic(child, (*path, str(index))) for index, child in enumerate(value)]
    if not isinstance(value, dict):
        return value

    source_xor = _source_xor_branches(value)
    if source_xor is not None:
        value = {"oneOf": list(source_xor)}

    compacted = {key: _compact_schema_logic(child, (*path, key)) for key, child in value.items()}

    # These keywords have the same Draft 2020-12 default when omitted. They are
    # generator noise, not scientific limits or client guidance.
    for keyword in ("minItems", "minLength", "minProperties"):
        if compacted.get(keyword) == 0:
            compacted.pop(keyword)
    if compacted.get("required") == []:
        compacted.pop("required")

    if "const" in compacted and isinstance(compacted.get("type"), str):
        constant = compacted["const"]
        expected_type = compacted["type"]
        if (
            (expected_type == "string" and isinstance(constant, str))
            or (expected_type == "boolean" and isinstance(constant, bool))
            or (expected_type == "integer" and isinstance(constant, int) and not isinstance(constant, bool))
            or (expected_type == "number" and isinstance(constant, (int, float)) and not isinstance(constant, bool))
            or (expected_type == "null" and constant is None)
        ):
            compacted.pop("type")
    enum = compacted.get("enum")
    if compacted.get("type") == "string" and isinstance(enum, list) and enum and all(isinstance(item, str) for item in enum):
        compacted.pop("type")

    variants = compacted.get("anyOf")
    if not _preserve_expanded_union(path) and isinstance(variants, list) and variants:
        simple_values: list[Any] = []
        simple_value_union = True
        for item in variants:
            if not isinstance(item, dict):
                simple_value_union = False
                break
            if set(item) == {"const"}:
                simple_values.append(item["const"])
            elif set(item) == {"enum"} and isinstance(item["enum"], list):
                simple_values.extend(item["enum"])
            elif item == {"type": "null"}:
                simple_values.append(None)
            else:
                simple_value_union = False
                break
        if simple_value_union and len({_canonical_schema(item) for item in simple_values}) == len(simple_values):
            compacted.pop("anyOf")
            compacted["enum"] = simple_values
        elif all(isinstance(item, dict) and set(item) == {"type"} and isinstance(item["type"], str) for item in variants):
            types: list[str] = []
            source_types = [item["type"] for item in variants]
            for item in source_types:
                normalized = "number" if item == "integer" and "number" in source_types else item
                if normalized not in types:
                    types.append(normalized)
            compacted.pop("anyOf")
            compacted["type"] = types[0] if len(types) == 1 else types
        elif len(variants) == 2 and {"type": "null"} in variants:
            typed = next(item for item in variants if item != {"type": "null"})
            if isinstance(typed, dict) and isinstance(typed.get("type"), str):
                compacted.pop("anyOf")
                compacted.update(typed)
                compacted["type"] = [typed["type"], "null"]

    required_property = _single_required_property(compacted)
    if required_property is not None:
        field, constraint = required_property
        candidate = {"not": {"properties": {field: _negated_schema(constraint)}}}
        if _schema_bytes(candidate) < _schema_bytes(compacted):
            compacted = candidate
    elif set(compacted) == {"not"}:
        required_property = _single_required_property(compacted["not"])
        if required_property is not None:
            field, constraint = required_property
            candidate = {"properties": {field: _negated_schema(constraint)}}
            if _schema_bytes(candidate) < _schema_bytes(compacted):
                compacted = candidate

    if set(compacted) == {"if", "then"} and isinstance(compacted["if"], dict) and isinstance(compacted["then"], dict):
        implication = {"anyOf": [_negated_schema(compacted["if"]), compacted["then"]]}
        if _schema_bytes(implication) < _schema_bytes(compacted):
            compacted = implication

    for combiner in ("allOf", "anyOf"):
        branches = compacted.get(combiner)
        if not isinstance(branches, list):
            continue
        flattened: list[Any] = []
        for branch in branches:
            if isinstance(branch, dict) and set(branch) == {combiner} and isinstance(branch[combiner], list):
                flattened.extend(branch[combiner])
            else:
                flattened.append(branch)
        if len(flattened) == 1 and set(compacted) == {combiner}:
            return flattened[0]
        compacted[combiner] = flattened

        negated = [branch["not"] for branch in flattened if isinstance(branch, dict) and set(branch) == {"not"} and isinstance(branch["not"], dict)]
        remainder = [branch for branch in flattened if not (isinstance(branch, dict) and set(branch) == {"not"} and isinstance(branch["not"], dict))]
        if len(negated) >= 2:
            opposite = "anyOf" if combiner == "allOf" else "allOf"
            grouped = {"not": {opposite: negated}}
            candidate = [grouped, *remainder]
            if _schema_bytes(candidate) < _schema_bytes(flattened):
                compacted[combiner] = candidate

        if combiner == "allOf":
            branches = compacted[combiner]
            property_branches = [branch for branch in branches if isinstance(branch, dict) and set(branch) == {"properties"} and isinstance(branch["properties"], dict)]
            occupied: set[str] = set()
            disjoint_properties = True
            for branch in property_branches:
                fields = set(branch["properties"])
                if occupied & fields:
                    disjoint_properties = False
                    break
                occupied.update(fields)
            if len(property_branches) >= 2 and disjoint_properties:
                merged_properties = {field: constraint for branch in property_branches for field, constraint in branch["properties"].items()}
                remaining = [branch for branch in branches if branch not in property_branches]
                candidate = [{"properties": merged_properties}, *remaining]
                if _schema_bytes(candidate) < _schema_bytes(branches):
                    compacted[combiner] = candidate
    return compacted


def _asserted_property(field: str, constraint: dict[str, Any]) -> dict[str, Any]:
    """Assert that a property is present and satisfies one validation schema."""
    return {"not": {"properties": {field: _negated_schema(constraint)}}}


def _all_asserted_properties(constraints: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Conjoin property assertions without repeating one outer negation per field."""
    return {"not": {"anyOf": [{"properties": {field: _negated_schema(constraint)}} for field, constraint in constraints.items()]}}


def _schema_implication(assertion: dict[str, Any], consequence: dict[str, Any]) -> dict[str, Any]:
    """Encode ``assertion -> consequence`` without repeating its selector."""
    return {"anyOf": [_negated_schema(assertion), consequence]}


def _compact_filter_rule_conditions(definition: dict[str, Any]) -> None:
    if not definition.get("allOf") or not {
        "operator",
        "value",
        "values",
        "minimum",
        "maximum",
    } <= set(definition.get("properties", {})):
        return
    scalar_operators = [
        "equal",
        "not_equal",
        "greater_than",
        "greater_than_or_equal",
        "less_than",
        "less_than_or_equal",
    ]
    non_null = {"not": {"type": "null"}}
    non_empty = {"minItems": 1}
    empty_array = {"maxItems": 0}
    empty_value_properties = {
        "value": {"type": "null"},
        "values": empty_array,
        "minimum": {"type": "null"},
        "maximum": {"type": "null"},
    }
    definition["allOf"] = [
        {
            "oneOf": [
                {
                    "properties": {
                        "operator": {"const": "not_null"},
                        **empty_value_properties,
                    }
                },
                {
                    "properties": {
                        "operator": {"enum": scalar_operators},
                        "values": empty_array,
                        "minimum": {"type": "null"},
                        "maximum": {"type": "null"},
                    },
                    **_asserted_property("value", non_null),
                },
                {
                    "properties": {
                        "operator": {"const": "between"},
                        "value": {"type": "null"},
                        "values": empty_array,
                    },
                    **_all_asserted_properties(
                        {
                            "minimum": non_null,
                            "maximum": non_null,
                        }
                    ),
                },
                {
                    "properties": {
                        "operator": {"const": "in"},
                        "value": {"type": "null"},
                        "minimum": {"type": "null"},
                        "maximum": {"type": "null"},
                    },
                    **_asserted_property("values", non_empty),
                },
            ]
        }
    ]


def _compact_preparation_conditions(definition: dict[str, Any]) -> None:
    if not definition.get("allOf") or not {
        "worksheets",
        "worksheet",
        "union_mode",
        "source_sheet_column",
        "source_row_column",
        "selected_columns",
        "excluded_columns",
        "header_row_index",
        "header_row_indices",
    } <= set(definition.get("properties", {})):
        return
    non_null = {"not": {"type": "null"}}
    non_empty = {"minItems": 1}
    empty_array = {"maxItems": 0}
    definition["allOf"] = [
        {
            "oneOf": [
                {
                    "properties": {
                        "worksheets": empty_array,
                        "union_mode": {"type": "null"},
                        "source_sheet_column": {"type": "null"},
                    }
                },
                {
                    "properties": {"worksheet": {"type": "null"}},
                    **_all_asserted_properties(
                        {
                            "worksheets": {"minItems": 2},
                            "union_mode": {"const": "rows"},
                            "source_sheet_column": non_null,
                            "source_row_column": non_null,
                            "selected_columns": non_empty,
                        }
                    ),
                },
            ]
        },
        {
            "anyOf": [
                {"properties": {"header_row_indices": empty_array}},
                {"not": {"required": ["header_row_index"]}},
            ]
        },
        {
            "anyOf": [
                {"properties": {"selected_columns": empty_array}},
                {"properties": {"excluded_columns": empty_array}},
            ]
        },
    ]


def _compact_row_identity_conditions(definition: dict[str, Any]) -> None:
    if not definition.get("allOf") or not {
        "strategy",
        "columns",
        "source_mapping_path",
        "source_mapping_sha256",
    } <= set(definition.get("properties", {})):
        return
    non_null = {"not": {"type": "null"}}
    non_empty = {"minItems": 1}
    definition["allOf"] = [
        {
            "not": {
                "oneOf": [
                    _asserted_property("strategy", {"const": "column_values"}),
                    _asserted_property("columns", non_empty),
                ]
            }
        },
        {
            "not": {
                "oneOf": [
                    _asserted_property("source_mapping_path", non_null),
                    _asserted_property("source_mapping_sha256", non_null),
                ]
            }
        },
    ]


def _compact_time_series_conditions(definition: dict[str, Any]) -> None:
    if definition.get("properties", {}).get("task", {}).get("const") != "time_series" or not definition.get("allOf"):
        return
    non_null = {"not": {"type": "null"}}
    non_empty = {"minItems": 1}
    empty_array = {"maxItems": 0}
    filter_dependency = {
        "anyOf": [
            {
                "properties": {
                    "filter_minimum": {"type": "null"},
                    "filter_maximum": {"type": "null"},
                }
            },
            _asserted_property("filter_column", non_null),
        ]
    }
    comparison_pair = {
        "not": {
            "oneOf": [
                _asserted_property("comparison_label_column", non_null),
                _asserted_property("comparison_positive_values", non_empty),
            ]
        }
    }
    event_filter_pair = {
        "not": {
            "oneOf": [
                _asserted_property("event_filter_column", non_null),
                _asserted_property("event_filter_values", non_empty),
            ]
        }
    }
    no_event_detail = {
        "properties": {
            "event_time_column": {"type": "null"},
            "event_identifier_column": {"type": "null"},
            "event_filter_column": {"type": "null"},
            "association_window_days": {"type": "null"},
            "event_filter_values": empty_array,
        }
    }
    event_detail_requires_dataset = {
        "anyOf": [
            no_event_detail,
            _asserted_property("event_dataset_path", non_null),
        ]
    }
    event_dataset_requires_time = {
        "anyOf": [
            {"properties": {"event_dataset_path": {"type": "null"}}},
            _asserted_property("event_time_column", non_null),
        ]
    }
    reference_mode = _asserted_property("mode", {"const": "reference_anomaly_series"})
    continuous_mode = _asserted_property("mode", {"const": "continuous"})
    element_mode = _asserted_property("mode", {"const": "element_mean"})
    mode_contract = {
        "allOf": [
            *_time_series_mode_ownership_json_conditions(),
            # Every mode except reference_anomaly_series requires bin_width;
            # an omitted mode retains the advertised subaerial default.
            {"anyOf": [reference_mode, _asserted_property("bin_width", non_null)]},
            _schema_implication(
                continuous_mode,
                {
                    **_all_asserted_properties(
                        {
                            "minimum_age_column": non_null,
                            "value_column": non_null,
                        }
                    ),
                    "allOf": [filter_dependency],
                },
            ),
            _schema_implication(
                element_mode,
                {
                    **_asserted_property("element_columns", non_empty),
                    "allOf": [filter_dependency],
                },
            ),
            _schema_implication(
                reference_mode,
                {
                    "properties": {
                        "bin_width": {"type": "null"},
                        "missing_values": {"properties": {"method": {"const": "error"}}},
                    },
                    **_all_asserted_properties(
                        {
                            "time_column": non_null,
                            "signal_columns": non_empty,
                            "reference_label_column": non_null,
                            "reference_positive_values": non_empty,
                        }
                    ),
                    "allOf": [
                        comparison_pair,
                        event_filter_pair,
                        event_detail_requires_dataset,
                        event_dataset_requires_time,
                    ],
                },
            ),
        ]
    }
    source_contract = definition["allOf"][0]
    definition["allOf"] = [source_contract, mode_contract]


def _compact_known_schema_contracts(schema: dict[str, Any]) -> dict[str, Any]:
    """Project verbose Pydantic conditions into equivalent standard schemas."""
    definitions = schema.get("$defs", {})
    if isinstance(definitions, dict):
        filter_rule = definitions.get("DatasetFilterRule")
        if isinstance(filter_rule, dict):
            _compact_filter_rule_conditions(filter_rule)
        preparation = definitions.get("DatasetPreparationContract")
        if isinstance(preparation, dict):
            _compact_preparation_conditions(preparation)
        row_identity = definitions.get("SourceRowIdentityContract")
        if isinstance(row_identity, dict):
            _compact_row_identity_conditions(row_identity)
        time_series = definitions.get("TimeSeriesRequest")
        if isinstance(time_series, dict):
            _compact_time_series_conditions(time_series)
    _compact_time_series_conditions(schema)
    return schema


def _is_protected_schema_path(path: tuple[str, ...]) -> bool:
    if not path:
        return True
    if len(path) == 2 and path[0] == "$defs":
        return True
    return len(path) >= 2 and path[-2:] in {
        ("properties", "task"),
        ("properties", "training_dataset"),
        ("properties", "bin_width"),
        ("properties", "artifact_view"),
        ("properties", "detail"),
        ("properties", "if_result_sha256"),
    }


def _iter_schema_nodes(value: Any, path: tuple[str, ...] = ()) -> Any:
    """Yield schema nodes while excluding maps whose values are not schemas."""
    if not isinstance(value, dict):
        return
    if path and not _is_protected_schema_path(path):
        annotations = {key: value[key] for key in value if key in _SCHEMA_ANNOTATION_KEYS}
        validation = {key: value[key] for key in value if key not in _SCHEMA_ANNOTATION_KEYS}
        if validation and set(validation) != {"$ref"} and "$defs" not in validation and _schema_bytes(validation) >= 20:
            yield path, value, annotations, validation
    for key, child in value.items():
        if key in _SCHEMA_OBJECT_MAP_KEYS and isinstance(child, dict):
            for name, schema in child.items():
                if key == "$defs" and name.startswith("_x"):
                    continue
                yield from _iter_schema_nodes(schema, (*path, key, name))
        elif key in _SCHEMA_ARRAY_KEYS and isinstance(child, list):
            for index, schema in enumerate(child):
                yield from _iter_schema_nodes(schema, (*path, key, str(index)))
        elif key in _SCHEMA_CHILD_KEYS and isinstance(child, dict):
            yield from _iter_schema_nodes(child, (*path, key))


def _shared_definition_name(validation: dict[str, Any], index: int) -> str:
    # These are implementation-only validation fragments. Existing scientific
    # model names in ``$defs`` remain untouched and therefore stay meaningful.
    return f"_x{index}"


def _replace_schema_core(
    value: Any,
    target: str,
    reference: dict[str, str],
    include_annotations: bool,
    path: tuple[str, ...] = (),
) -> Any:
    if not isinstance(value, dict):
        return value
    if path and not _is_protected_schema_path(path):
        annotations = {key: value[key] for key in value if key in _SCHEMA_ANNOTATION_KEYS}
        validation = {key: value[key] for key in value if key not in _SCHEMA_ANNOTATION_KEYS}
        candidate = value if include_annotations else validation
        if _canonical_schema(candidate) == target:
            return reference if include_annotations else {**annotations, **reference}
    replaced: dict[str, Any] = {}
    for key, child in value.items():
        if key in _SCHEMA_OBJECT_MAP_KEYS and isinstance(child, dict):
            replaced[key] = {
                name: (
                    schema
                    if key == "$defs" and name.startswith("_x")
                    else _replace_schema_core(
                        schema,
                        target,
                        reference,
                        include_annotations,
                        (*path, key, name),
                    )
                )
                for name, schema in child.items()
            }
        elif key in _SCHEMA_ARRAY_KEYS and isinstance(child, list):
            replaced[key] = [
                _replace_schema_core(
                    schema,
                    target,
                    reference,
                    include_annotations,
                    (*path, key, str(index)),
                )
                for index, schema in enumerate(child)
            ]
        elif key in _SCHEMA_CHILD_KEYS and isinstance(child, dict):
            replaced[key] = _replace_schema_core(child, target, reference, include_annotations, (*path, key))
        else:
            replaced[key] = child
    return replaced


def _deduplicate_schema_fragments(schema: dict[str, Any]) -> dict[str, Any]:
    """Factor repeated validation fragments while retaining public annotations."""
    compacted = schema
    for index in range(1, 129):
        found: dict[tuple[bool, str], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
        examples: dict[tuple[bool, str], dict[str, Any]] = {}
        for _, node, annotations, validation in _iter_schema_nodes(compacted):
            core_key = False, _canonical_schema(validation)
            found.setdefault(core_key, []).append((node, annotations))
            examples[core_key] = validation
            if annotations:
                full_key = True, _canonical_schema(node)
                found.setdefault(full_key, []).append((node, {}))
                examples[full_key] = node

        best: tuple[int, str, dict[str, Any], str, bool] | None = None
        for (include_annotations, key), occurrences in found.items():
            if len(occurrences) < 2:
                continue
            definition = examples[(include_annotations, key)]
            name = _shared_definition_name(definition, index)
            while name in compacted.get("$defs", {}):
                name = f"{name}x"
            reference = {"$ref": f"#/$defs/{name}"}
            original_size = sum(_schema_bytes(node) for node, _ in occurrences)
            replacement_size = sum(_schema_bytes({**annotations, **reference}) for _, annotations in occurrences)
            definition_size = _schema_bytes({name: definition}) - 2
            savings = original_size - replacement_size - definition_size
            if savings > 0 and (best is None or savings > best[0]):
                best = savings, key, definition, name, include_annotations
        if best is None:
            break
        _, target, definition, name, include_annotations = best
        reference = {"$ref": f"#/$defs/{name}"}
        compacted = _replace_schema_core(compacted, target, reference, include_annotations)
        compacted.setdefault("$defs", {})[name] = definition
    return compacted


def _optimized_advertised_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Compact only redundant representation; preserve every public contract."""
    return _deduplicate_schema_fragments(_compact_schema_logic(_compact_known_schema_contracts(schema)))


def _schema_exactly_one_groups(
    schema: dict[str, Any],
) -> tuple[tuple[str, str], ...]:
    """Read exact-one field groups from Pydantic's generated JSON Schema."""
    groups: list[tuple[str, str]] = []
    for condition in schema.get("allOf", ()):
        branches = _source_xor_branches(condition)
        if branches is None:
            continue
        left = _single_required_property(branches[0])
        right = _single_required_property(branches[1])
        if left is None or right is None:
            continue
        groups.append((left[0], right[0]))
    return tuple(groups)


def _minimal_required_placeholder(
    field: str,
    definition: dict[str, Any],
) -> Any:
    """Create one non-scientific placeholder from a required field schema."""
    if "const" in definition:
        return definition["const"]
    if definition.get("type") == "array":
        return [f"{field}_item"]
    for alternative in definition.get("anyOf", ()):
        if isinstance(alternative, dict) and alternative.get("type") == "array":
            return [f"{field}_item"]
    return field


def _minimal_regression_request_example(
    model: type[BaseModel],
    schema: dict[str, Any],
    xor_groups: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    """Build and validate a placeholder-only request from the live regression schema."""
    properties = schema.get("properties", {})
    example = {field: _minimal_required_placeholder(field, properties[field]) for field in schema.get("required", ())}
    training_group = next(group for group in xor_groups if set(group) == {"training_dataset", "training_dataset_path"})
    target_group = next(group for group in xor_groups if set(group) == {"target_column", "target_columns"})
    training_field = "training_dataset" if "training_dataset" in training_group else training_group[0]
    example[training_field] = {"source": "path", "path": "input.csv"}
    example[target_group[0]] = "target"
    validated = model.model_validate(example)
    return validated.model_dump(mode="json", exclude_unset=True)


def _task_validation_request_contract(task: str) -> TaskValidationRequestContract:
    """Build one exact task schema and stable navigation receipt from runtime models."""
    model = _ANALYSIS_REQUEST_MODELS[task]
    schema = _advertised_input_schema(model.model_json_schema())
    schema = _require_advertised_analysis_task(schema)
    top_level_fields = tuple(schema.get("properties", ()))
    top_level_required_fields = tuple(schema.get("required", ()))
    has_application_dataset = "application_dataset" in top_level_fields
    has_model = "model" in top_level_fields
    has_model_selection = "model_selection" in top_level_fields
    xor_groups = _schema_exactly_one_groups(schema)
    regression_target_group = next(
        (group for group in xor_groups if set(group) == {"target_column", "target_columns"}),
        (),
    )
    minimal_example = _minimal_regression_request_example(model, schema, xor_groups) if task == "regression" else None
    optimized_schema = _optimized_advertised_schema(schema)
    schema_bytes = json.dumps(
        optimized_schema,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return TaskValidationRequestContract(
        task=task,
        top_level_fields=top_level_fields,
        top_level_required_fields=top_level_required_fields,
        navigation=ValidationRequestNavigation(
            application_dataset_at_most_one_of=(("application_dataset", "application_dataset_path") if has_application_dataset else ()),
            regression_target_exactly_one_of=regression_target_group,
            model_selection_discriminator_path=("model_selection.mode" if has_model_selection else None),
            model_settings_discriminator_path="model.type" if has_model else None,
            split_seed_path=(None if task == "time_series" else "reproducibility.split_seed"),
            model_seed_path=(None if task == "time_series" else "reproducibility.model_seed"),
            tuning_seed_path=(None if task == "time_series" else "reproducibility.tuning_seed"),
            workflow_seed_path="seed" if task == "time_series" else None,
        ),
        minimal_legal_request_example=minimal_example,
        request_schema_utf8_bytes=len(schema_bytes),
        request_schema_sha256=hashlib.sha256(schema_bytes).hexdigest(),
        request_schema=optimized_schema,
    )


def _placeholder_paths(value: Any, prefix: str = "") -> tuple[str, ...]:
    """List every explicit placeholder path in one structural request template."""
    paths: list[str] = []
    if isinstance(value, str) and value.startswith("__") and value.endswith("__"):
        paths.append(prefix)
    elif isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            paths.extend(_placeholder_paths(child, child_prefix))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            paths.extend(_placeholder_paths(child, f"{prefix}[{index}]"))
    return tuple(paths)


def _start_ready_request_template(task: str, method: str) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Build one runtime-valid structural template without choosing scientific values."""
    request: dict[str, Any] = {
        "task": task,
        "training_dataset": {
            "source": "path",
            "path": "__TRAINING_DATASET_PATH__",
        },
        "experiment_name": "__EXPERIMENT_NAME__",
        "run_name": "__RUN_NAME__",
    }
    numeric_placeholder_paths: tuple[str, ...] = ()
    if task == "time_series":
        request["mode"] = method
        if method == "subaerial_proportion":
            request.update(
                {
                    "bin_width": 1.0,
                    "age_column": "__CENTRAL_AGE_COLUMN__",
                    "maximum_age_column": "__MAXIMUM_AGE_COLUMN__",
                    "probability_column": "__PROBABILITY_COLUMN__",
                    "latitude_column": "__LATITUDE_COLUMN__",
                    "longitude_column": "__LONGITUDE_COLUMN__",
                }
            )
            numeric_placeholder_paths = ("bin_width",)
        elif method == "continuous":
            request.update(
                {
                    "bin_width": 1.0,
                    "age_column": "__CENTRAL_AGE_COLUMN__",
                    "minimum_age_column": "__MINIMUM_AGE_COLUMN__",
                    "maximum_age_column": "__MAXIMUM_AGE_COLUMN__",
                    "value_column": "__VALUE_COLUMN__",
                    "latitude_column": "__LATITUDE_COLUMN__",
                    "longitude_column": "__LONGITUDE_COLUMN__",
                }
            )
            numeric_placeholder_paths = ("bin_width",)
        elif method == "element_mean":
            request.update(
                {
                    "bin_width": 1.0,
                    "age_column": "__CENTRAL_AGE_COLUMN__",
                    "element_columns": ["__ELEMENT_COLUMN__"],
                }
            )
            numeric_placeholder_paths = ("bin_width",)
        else:
            request.update(
                {
                    "time_column": "__TIME_COLUMN__",
                    "signal_columns": ["__SIGNAL_COLUMN__"],
                    "reference_label_column": "__REFERENCE_LABEL_COLUMN__",
                    "reference_positive_values": ["__REFERENCE_POSITIVE_VALUE__"],
                }
            )
    else:
        request.update(
            {
                "identifier_column": "__IDENTIFIER_COLUMN__",
                "feature_columns": [
                    "__FEATURE_COLUMN_1__",
                    "__FEATURE_COLUMN_2__",
                ],
            }
        )
        if task in {"classification", "regression"}:
            request["target_column"] = "__TARGET_COLUMN__"
        if task == "decomposition" and method == "embedding_label_overlay":
            request.update(
                {
                    "mode": "embedding_label_overlay",
                    "application_dataset": {
                        "source": "path",
                        "path": "__LABEL_DATASET_PATH__",
                    },
                    "label_identifier_column": "__LABEL_IDENTIFIER_COLUMN__",
                    "label_column": "__LABEL_COLUMN__",
                    "positive_label_values": ["__POSITIVE_LABEL_VALUE__"],
                    "scaling": "none",
                }
            )
        else:
            request["model"] = {"type": method}
    model = _ANALYSIS_REQUEST_MODELS[task]
    validated = model.model_validate(request)
    template = validated.model_dump(mode="json", exclude_unset=True)
    model.model_validate(template)
    return template, (*_placeholder_paths(template), *numeric_placeholder_paths)


def _find_discriminator_schema(value: Any, method: str) -> dict[str, Any] | None:
    """Find the real Pydantic settings branch selected by one public method."""
    if isinstance(value, dict):
        type_schema = value.get("properties", {}).get("type", {})
        if type_schema.get("const") == method or type_schema.get("enum") == [method]:
            return value
        for child in value.values():
            found = _find_discriminator_schema(child, method)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_discriminator_schema(child, method)
            if found is not None:
                return found
    return None


def _selected_method_constraints(task: str, method: str) -> dict[str, Any]:
    """Project only the selected method/mode constraints from the live request model."""
    schema = _advertised_input_schema(_ANALYSIS_REQUEST_MODELS[task].model_json_schema())
    if task != "time_series" and not (task == "decomposition" and method == "embedding_label_overlay"):
        branch = _find_discriminator_schema(schema, method)
        if branch is None:
            raise RuntimeError(f"No runtime request-schema branch was found for {task}.{method}.")
        return {
            "discriminator_path": "model.type",
            "discriminator_value": method,
            "model_settings_schema": branch,
        }
    properties = schema.get("properties", {})
    if task == "decomposition":
        fields = (
            "mode",
            "application_dataset",
            "application_dataset_path",
            "feature_columns",
            "label_identifier_column",
            "label_column",
            "positive_label_values",
            "scaling",
        )
        discriminator_path = "mode"
    else:
        fields_by_mode = {
            "subaerial_proportion": (
                "mode",
                "bin_width",
                "age_column",
                "maximum_age_column",
                "probability_column",
                "latitude_column",
                "longitude_column",
            ),
            "continuous": (
                "mode",
                "bin_width",
                "age_column",
                "minimum_age_column",
                "maximum_age_column",
                "value_column",
                "latitude_column",
                "longitude_column",
            ),
            "element_mean": (
                "mode",
                "bin_width",
                "age_column",
                "element_columns",
            ),
            "reference_anomaly_series": (
                "mode",
                "time_column",
                "signal_columns",
                "reference_label_column",
                "reference_positive_values",
            ),
        }
        fields = fields_by_mode[method]
        discriminator_path = "mode"
    return {
        "discriminator_path": discriminator_path,
        "discriminator_value": method,
        "field_schemas": {field: properties[field] for field in fields if field in properties},
    }


def _request_schema_response(request_schema_sha256: str) -> RequestSchemaResponse:
    """Resolve one exact task request schema without replaying capability metadata."""
    for task in _ANALYSIS_REQUEST_MODELS:
        contract = _task_validation_request_contract(task)
        if contract.request_schema_sha256 == request_schema_sha256:
            return RequestSchemaResponse(
                task=task,
                request_schema_utf8_bytes=contract.request_schema_utf8_bytes,
                request_schema_sha256=contract.request_schema_sha256,
                request_schema=contract.request_schema,
            )
    raise ContractLookupError("The requested validate_analysis request schema is not registered by this server; " "request one current start_ready view and use its exact request_schema_sha256.")


def _start_ready_capabilities_view(
    response: CapabilitiesResponse,
    request: CapabilitiesRequest,
    contract: TaskValidationRequestContract,
) -> StartReadyCapabilitiesResponse | StartReadyCapabilitiesNotModifiedResponse:
    """Return one small stateless method-specific construction view."""
    assert request.task is not None
    assert request.method is not None
    template, placeholder_paths = _start_ready_request_template(
        request.task,
        request.method,
    )
    identity = capabilities_sha256(response)
    projected = StartReadyCapabilitiesResponse(
        capabilities_sha256=identity,
        capability_view_sha256="0" * 64,
        task_filter=request.task,
        method_filter=request.method,
        server_version=response.server_version,
        available_methods=START_READY_METHODS_BY_TASK[request.task],
        top_level_required_fields=contract.top_level_required_fields,
        navigation=contract.navigation,
        selected_method_constraints=_selected_method_constraints(
            request.task,
            request.method,
        ),
        request_template=template,
        placeholder_paths=placeholder_paths,
        request_schema_utf8_bytes=contract.request_schema_utf8_bytes,
        request_schema_sha256=contract.request_schema_sha256,
        request_schema_resolver=RequestSchemaResolver(
            arguments=RequestSchemaLookupArguments(
                request_schema_sha256=contract.request_schema_sha256,
            )
        ),
        guidance=(
            "Replace every listed placeholder, then overlay and preserve every user-supplied scientific "
            "parameter. If the exact dataset path and columns are already known, call validate_analysis "
            "directly; call inspect_dataset only when required dataset facts are unknown. detail='full' is "
            "audit-only and is not required to construct this request. Send a conditional view hash only "
            "while the original structured payload is still retained."
        ),
    )
    view_identity = capability_projection_sha256(projected)
    if request.if_capability_view_sha256 == view_identity:
        return StartReadyCapabilitiesNotModifiedResponse(
            capabilities_sha256=identity,
            capability_view_sha256=view_identity,
            task_filter=request.task,
            method_filter=request.method,
            server_version=response.server_version,
        )
    return projected.model_copy(update={"capability_view_sha256": view_identity})


def _validation_tool_input_schema(
    request: type[BaseModel],
) -> dict[str, Any]:
    """Advertise exact scoped schemas or one generic contract-discovery route.

    The unscoped server accepts all six strict runtime request models.  Their
    complete task-specific schemas are returned, unchanged and hash-bound, by
    ``get_capabilities(task=...)``.  Repeating the six-schema union in every
    tools/list replay is therefore redundant.  An explicitly task-scoped
    server continues to advertise that task's complete schema directly.
    """

    if request is AnalysisRequest:
        analysis_schema = {
            "additionalProperties": True,
            "description": (
                "New scientific request routing envelope. When request shape is not already retained, "
                "call get_capabilities once with detail='start_ready', the selected task, and the exact "
                "public method or workflow mode. Replace its placeholders, preserve all user-supplied "
                "scientific values, and validate once. Known dataset paths and columns do not require "
                "inspection, and detail='full' is audit-only. The server still "
                "validates the complete strict task contract and rejects unknown, "
                "misplaced, or incompatible fields before starting any process."
            ),
            "properties": {
                "scientific_contract_version": {
                    "const": 2,
                    "default": 2,
                    "description": "Current strict scientific request contract version.",
                    "type": "integer",
                },
                "task": {
                    "description": "Selects the exact contract returned by get_capabilities.",
                    "enum": list(_ANALYSIS_REQUEST_MODELS),
                    "type": "string",
                },
            },
            "required": ["task"],
            "type": "object",
        }
    else:
        analysis_schema = _require_advertised_analysis_task(_advertised_input_schema(request.model_json_schema()))
    detail_schema = _advertised_input_schema(AnalysisValidationDetailRequest.model_json_schema())
    definitions = {
        **analysis_schema.pop("$defs", {}),
        **detail_schema.pop("$defs", {}),
    }
    if "oneOf" in analysis_schema:
        analysis_schema["oneOf"] = [
            *analysis_schema["oneOf"],
            detail_schema,
        ]
        combined = analysis_schema
    else:
        combined = {
            "oneOf": [analysis_schema, detail_schema],
            "type": "object",
        }
    if definitions:
        combined["$defs"] = definitions
    return combined


def _full_output_contract_schema(
    response: type[BaseModel] | tuple[type[BaseModel], ...],
) -> dict[str, Any]:
    """Generate the complete strict serialized response union for audit/tests."""

    success_models = response if isinstance(response, tuple) else (response,)
    output_adapter = TypeAdapter(reduce(or_, (*success_models, PublicToolErrorResponse)))
    output_schema = output_adapter.json_schema(mode="serialization")
    output_schema["type"] = "object"
    return _optimized_advertised_schema(_advertised_input_schema(output_schema))


def full_output_contract_schema(contract_sha256: str) -> dict[str, Any]:
    """Return one internally retained full schema by its advertised identity."""

    try:
        return deepcopy(_FULL_OUTPUT_SCHEMA_REGISTRY[contract_sha256])
    except KeyError as exc:
        raise KeyError(f"Unknown GeochemistryPi output contract: {contract_sha256}") from exc


def _hash_addressed_output_schema(
    response: type[BaseModel] | tuple[type[BaseModel], ...],
) -> dict[str, Any]:
    """Advertise a small envelope while retaining the exact full contract."""

    full_schema = _full_output_contract_schema(response)
    encoded = _canonical_schema(full_schema).encode("utf-8")
    contract_sha256 = hashlib.sha256(encoded).hexdigest()
    existing = _FULL_OUTPUT_SCHEMA_REGISTRY.setdefault(contract_sha256, full_schema)
    if existing != full_schema:
        raise RuntimeError("An output-contract SHA-256 collision was detected.")
    return {
        "additionalProperties": True,
        "type": "object",
        "x-geochemistrypi-full-output-schema-sha256": contract_sha256,
        "x-geochemistrypi-full-output-schema-utf8-bytes": len(encoded),
        "x-geochemistrypi-output-contract-delivery": "hash-addressed-server-enforced",
        "x-geochemistrypi-output-contract-resolver": {
            "tool": "get_capabilities",
            "argument": "output_contract_sha256",
            "response_field": "output_contract_schema",
        },
    }


def _tool(
    name: str,
    description: str,
    request: type[BaseModel],
    response: type[BaseModel] | tuple[type[BaseModel], ...],
) -> Tool:
    input_schema = _validation_tool_input_schema(request) if name == "validate_analysis" else _advertised_input_schema(request.model_json_schema())
    input_schema = _optimized_advertised_schema(input_schema)
    return Tool(
        name=name,
        description=description,
        input_schema=input_schema,
        output_schema=_hash_addressed_output_schema(response),
    )


def _compact_text(name: str, response: BaseModel) -> str:
    data = response.model_dump(mode="json")
    if name == "get_capabilities":
        if data.get("response_detail") == "output_contract":
            return ("Complete serialized tool-output contract returned for SHA-256 " f"{data['output_contract_sha256']} ({data['output_contract_utf8_bytes']} UTF-8 bytes).")[:_MAX_MODEL_TEXT]
        if data.get("response_detail") == "request_schema":
            return ("Complete validate_analysis request schema returned for " f"{data['task']} at SHA-256 {data['request_schema_sha256']} " f"({data['request_schema_utf8_bytes']} UTF-8 bytes).")[
                :_MAX_MODEL_TEXT
            ]
        if data.get("response_detail") in {"not_modified", "start_ready_not_modified"}:
            return (
                f"GeochemistryPi MCP capability view {data['capability_view_sha256']} is unchanged; "
                "the projection was not replayed. This receipt is useful only while the original "
                "structured payload is still retained."
            )[:_MAX_MODEL_TEXT]
        if data.get("response_detail") == "start_ready":
            return (
                f"Start-ready {data['task_filter']}.{data['method_filter']} request template returned. "
                "Replace every listed placeholder and preserve all user-supplied scientific values, then "
                "call validate_analysis directly when the dataset path and columns are known. Inspect only "
                "unknown dataset facts; full capabilities are audit-only. Request-schema SHA-256: "
                f"{data['request_schema_sha256']}."
            )[:_MAX_MODEL_TEXT]
        models_by_task = data.get("supported_models_by_task", {})
        model_count = len({model for models in models_by_task.values() for model in models})
        if not model_count:
            model_count = len(data.get("supported_models", ()))
        task_filter = data.get("task_filter")
        filter_text = f" for {task_filter}" if task_filter else ""
        if data.get("response_detail") == "full":
            conditional_guidance = (
                f"Complete audit-only snapshot SHA-256: {data['capabilities_sha256']}. "
                "It is not required for request construction. Reuse it with if_capabilities_sha256 "
                "only while its original structured payload is retained."
            )
        else:
            conditional_guidance = (
                f"Capability-view SHA-256: {data['capability_view_sha256']}. Reuse it with "
                "if_capability_view_sha256 for this exact compact projection only while its original "
                "structured payload is retained."
            )
        text = (
            f"GeochemistryPi MCP {data['server_version']} capability snapshot{filter_text} supports "
            f"{', '.join(data['supported_tasks'])} and indexes {model_count} model(s). "
            f"{conditional_guidance}"
        )
        request_contract = data.get("validation_request_contract")
        if request_contract is not None:
            text += " Use the returned task-level validate_analysis request contract once; " f"schema SHA-256: {request_contract['request_schema_sha256']}."
        return text[:_MAX_MODEL_TEXT]
    if name == "inspect_dataset":
        column_names = data["column_names"] or [column["name"] for column in data["columns"]]
        returned_column_names = column_names[:_MAX_COMPACT_DATASET_COLUMNS]
        columns = ", ".join(returned_column_names)
        column_guidance = " The complete ordered column list remains in structured_content." if len(returned_column_names) < len(column_names) else ""
        text = (
            f"Dataset: {data['row_count']} rows, {data['column_count']} columns. "
            f"Showing {len(returned_column_names)} of {len(column_names)} ordered columns: {columns}."
            f"{column_guidance} "
            f"Source SHA-256: {data['source_sha256']}. Prepared-view SHA-256: "
            f"{data['prepared_view_sha256']}."
        )
        return text[:_MAX_MODEL_TEXT]
    if name == "list_datasets":
        if data.get("response_detail") == "not_modified":
            return (f"GeochemistryPi dataset view {data['view_sha256']} is unchanged for source " f"{data['source_filter']}; the requested page was not replayed.")[:_MAX_MODEL_TEXT]
        continuation = f" Continue at offset {data['next_offset']}." if data.get("next_offset") is not None else " This is the final page."
        text = (
            f"Returned {data['returned_count']} of {data['total_count']} safe GeochemistryPi datasets "
            f"for source {data['source_filter']} at offset {data['offset']}.{continuation} "
            f"View SHA-256: {data['view_sha256']}."
        )
        return text[:_MAX_MODEL_TEXT]
    if name == "list_experiments":
        if data.get("response_detail") == "not_modified":
            return (f"MLflow experiment-directory view {data['view_sha256']} is unchanged; " "the requested page was not replayed.")[:_MAX_MODEL_TEXT]
        continuation = f" Continue at offset {data['next_offset']}." if data.get("next_offset") is not None else " This is the final page."
        return (f"Returned {data['returned_count']} of {data['total_count']} active persistent MLflow " f"experiments at offset {data['offset']}.{continuation} View SHA-256: {data['view_sha256']}.")[
            :_MAX_MODEL_TEXT
        ]
    if name == "get_experiment":
        if data.get("response_detail") == "not_modified":
            return (f"MLflow experiment {data['experiment_id']} run-history view {data['view_sha256']} is " "unchanged; the requested page was not replayed.")[:_MAX_MODEL_TEXT]
        continuation = f" Continue at offset {data['next_offset']}." if data.get("next_offset") is not None else " This is the final run page."
        return (
            f"Experiment {data['experiment']['experiment_id']} is {data['experiment']['name']}; "
            f"returned {data['returned_count']} of {data['total_count']} runs at offset "
            f"{data['offset']}.{continuation} View SHA-256: {data['view_sha256']}."
        )[:_MAX_MODEL_TEXT]
    if name in {"start_mlflow_ui", "mlflow_ui_status", "stop_mlflow_ui"}:
        location = f" at {data['url']}" if data.get("url") else ""
        return f"Managed MLflow UI is {data['state']}{location}. {data['message']}"[:_MAX_MODEL_TEXT]
    if name == "start_analysis":
        return (f"Queued GeochemistryPi run {data['run_id']} for {data['estimated_model_count']} model(s): " f"{', '.join(data['models'])}. {data['status_hint']}")[:_MAX_MODEL_TEXT]
    if name == "validate_analysis":
        if data.get("response_detail") == "full":
            if data["execution_ready"]:
                return (
                    f"Complete validation decision details recovered for {data['task']} at validation_id "
                    f"{data['validation_id']}; {data['artifact_requirement_count']} complete artifact "
                    "requirement(s) returned without repeating validation."
                )[:_MAX_MODEL_TEXT]
            blocker_text = "; ".join(data["blocking_issues"])
            return (
                f"Complete validation decision details recovered for blocked {data['task']} validation_id "
                f"{data['validation_id']}. Do not start it. Blocking issues: {blocker_text}. No validation "
                "or analysis process was repeated."
            )[:_MAX_MODEL_TEXT]
        readiness = data.get("readiness", {})
        if not readiness.get("execution_ready", data.get("execution_ready", False)):
            blocker_receipt = data.get("blocking_issues", {})
            blockers = blocker_receipt.get("prefix", ()) if isinstance(blocker_receipt, dict) else blocker_receipt
            blocker_values = tuple(blocker.get("text", "") if isinstance(blocker, dict) else str(blocker) for blocker in blockers)
            blocker_text = "; ".join(value for value in blocker_values if value) if blocker_values else "execution readiness was not established"
            if isinstance(blocker_receipt, dict) and blocker_receipt.get("truncated"):
                blocker_text += (
                    f" [showing {len(blocker_values)} of " f"{blocker_receipt.get('total_count', len(blocker_values))}; complete-sequence " f"SHA-256 {blocker_receipt.get('sha256', 'unavailable')}]"
                )
                next_step = (
                    "Read the complete stored validation once by calling validate_analysis with validation_id "
                    f"{data['validation_id']}, request_hash {data['request_hash']}, and detail=full. Do not repeat "
                    "the scientific validation or guess omitted blockers."
                )
            else:
                next_step = "Resolve every reported issue, then call validate_analysis once with the corrected request."
            return (f"Analysis validation is blocked for {data['task']}. Do not start this validation reference. " f"Blocking issues: {blocker_text}. {next_step} No analysis process was started.")[
                :_MAX_MODEL_TEXT
            ]
        detail_guidance = (
            " This compact record contains start-relevant truncated content; read the complete stored validation "
            f"once with validate_analysis(validation_id={data['validation_id']}, "
            f"request_hash={data['request_hash']}, detail=full) before deciding whether to start. "
            "Do not validate the same request again."
            if data.get("contains_truncated_content") and not data.get("start_relevant_content_complete", False)
            else (
                " Only the supplemental unselected observed-column inventory is truncated; all selected roles, "
                "scientific decisions, blockers, warnings, and artifact requirements needed to start are complete. "
                "Do not read full validation detail before starting unless the complete unselected-column inventory "
                "was explicitly requested."
                if data.get("contains_truncated_content")
                else ""
            )
        )
        return (
            f"Analysis is execution-ready for {data['task']} with {data['estimated_model_count']} model(s): "
            f"{', '.join(data['models'])}. Start this exact request with validation_id "
            f"{data['validation_id']} and request_hash {data['request_hash']}. No analysis process was started."
            f"{detail_guidance}"
        )[:_MAX_MODEL_TEXT]
    if name == "get_run_status":
        return f"Run {data['run_id']} is {data['state']} at stage {data['stage']}: {data['progress_message']}"[:_MAX_MODEL_TEXT]
    if name == "get_run_result":
        if data.get("response_detail") == "pending":
            return (
                f"Run {data['run_id']} is still {data['state']} at stage {data['stage']}; the bounded "
                f"{data['wait_seconds']}-second result wait ended without a terminal result. This is not a "
                "scientific failure. Do not infer results or poll tightly; make one later bounded result call "
                f"after at least {data['recommended_wait_seconds']} seconds if completion is still needed."
            )[:_MAX_MODEL_TEXT]
        if data.get("response_detail") == "artifact_page":
            continuation = f" Continue only if needed at artifact_offset {data['next_artifact_offset']}." if data.get("next_artifact_offset") is not None else " This is the final artifact page."
            return (
                f"Run {data['run_id']} additive artifact page {data['artifact_page_number']} returned "
                f"{data['returned_artifact_count']} {data['artifact_view']} receipt(s) at offset "
                f"{data['artifact_offset']} with limit {data['artifact_limit']}. Result SHA-256: "
                f"{data['result_record_sha256']}. Artifact-index SHA-256: {data['artifact_index_sha256']}. "
                f"Page SHA-256: {data['artifact_page_sha256']}.{continuation} Scientific core fields were "
                "not replayed."
            )[:_MAX_MODEL_TEXT]
        if data.get("response_detail") == "terminal":
            return (
                f"Run {data['run_id']} is terminal with state {data['state']}. "
                f"Scientific validity was not established, the artifact contract was not evaluated, "
                f"and verified artifacts are 0. Immutable terminal receipt SHA-256: "
                f"{data['result_record_sha256']}. Do not retry or infer scientific results from this receipt."
            )[:_MAX_MODEL_TEXT]
        if data.get("response_detail") == "not_modified":
            receipt_kind = " failure/cancellation receipt" if data.get("terminal_receipt") else " result"
            return (f"Run {data['run_id']} terminal{receipt_kind} is unchanged at SHA-256 " f"{data['result_record_sha256']}. Terminal payload was not replayed; no further result query is required.")[
                :_MAX_MODEL_TEXT
            ]
        preprocessing = data.get("preprocessing_summary")
        row_summary = (
            " Preprocessing rows: " f"input={preprocessing['input_row_count']}, " f"analysis={preprocessing['analysis_row_count']}, " f"dropped={preprocessing['dropped_row_count']}."
            if preprocessing is not None
            else ""
        )
        metrics_summary = (
            f" Compact metrics omitted {data['reported_metric_groups_omitted']} oversized group(s); " "the immutable result record contains the complete values."
            if data.get("reported_metrics_truncated")
            else " Metrics are included once in structured_content."
        )
        tabular = data.get("required_tabular_observations", {})
        tabular_observations = tabular.get("observations", ()) if isinstance(tabular, dict) else ()
        complete_table_count = sum(bool(item.get("rows_included")) for item in tabular_observations if isinstance(item, dict))
        tabular_summary = (
            f" Native table observations: {tabular.get('returned_count', 0)} of "
            f"{tabular.get('total_count', 0)} requirement-bound canonical table(s); complete rows are included "
            f"for {complete_table_count} small table(s), while large tables retain verified output row/column metadata. "
            f"Safely omitted non-tabular or over-budget artifacts: {tabular.get('omitted_artifact_count', 0)}."
            if isinstance(tabular, dict) and tabular.get("total_count", 0)
            else " No requirement-bound canonical native table was readable for this run."
        )
        missing_requirements = data.get("missing_artifact_requirement_ids", ())
        missing_requirement_count = data.get(
            "missing_artifact_requirement_ids_total_count",
            len(missing_requirements),
        )
        missing_requirements_truncated = data.get(
            "missing_artifact_requirement_ids_truncated",
            False,
        )
        contract_summary = (
            " Artifact contract is complete."
            if data.get("contract_status") == "complete"
            else (
                f" Artifact contract is incomplete; showing {len(missing_requirements)} of "
                f"{missing_requirement_count} missing required IDs: {', '.join(missing_requirements)}."
                + (
                    " The complete ordered ID list is preserved in the immutable result record at SHA-256 " f"{data['missing_artifact_requirement_ids_sha256']}."
                    if missing_requirements_truncated
                    else ""
                )
            )
        )
        compact_content_truncated = any(
            data.get(field, False)
            for field in (
                "reported_metrics_truncated",
                "missing_artifact_requirement_ids_truncated",
                "children_truncated",
                "limitations_truncated",
            )
        )
        if data.get("next_artifact_offset") is None:
            delivery_guidance = (
                " This compact terminal page is complete. Request detail=full only when the explicitly requested " "report requires omitted structured values; otherwise do not replay it."
                if compact_content_truncated
                else " This terminal view is complete; do not call get_run_result again."
            ) + " If an external confirmation is required, pass this SHA-256 as if_result_sha256 for a short " "unchanged receipt."
        else:
            delivery_guidance = (
                f" Additional {data['artifact_view']} artifact receipts remain at artifact_offset " f"{data['next_artifact_offset']}; request that page only if those additional receipts are needed."
            )
        text = (
            f"Run {data['run_id']} is terminal with state {data['state']}. Original output: {data['output_directory']}. "
            f"Artifacts returned: {data['returned_artifact_count']} of {data['artifact_view_count']} in the "
            f"{data['artifact_view']} view; the complete immutable index contains {data['artifact_count']} and "
            f"{data['summary_mirror_count']} summary mirror(s). Result SHA-256: {data['result_record_sha256']}."
            f"{contract_summary}{row_summary}{metrics_summary}{tabular_summary}{delivery_guidance}"
        )
        return text[:_MAX_MODEL_TEXT]
    return f"Run {data['run_id']}: {data['message']}"[:_MAX_MODEL_TEXT]


def _bounded_actual(value: Any) -> str:
    if isinstance(value, dict):
        return f"object with {len(value)} field(s)"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__} with {len(value)} item(s)"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return f"string with {len(value)} character(s)"
    return "value of an unsupported type"


def _bounded_error_identifier(value: str) -> str:
    """Bound a sanitized field path without making long paths indistinguishable."""
    if len(value) <= _MAX_PUBLIC_ERROR_FIELD_TEXT:
        return value
    suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{value[: _MAX_PUBLIC_ERROR_FIELD_TEXT - len(suffix) - 2]}~{suffix}"


def _safe_error_field(error: dict[str, Any]) -> str:
    parts = []
    for part in error.get("loc", ()):
        if isinstance(part, int):
            parts.append(str(part))
            continue
        text = str(part)
        parts.append(text if _SAFE_FIELD_PART.fullmatch(text) else "unknown_field")
    return _bounded_error_identifier(".".join(parts) or "request")


def _validation_error_kind(error_type: str) -> str:
    if error_type == "missing":
        return "missing"
    if error_type == "extra_forbidden":
        return "extra_forbidden"
    if error_type == "string_pattern_mismatch":
        return "pattern"
    if error_type in {
        "greater_than",
        "greater_than_equal",
        "less_than",
        "less_than_equal",
    }:
        return "range"
    if error_type in {"literal_error", "union_tag_invalid", "union_tag_not_found"}:
        return "literal"
    if error_type.endswith("_type") or error_type.endswith("_parsing"):
        return "type"
    return "value"


def _validation_problem(kind: str) -> str:
    return {
        "missing": "Required field is missing",
        "extra_forbidden": "Extra input is not permitted",
        "pattern": "String does not match the declared pattern",
        "range": "Value is outside the declared range",
        "literal": "Value is not one of the declared literals",
        "type": "Value has the wrong JSON type",
        "value": "Value does not satisfy the declared field contract",
    }[kind]


def _validation_alternative(field: str, kind: str) -> str:
    leaf = field.rsplit(".", 1)[-1]
    if field.startswith("time_series.") and kind == "extra_forbidden":
        replacement = {
            "dataset": "remove it and use top-level 'training_dataset'",
            "model": "remove it because the Time Series workflow has a fixed model",
            "model_parameters": "remove it and place supported fields such as 'bin_width', 'iterations', and 'seed' at the top level",
            "bin_width_ma": "remove it and use top-level 'bin_width'",
            "bootstrap_iterations": "remove it and use top-level 'iterations'",
            "random_seed": "remove it and use top-level 'seed'",
        }.get(leaf)
        if replacement is not None:
            return replacement
    if leaf == "dataset" and kind == "extra_forbidden":
        return "remove it and provide exactly one top-level training_dataset reference " "(with source='path', 'builtin', or 'desktop') or training_dataset_path"
    if "training_dataset" in field:
        if leaf == "source":
            return "set training_dataset.source to 'path', 'builtin', or 'desktop' and " "supply that variant's declared fields"
        return "provide exactly one top-level training_dataset reference (discriminated by " "source) or top-level training_dataset_path, never both"
    if "application_dataset" in field:
        return "when application data is supported, provide at most one top-level " "application_dataset reference (discriminated by source) or " "application_dataset_path"
    if leaf == "model_selection" or ".model_selection." in field:
        return "use top-level model_selection.mode as the 'single' or 'all' discriminator"
    if leaf == "model_parameters" and kind == "extra_forbidden":
        return "remove it, set the discriminator at top-level model.type, and place " "supported parameters beside type inside that model object"
    if ".model." in field or field.endswith(".model") or leaf == "model":
        return "use the top-level model object, set model.type to one supported model " "discriminator, and place that model's parameters beside type"
    if leaf in {"random_seed", "split_seed", "model_seed", "tuning_seed"}:
        if field.startswith("time_series.") or leaf == "random_seed":
            return "for Time Series use top-level seed; for other tasks place split_seed, " "model_seed, or tuning_seed inside top-level reproducibility"
        return f"place {leaf} inside the top-level reproducibility object"
    if leaf == "reproducibility":
        return "use the top-level reproducibility object; place task-supported seed fields " "inside it (Time Series instead uses top-level seed)"
    if kind == "missing":
        return "provide this required field at the location shown"
    if kind == "extra_forbidden":
        return "remove this unsupported field"
    if kind == "pattern" and field.endswith("dataset_id"):
        return "use the exact 'builtin:<id>' value returned by list_datasets, including its prefix"
    if kind == "pattern":
        return "use a value matching the field's declared pattern"
    if kind == "range":
        return "use a value within the field's declared minimum and maximum"
    if kind == "literal":
        return "use one of the field's declared literal values"
    if kind == "type":
        return "use the field's declared JSON type"
    return "inspect the field description and supported capabilities"


def _validation_root_field(field: str) -> str:
    """Collapse array indexes so repeated element failures share one cause."""
    return _bounded_error_identifier(".".join(part for part in field.split(".") if not part.isdecimal()))


def _validation_issue(error: dict[str, Any]) -> dict[str, Any]:
    location = _safe_error_field(error)
    field = _validation_root_field(location)
    error_type = str(error.get("type", ""))
    kind = _validation_error_kind(error_type)
    return {
        "field": field,
        "kind": kind,
        "problem": _validation_problem(kind),
        "valid_alternative": _validation_alternative(field, kind),
        "locations": [location],
        "actual_value_summaries": [_bounded_actual(error.get("input"))],
        "occurrences": 1,
    }


def _redacted_public_error(exc: BaseException) -> str:
    message = " ".join(str(exc).split())
    message = _WINDOWS_LOCAL_PATH.sub("<local-path>", message)
    message = _POSIX_LOCAL_PATH.sub("<local-path>", message)
    return message or type(exc).__name__


def _validation_root_causes(exc: ValidationError) -> list[dict[str, Any]]:
    """Return every independent validation cause with repeated items grouped."""
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for error in exc.errors(include_url=False):
        issue = _validation_issue(error)
        key = (issue["field"], issue["kind"])
        cause = grouped.setdefault(key, issue)
        if cause is issue:
            continue
        cause["occurrences"] += 1
        for location in issue["locations"]:
            if location not in cause["locations"]:
                cause["locations"].append(location)
        for actual in issue["actual_value_summaries"]:
            if actual not in cause["actual_value_summaries"]:
                cause["actual_value_summaries"].append(actual)
    return [grouped[key] for key in sorted(grouped)]


def _canonical_error_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _bounded_utf8_text(value: str, maximum_bytes: int) -> str:
    """Return a valid UTF-8 prefix, with an omission marker, within one budget."""
    encoded = value.encode("utf-8")
    if len(encoded) <= maximum_bytes:
        return value
    marker = " …"
    marker_bytes = marker.encode("utf-8")
    if maximum_bytes <= len(marker_bytes):
        return marker_bytes[:maximum_bytes].decode("utf-8", errors="ignore")
    prefix = encoded[: maximum_bytes - len(marker_bytes)].decode(
        "utf-8",
        errors="ignore",
    )
    return f"{prefix}{marker}"


def _bounded_public_error_problem(value: str) -> tuple[str, bool]:
    """Bound the displayed problem while the complete cause stays hash-bound."""
    if len(value) <= MAX_PUBLIC_TOOL_ERROR_PROBLEM_CHARS:
        return value, False
    marker = "…"
    complete_size = len(value.encode("utf-8"))
    maximum_prefix_bytes = complete_size - len(marker.encode("utf-8")) - 1
    prefix = value[: MAX_PUBLIC_TOOL_ERROR_PROBLEM_CHARS - 1]
    prefix = prefix.encode("utf-8")[:maximum_prefix_bytes].decode(
        "utf-8",
        errors="ignore",
    )
    return f"{prefix}{marker}", True


def _project_public_root_cause(cause: dict[str, Any], *, bound_repeated_details: bool) -> dict[str, Any]:
    locations = list(cause["locations"])
    actuals = list(cause["actual_value_summaries"])
    complete_problem = cause["problem"]
    complete_problem_bytes = complete_problem.encode("utf-8")
    if bound_repeated_details:
        problem, problem_truncated = _bounded_public_error_problem(complete_problem)
    else:
        problem, problem_truncated = complete_problem, False
    returned_locations = locations[:MAX_PUBLIC_TOOL_ERROR_LOCATIONS] if bound_repeated_details else locations
    returned_actuals = actuals[:MAX_PUBLIC_TOOL_ERROR_ACTUAL_VALUE_SUMMARIES] if bound_repeated_details else actuals
    return {
        "field": cause["field"],
        "kind": cause["kind"],
        "problem": problem,
        "problem_truncated": problem_truncated,
        "problem_sha256": hashlib.sha256(complete_problem_bytes).hexdigest(),
        "problem_total_utf8_bytes": len(complete_problem_bytes),
        "valid_alternative": cause["valid_alternative"],
        "locations": returned_locations,
        "locations_total_count": len(locations),
        "locations_truncated": len(returned_locations) < len(locations),
        "locations_sha256": _canonical_error_sha256(locations),
        "actual_value_summaries": returned_actuals,
        "actual_value_summaries_total_count": len(actuals),
        "actual_value_summaries_truncated": len(returned_actuals) < len(actuals),
        "actual_value_summaries_sha256": _canonical_error_sha256(actuals),
        "occurrences": cause["occurrences"],
    }


def _public_error_payload(
    *,
    error_type: str,
    result_type: str,
    retryable: bool,
    root_causes: list[dict[str, Any]],
    next_action: str,
) -> dict[str, Any]:
    full_projection = [_project_public_root_cause(cause, bound_repeated_details=False) for cause in root_causes]
    returned = [_project_public_root_cause(cause, bound_repeated_details=True) for cause in root_causes[:MAX_PUBLIC_TOOL_ERROR_ROOT_CAUSES]]
    return {
        "error_schema_version": 2,
        "error_type": error_type,
        "result_type": result_type,
        "retryable": retryable,
        "root_cause_count": len(returned),
        "root_causes": returned,
        "root_causes_total_count": len(root_causes),
        "root_causes_truncated": len(returned) < len(root_causes),
        "root_causes_sha256": _canonical_error_sha256(full_projection),
        "next_action": next_action,
    }


def _bounded_validation_message(root_causes: list[dict[str, Any]], next_action: str) -> str:
    returned = root_causes[:MAX_PUBLIC_TOOL_ERROR_ROOT_CAUSES]
    issue_text = " ".join(f"{index}. {_validation_cause_text(cause)}" for index, cause in enumerate(returned, start=1))
    header = f"Validation failed with {len(root_causes)} independent root cause(s); " f"the structured error returns {len(returned)} actionable representative(s): "
    projection_note = " The remaining root causes are represented by total-count and SHA-256 metadata." if len(returned) < len(root_causes) else ""
    suffix = f"{projection_note} {next_action}"
    issue_budget_bytes = max(
        0,
        _MAX_PUBLIC_ERROR_MESSAGE_BYTES - len(header.encode("utf-8")) - len(suffix.encode("utf-8")),
    )
    issue_text = _bounded_utf8_text(issue_text, issue_budget_bytes)
    return f"{header}{issue_text}{suffix}"


def _validation_cause_text(cause: dict[str, Any]) -> str:
    occurrence_text = f" ({cause['occurrences']} occurrences)" if cause["occurrences"] > 1 else ""
    actual = ", ".join(cause["actual_value_summaries"])
    return f"[{cause['kind']}] Invalid field '{cause['field']}'{occurrence_text}. " f"Actual value: {actual}. Problem: {cause['problem']}. " f"Valid alternatives: {cause['valid_alternative']}."


def _bounded_request_error_message(
    message: str,
    next_action: str,
    *,
    header: str = "Invalid analysis configuration. Problem: ",
    valid_alternatives: str = "inspect the dataset for exact columns and check the supported scientific workflows",
) -> str:
    """Reserve useful alternatives and next-action guidance in bounded text."""
    suffix = f". Valid alternatives: {valid_alternatives}. Next action: {next_action}"
    problem_budget_bytes = max(
        0,
        _MAX_PUBLIC_ERROR_MESSAGE_BYTES - len(header.encode("utf-8")) - len(suffix.encode("utf-8")),
    )
    message = _bounded_utf8_text(message, problem_budget_bytes)
    return f"{header}{message}{suffix}"


def _actionable_error(exc: BaseException) -> tuple[str, dict[str, Any]]:
    if isinstance(exc, ValidationError):
        root_causes = _validation_root_causes(exc)
        next_action = (
            "Next action: Resolve it from the conversation or earlier tool results when possible, "
            "use one get_capabilities detail='start_ready' response for the selected task and method when field "
            "placement is unclear, apply every listed fix in one request, and validate once. If the dataset path "
            "and exact columns are already known, do not inspect them again or request full capabilities. "
            "Do not probe the schema with repeated guessed requests. Only if an unresolved scientific "
            "choice belongs to the user, ask one plain-language question; never ask the user for a schema "
            "field name or JSON, and do not ask the user to edit JSON."
        )
        return _bounded_validation_message(root_causes, next_action), _public_error_payload(
            error_type="validation_error",
            result_type="invalid_arguments",
            retryable=False,
            root_causes=root_causes,
            next_action=next_action,
        )
    typed_error: tuple[str, bool, str, str, str] | None = None
    if isinstance(exc, InputIntegrityError):
        typed_error = (
            "input_integrity_changed",
            False,
            "dataset",
            "the exact selected dataset identity after all writes have stopped",
            "Do not reuse this validation receipt or retry the same start. Wait until all dataset writes stop, "
            "inspect or resolve the exact selected input again, call validate_analysis once, and start only with "
            "the new validation_id and request_hash.",
        )
    elif isinstance(exc, EnvironmentInspectionError):
        typed_error = (
            "environment_inspection_failed",
            False,
            "cli_environment",
            "a healthy installer-owned CLI environment verified by geochemistrypi-mcp-doctor",
            "Do not change scientific parameters or retry the same request. Run geochemistrypi-mcp-doctor, repair "
            "or reinstall the declared CLI environment if it fails, restart the MCP server, and then call "
            "validate_analysis once.",
        )
    elif isinstance(exc, CapabilityManifestError):
        typed_error = (
            "capability_manifest_invalid",
            False,
            "capability_manifest",
            "the exact installed MCP package with a doctor-verified capability manifest",
            "Do not infer capabilities or start an analysis. Repair or reinstall the exact GeochemistryPi MCP "
            "package, run geochemistrypi-mcp-doctor, restart the MCP server, and call get_capabilities once.",
        )
    elif isinstance(exc, RunNotFoundError):
        typed_error = (
            "run_not_found",
            False,
            "run_id",
            "the exact run_id returned by the successful start_analysis acknowledgement",
            "Do not guess run IDs, retry this lookup, or start a replacement run automatically. Use the exact "
            "run_id retained from start_analysis; if that acknowledgement is unavailable, stop and report that "
            "the run cannot be recovered through the public protocol.",
        )
    elif isinstance(exc, RunStateError):
        typed_error = (
            "run_state_invalid",
            False,
            "run_state",
            "the exact durable run or validation state and its immutable identity hashes",
            "Do not repeat the failed state-changing call or alter scientific parameters. If a run_id exists, "
            "read get_run_status or get_run_result once as directed by the error; if a validation reference is "
            "expired or changed, validate the unchanged scientific request once and use only the new reference.",
        )
    elif isinstance(exc, DatasetCatalogError):
        typed_error = (
            "dataset_catalog_failed",
            False,
            "dataset_reference",
            "an exact dataset_id and source returned by one current list_datasets response for the requested task and role",
            "Do not guess a dataset ID, path, role, or hash. Call list_datasets once for the intended source, "
            "select an exact returned reference that satisfies the user's request, and stop if no such dataset is "
            "listed; preserve every user-specified scientific parameter.",
        )
    elif isinstance(exc, DatasetInspectionError):
        typed_error = (
            "dataset_inspection_failed",
            False,
            "dataset_path_or_reference",
            "one stable supported CSV/XLSX file given by an absolute path, a safe startup-relative path, or an exact catalog reference",
            "Do not repeat the same inspection or invent a replacement path. Correct the reported path, format, "
            "stability, size, or worksheet issue once from known user or catalog evidence, then inspect once; stop "
            "if the exact input cannot be satisfied.",
        )
    elif isinstance(exc, DatasetPreparationError):
        typed_error = (
            "dataset_preparation_failed",
            False,
            "dataset.preparation",
            "a preparation object using exact observed worksheet, header, column, filter, and row-identity values from the selected task contract",
            "Do not insert defaults, rename columns, or retry unchanged preparation. Use the task-filtered "
            "validation_request_contract and exact dataset evidence to correct all reported preparation fields in "
            "one request; ask the user only when a scientific preparation choice is genuinely missing.",
        )
    elif isinstance(exc, PlanCompilationError):
        typed_error = (
            "plan_compilation_failed",
            False,
            "scientific_contract",
            "one scientifically supported combination declared by the selected task contract and capability boundaries",
            "Do not substitute a model, workflow, column role, preprocessing step, or evaluation rule and do not "
            "retry the same plan. Correct only contradictions identified by the task contract, then validate once; "
            "if the exact requested adapter is unavailable, stop and report that boundary.",
        )
    elif isinstance(exc, CliDriverError):
        typed_error = (
            "cli_execution_failed",
            False,
            "cli_execution",
            "a doctor-verified CLI runtime and a newly authorized run after the failed run evidence is preserved",
            "Preserve the run directory and terminal evidence. Do not retry, repair, or replace the same measured "
            "run. Read its terminal status/result once, run geochemistrypi-mcp-doctor outside the run if the error "
            "indicates an environment fault, and create a new run only with explicit authorization.",
        )
    elif isinstance(exc, ExperimentStoreError):
        typed_error = (
            "experiment_store_failed",
            False,
            "experiment_id_or_tracking_store",
            "an exact experiment_id returned by list_experiments and a healthy persistent tracking store",
            "Do not guess an experiment ID or silently create a replacement experiment. List experiments once when "
            "identity is uncertain; if the tracking store is unreadable or inconsistent, preserve it, run Doctor "
            "or repair outside the scientific request, restart MCP, and retry only after the store is healthy.",
        )
    elif isinstance(exc, MlflowUiError):
        typed_error = (
            "mlflow_ui_failed",
            False,
            "mlflow_ui_state",
            "one verified managed MLflow UI state using the configured local tracking root and available port",
            "Do not loop start/stop calls or change the scientific request. Read mlflow_ui_status once, correct the "
            "reported managed-process, port, or tracking configuration outside analysis, and retry one UI action "
            "only after that condition is resolved.",
        )
    elif isinstance(exc, ContractLookupError):
        typed_error = (
            "contract_not_found",
            False,
            "contract_sha256",
            "an exact current request-schema or output-contract SHA-256 published by this server",
            "Do not guess hashes or repeat the failed lookup. Request one current start_ready view for the "
            "selected task and method, or refresh tools/list for an output contract, then use the exact "
            "published resolver arguments once.",
        )
    elif isinstance(exc, SettingsError):
        typed_error = (
            "settings_invalid",
            False,
            "mcp_settings",
            "a complete installer-owned configuration verified by geochemistrypi-mcp-doctor",
            "Do not change scientific parameters or repeat the failed tool call. Run geochemistrypi-mcp-doctor, "
            "repair the reported MCP configuration, restart the MCP server, and retry only after Doctor succeeds.",
        )
    elif isinstance(exc, DirectoryViewError):
        typed_error = (
            "directory_view_rejected",
            True,
            "detail_or_limit",
            "detail=compact, or a smaller limit when several full records share the page",
            "Call the same directory tool once with detail=compact, or reduce limit for a multi-record full page. "
            "If one lossless full record alone exceeds the declared full-detail budget, reduce that source "
            "record's metadata before requesting full detail again; no legacy field was silently dropped.",
        )
    if typed_error is not None:
        result_type, retryable, field, valid_alternative, next_action = typed_error
        message = _redacted_public_error(exc)
        root_causes = [
            {
                "field": field,
                "kind": "value",
                "problem": message,
                "valid_alternative": valid_alternative,
                "locations": [field],
                "actual_value_summaries": [],
                "occurrences": 1,
            }
        ]
        return _bounded_request_error_message(message, next_action, header=f"{field.replace('_', ' ').title()} check failed. Problem: ", valid_alternatives=valid_alternative,), _public_error_payload(
            error_type="request_error",
            result_type=result_type,
            retryable=retryable,
            root_causes=root_causes,
            next_action=next_action,
        )
    message = _redacted_public_error(exc)
    next_action = (
        "Do not retry automatically, invent defaults, or alter scientific parameters. Preserve the exact public "
        "error and stop; this installed public-error category has no more specific recovery contract."
    )
    root_causes = [
        {
            "field": "request",
            "kind": "value",
            "problem": message,
            "valid_alternative": "the exact documented public contract for this error category",
            "locations": ["request"],
            "actual_value_summaries": [],
            "occurrences": 1,
        }
    ]
    return _bounded_request_error_message(message, next_action), _public_error_payload(
        error_type="request_error",
        result_type="request_rejected",
        retryable=False,
        root_causes=root_causes,
        next_action=next_action,
    )


def build_tool_handlers(
    settings: McpSettings,
    runs: RunManager,
    experiments: ExperimentManager | None = None,
    mlflow_ui: MlflowUiManager | None = None,
) -> tuple[Callable[[ServerRequestContext[Any], PaginatedRequestParams | None], Awaitable[ListToolsResult],], Callable[[ServerRequestContext[Any], CallToolRequestParams], Awaitable[CallToolResult]],]:
    """Build strict schemas and a sanitized dispatcher for one server."""
    experiment_store = experiments or getattr(runs, "experiment_manager", None) or ExperimentManager(settings)
    ui_manager = mlflow_ui or MlflowUiManager(settings)
    analysis_schema_task_scope = os.environ.get(ANALYSIS_SCHEMA_TASK_ENV)
    if analysis_schema_task_scope is None:
        advertised_analysis_request_model: type[BaseModel] = AnalysisRequest
    else:
        try:
            advertised_analysis_request_model = _ANALYSIS_REQUEST_MODELS[analysis_schema_task_scope]
        except KeyError as exc:
            allowed = ", ".join(_ANALYSIS_REQUEST_MODELS)
            raise SettingsError(f"{ANALYSIS_SCHEMA_TASK_ENV} must be one of: {allowed}.") from exc
    definitions = (
        _tool(
            "get_capabilities",
            "Construct a request with one small detail='start_ready' call using the selected task and public "
            "method or workflow mode. detail='compact' retains the backward-compatible task index/exact schema; "
            "detail='full' is audit-only. Send a conditional hash only while retaining the original payload. "
            "Exact request and output schemas remain losslessly hash-resolvable. All tools and tasks remain available.",
            CapabilitiesRequest,
            (
                CapabilitiesResponse,
                CompactCapabilitiesResponse,
                CapabilitiesNotModifiedResponse,
                StartReadyCapabilitiesResponse,
                StartReadyCapabilitiesNotModifiedResponse,
                RequestSchemaResponse,
                OutputContractSchemaResponse,
            ),
        ),
        _tool(
            "list_datasets",
            "List trusted built-in and Desktop datasets in deterministic bounded pages. Compact detail is the "
            "default and omits absolute installation paths; full detail retains every legacy catalog field. "
            "Reuse view_sha256 only with the exact same source, detail, offset, and limit for an unchanged receipt.",
            ListDatasetsViewRequest,
            (
                CompactListDatasetsResponse,
                FullListDatasetsResponse,
                ListDatasetsNotModifiedResponse,
            ),
        ),
        _tool(
            "inspect_dataset",
            "Inspect one bounded dataset. Names-only is the default and returns columns, counts, and separate "
            "source/prepared hashes. Skip inspection and validate directly when the exact path and column roles "
            "are already known; request full detail only for types, samples, or the complete preparation record.",
            DatasetInspectionRequest,
            (DatasetInspectionResponse, CompactDatasetInspectionResponse),
        ),
        _tool(
            "list_experiments",
            "List MLflow experiments by stable ID in deterministic bounded pages. Compact detail is the default; "
            "full detail retains tracking locations and all tags. Reuse view_sha256 only with the exact same "
            "query, detail, offset, and limit for an unchanged receipt.",
            ListExperimentsViewRequest,
            (
                CompactListExperimentsResponse,
                FullListExperimentsResponse,
                ListExperimentsNotModifiedResponse,
            ),
        ),
        _tool(
            "get_experiment",
            "Read one experiment and a deterministic bounded page of recent runs. Compact detail is the default; "
            "full detail retains tracking locations, tags, metrics, parameters, and artifact paths. Reuse "
            "view_sha256 only with the exact same experiment, detail, offset, and limit for an unchanged receipt.",
            GetExperimentViewRequest,
            (
                CompactGetExperimentResponse,
                FullGetExperimentResponse,
                GetExperimentNotModifiedResponse,
            ),
        ),
        _tool(
            "start_mlflow_ui",
            "Start the managed local MLflow UI.",
            StartMlflowUiRequest,
            MlflowUiStatusResponse,
        ),
        _tool(
            "mlflow_ui_status",
            "Read managed MLflow UI state.",
            EmptyRequest,
            MlflowUiStatusResponse,
        ),
        _tool(
            "stop_mlflow_ui",
            "Stop the verified managed MLflow UI.",
            EmptyRequest,
            MlflowUiStatusResponse,
        ),
        _tool(
            "validate_analysis",
            "Fail-closed preview of one of six scientific workflows. Starts no process and returns a compact "
            "immutable validation reference plus every execution-readiness decision. Read detail=full once only "
            "when start_relevant_content_complete is false or a complete supplemental record is explicitly needed; "
            "that reads the stored full validation record without validating again.",
            advertised_analysis_request_model,
            (
                CompactAnalysisValidationResponse,
                FullAnalysisValidationDetailResponse,
            ),
        ),
        _tool(
            "start_analysis",
            "Start the exact immutable validation reference. A legacy complete scientific request remains accepted " "for compatibility.",
            StartAnalysisByValidationRequest,
            StartAnalysisResponse,
        ),
        _tool(
            "get_run_status",
            "Read state or wait once for at most 300 seconds.",
            RunStatusRequest,
            RunStatusResponse,
        ),
        _tool(
            "get_run_result",
            "Wait once for a terminal receipt. Active runs return pending. Success defaults to compact canonical "
            "artifacts plus bounded native-table observations; offsets are additive artifact-only pages. "
            "Failure/cancellation returns a bounded non-scientific receipt. Use full/all only for explicit replay.",
            RunResultRequest,
            (
                RunResultResponse,
                CompactRunResultResponse,
                CompactRunArtifactPageResponse,
                RunResultNotModifiedResponse,
                PendingRunResultResponse,
                TerminalRunReceipt,
                TerminalRunNotModifiedResponse,
            ),
        ),
        _tool(
            "cancel_run",
            "Cancel the recorded live CLI process tree.",
            RunLookupRequest,
            CancelRunResponse,
        ),
    )
    request_models: dict[str, type[BaseModel]] = {
        "get_capabilities": CapabilitiesRequest,
        "list_datasets": ListDatasetsViewRequest,
        "inspect_dataset": DatasetInspectionRequest,
        "list_experiments": ListExperimentsViewRequest,
        "get_experiment": GetExperimentViewRequest,
        "validate_analysis": AnalysisRequest,
        "start_mlflow_ui": StartMlflowUiRequest,
        "mlflow_ui_status": EmptyRequest,
        "stop_mlflow_ui": EmptyRequest,
        "start_analysis": StartAnalysisByValidationRequest,
        "get_run_status": RunStatusRequest,
        "get_run_result": RunResultRequest,
        "cancel_run": RunLookupRequest,
    }

    def capabilities(request: CapabilitiesRequest) -> BaseModel:
        if request.output_contract_sha256 is not None:
            try:
                output_contract_schema = full_output_contract_schema(request.output_contract_sha256)
            except KeyError as exc:
                raise SettingsError("The requested output contract is not registered by this server; " "refresh tools/list and use one of its current contract SHA-256 values.") from exc
            encoded = _canonical_schema(output_contract_schema).encode("utf-8")
            return OutputContractSchemaResponse(
                output_contract_sha256=request.output_contract_sha256,
                output_contract_utf8_bytes=len(encoded),
                output_contract_schema=output_contract_schema,
            )
        if request.request_schema_sha256 is not None:
            return _request_schema_response(request.request_schema_sha256)
        manifest = load_capability_manifest()
        capability_requirements = (
            "command:time-series",
            "command:reference-anomaly-time-series",
            "command:embedding-label-overlay",
            "option:data-mining:--scientific-config",
        )
        cli_resolver = getattr(runs, "cli_resolver", None)
        try:
            cli_executable = cli_resolver()[0] if callable(cli_resolver) else None
        except Exception:
            cli_executable = None
        cli_probe = probe_cli_capabilities(cli_executable, capability_requirements) if cli_executable is not None and cli_executable.is_file() else probe_cli_capabilities(Path("."), ())
        available_cli_capabilities = set(cli_probe.available)
        executable_time_series_modes = tuple(
            mode
            for mode, requirement in (
                ("subaerial_proportion", "command:time-series"),
                ("continuous", "command:time-series"),
                (
                    "reference_anomaly_series",
                    "command:reference-anomaly-time-series",
                ),
            )
            if requirement in available_cli_capabilities
        )
        scientific_time_series_modes = (
            "subaerial_proportion",
            "continuous",
            "element_mean",
            "reference_anomaly_series",
        )
        schema_only_time_series_modes = tuple(mode for mode in scientific_time_series_modes if mode not in executable_time_series_modes)
        scientific_attestation = ScientificAttestationCapabilities(
            v4_attested_methods_by_task=dict(SCIENTIFIC_EXECUTION_METHODS_BY_TASK),
            legacy_methods_without_v4_attestation_by_task=dict(LEGACY_METHODS_WITHOUT_V4_ATTESTATION_BY_TASK),
        )
        if sum(len(methods) for methods in PUBLIC_MANUAL_METHODS_BY_TASK.values()) != scientific_attestation.public_manual_method_count:
            raise RuntimeError("The public capability method registry no longer matches the 36-method contract.")
        full_response = CapabilitiesResponse(
            server_name=SERVER_NAME,
            server_version=SERVER_VERSION,
            supported_cli_versions=SUPPORTED_CLI_VERSIONS,
            supported_tasks=(
                "classification",
                "regression",
                "clustering",
                "decomposition",
                "anomaly_detection",
                "time_series",
            ),
            analysis_schema_task_scope=analysis_schema_task_scope,
            supported_models=tuple(
                dict.fromkeys(
                    (
                        *CLASSIFICATION_MODEL_ORDER,
                        *REGRESSION_MODEL_ORDER,
                        *CLUSTERING_MODEL_ORDER,
                        *DECOMPOSITION_MODEL_ORDER,
                        *ANOMALY_DETECTION_MODEL_ORDER,
                    )
                )
            ),
            supported_dataset_formats=("csv", "xlsx"),
            maximum_dataset_bytes=settings.maximum_dataset_bytes,
            default_concurrency=settings.concurrency,
            capability_manifest_schema_version=manifest["schema_version"],
            capability_manifest_id=manifest["manifest_id"],
            cli_automation_contract_version=CLI_AUTOMATION_CONTRACT_VERSION,
            capabilities=public_capabilities(),
            known_gaps=known_gap_ids(),
            supported_data_sources=("path", "builtin", "desktop"),
            supported_clients=SUPPORTED_CLIENTS,
            compatibility=CompatibilityPolicy(
                schema_version=COMPATIBILITY_POLICY_VERSION,
                release_channel=RELEASE_CHANNEL,
                public_release_ready=PUBLIC_RELEASE_READY,
                mcp_python_requires=MCP_PYTHON_REQUIRES,
                cli_python_requires=CLI_PYTHON_REQUIRES,
                mcp_sdk_requires=MCP_SDK_REQUIRES,
                supported_cli_versions=SUPPORTED_CLI_VERSIONS,
                interaction_plan_version=INTERACTION_PLAN_VERSION,
                cli_automation_contract_version=CLI_AUTOMATION_CONTRACT_VERSION,
                artifact_index_schema_version=ARTIFACT_INDEX_SCHEMA_VERSION,
                target_operating_systems=TARGET_OPERATING_SYSTEMS,
                pending_release_gates=PENDING_RELEASE_GATES,
            ),
            resource_limits=ResourceLimits(
                maximum_dataset_bytes=settings.maximum_dataset_bytes,
                maximum_columns=settings.maximum_columns,
                maximum_artifact_references=settings.maximum_artifact_references,
                maximum_concurrent_runs=settings.concurrency,
                maximum_pending_runs=settings.maximum_pending_runs,
                maximum_process_seconds=settings.maximum_process_seconds,
            ),
            classification_options={
                "label_customization": LABEL_STRATEGIES,
                "missing_values": MISSING_VALUE_METHODS,
                "scaling": SCALING_METHODS,
                "feature_selection": FEATURE_SELECTION_METHODS,
                "tuning": TUNING_MODES,
                "application_data": ("disabled", "enabled"),
                "sample_balancing": ("none",),
                "model_selection": ("single", "all"),
                "split_strategy": ("stratified_holdout", "random_holdout"),
                "xgboost_objective": (
                    "auto",
                    "binary:logistic",
                    "multi:softprob",
                    "multi:softmax",
                ),
                "xgboost_importance_type": (
                    "gain",
                    "weight",
                    "cover",
                    "total_gain",
                    "total_cover",
                ),
                "scientific_methods": SCIENTIFIC_EXECUTION_METHODS_BY_TASK["classification"],
            },
            regression_options={
                "missing_values": MISSING_VALUE_METHODS,
                "scaling": SCALING_METHODS,
                "feature_selection": FEATURE_SELECTION_METHODS,
                "tuning": TUNING_MODES,
                "automl_models": tuple(model for model in REGRESSION_MODEL_ORDER if model not in MODELS_WITHOUT_AUTOML),
                "application_data": ("disabled", "enabled"),
                "target_columns": ("single_numeric", "multiple_numeric"),
                "model_selection": ("single", "all"),
                "scientific_methods": SCIENTIFIC_EXECUTION_METHODS_BY_TASK["regression"],
            },
            clustering_options={
                "missing_values": CLUSTERING_MISSING_VALUE_METHODS,
                "scaling": CLUSTERING_SCALING_METHODS,
                "application_data": ("disabled",),
                "target_columns": ("not_applicable",),
                "tuning": ("not_applicable",),
                "model_selection": ("single", "all"),
                "scientific_methods": SCIENTIFIC_EXECUTION_METHODS_BY_TASK["clustering"],
            },
            decomposition_options={
                "missing_values": DECOMPOSITION_MISSING_VALUE_METHODS,
                "scaling": DECOMPOSITION_SCALING_METHODS,
                "application_data": ("disabled",),
                "target_columns": ("not_applicable",),
                "tuning": ("not_applicable",),
                "transformed_data": ("enabled",),
                "model_selection": ("single", "all"),
                "scientific_methods": SCIENTIFIC_EXECUTION_METHODS_BY_TASK["decomposition"],
                "artifact_composition": (("embedding_label_overlay",) if "command:embedding-label-overlay" in available_cli_capabilities else ()),
            },
            anomaly_detection_options={
                "missing_values": ANOMALY_DETECTION_MISSING_VALUE_METHODS,
                "scaling": ANOMALY_DETECTION_SCALING_METHODS,
                "application_data": ("disabled",),
                "target_columns": ("not_applicable",),
                "tuning": ("not_applicable",),
                "detection_labels": ("1_inlier", "-1_outlier"),
                "model_selection": ("single", "all"),
                "scientific_methods": SCIENTIFIC_EXECUTION_METHODS_BY_TASK["anomaly_detection"],
            },
            time_series_options={
                "model": tuple(
                    model
                    for model, mode in (
                        (
                            "subaerial_proportion_bootstrap",
                            "subaerial_proportion",
                        ),
                        (
                            "spatiotemporal_weighted_continuous_bootstrap",
                            "continuous",
                        ),
                        (
                            "reference_label_event_overlay",
                            "reference_anomaly_series",
                        ),
                    )
                    if mode in executable_time_series_modes
                ),
                "scientific_modes": scientific_time_series_modes,
                "cli_executable_modes": executable_time_series_modes,
                "schema_only_modes": schema_only_time_series_modes,
                "age_units": ("Ma", "Ga"),
                "fit_curve": ("disabled", "enabled"),
                "randomness": ("explicit_seed",),
            },
            supported_models_by_task={
                "classification": CLASSIFICATION_MODEL_ORDER,
                "regression": REGRESSION_MODEL_ORDER,
                "clustering": CLUSTERING_MODEL_ORDER,
                "decomposition": DECOMPOSITION_MODEL_ORDER,
                "anomaly_detection": ANOMALY_DETECTION_MODEL_ORDER,
                "time_series": tuple(
                    model
                    for model, mode in (
                        (
                            "subaerial_proportion_bootstrap",
                            "subaerial_proportion",
                        ),
                        (
                            "spatiotemporal_weighted_continuous_bootstrap",
                            "continuous",
                        ),
                        (
                            "reference_label_event_overlay",
                            "reference_anomaly_series",
                        ),
                    )
                    if mode in executable_time_series_modes
                ),
            },
            scientific_attestation=scientific_attestation,
            unsupported_interactions=(
                *CLASSIFICATION_UNSUPPORTED_INTERACTIONS,
                *REGRESSION_UNSUPPORTED_INTERACTIONS,
                *CLUSTERING_UNSUPPORTED_INTERACTIONS,
                *DECOMPOSITION_UNSUPPORTED_INTERACTIONS,
                *ANOMALY_DETECTION_UNSUPPORTED_INTERACTIONS,
            ),
            notes=(
                "PR9 covers semantic world maps, seeded Time Series analysis, and isolated "
                "all-models aggregates in addition to every single-model family offered by "
                "the GeochemistryPi 0.8.1 public CLI.",
                "The stable compatibility policy is published only by the protected workflow after every required release gate passes.",
                "The existing GeochemistryPi CLI creates every scientific result and original output file.",
                "Unsupported or scientifically invalid combinations are rejected before the CLI starts.",
                "Use inspect_dataset when column roles are not already known.",
            ),
        )
        validation_request_contract = _task_validation_request_contract(request.task) if request.detail in {"compact", "start_ready"} and request.task is not None else None
        if request.detail == "start_ready":
            assert validation_request_contract is not None
            return _start_ready_capabilities_view(
                full_response,
                request,
                validation_request_contract,
            )
        return capabilities_response_view(
            full_response,
            request,
            validation_request_contract,
        )

    def inspect_dataset_request(request: DatasetInspectionRequest) -> BaseModel:
        if request.dataset is None:
            response = inspect_local_dataset(request, settings)
        else:
            resolved = runs.dataset_catalog.resolve(request.dataset)
            source_snapshot = snapshot_dataset(resolved.path, settings.maximum_dataset_bytes)
            if resolved.expected_sha256 is not None and source_snapshot.sha256 != resolved.expected_sha256:
                raise DatasetCatalogError("The dataset changed between source resolution and inspection.")
            if settings.service_state_root is None:
                raise SettingsError("The MCP service-state root is not configured.")
            prepared = prepare_dataset_view(
                source_snapshot,
                request.dataset.preparation,
                settings.service_state_root,
                settings.maximum_dataset_bytes,
                settings.maximum_columns,
                allow_pandas_duplicate_mangling=source_allows_pandas_duplicate_mangling(resolved.source),
            )
            response = inspect_local_dataset(
                request.model_copy(
                    update={
                        "dataset_path": prepared.snapshot.resolved_path,
                        "dataset": None,
                    }
                ),
                settings,
                allow_pandas_duplicate_mangling=(source_allows_pandas_duplicate_mangling(resolved.source) if prepared.snapshot.resolved_path == source_snapshot.resolved_path else False),
            )
            response = response.model_copy(
                update={
                    "original_source_path": str(source_snapshot.resolved_path),
                    "original_source_sha256": source_snapshot.sha256,
                    "dataset_preparation": prepared.record,
                }
            )
        return dataset_inspection_response_view(response, request)

    def analysis_value(request: BaseModel) -> BaseModel:
        return request.root if isinstance(request, AnalysisRequest) else request

    def validate_analysis_request(request: BaseModel) -> BaseModel:
        if isinstance(request, AnalysisValidationDetailRequest):
            return full_analysis_validation_detail(
                runs.get_validation_detail(
                    request.validation_id,
                    request.request_hash,
                )
            )
        return compact_analysis_validation(runs.validate(analysis_value(request)))

    def start_analysis_request(request: BaseModel) -> StartAnalysisResponse:
        if isinstance(request, StartAnalysisByValidationRequest):
            return runs.start_validated(
                request.validation_id,
                request.request_hash,
            )
        return runs.start(analysis_value(request))

    def get_run_result_request(request: RunResultRequest) -> BaseModel:
        artifact_limit = request.artifact_limit
        if request.detail == "compact":
            artifact_limit = min(
                artifact_limit or settings.maximum_artifact_references,
                settings.maximum_artifact_references,
                _MAX_COMPACT_ARTIFACT_REFERENCES,
            )
        response = runs.get_result(
            request.run_id,
            wait_seconds=request.wait_seconds,
            artifact_offset=request.artifact_offset,
            artifact_limit=artifact_limit,
            artifact_view=request.artifact_view,
        )
        if isinstance(response, TerminalRunReceipt):
            return terminal_result_response_view(
                response,
                request.if_result_sha256,
            )
        if isinstance(response, PendingRunResultResponse):
            return response
        if (
            response.result_record_path is None
            or response.result_record_sha256 is None
            or getattr(response, "request_hash", None) is None
            or getattr(response, "validation_id", None) is None
            or getattr(response, "canonical_contract_hash", None) is None
            or getattr(response, "compiled_plan_hash", None) is None
            or getattr(response, "provenance_manifest_path", None) is None
            or getattr(response, "provenance_manifest_sha256", None) is None
            or response.artifact_index_path is None
            or response.artifact_index_sha256 is None
            or response.canonical_artifact_count is None
            or response.artifact_view_count is None
        ):
            raise RunStateError("The immutable successful result identity is incomplete.")
        if request.if_result_sha256 == response.result_record_sha256 and request.if_result_sha256 is not None:
            return RunResultNotModifiedResponse(
                run_id=response.run_id,
                state=response.state,
                request_hash=response.request_hash,
                validation_id=response.validation_id,
                canonical_contract_hash=response.canonical_contract_hash,
                compiled_plan_hash=response.compiled_plan_hash,
                scientific_contract_id=response.scientific_contract_id,
                scientific_execution_contract_bound=response.scientific_execution_contract_bound,
                provenance_manifest_path=response.provenance_manifest_path,
                provenance_manifest_sha256=response.provenance_manifest_sha256,
                result_record_path=response.result_record_path,
                result_record_sha256=response.result_record_sha256,
                artifact_index_path=response.artifact_index_path,
                artifact_index_sha256=response.artifact_index_sha256,
                artifact_count=response.artifact_count,
                canonical_artifact_count=response.canonical_artifact_count,
                summary_mirror_count=response.summary_mirror_count,
                artifact_view=response.artifact_view,
                artifact_view_count=response.artifact_view_count,
                output_directory=response.output_directory,
            )
        if request.detail == "full":
            return response
        if request.artifact_offset > 0:
            if artifact_limit is None:
                raise RunStateError("A compact artifact continuation requires an effective artifact limit.")
            return CompactRunArtifactPageResponse.from_full(
                response,
                artifact_limit=artifact_limit,
            )
        return CompactRunResultResponse.from_full(response)

    def list_datasets_request(request: ListDatasetsViewRequest) -> BaseModel:
        complete = runs.dataset_catalog.list(ListDatasetsRequest(source=request.source))
        return list_datasets_response_view(complete, request)

    def list_experiments_request(request: ListExperimentsViewRequest) -> BaseModel:
        complete = experiment_store.list(ListExperimentsRequest(maximum_experiments=request.maximum_experiments))
        return list_experiments_response_view(complete, request)

    def get_experiment_request(request: GetExperimentViewRequest) -> BaseModel:
        complete = experiment_store.get(
            GetExperimentRequest(
                experiment_id=request.experiment_id,
                maximum_runs=request.maximum_runs,
            )
        )
        return get_experiment_response_view(complete, request)

    functions: dict[str, _ToolFunction] = {
        "get_capabilities": capabilities,
        "list_datasets": list_datasets_request,
        "inspect_dataset": inspect_dataset_request,
        "list_experiments": list_experiments_request,
        "get_experiment": get_experiment_request,
        "validate_analysis": validate_analysis_request,
        "start_mlflow_ui": lambda request: ui_manager.start(request),
        "mlflow_ui_status": lambda request: ui_manager.status(),
        "stop_mlflow_ui": lambda request: ui_manager.stop(),
        "start_analysis": start_analysis_request,
        "get_run_status": lambda request: runs.get_status(
            request.run_id,
            wait_seconds=request.wait_seconds,
        ),
        "get_run_result": get_run_result_request,
        "cancel_run": lambda request: runs.cancel(request.run_id),
    }

    async def list_tools(_: ServerRequestContext[Any], __: PaginatedRequestParams | None) -> ListToolsResult:
        return ListToolsResult(tools=list(definitions))

    def public_error_result(tool_name: str, exc: BaseException) -> CallToolResult:
        message, structured = _actionable_error(exc)
        structured["tool_name"] = tool_name
        structured = PublicToolErrorResponse.model_validate(structured).model_dump(mode="json")
        return CallToolResult(
            content=[TextContent(type="text", text=f"{_PUBLIC_ERROR_TEXT_PREFIX}{message}")],
            structured_content=structured,
            is_error=True,
        )

    async def call_tool(_: ServerRequestContext[Any], params: CallToolRequestParams) -> CallToolResult:
        if params.name not in functions:
            raise MCPError(INVALID_PARAMS, f"Unknown tool: {params.name}")
        arguments = params.arguments or {}
        try:
            if params.name == "validate_analysis" and "validation_id" in arguments:
                request = AnalysisValidationDetailRequest.model_validate(arguments)
            elif params.name == "start_analysis" and not ("validation_id" in arguments or "request_hash" in arguments):
                request = AnalysisRequest.model_validate(arguments)
            else:
                request = request_models[params.name].model_validate(arguments)
        except ValidationError as exc:
            return public_error_result(params.name, exc)
        try:
            response = await anyio.to_thread.run_sync(functions[params.name], request)
            structured = response.model_dump(mode="json")
            return CallToolResult(
                content=[TextContent(type="text", text=_compact_text(params.name, response))],
                structured_content=structured,
            )
        except _PUBLIC_ERRORS as exc:
            return public_error_result(params.name, exc)
        except Exception:
            LOGGER.exception("Unexpected GeochemistryPi MCP tool failure in %s", params.name)
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="GeochemistryPi encountered an internal wrapper error. Check the server's stderr log.",
                    )
                ],
                is_error=True,
            )

    return list_tools, call_tool
