"""Machine-readable MCP output contracts shared by every public tool."""

import hashlib
from typing import Literal

from pydantic import Field, model_validator

from .schemas import StrictModel

MAX_PUBLIC_TOOL_ERROR_JSON_BYTES = 64 * 1024
MAX_PUBLIC_TOOL_ERROR_ROOT_CAUSES = 24
MAX_PUBLIC_TOOL_ERROR_LOCATIONS = 4
MAX_PUBLIC_TOOL_ERROR_ACTUAL_VALUE_SUMMARIES = 4
MAX_PUBLIC_TOOL_ERROR_PROBLEM_CHARS = 256
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class PublicToolErrorRootCause(StrictModel):
    """One grouped, actionable cause in a rejected public tool request."""

    field: str = Field(min_length=1, max_length=256)
    kind: Literal[
        "missing",
        "extra_forbidden",
        "pattern",
        "range",
        "literal",
        "type",
        "value",
    ]
    problem: str = Field(
        min_length=1,
        max_length=MAX_PUBLIC_TOOL_ERROR_PROBLEM_CHARS,
    )
    problem_truncated: bool
    problem_sha256: str = Field(pattern=_SHA256_PATTERN)
    problem_total_utf8_bytes: int = Field(ge=1)
    valid_alternative: str = Field(min_length=1, max_length=512)
    locations: tuple[str, ...] = Field(min_length=1, max_length=MAX_PUBLIC_TOOL_ERROR_LOCATIONS)
    locations_total_count: int = Field(ge=1)
    locations_truncated: bool
    locations_sha256: str = Field(pattern=_SHA256_PATTERN)
    actual_value_summaries: tuple[str, ...] = Field(default=(), max_length=MAX_PUBLIC_TOOL_ERROR_ACTUAL_VALUE_SUMMARIES)
    actual_value_summaries_total_count: int = Field(ge=0)
    actual_value_summaries_truncated: bool
    actual_value_summaries_sha256: str = Field(pattern=_SHA256_PATTERN)
    occurrences: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_bounded_repeated_details(self) -> "PublicToolErrorRootCause":
        displayed_problem = self.problem.encode("utf-8")
        if self.problem_truncated:
            if not self.problem.endswith("…"):
                raise ValueError("a truncated problem must end with the omission marker")
            if len(displayed_problem) >= self.problem_total_utf8_bytes:
                raise ValueError("a truncated problem must be smaller than its complete UTF-8 text")
        else:
            if len(displayed_problem) != self.problem_total_utf8_bytes:
                raise ValueError("an untruncated problem must report its exact UTF-8 byte length")
            if hashlib.sha256(displayed_problem).hexdigest() != self.problem_sha256:
                raise ValueError("an untruncated problem must match problem_sha256")
        if len(self.locations) > self.locations_total_count:
            raise ValueError("locations cannot exceed locations_total_count")
        if self.locations_truncated != (len(self.locations) < self.locations_total_count):
            raise ValueError("locations_truncated must describe the returned locations prefix")
        if self.occurrences < self.locations_total_count:
            raise ValueError("occurrences cannot be smaller than the distinct location count")
        if len(self.actual_value_summaries) > self.actual_value_summaries_total_count:
            raise ValueError("actual_value_summaries cannot exceed their total count")
        if self.actual_value_summaries_truncated != (len(self.actual_value_summaries) < self.actual_value_summaries_total_count):
            raise ValueError("actual_value_summaries_truncated must describe the returned prefix")
        return self


class PublicToolErrorResponse(StrictModel):
    """Structured fail-closed error envelope returned by any public MCP tool."""

    error_schema_version: Literal[2] = 2
    error_type: Literal["validation_error", "request_error"]
    result_type: Literal[
        "invalid_arguments",
        "request_rejected",
        "input_integrity_changed",
        "environment_inspection_failed",
        "capability_manifest_invalid",
        "directory_view_rejected",
        "run_not_found",
        "run_state_invalid",
        "dataset_catalog_failed",
        "dataset_inspection_failed",
        "dataset_preparation_failed",
        "plan_compilation_failed",
        "cli_execution_failed",
        "internal_error",
        "experiment_store_failed",
        "mlflow_ui_failed",
        "contract_not_found",
        "settings_invalid",
    ]
    retryable: bool
    root_cause_count: int = Field(ge=1, le=MAX_PUBLIC_TOOL_ERROR_ROOT_CAUSES)
    root_causes: tuple[PublicToolErrorRootCause, ...] = Field(min_length=1, max_length=MAX_PUBLIC_TOOL_ERROR_ROOT_CAUSES)
    root_causes_total_count: int = Field(ge=1)
    root_causes_truncated: bool
    root_causes_sha256: str = Field(pattern=_SHA256_PATTERN)
    next_action: str = Field(min_length=1, max_length=1024)
    tool_name: Literal[
        "get_capabilities",
        "list_datasets",
        "inspect_dataset",
        "list_experiments",
        "get_experiment",
        "start_mlflow_ui",
        "mlflow_ui_status",
        "stop_mlflow_ui",
        "validate_analysis",
        "start_analysis",
        "get_run_status",
        "get_run_result",
        "cancel_run",
    ]

    @model_validator(mode="after")
    def validate_root_cause_projection(self) -> "PublicToolErrorResponse":
        if self.root_cause_count != len(self.root_causes):
            raise ValueError("root_cause_count must equal the number of returned root causes")
        if self.root_cause_count > self.root_causes_total_count:
            raise ValueError("root causes cannot exceed root_causes_total_count")
        if self.root_causes_truncated != (self.root_cause_count < self.root_causes_total_count):
            raise ValueError("root_causes_truncated must describe the returned root-cause prefix")
        if len(self.model_dump_json().encode("utf-8")) > MAX_PUBLIC_TOOL_ERROR_JSON_BYTES:
            raise ValueError("public tool error exceeds the structured-content byte budget")
        return self
