"""Strict, non-scientific receipts for failed and cancelled managed runs."""

import hashlib
import hmac
from datetime import datetime
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from .schemas import StrictModel, validate_cli_execution_interval, validate_terminal_error_projection

MAX_TERMINAL_ERROR_LENGTH = 1000


def normalize_terminal_error(value: object | None) -> str | None:
    """Collapse control whitespace without discarding the diagnostic identity."""
    if value is None:
        return None
    normalized = " ".join(str(value).split())
    return normalized or type(value).__name__


def sanitize_terminal_error(value: object | None) -> str | None:
    """Return the bounded display prefix used by a terminal receipt."""

    normalized = normalize_terminal_error(value)
    return normalized[:MAX_TERMINAL_ERROR_LENGTH] if normalized is not None else None


def terminal_error_projection(
    value: object | None,
) -> tuple[str | None, bool, str | None, int | None]:
    """Return a bounded error plus verifiable metadata for its complete text."""

    normalized = normalize_terminal_error(value)
    if normalized is None:
        return None, False, None, None
    bounded = normalized[:MAX_TERMINAL_ERROR_LENGTH]
    normalized_bytes = normalized.encode("utf-8")
    return (
        bounded,
        bounded != normalized,
        hashlib.sha256(normalized_bytes).hexdigest(),
        len(normalized_bytes),
    )


class TerminalEvidenceIdentity(StrictModel):
    """Content identity for one allowlisted wrapper log or interaction trace."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["interaction_trace", "cli_stdout_log", "cli_stderr_log"]
    path: str = Field(min_length=1, max_length=4096)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class TerminalRunReceipt(StrictModel):
    """Terminal receipt that makes no scientific or artifact-validity claim."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1, 2] = 2
    response_detail: Literal["terminal"] = "terminal"
    run_id: str = Field(pattern=r"^run-[0-9a-f]{16}$")
    result_record_path: str | None = Field(None, min_length=1, max_length=4096)
    result_record_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    scientific_contract_id: str = Field(min_length=1, max_length=256)
    scientific_execution_contract_bound: bool
    state: Literal["failed", "cancelled"]
    stage: Literal["failed", "cancelled"]
    created_at: str
    started_at: str | None = None
    finished_at: str
    progress_message: str = Field(min_length=1, max_length=1000)
    error: str | None = Field(None, max_length=MAX_TERMINAL_ERROR_LENGTH)
    error_truncated: bool = False
    error_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    error_total_utf8_bytes: int | None = Field(None, ge=1)
    result_type: Literal[
        "input_integrity_changed",
        "cli_execution_failed",
        "run_state_invalid",
        "internal_error",
    ] | None = None
    retryable: bool | None = None
    analysis_process_started: bool
    cli_exit_code: int | None = None
    cli_started_at: str | None = Field(
        None,
        description="Actual CLI child start from the immutable trace; null without a child interval.",
    )
    cli_finished_at: str | None = Field(
        None,
        description="Actual CLI child finish from the immutable trace; never managed-run time.",
    )
    cli_execution_duration_seconds: float | None = Field(
        None,
        ge=0,
        allow_inf_nan=False,
        description="CLI child monotonic seconds; null without a child interval.",
    )
    scientific_validity: Literal["not_established"] = "not_established"
    artifact_contract_status: Literal["not_evaluated"] = "not_evaluated"
    verified_artifact_count: Literal[0] = 0
    interaction_trace: TerminalEvidenceIdentity | None = None
    cli_stdout_log: TerminalEvidenceIdentity | None = None
    cli_stderr_log: TerminalEvidenceIdentity | None = None

    @field_validator("created_at", "started_at", "finished_at")
    @classmethod
    def validate_aware_timestamp(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("terminal receipt timestamps must be ISO 8601") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("terminal receipt timestamps must include a timezone")
        return value

    @field_validator("progress_message", "error")
    @classmethod
    def validate_canonical_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if sanitize_terminal_error(value) != value:
            raise ValueError("terminal receipt text must be whitespace-normalized and bounded")
        return value

    @model_validator(mode="after")
    def validate_terminal_invariants(self) -> "TerminalRunReceipt":
        validate_cli_execution_interval(
            self.cli_started_at,
            self.cli_finished_at,
            self.cli_execution_duration_seconds,
        )
        if self.state != self.stage:
            raise ValueError("terminal receipt state and stage must match")
        if self.state == "failed" and not self.error:
            raise ValueError("a failed terminal receipt requires a bounded error")
        if self.schema_version == 2 and self.state == "failed":
            if self.result_type is None or self.retryable is None:
                raise ValueError("a failed terminal receipt requires a typed recovery contract")
        elif self.state != "failed" and (self.result_type is not None or self.retryable is not None):
            raise ValueError("failure recovery fields are available only for failed receipts")
        validate_terminal_error_projection(
            self.error,
            self.error_truncated,
            self.error_sha256,
            self.error_total_utf8_bytes,
        )
        if not self.analysis_process_started and self.cli_exit_code is not None:
            raise ValueError("a CLI exit code requires a started analysis process")
        if not self.analysis_process_started and self.cli_started_at is not None:
            raise ValueError("CLI execution timing requires a started analysis process")
        expected_kinds = {
            "interaction_trace": "interaction_trace",
            "cli_stdout_log": "cli_stdout_log",
            "cli_stderr_log": "cli_stderr_log",
        }
        for field_name, expected_kind in expected_kinds.items():
            identity = getattr(self, field_name)
            if identity is not None and identity.kind != expected_kind:
                raise ValueError(f"{field_name} must use evidence kind {expected_kind!r}")
        if (self.result_record_path is None) != (self.result_record_sha256 is None):
            raise ValueError("terminal result record path and SHA-256 must be provided together")
        return self


class TerminalRunNotModifiedResponse(StrictModel):
    """Small identity-only receipt for an unchanged failure or cancellation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[2] = 2
    response_detail: Literal["not_modified"] = "not_modified"
    terminal_receipt: Literal[True] = True
    not_modified: Literal[True] = True
    run_id: str = Field(pattern=r"^run-[0-9a-f]{16}$")
    state: Literal["failed", "cancelled"]
    result_record_path: str = Field(min_length=1, max_length=4096)
    result_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scientific_contract_id: str = Field(min_length=1, max_length=256)
    scientific_execution_contract_bound: bool
    result_type: Literal[
        "input_integrity_changed",
        "cli_execution_failed",
        "run_state_invalid",
        "internal_error",
    ] | None = None
    retryable: bool | None = None
    scientific_validity: Literal["not_established"] = "not_established"
    artifact_contract_status: Literal["not_evaluated"] = "not_evaluated"
    verified_artifact_count: Literal[0] = 0
    requery_required: Literal[False] = False
    message: Literal["Terminal failure/cancellation receipt is unchanged; diagnostics were not replayed."] = "Terminal failure/cancellation receipt is unchanged; diagnostics were not replayed."

    @model_validator(mode="after")
    def validate_failure_contract(self) -> "TerminalRunNotModifiedResponse":
        if self.state == "failed":
            if self.result_type is None or self.retryable is None:
                raise ValueError("an unchanged failed receipt requires a typed recovery contract")
        elif self.result_type is not None or self.retryable is not None:
            raise ValueError("failure recovery fields are available only for failed receipts")
        return self


def terminal_result_response_view(
    receipt: TerminalRunReceipt,
    if_result_sha256: str | None,
) -> TerminalRunReceipt | TerminalRunNotModifiedResponse:
    """Return a reusable identity-only view when a terminal receipt matches."""

    if if_result_sha256 is None or receipt.result_record_sha256 is None:
        return receipt
    if not hmac.compare_digest(if_result_sha256, receipt.result_record_sha256):
        return receipt
    if receipt.state == "failed" and receipt.result_type is None:
        # A sealed schema-v1 receipt predates typed recovery metadata.  Do not
        # fabricate a category in a new identity-only projection.
        return receipt
    if receipt.result_record_path is None:
        raise ValueError("a conditional terminal receipt requires its immutable record path")
    return TerminalRunNotModifiedResponse(
        run_id=receipt.run_id,
        state=receipt.state,
        result_type=receipt.result_type,
        retryable=receipt.retryable,
        result_record_path=receipt.result_record_path,
        result_record_sha256=receipt.result_record_sha256,
        scientific_contract_id=receipt.scientific_contract_id,
        scientific_execution_contract_bound=receipt.scientific_execution_contract_bound,
    )
