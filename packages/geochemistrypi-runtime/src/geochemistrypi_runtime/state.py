"""Run status and cancellation records with explicit ownership."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional

from ._validation import identifier, optional_string, require_fields, revision, run_id, utc_timestamp
from .exceptions import InvalidStateTransitionError

RECORD_FORMAT_VERSION = "1.0"


class RunState(str, Enum):
    """Durable runtime states."""

    QUEUED = "queued"
    VALIDATING = "validating"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    ORPHANED = "orphaned"
    CORRUPTED = "corrupted"


class StatusOwner(str, Enum):
    """Components permitted to own a status revision."""

    RUN_MANAGER = "run_manager"
    WORKER = "worker"
    RECOVERY = "recovery"


@dataclass(frozen=True)
class StatusWriter:
    """Identity used to authorize a status update."""

    owner: StatusOwner
    owner_id: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner", StatusOwner(self.owner))
        normalized_id = None if self.owner_id is None else identifier(self.owner_id, "owner_id")
        if self.owner is StatusOwner.WORKER and normalized_id is None:
            raise ValueError("A worker status writer requires owner_id.")
        if self.owner is not StatusOwner.WORKER and normalized_id is not None:
            raise ValueError("Only a worker status writer may have owner_id.")
        object.__setattr__(self, "owner_id", normalized_id)


@dataclass(frozen=True)
class StatusRecord:
    """One revision of the durable run status."""

    run_id: str
    state: RunState
    revision: int
    updated_at: str
    owner: StatusOwner
    owner_id: Optional[str] = None
    detail: Optional[str] = None
    format_version: str = RECORD_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != RECORD_FORMAT_VERSION:
            raise ValueError(f"format_version must be {RECORD_FORMAT_VERSION!r}.")
        object.__setattr__(self, "run_id", run_id(self.run_id))
        object.__setattr__(self, "state", RunState(self.state))
        object.__setattr__(self, "revision", revision(self.revision))
        object.__setattr__(
            self,
            "updated_at",
            utc_timestamp(self.updated_at, "updated_at"),
        )
        object.__setattr__(self, "owner", StatusOwner(self.owner))
        normalized_owner_id = None if self.owner_id is None else identifier(self.owner_id, "owner_id")
        if self.owner is StatusOwner.WORKER and normalized_owner_id is None:
            raise ValueError("Worker-owned status requires owner_id.")
        if self.owner is not StatusOwner.WORKER and normalized_owner_id is not None:
            raise ValueError("Only worker-owned status may have owner_id.")
        object.__setattr__(self, "owner_id", normalized_owner_id)
        object.__setattr__(
            self,
            "detail",
            optional_string(self.detail, "detail", 2000),
        )

    def to_dict(self) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "format_version": self.format_version,
            "run_id": self.run_id,
            "state": self.state.value,
            "revision": self.revision,
            "updated_at": self.updated_at,
            "owner": self.owner.value,
            "owner_id": self.owner_id,
        }
        if self.detail is not None:
            value["detail"] = self.detail
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StatusRecord":
        fields = require_fields(
            value,
            required={
                "format_version",
                "run_id",
                "state",
                "revision",
                "updated_at",
                "owner",
                "owner_id",
            },
            optional={"detail"},
            label="StatusRecord",
        )
        return cls(**fields)


@dataclass(frozen=True)
class ControlRecord:
    """Cancellation request stored separately from worker-owned status."""

    run_id: str
    revision: int
    cancel_requested: bool
    requested_at: Optional[str] = None
    requested_by: Optional[str] = None
    reason: Optional[str] = None
    format_version: str = RECORD_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != RECORD_FORMAT_VERSION:
            raise ValueError(f"format_version must be {RECORD_FORMAT_VERSION!r}.")
        object.__setattr__(self, "run_id", run_id(self.run_id))
        object.__setattr__(self, "revision", revision(self.revision))
        if not isinstance(self.cancel_requested, bool):
            raise TypeError("cancel_requested must be a boolean.")
        timestamp = None if self.requested_at is None else utc_timestamp(self.requested_at, "requested_at")
        requester = optional_string(self.requested_by, "requested_by", 128)
        reason = optional_string(self.reason, "reason", 1000)
        if self.cancel_requested:
            if timestamp is None or requester is None:
                raise ValueError("A cancellation request requires requested_at and requested_by.")
        elif any(item is not None for item in (timestamp, requester, reason)):
            raise ValueError("An inactive control record cannot contain cancellation details.")
        object.__setattr__(self, "requested_at", timestamp)
        object.__setattr__(self, "requested_by", requester)
        object.__setattr__(self, "reason", reason)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": self.format_version,
            "run_id": self.run_id,
            "revision": self.revision,
            "cancel_requested": self.cancel_requested,
            "requested_at": self.requested_at,
            "requested_by": self.requested_by,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ControlRecord":
        fields = require_fields(
            value,
            required={
                "format_version",
                "run_id",
                "revision",
                "cancel_requested",
                "requested_at",
                "requested_by",
                "reason",
            },
            label="ControlRecord",
        )
        return cls(**fields)


ALLOWED_TRANSITIONS = {
    RunState.QUEUED: frozenset({RunState.VALIDATING, RunState.FAILED}),
    RunState.VALIDATING: frozenset({RunState.RUNNING, RunState.CANCEL_REQUESTED, RunState.FAILED}),
    RunState.RUNNING: frozenset({RunState.COMPLETED, RunState.CANCEL_REQUESTED, RunState.FAILED}),
    RunState.CANCEL_REQUESTED: frozenset({RunState.CANCELLED, RunState.FAILED}),
    RunState.COMPLETED: frozenset(),
    RunState.CANCELLED: frozenset(),
    RunState.FAILED: frozenset(),
    RunState.ORPHANED: frozenset(),
    RunState.CORRUPTED: frozenset(),
}

ACTIVE_STATES = frozenset(
    {
        RunState.QUEUED,
        RunState.VALIDATING,
        RunState.RUNNING,
        RunState.CANCEL_REQUESTED,
    }
)


def validate_transition(current: RunState, target: RunState) -> None:
    """Raise when a normal state-machine transition is not allowed."""

    current_state = RunState(current)
    target_state = RunState(target)
    if target_state not in ALLOWED_TRANSITIONS[current_state]:
        raise InvalidStateTransitionError(f"Cannot transition run status from {current_state.value!r} " f"to {target_state.value!r}.")
