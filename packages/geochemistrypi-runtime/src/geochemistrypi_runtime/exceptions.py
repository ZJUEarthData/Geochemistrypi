"""Public exceptions raised by the runtime storage package."""


class GeochemistryPiRuntimeError(Exception):
    """Base class for runtime storage failures."""


class InvalidRunIdError(GeochemistryPiRuntimeError, ValueError):
    """The requested run identifier is unsafe or unsupported."""


class RunAlreadyExistsError(GeochemistryPiRuntimeError):
    """A run directory already exists for the requested identifier."""


class RunNotFoundError(GeochemistryPiRuntimeError):
    """The requested run directory does not exist."""


class CorruptedRecordError(GeochemistryPiRuntimeError):
    """A persisted record is missing, malformed, or fails integrity checks."""


class RevisionConflictError(GeochemistryPiRuntimeError):
    """A caller attempted to update a stale record revision."""


class InvalidStateTransitionError(GeochemistryPiRuntimeError):
    """The requested run-state transition is not allowed."""


class StatusOwnershipError(GeochemistryPiRuntimeError):
    """A status writer does not own the current run state."""


class UnsafePathError(GeochemistryPiRuntimeError, ValueError):
    """A path could escape or alias the configured run directory."""


class ArtifactIntegrityError(GeochemistryPiRuntimeError):
    """An artifact no longer matches its recorded size or digest."""


class RuntimeLockTimeout(GeochemistryPiRuntimeError, TimeoutError):
    """A runtime record lock could not be acquired in time."""
