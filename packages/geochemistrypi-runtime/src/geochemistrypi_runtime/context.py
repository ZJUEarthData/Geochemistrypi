"""Durable run-directory lifecycle and integrity enforcement."""

from __future__ import annotations

import os
import secrets
import shutil
import stat
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple, Type, TypeVar

from filelock import FileLock, Timeout
from geochemistrypi_contracts import (
    CONTRACT_VERSION,
    ArtifactRef,
    ClassificationExperimentSpec,
    ExperimentResult,
    SchemaName,
    schema_id,
    schema_sha256,
)

from ._validation import (
    RUN_ID_PATTERN,
    json_mapping,
    nonempty_string,
    portable_relative_path,
    sha256,
    utc_now,
)
from ._version import __version__
from .atomic import (
    DEFAULT_MAX_JSON_BYTES,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    read_json_object,
    sha256_bytes,
    sha256_file,
)
from .exceptions import (
    ArtifactIntegrityError,
    CorruptedRecordError,
    InvalidRunIdError,
    InvalidStateTransitionError,
    RevisionConflictError,
    RunAlreadyExistsError,
    RunNotFoundError,
    RuntimeLockTimeout,
    StatusOwnershipError,
    UnsafePathError,
)
from .records import ManifestRecord, ProvenanceRecord, ProvenanceSection
from .state import (
    ACTIVE_STATES,
    ControlRecord,
    RunState,
    StatusOwner,
    StatusRecord,
    StatusWriter,
    validate_transition,
)

_RecordT = TypeVar("_RecordT")

_REQUEST_FILE = "request.json"
_REQUEST_HASH_FILE = "request.sha256"
_STATUS_FILE = "status.json"
_CONTROL_FILE = "control.json"
_RESULT_FILE = "result.json"
_MANIFEST_FILE = "manifest.json"
_PROVENANCE_FILE = "provenance.json"
_REQUIRED_DIRECTORIES = ("inputs", "artifacts", "errors", ".locks")
_MAX_STATUS_EVIDENCE_BYTES = 1024 * 1024


def _package_version(name: str, fallback: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return fallback


def _is_within(path: Path, parent: Path) -> bool:
    try:
        return os.path.commonpath((str(path), str(parent))) == str(parent)
    except ValueError:
        return False


def _validated_run_id(value: str) -> str:
    if (
        not isinstance(value, str)
        or not RUN_ID_PATTERN.fullmatch(value)
        or value in {".", ".."}
    ):
        raise InvalidRunIdError(
            "run_id must start with a letter or digit and contain only letters, "
            "digits, dots, underscores, or hyphens (maximum 128 characters)."
        )
    return value


def _generated_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")[:-3] + "Z"
    return f"run-{timestamp}-{secrets.token_hex(6)}"


@dataclass(frozen=True)
class RunContext:
    """Validated access to one atomically created local run directory."""

    runs_root: Path
    run_id: str
    lock_timeout: float = 10.0

    def __post_init__(self) -> None:
        root = Path(self.runs_root)
        if not root.is_absolute():
            raise UnsafePathError("runs_root must be an absolute path.")
        if isinstance(self.lock_timeout, bool) or self.lock_timeout <= 0:
            raise ValueError("lock_timeout must be a positive number.")
        object.__setattr__(self, "runs_root", root.resolve())
        object.__setattr__(self, "run_id", _validated_run_id(self.run_id))

    @property
    def path(self) -> Path:
        return self.runs_root / self.run_id

    @classmethod
    def create(
        cls,
        runs_root: Path,
        request: ClassificationExperimentSpec,
        *,
        run_id: Optional[str] = None,
        git_commit: Optional[str] = None,
        lock_timeout: float = 10.0,
    ) -> "RunContext":
        """Build a complete run privately, then publish it with one rename."""

        if not isinstance(request, ClassificationExperimentSpec):
            raise TypeError("request must be a ClassificationExperimentSpec.")
        root = cls._prepare_root(runs_root)
        chosen_run_id = _validated_run_id(run_id or _generated_run_id())
        context = cls(root, chosen_run_id, lock_timeout)
        final_path = context.path
        staging_name = f".creating-{chosen_run_id}-{secrets.token_hex(6)}"
        staging_path = root / staging_name
        if staging_path.parent != root:
            raise UnsafePathError("Generated staging directory escaped runs_root.")

        create_lock = FileLock(
            str(root / ".runtime-create.lock"),
            timeout=lock_timeout,
        )
        try:
            with create_lock:
                if final_path.exists() or final_path.is_symlink():
                    raise RunAlreadyExistsError(
                        f"Run already exists: {chosen_run_id}"
                    )
                staging_path.mkdir(mode=0o700)
                try:
                    context._initialize_staging(
                        staging_path,
                        request,
                        git_commit=git_commit,
                    )
                    if final_path.exists() or final_path.is_symlink():
                        raise RunAlreadyExistsError(
                            f"Run already exists: {chosen_run_id}"
                        )
                    os.rename(str(staging_path), str(final_path))
                except Exception:
                    if staging_path.exists() and staging_path.parent == root:
                        shutil.rmtree(staging_path)
                    raise
        except Timeout as exc:
            raise RuntimeLockTimeout(
                f"Timed out while creating run {chosen_run_id!r}."
            ) from exc
        return context

    @classmethod
    def open(
        cls,
        runs_root: Path,
        run_id: str,
        *,
        lock_timeout: float = 10.0,
    ) -> "RunContext":
        """Open an existing run without following an aliased run directory."""

        root = cls._prepare_root(runs_root)
        context = cls(root, _validated_run_id(run_id), lock_timeout)
        context._validate_run_directory()
        return context

    @staticmethod
    def _prepare_root(runs_root: Path) -> Path:
        root = Path(runs_root)
        if not root.is_absolute():
            raise UnsafePathError("runs_root must be an absolute path.")
        root.mkdir(parents=True, exist_ok=True)
        root_stat = root.lstat()
        if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
            raise UnsafePathError(
                "runs_root must be a real directory, not a symbolic link."
            )
        return root.resolve(strict=True)

    def _initialize_staging(
        self,
        staging_path: Path,
        request: ClassificationExperimentSpec,
        *,
        git_commit: Optional[str],
    ) -> None:
        for directory_name in _REQUIRED_DIRECTORIES:
            (staging_path / directory_name).mkdir(mode=0o700)

        request_payload = canonical_json_bytes(request.to_dict())
        request_digest = sha256_bytes(request_payload)
        created_at = utc_now()
        request_schema_name = SchemaName.CLASSIFICATION_EXPERIMENT_SPEC
        request_schema_id = schema_id(request_schema_name)
        request_schema_digest = schema_sha256(request_schema_name)

        initial_status = StatusRecord(
            run_id=self.run_id,
            state=RunState.QUEUED,
            revision=0,
            updated_at=created_at,
            owner=StatusOwner.RUN_MANAGER,
        )
        initial_control = ControlRecord(
            run_id=self.run_id,
            revision=0,
            cancel_requested=False,
        )
        initial_manifest = ManifestRecord(
            run_id=self.run_id,
            revision=0,
            created_at=created_at,
            updated_at=created_at,
            request_sha256=request_digest,
            contract_version=CONTRACT_VERSION,
            request_schema_id=request_schema_id,
            request_schema_sha256=request_schema_digest,
            status_path=_STATUS_FILE,
            provenance_path=_PROVENANCE_FILE,
        )
        initial_provenance = ProvenanceRecord.for_current_process(
            run_id_value=self.run_id,
            created_at=created_at,
            contract_version=CONTRACT_VERSION,
            request_schema_id=request_schema_id,
            request_schema_sha256=request_schema_digest,
            runtime_version=__version__,
            dependency_versions={
                "filelock": _package_version("filelock", "unknown"),
                "geochemistrypi-contracts": _package_version(
                    "geochemistrypi-contracts", "0.1.0"
                ),
                "geochemistrypi-runtime": __version__,
            },
            git_commit=git_commit,
        )

        atomic_write_bytes(staging_path / _REQUEST_FILE, request_payload)
        atomic_write_bytes(
            staging_path / _REQUEST_HASH_FILE,
            f"{request_digest}\n".encode("ascii"),
        )
        atomic_write_json(staging_path / _STATUS_FILE, initial_status.to_dict())
        atomic_write_json(staging_path / _CONTROL_FILE, initial_control.to_dict())
        atomic_write_json(
            staging_path / _MANIFEST_FILE,
            initial_manifest.to_dict(),
        )
        atomic_write_json(
            staging_path / _PROVENANCE_FILE,
            initial_provenance.to_dict(),
        )

    def _validate_run_directory(self) -> None:
        path = self.path
        try:
            path_stat = path.lstat()
        except FileNotFoundError as exc:
            raise RunNotFoundError(f"Run does not exist: {self.run_id}") from exc
        if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISDIR(path_stat.st_mode):
            raise UnsafePathError("Run path must be a real directory.")
        resolved = path.resolve(strict=True)
        if resolved.parent != self.runs_root:
            raise UnsafePathError("Run directory escaped runs_root.")
        for directory_name in _REQUIRED_DIRECTORIES:
            directory = path / directory_name
            try:
                directory_stat = directory.lstat()
            except FileNotFoundError as exc:
                raise CorruptedRecordError(
                    f"Run directory is missing {directory_name}/."
                ) from exc
            if stat.S_ISLNK(directory_stat.st_mode) or not stat.S_ISDIR(
                directory_stat.st_mode
            ):
                raise UnsafePathError(
                    f"Run member {directory_name}/ must be a real directory."
                )
            if not _is_within(directory.resolve(strict=True), resolved):
                raise UnsafePathError(
                    f"Run member {directory_name}/ escaped the run directory."
                )

    @contextmanager
    def _locked(self, record_name: str) -> Iterator[None]:
        self._validate_run_directory()
        lock = FileLock(
            str(self.path / ".locks" / f"{record_name}.lock"),
            timeout=self.lock_timeout,
        )
        try:
            with lock:
                yield
        except Timeout as exc:
            raise RuntimeLockTimeout(
                f"Timed out waiting for {record_name!r} lock "
                f"for run {self.run_id!r}."
            ) from exc

    def _read_record(
        self,
        filename: str,
        record_type: Type[_RecordT],
    ) -> _RecordT:
        try:
            record = record_type.from_dict(read_json_object(self.path / filename))
        except CorruptedRecordError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise CorruptedRecordError(
                f"Record {filename} does not match its runtime format."
            ) from exc
        if getattr(record, "run_id", None) != self.run_id:
            raise CorruptedRecordError(
                f"Record {filename} belongs to a different run."
            )
        return record

    def read_request(self) -> ClassificationExperimentSpec:
        """Verify the immutable request digest before parsing the contract."""

        self._validate_run_directory()
        request_path = self.path / _REQUEST_FILE
        hash_path = self.path / _REQUEST_HASH_FILE
        try:
            hash_stat = hash_path.lstat()
            if stat.S_ISLNK(hash_stat.st_mode) or not stat.S_ISREG(
                hash_stat.st_mode
            ):
                raise CorruptedRecordError(
                    "request.sha256 must be a regular file."
                )
            if hash_stat.st_size > 128:
                raise CorruptedRecordError("request.sha256 is unexpectedly large.")
            request_stat = request_path.lstat()
            if request_stat.st_size > DEFAULT_MAX_JSON_BYTES:
                raise CorruptedRecordError(
                    "request.json exceeds the runtime JSON safety limit."
                )
            expected_digest = sha256(
                hash_path.read_text(encoding="ascii").strip(),
                "request.sha256",
            )
            actual_digest = sha256_file(request_path)
        except (OSError, UnicodeError, ValueError) as exc:
            raise CorruptedRecordError("request.sha256 is invalid.") from exc
        if actual_digest != expected_digest:
            raise CorruptedRecordError(
                "request.json no longer matches request.sha256."
            )
        try:
            request = ClassificationExperimentSpec.from_dict(
                read_json_object(request_path)
            )
        except CorruptedRecordError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise CorruptedRecordError(
                "request.json does not match the classification contract."
            ) from exc
        return request

    def request_sha256(self) -> str:
        """Return the verified request digest."""

        self.read_request()
        return (self.path / _REQUEST_HASH_FILE).read_text(
            encoding="ascii"
        ).strip()

    def read_status(self) -> StatusRecord:
        return self._read_record(_STATUS_FILE, StatusRecord)

    def transition_status(
        self,
        target: RunState,
        *,
        writer: StatusWriter,
        expected_revision: int,
        detail: Optional[str] = None,
    ) -> StatusRecord:
        """Compare-and-swap a normal state transition under the status lock."""

        target_state = RunState(target)
        if not isinstance(writer, StatusWriter):
            raise TypeError("writer must be a StatusWriter.")
        with self._locked("status"):
            current = self.read_status()
            if current.revision != expected_revision:
                raise RevisionConflictError(
                    f"Expected status revision {expected_revision}, "
                    f"found {current.revision}."
                )
            validate_transition(current.state, target_state)
            self._authorize_status_writer(current, target_state, writer)
            updated = StatusRecord(
                run_id=self.run_id,
                state=target_state,
                revision=current.revision + 1,
                updated_at=utc_now(),
                owner=writer.owner,
                owner_id=writer.owner_id,
                detail=detail,
            )
            atomic_write_json(self.path / _STATUS_FILE, updated.to_dict())
            return updated

    @staticmethod
    def _authorize_status_writer(
        current: StatusRecord,
        target: RunState,
        writer: StatusWriter,
    ) -> None:
        if current.owner is StatusOwner.RUN_MANAGER:
            if (
                current.state is RunState.QUEUED
                and target is RunState.VALIDATING
                and writer.owner is StatusOwner.WORKER
            ):
                return
            if (
                current.state is RunState.QUEUED
                and target is RunState.FAILED
                and writer.owner is StatusOwner.RUN_MANAGER
            ):
                return
            raise StatusOwnershipError(
                "The run manager may only fail a queued run; a worker must "
                "claim queued work by entering validating."
            )
        if current.owner is StatusOwner.WORKER:
            if (
                writer.owner is StatusOwner.WORKER
                and writer.owner_id == current.owner_id
            ):
                return
            raise StatusOwnershipError(
                "Only the worker that owns this status may update it."
            )
        raise StatusOwnershipError(
            "Recovery-owned status cannot be changed through normal transitions."
        )

    def repair_orphaned(
        self,
        *,
        expected_revision: int,
        worker_confirmed_stopped: bool,
        detail: str,
    ) -> StatusRecord:
        """Mark lost worker-owned active work as orphaned."""

        if not worker_confirmed_stopped:
            raise StatusOwnershipError(
                "Orphan repair requires confirmation that the worker stopped."
            )
        with self._locked("status"):
            current = self.read_status()
            if current.revision != expected_revision:
                raise RevisionConflictError(
                    f"Expected status revision {expected_revision}, "
                    f"found {current.revision}."
                )
            if (
                current.owner is not StatusOwner.WORKER
                or current.state not in ACTIVE_STATES
            ):
                raise InvalidStateTransitionError(
                    "Only active worker-owned status can become orphaned."
                )
            repaired = StatusRecord(
                run_id=self.run_id,
                state=RunState.ORPHANED,
                revision=current.revision + 1,
                updated_at=utc_now(),
                owner=StatusOwner.RECOVERY,
                detail=nonempty_string(detail, "detail", 2000),
            )
            atomic_write_json(self.path / _STATUS_FILE, repaired.to_dict())
            return repaired

    def repair_corrupted_status(
        self,
        *,
        worker_confirmed_stopped: bool,
        detail: str,
    ) -> Tuple[StatusRecord, Path]:
        """Archive status evidence, then publish an explicit corrupted state."""

        if not worker_confirmed_stopped:
            raise StatusOwnershipError(
                "Corruption repair requires confirmation that the worker stopped."
            )
        normalized_detail = nonempty_string(detail, "detail", 2000)
        with self._locked("status"):
            status_path = self.path / _STATUS_FILE
            evidence_payload = self._status_evidence(status_path)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            evidence_path = (
                self.path
                / "errors"
                / f"status.corrupted.{timestamp}-{secrets.token_hex(4)}.bin"
            )
            atomic_write_bytes(evidence_path, evidence_payload)
            try:
                current = self.read_status()
                next_revision = current.revision + 1
            except CorruptedRecordError:
                next_revision = 0
            repaired = StatusRecord(
                run_id=self.run_id,
                state=RunState.CORRUPTED,
                revision=next_revision,
                updated_at=utc_now(),
                owner=StatusOwner.RECOVERY,
                detail=normalized_detail,
            )
            atomic_write_json(status_path, repaired.to_dict())
            return repaired, evidence_path

    @staticmethod
    def _status_evidence(status_path: Path) -> bytes:
        try:
            status_stat = status_path.lstat()
        except FileNotFoundError:
            return b"status.json was missing at recovery time.\n"
        if stat.S_ISLNK(status_stat.st_mode):
            return b"status.json was a symbolic link at recovery time.\n"
        if not stat.S_ISREG(status_stat.st_mode):
            return b"status.json was not a regular file at recovery time.\n"
        payload = status_path.read_bytes()
        if len(payload) <= _MAX_STATUS_EVIDENCE_BYTES:
            return payload
        marker = (
            b"\n[truncated by geochemistrypi-runtime after "
            + str(_MAX_STATUS_EVIDENCE_BYTES).encode("ascii")
            + b" bytes]\n"
        )
        return payload[:_MAX_STATUS_EVIDENCE_BYTES] + marker

    def read_control(self) -> ControlRecord:
        return self._read_record(_CONTROL_FILE, ControlRecord)

    def request_cancel(
        self,
        *,
        requested_by: str,
        reason: Optional[str] = None,
        expected_revision: Optional[int] = None,
    ) -> ControlRecord:
        """Persist an idempotent cancellation request without changing status."""

        requester = nonempty_string(requested_by, "requested_by", 128)
        normalized_reason = (
            None
            if reason is None
            else nonempty_string(reason, "reason", 1000)
        )
        with self._locked("control"):
            current = self.read_control()
            if current.cancel_requested:
                return current
            if (
                expected_revision is not None
                and current.revision != expected_revision
            ):
                raise RevisionConflictError(
                    f"Expected control revision {expected_revision}, "
                    f"found {current.revision}."
                )
            updated = ControlRecord(
                run_id=self.run_id,
                revision=current.revision + 1,
                cancel_requested=True,
                requested_at=utc_now(),
                requested_by=requester,
                reason=normalized_reason,
            )
            atomic_write_json(self.path / _CONTROL_FILE, updated.to_dict())
            return updated

    def read_manifest(self) -> ManifestRecord:
        return self._read_record(_MANIFEST_FILE, ManifestRecord)

    def read_provenance(self) -> ProvenanceRecord:
        return self._read_record(_PROVENANCE_FILE, ProvenanceRecord)

    def update_provenance(
        self,
        section: ProvenanceSection,
        details: Mapping[str, Any],
        *,
        expected_revision: int,
        engine_version: Optional[str] = None,
    ) -> ProvenanceRecord:
        """Replace one named provenance section using compare-and-swap."""

        section_name = ProvenanceSection(section).value
        normalized_details = json_mapping(details, "provenance details")
        with self._locked("provenance"):
            current = self.read_provenance()
            if current.revision != expected_revision:
                raise RevisionConflictError(
                    f"Expected provenance revision {expected_revision}, "
                    f"found {current.revision}."
                )
            sections: Dict[str, Mapping[str, Any]] = dict(current.sections)
            sections[section_name] = normalized_details
            updated = replace(
                current,
                revision=current.revision + 1,
                updated_at=utc_now(),
                engine_version=(
                    current.engine_version
                    if engine_version is None
                    else nonempty_string(
                        engine_version, "engine_version", 128
                    )
                ),
                sections=sections,
            )
            atomic_write_json(self.path / _PROVENANCE_FILE, updated.to_dict())
            return updated

    def register_artifact(
        self,
        *,
        artifact_id: str,
        role: str,
        media_type: str,
        relative_path: str,
        expected_manifest_revision: int,
    ) -> ArtifactRef:
        """Hash and register a completed artifact contained by artifacts/."""

        portable_path = portable_relative_path(
            relative_path, "artifact.relative_path"
        )
        if (
            PurePosixPath(portable_path).parts[0] != "artifacts"
            or len(PurePosixPath(portable_path).parts) < 2
        ):
            raise UnsafePathError(
                "Artifacts must use a path below the run's artifacts/ directory."
            )
        with self._locked("manifest"):
            current = self.read_manifest()
            if current.revision != expected_manifest_revision:
                raise RevisionConflictError(
                    f"Expected manifest revision {expected_manifest_revision}, "
                    f"found {current.revision}."
                )
            if any(
                artifact.artifact_id == artifact_id
                for artifact in current.artifacts
            ):
                raise ValueError(f"Duplicate artifact_id: {artifact_id!r}")
            if any(
                artifact.relative_path == portable_path
                for artifact in current.artifacts
            ):
                raise ValueError(f"Artifact path is already registered: {portable_path}")
            artifact_path = self._resolve_artifact_path(portable_path)
            before = artifact_path.stat()
            digest = sha256_file(artifact_path)
            after = artifact_path.stat()
            if (
                before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
            ):
                raise ArtifactIntegrityError(
                    "Artifact changed while its digest was being calculated."
                )
            artifact = ArtifactRef(
                artifact_id=artifact_id,
                role=role,
                media_type=media_type,
                relative_path=portable_path,
                size_bytes=after.st_size,
                sha256=digest,
            )
            updated = replace(
                current,
                revision=current.revision + 1,
                updated_at=utc_now(),
                artifacts=current.artifacts + (artifact,),
            )
            atomic_write_json(self.path / _MANIFEST_FILE, updated.to_dict())
            return artifact

    def verify_artifact(self, artifact_id: str) -> Path:
        """Return an artifact path only after size and SHA-256 verification."""

        manifest = self.read_manifest()
        artifact = next(
            (
                item
                for item in manifest.artifacts
                if item.artifact_id == artifact_id
            ),
            None,
        )
        if artifact is None:
            raise KeyError(f"Unknown artifact_id: {artifact_id!r}")
        path = self._resolve_artifact_path(artifact.relative_path)
        self._verify_artifact_record(artifact, path)
        return path

    def _resolve_artifact_path(self, relative_path: str) -> Path:
        portable_path = portable_relative_path(
            relative_path, "artifact.relative_path"
        )
        parts = PurePosixPath(portable_path).parts
        if not parts or parts[0] != "artifacts":
            raise UnsafePathError("Artifact path must be below artifacts/.")
        candidate = self.path.joinpath(*parts)
        try:
            candidate_stat = candidate.lstat()
        except FileNotFoundError as exc:
            raise ArtifactIntegrityError(
                f"Artifact file is missing: {portable_path}"
            ) from exc
        if stat.S_ISLNK(candidate_stat.st_mode) or not stat.S_ISREG(
            candidate_stat.st_mode
        ):
            raise UnsafePathError("Artifact must be a regular, non-link file.")
        run_path = self.path.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        artifacts_root = (run_path / "artifacts").resolve(strict=True)
        if not _is_within(resolved, artifacts_root):
            raise UnsafePathError("Artifact path escaped artifacts/.")
        ancestor = candidate.parent
        while ancestor != self.path:
            ancestor_stat = ancestor.lstat()
            if stat.S_ISLNK(ancestor_stat.st_mode):
                raise UnsafePathError("Artifact path contains a symbolic link.")
            ancestor = ancestor.parent
        return resolved

    @staticmethod
    def _verify_artifact_record(artifact: ArtifactRef, path: Path) -> None:
        actual_size = path.stat().st_size
        if actual_size != artifact.size_bytes:
            raise ArtifactIntegrityError(
                f"Artifact {artifact.artifact_id!r} size changed: "
                f"expected {artifact.size_bytes}, found {actual_size}."
            )
        actual_digest = sha256_file(path)
        if actual_digest != artifact.sha256:
            raise ArtifactIntegrityError(
                f"Artifact {artifact.artifact_id!r} SHA-256 changed."
            )

    def write_result(
        self,
        result: ExperimentResult,
        *,
        expected_manifest_revision: int,
    ) -> ExperimentResult:
        """Persist a contract result and link it from the manifest."""

        if not isinstance(result, ExperimentResult):
            raise TypeError("result must be an ExperimentResult.")
        if result.run_id != self.run_id:
            raise ValueError("Result belongs to a different run.")
        verified_request_hash = self.request_sha256()
        if result.request_hash != verified_request_hash:
            raise ValueError("Result request_hash does not match request.json.")
        if result.manifest_path != _MANIFEST_FILE:
            raise ValueError("Result manifest_path must be 'manifest.json'.")
        if result.provenance_path != _PROVENANCE_FILE:
            raise ValueError("Result provenance_path must be 'provenance.json'.")

        with self._locked("manifest"):
            current = self.read_manifest()
            if current.revision != expected_manifest_revision:
                raise RevisionConflictError(
                    f"Expected manifest revision {expected_manifest_revision}, "
                    f"found {current.revision}."
                )
            artifacts_by_id = {
                artifact.artifact_id: artifact for artifact in current.artifacts
            }
            for artifact in result.artifacts:
                recorded = artifacts_by_id.get(artifact.artifact_id)
                if recorded is None or recorded.to_dict() != artifact.to_dict():
                    raise ArtifactIntegrityError(
                        f"Result artifact {artifact.artifact_id!r} is not "
                        "identical to the manifest record."
                    )
                artifact_path = self._resolve_artifact_path(
                    artifact.relative_path
                )
                self._verify_artifact_record(artifact, artifact_path)

            result_path = self.path / _RESULT_FILE
            result_payload = canonical_json_bytes(result.to_dict())
            if result_path.exists():
                if result_path.is_symlink() or result_path.read_bytes() != result_payload:
                    raise RevisionConflictError(
                        "A different result.json already exists for this run."
                    )
                if current.result_path == _RESULT_FILE:
                    return result
            else:
                atomic_write_bytes(result_path, result_payload)

            updated = replace(
                current,
                revision=current.revision + 1,
                updated_at=utc_now(),
                result_path=_RESULT_FILE,
            )
            atomic_write_json(self.path / _MANIFEST_FILE, updated.to_dict())
            return result

    def read_result(self) -> ExperimentResult:
        """Read and cross-check the persisted terminal result."""

        manifest = self.read_manifest()
        if manifest.result_path != _RESULT_FILE:
            raise CorruptedRecordError(
                "manifest.json does not declare a completed result."
            )
        try:
            result = ExperimentResult.from_dict(
                read_json_object(self.path / _RESULT_FILE)
            )
        except CorruptedRecordError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise CorruptedRecordError(
                "result.json does not match the experiment result contract."
            ) from exc
        if result.run_id != self.run_id:
            raise CorruptedRecordError("result.json belongs to a different run.")
        if result.request_hash != manifest.request_sha256:
            raise CorruptedRecordError(
                "result.json does not match the manifest request digest."
            )
        manifest_artifacts = {
            artifact.artifact_id: artifact for artifact in manifest.artifacts
        }
        for artifact in result.artifacts:
            recorded = manifest_artifacts.get(artifact.artifact_id)
            if recorded is None or recorded.to_dict() != artifact.to_dict():
                raise CorruptedRecordError(
                    "result.json contains an artifact absent from manifest.json."
                )
        return result
