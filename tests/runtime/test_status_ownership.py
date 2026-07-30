from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
from geochemistrypi_contracts import ClassificationExperimentSpec
from geochemistrypi_runtime import CorruptedRecordError, InvalidStateTransitionError, RevisionConflictError, RunContext, RunState, StatusOwner, StatusOwnershipError, StatusWriter


def _context(
    runs_root: Path,
    request: ClassificationExperimentSpec,
) -> RunContext:
    return RunContext.create(runs_root.resolve(), request, run_id="status-run")


def test_worker_claim_and_owner_nonce_are_enforced(
    runs_root: Path,
    classification_request: ClassificationExperimentSpec,
) -> None:
    context = _context(runs_root, classification_request)
    manager = StatusWriter(StatusOwner.RUN_MANAGER)
    worker = StatusWriter(StatusOwner.WORKER, "worker-nonce-a")
    other_worker = StatusWriter(StatusOwner.WORKER, "worker-nonce-b")

    with pytest.raises(StatusOwnershipError):
        context.transition_status(
            RunState.VALIDATING,
            writer=manager,
            expected_revision=0,
        )

    validating = context.transition_status(
        RunState.VALIDATING,
        writer=worker,
        expected_revision=0,
    )
    assert validating.owner is StatusOwner.WORKER
    assert validating.owner_id == "worker-nonce-a"

    with pytest.raises(StatusOwnershipError):
        context.transition_status(
            RunState.RUNNING,
            writer=other_worker,
            expected_revision=1,
        )
    with pytest.raises(StatusOwnershipError):
        context.transition_status(
            RunState.RUNNING,
            writer=manager,
            expected_revision=1,
        )

    running = context.transition_status(
        RunState.RUNNING,
        writer=worker,
        expected_revision=1,
    )
    assert running.revision == 2
    completed = context.transition_status(
        RunState.COMPLETED,
        writer=worker,
        expected_revision=2,
    )
    assert completed.state is RunState.COMPLETED
    with pytest.raises(InvalidStateTransitionError):
        context.transition_status(
            RunState.RUNNING,
            writer=worker,
            expected_revision=3,
        )


def test_revision_compare_and_swap_allows_only_one_concurrent_update(
    runs_root: Path,
    classification_request: ClassificationExperimentSpec,
) -> None:
    context = _context(runs_root, classification_request)
    worker = StatusWriter(StatusOwner.WORKER, "worker-race")
    context.transition_status(
        RunState.VALIDATING,
        writer=worker,
        expected_revision=0,
    )
    barrier = Barrier(2)

    def update() -> object:
        barrier.wait()
        try:
            return context.transition_status(
                RunState.RUNNING,
                writer=worker,
                expected_revision=1,
            )
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: update(), range(2)))

    assert sum(not isinstance(item, Exception) for item in outcomes) == 1
    conflicts = [item for item in outcomes if isinstance(item, Exception)]
    assert len(conflicts) == 1
    assert isinstance(conflicts[0], RevisionConflictError)
    assert context.read_status().revision == 2


def test_cancellation_control_does_not_overwrite_worker_status(
    runs_root: Path,
    classification_request: ClassificationExperimentSpec,
) -> None:
    context = _context(runs_root, classification_request)
    worker = StatusWriter(StatusOwner.WORKER, "worker-cancel")
    context.transition_status(
        RunState.VALIDATING,
        writer=worker,
        expected_revision=0,
    )
    before = context.read_status()

    control = context.request_cancel(
        requested_by="mcp-client",
        reason="User requested cancellation.",
        expected_revision=0,
    )
    repeated = context.request_cancel(requested_by="another-client")

    assert control.cancel_requested is True
    assert repeated == control
    assert context.read_status() == before


def test_orphan_repair_requires_stopped_worker_and_preserves_revision(
    runs_root: Path,
    classification_request: ClassificationExperimentSpec,
) -> None:
    context = _context(runs_root, classification_request)
    worker = StatusWriter(StatusOwner.WORKER, "lost-worker")
    context.transition_status(
        RunState.VALIDATING,
        writer=worker,
        expected_revision=0,
    )
    context.transition_status(
        RunState.RUNNING,
        writer=worker,
        expected_revision=1,
    )

    with pytest.raises(StatusOwnershipError):
        context.repair_orphaned(
            expected_revision=2,
            worker_confirmed_stopped=False,
            detail="No heartbeat.",
        )
    repaired = context.repair_orphaned(
        expected_revision=2,
        worker_confirmed_stopped=True,
        detail="Worker process no longer exists.",
    )
    assert repaired.state is RunState.ORPHANED
    assert repaired.owner is StatusOwner.RECOVERY
    assert repaired.revision == 3


def test_corrupt_status_is_archived_before_recovery(
    runs_root: Path,
    classification_request: ClassificationExperimentSpec,
) -> None:
    context = _context(runs_root, classification_request)
    corrupt_bytes = b'{"revision":'
    (context.path / "status.json").write_bytes(corrupt_bytes)

    with pytest.raises(CorruptedRecordError):
        context.read_status()
    repaired, evidence_path = context.repair_corrupted_status(
        worker_confirmed_stopped=True,
        detail="Status JSON was truncated after an interrupted write.",
    )

    assert evidence_path.parent == context.path / "errors"
    assert evidence_path.read_bytes() == corrupt_bytes
    assert repaired.state is RunState.CORRUPTED
    assert repaired.revision == 0
    assert context.read_status() == repaired
