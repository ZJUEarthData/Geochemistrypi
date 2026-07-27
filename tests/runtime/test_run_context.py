import json
from pathlib import Path

import pytest
from geochemistrypi_contracts import (
    CONTRACT_VERSION,
    ClassificationExperimentSpec,
    SchemaName,
    schema_id,
    schema_sha256,
)

import geochemistrypi_runtime.context as context_module
from geochemistrypi_runtime import (
    CorruptedRecordError,
    RunAlreadyExistsError,
    RunContext,
    RunNotFoundError,
    RunState,
    StatusOwner,
    UnsafePathError,
)
from geochemistrypi_runtime.atomic import canonical_json_bytes, sha256_bytes


def test_create_publishes_complete_verified_run_directory(
    runs_root: Path,
    classification_request: ClassificationExperimentSpec,
) -> None:
    context = RunContext.create(
        runs_root.resolve(),
        classification_request,
        run_id="run-pr3-baseline",
        git_commit="abc123",
    )

    assert context.path == runs_root.resolve() / "run-pr3-baseline"
    assert {path.name for path in context.path.iterdir()} == {
        ".locks",
        "artifacts",
        "control.json",
        "errors",
        "inputs",
        "manifest.json",
        "provenance.json",
        "request.json",
        "request.sha256",
        "status.json",
    }
    assert context.read_request().to_dict() == classification_request.to_dict()
    expected_request_bytes = canonical_json_bytes(classification_request.to_dict())
    assert context.path.joinpath("request.json").read_bytes() == expected_request_bytes
    assert context.request_sha256() == sha256_bytes(expected_request_bytes)

    status = context.read_status()
    assert status.state is RunState.QUEUED
    assert status.revision == 0
    assert status.owner is StatusOwner.RUN_MANAGER
    assert status.owner_id is None

    control = context.read_control()
    assert control.cancel_requested is False
    assert control.revision == 0

    manifest = context.read_manifest()
    assert manifest.request_sha256 == context.request_sha256()
    assert manifest.contract_version == CONTRACT_VERSION
    assert manifest.request_schema_id == schema_id(
        SchemaName.CLASSIFICATION_EXPERIMENT_SPEC
    )
    assert manifest.request_schema_sha256 == schema_sha256(
        SchemaName.CLASSIFICATION_EXPERIMENT_SPEC
    )
    assert manifest.result_path is None
    assert manifest.artifacts == ()

    provenance = context.read_provenance()
    assert provenance.git_commit == "abc123"
    assert provenance.contract_version == CONTRACT_VERSION
    assert provenance.dependency_versions["geochemistrypi-runtime"] == "0.1.0"


def test_create_rejects_duplicate_and_unsafe_ids(
    runs_root: Path,
    classification_request: ClassificationExperimentSpec,
) -> None:
    absolute_root = runs_root.resolve()
    RunContext.create(absolute_root, classification_request, run_id="same-run")
    with pytest.raises(RunAlreadyExistsError):
        RunContext.create(absolute_root, classification_request, run_id="same-run")
    with pytest.raises(ValueError):
        RunContext.create(absolute_root, classification_request, run_id="../escape")
    with pytest.raises(UnsafePathError):
        RunContext.create(Path("relative-runs"), classification_request)


def test_failed_initialization_never_publishes_partial_run(
    runs_root: Path,
    classification_request: ClassificationExperimentSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    absolute_root = runs_root.resolve()

    def fail_write(*args: object, **kwargs: object) -> None:
        raise OSError("simulated storage failure")

    monkeypatch.setattr(context_module, "atomic_write_json", fail_write)
    with pytest.raises(OSError, match="simulated storage failure"):
        RunContext.create(
            absolute_root,
            classification_request,
            run_id="never-visible",
        )

    assert not (absolute_root / "never-visible").exists()
    assert not list(absolute_root.glob(".creating-never-visible-*"))


def test_request_tampering_is_detected(
    runs_root: Path,
    classification_request: ClassificationExperimentSpec,
) -> None:
    context = RunContext.create(runs_root.resolve(), classification_request)
    request_path = context.path / "request.json"
    value = json.loads(request_path.read_text(encoding="utf-8"))
    value["target_column"] = "tampered"
    request_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(CorruptedRecordError, match="request.sha256"):
        context.read_request()


def test_open_rejects_missing_or_aliased_run_directory(
    runs_root: Path,
    classification_request: ClassificationExperimentSpec,
) -> None:
    absolute_root = runs_root.resolve()
    context = RunContext.create(
        absolute_root,
        classification_request,
        run_id="real-run",
    )
    with pytest.raises(RunNotFoundError):
        RunContext.open(absolute_root, "missing-run")

    alias = absolute_root / "alias-run"
    try:
        alias.symlink_to(context.path, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symbolic links are not available on this system.")
    with pytest.raises(UnsafePathError):
        RunContext.open(absolute_root, "alias-run")
