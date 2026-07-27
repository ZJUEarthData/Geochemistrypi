from pathlib import Path

import pytest
from geochemistrypi_contracts import (
    CONTRACT_VERSION,
    ClassificationExperimentSpec,
    ExperimentResult,
    RunStatus,
)

from geochemistrypi_runtime import (
    ArtifactIntegrityError,
    ProvenanceSection,
    RevisionConflictError,
    RunContext,
    UnsafePathError,
)


def _context(
    runs_root: Path,
    request: ClassificationExperimentSpec,
) -> RunContext:
    return RunContext.create(runs_root.resolve(), request, run_id="artifact-run")


def test_artifact_registration_records_size_hash_and_manifest_revision(
    runs_root: Path,
    classification_request: ClassificationExperimentSpec,
) -> None:
    context = _context(runs_root, classification_request)
    artifact_path = context.path / "artifacts" / "metrics.json"
    artifact_path.write_bytes(b'{"accuracy":0.875}\n')

    artifact = context.register_artifact(
        artifact_id="metrics",
        role="metrics",
        media_type="application/json",
        relative_path="artifacts/metrics.json",
        expected_manifest_revision=0,
    )

    assert artifact.size_bytes == artifact_path.stat().st_size
    assert len(artifact.sha256) == 64
    assert context.verify_artifact("metrics") == artifact_path.resolve()
    manifest = context.read_manifest()
    assert manifest.revision == 1
    assert manifest.artifacts == (artifact,)

    with pytest.raises(RevisionConflictError):
        context.register_artifact(
            artifact_id="stale",
            role="metrics",
            media_type="application/json",
            relative_path="artifacts/metrics.json",
            expected_manifest_revision=0,
        )


def test_artifact_escape_and_symlink_are_rejected(
    runs_root: Path,
    classification_request: ClassificationExperimentSpec,
    tmp_path: Path,
) -> None:
    context = _context(runs_root, classification_request)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")

    with pytest.raises((ValueError, UnsafePathError)):
        context.register_artifact(
            artifact_id="escape",
            role="model",
            media_type="text/plain",
            relative_path="../outside.txt",
            expected_manifest_revision=0,
        )
    with pytest.raises(UnsafePathError):
        context.register_artifact(
            artifact_id="input",
            role="model",
            media_type="text/plain",
            relative_path="inputs/outside.txt",
            expected_manifest_revision=0,
        )
    with pytest.raises(ValueError):
        context.register_artifact(
            artifact_id="ambiguous",
            role="model",
            media_type="text/plain",
            relative_path="artifacts//outside.txt",
            expected_manifest_revision=0,
        )

    link = context.path / "artifacts" / "linked.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("File symbolic links are not available on this system.")
    with pytest.raises(UnsafePathError):
        context.register_artifact(
            artifact_id="link",
            role="model",
            media_type="text/plain",
            relative_path="artifacts/linked.txt",
            expected_manifest_revision=0,
        )


def test_artifact_tampering_is_detected(
    runs_root: Path,
    classification_request: ClassificationExperimentSpec,
) -> None:
    context = _context(runs_root, classification_request)
    artifact_path = context.path / "artifacts" / "model.bin"
    artifact_path.write_bytes(b"model-v1")
    context.register_artifact(
        artifact_id="model",
        role="model",
        media_type="application/octet-stream",
        relative_path="artifacts/model.bin",
        expected_manifest_revision=0,
    )
    artifact_path.write_bytes(b"model-v2")

    with pytest.raises(ArtifactIntegrityError):
        context.verify_artifact("model")


def test_provenance_updates_use_named_sections_and_revisions(
    runs_root: Path,
    classification_request: ClassificationExperimentSpec,
) -> None:
    context = _context(runs_root, classification_request)
    updated = context.update_provenance(
        ProvenanceSection.DATASET,
        {
            "sha256": "a" * 64,
            "rows": 100,
            "columns": 12,
            "read_options": {"encoding": "utf-8"},
        },
        expected_revision=0,
        engine_version="0.8.0",
    )
    assert updated.revision == 1
    assert updated.engine_version == "0.8.0"
    assert updated.sections["dataset"]["rows"] == 100

    with pytest.raises(RevisionConflictError):
        context.update_provenance(
            ProvenanceSection.MODEL,
            {"name": "random_forest"},
            expected_revision=0,
        )


def test_result_must_match_request_manifest_and_artifact_integrity(
    runs_root: Path,
    classification_request: ClassificationExperimentSpec,
) -> None:
    context = _context(runs_root, classification_request)
    artifact_path = context.path / "artifacts" / "metrics.json"
    artifact_path.write_bytes(b'{"accuracy":0.9}\n')
    artifact = context.register_artifact(
        artifact_id="metrics",
        role="metrics",
        media_type="application/json",
        relative_path="artifacts/metrics.json",
        expected_manifest_revision=0,
    )
    result = ExperimentResult(
        schema_version=CONTRACT_VERSION,
        run_id=context.run_id,
        request_hash=context.request_sha256(),
        status=RunStatus.COMPLETED,
        metrics={"accuracy": 0.9},
        artifacts=(artifact,),
        warnings=(),
        manifest_path="manifest.json",
        provenance_path="provenance.json",
    )

    assert context.write_result(result, expected_manifest_revision=1) == result
    assert context.read_manifest().result_path == "result.json"
    assert context.read_manifest().revision == 2
    assert context.read_result().to_dict() == result.to_dict()

    assert context.write_result(result, expected_manifest_revision=2) == result
    assert context.read_manifest().revision == 2
