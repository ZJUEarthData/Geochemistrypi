import hashlib
import json

import pytest
from geochemistrypi_mcp.api.schemas import ArtifactReference, CompactRunResultResponse, RunResultRequest, RunResultResponse
from geochemistrypi_mcp.api.terminal_receipts import TerminalRunNotModifiedResponse, TerminalRunReceipt, terminal_error_projection, terminal_result_response_view
from geochemistrypi_mcp.runtime.result_views import partition_artifact_views
from pydantic import ValidationError


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _scientific_result_identity() -> dict[str, object]:
    return {
        "request_hash": "b" * 64,
        "validation_id": "val-" + "c" * 32,
        "canonical_contract_hash": "d" * 64,
        "compiled_plan_hash": "e" * 64,
        "scientific_contract_id": "scientific-contract-v4/supervised_learning/classification/xgboost",
        "scientific_execution_contract_bound": True,
        "provenance_manifest_path": "D:/managed/wrapper/provenance-manifest.json",
        "provenance_manifest_sha256": "f" * 64,
    }


def _valid_partial_aggregate_result() -> RunResultResponse:
    children = (
        {
            "model": "logistic_regression",
            "state": "succeeded",
            "output_relative_path": "children/logistic_regression",
            "artifact_count": 0,
        },
        {
            "model": "xgboost",
            "state": "failed",
            "output_relative_path": "children/xgboost",
            "artifact_count": 0,
            "error": "native child failed",
        },
    )
    return RunResultResponse(
        **_scientific_result_identity(),
        run_id="run-abcdef0123456789",
        result_record_path="D:/managed/wrapper/result.json",
        result_record_sha256="0" * 64,
        state="partial_failure",
        task="classification",
        model="all_models",
        output_directory="D:/managed/output",
        interaction_trace="D:/managed/wrapper/interaction-trace.json",
        cli_stdout_log="D:/managed/wrapper/stdout.log",
        cli_stderr_log="D:/managed/wrapper/stderr.log",
        cli_exit_code=0,
        cli_started_at="2026-08-30T00:00:00+00:00",
        cli_finished_at="2026-08-30T00:00:01+00:00",
        cli_execution_duration_seconds=1.0,
        cli_version="0.8.1",
        input_sha256="1" * 64,
        input_hash_verified=True,
        reported_metrics={},
        artifact_count=0,
        canonical_artifact_count=0,
        artifact_index_path="D:/managed/wrapper/artifact-index.json",
        artifact_index_sha256="2" * 64,
        artifact_view="canonical",
        artifact_view_count=0,
        artifacts=(),
        artifacts_truncated=False,
        aggregate_state="partial_failure",
        aggregate_summary={
            "expected_model_count": 2,
            "succeeded_count": 1,
            "failed_count": 1,
        },
        children=children,
        limitations=(),
    )


@pytest.mark.parametrize("response_detail", ("full", "compact"))
def test_partial_aggregate_cannot_claim_top_level_success(
    response_detail: str,
) -> None:
    full = _valid_partial_aggregate_result()
    response = full if response_detail == "full" else CompactRunResultResponse.from_full(full)
    payload = response.model_dump(mode="json")
    payload["state"] = "succeeded"
    schema = RunResultResponse if response_detail == "full" else CompactRunResultResponse

    with pytest.raises(
        ValidationError,
        match="top-level state must reflect artifact-contract and aggregate completeness",
    ):
        schema.model_validate(payload)


def test_aggregate_state_and_complete_children_must_match_summary() -> None:
    payload = _valid_partial_aggregate_result().model_dump(mode="json")
    payload["aggregate_state"] = "complete"
    with pytest.raises(
        ValidationError,
        match="aggregate_state must reflect whether any aggregate child failed",
    ):
        RunResultResponse.model_validate(payload)

    payload = _valid_partial_aggregate_result().model_dump(mode="json")
    payload["aggregate_summary"] = {
        "expected_model_count": 2,
        "succeeded_count": 2,
        "failed_count": 0,
    }
    with pytest.raises(
        ValidationError,
        match="aggregate child states exceed the published summary counts",
    ):
        RunResultResponse.model_validate(payload)


def test_all_models_result_requires_complete_aggregate_identity() -> None:
    payload = _valid_partial_aggregate_result().model_dump(mode="json")
    payload["aggregate_state"] = None
    payload["aggregate_summary"] = None

    with pytest.raises(
        ValidationError,
        match="all-models result requires aggregate_state and aggregate_summary",
    ):
        RunResultResponse.model_validate(payload)


def _reference(
    relative_path: str,
    *,
    sha256: str,
    requirement_ids: tuple[str, ...] = (),
) -> ArtifactReference:
    parts = relative_path.split("/")
    category = next(part for part in parts if part in {"artifacts", "metrics", "parameters", "summary"})
    return ArtifactReference(
        artifact_id=f"artifact-{len(relative_path):016x}",
        category=category,
        relative_path=relative_path,
        local_path=f"C:/managed/output/{relative_path}",
        size_bytes=17,
        media_type="application/json",
        sha256=sha256,
        requirement_id=requirement_ids[0] if requirement_ids else None,
        requirement_ids=requirement_ids,
        scientific_type="test_artifact",
        metadata={"producer": "geochemistrypi_cli"},
    )


def test_canonical_view_suppresses_only_proven_summary_mirrors() -> None:
    entries = (
        _reference("metrics/Score.json", sha256="1" * 64),
        _reference("summary/Score.json", sha256="1" * 64),
        _reference("summary/Aggregate Model Results.json", sha256="2" * 64),
        _reference("metrics/Different.json", sha256="3" * 64),
        _reference("summary/Different.json", sha256="4" * 64),
        _reference("artifacts/model.bin", sha256="5" * 64),
        _reference("summary/renamed.bin", sha256="5" * 64),
        _reference("parameters/Config.json", sha256="6" * 64),
        _reference(
            "summary/Config.json",
            sha256="6" * 64,
            requirement_ids=("summary.config",),
        ),
    )

    views = partition_artifact_views(entries)

    assert views.all_entries == entries
    assert views.summary_mirror_count == 1
    assert {item.relative_path for item in views.canonical_entries} == {
        "metrics/Score.json",
        "summary/Aggregate Model Results.json",
        "metrics/Different.json",
        "summary/Different.json",
        "artifacts/model.bin",
        "summary/renamed.bin",
        "parameters/Config.json",
        "summary/Config.json",
    }


def test_canonical_view_respects_child_scope_and_ambiguous_sources() -> None:
    entries = (
        _reference("ChildA/metrics/Score.json", sha256="7" * 64),
        _reference("ChildA/summary/Score.json", sha256="7" * 64),
        _reference("ChildB/summary/Score.json", sha256="7" * 64),
        _reference("ChildA/artifacts/Duplicate.txt", sha256="8" * 64),
        _reference("ChildA/metrics/Duplicate.txt", sha256="8" * 64),
        _reference("ChildA/summary/Duplicate.txt", sha256="8" * 64),
    )

    views = partition_artifact_views(entries)

    assert views.summary_mirror_count == 1
    assert "ChildA/summary/Score.json" not in {item.relative_path for item in views.canonical_entries}
    assert "ChildB/summary/Score.json" in {item.relative_path for item in views.canonical_entries}
    assert "ChildA/summary/Duplicate.txt" in {item.relative_path for item in views.canonical_entries}


def test_canonical_view_suppresses_all_63_proven_r10_shaped_summary_mirrors() -> None:
    artifact_sources = tuple(
        _reference(
            (
                f"artifacts/Native Scientific Output {index:02d}.xlsx"
                if index == 0
                else f"artifacts/data/Native Scientific Output {index:02d}.xlsx"
                if index < 40
                else f"artifacts/image/model_output/Native Scientific Output {index:02d}.png"
            ),
            sha256=f"{index + 1:064x}",
        )
        for index in range(57)
    )
    metric_sources = tuple(
        _reference(
            f"metrics/Native Metric {index:02d}.txt",
            sha256=f"{index + 58:064x}",
        )
        for index in range(4)
    )
    parameter_sources = tuple(
        _reference(
            f"parameters/Native Parameter {index:02d}.json",
            sha256=f"{index + 62:064x}",
        )
        for index in range(2)
    )
    sources = (*artifact_sources, *metric_sources, *parameter_sources)
    mirrors = tuple(
        _reference(
            f"summary/{source.relative_path.rsplit('/', 1)[-1]}",
            sha256=source.sha256,
        )
        for source in sources
    )

    views = partition_artifact_views((*sources, *mirrors))

    assert len(views.all_entries) == 126
    assert views.summary_mirror_count == 63
    assert views.canonical_entries == sources


def test_canonical_view_keeps_flat_summary_when_nested_sources_are_ambiguous() -> None:
    entries = (
        _reference("artifacts/data/Duplicate.txt", sha256="b" * 64),
        _reference("artifacts/image/model_output/Duplicate.txt", sha256="b" * 64),
        _reference("summary/Duplicate.txt", sha256="b" * 64),
    )

    views = partition_artifact_views(entries)

    assert views.summary_mirror_count == 0
    assert views.canonical_entries == entries


def test_canonical_view_never_hides_nested_same_name_same_hash_products() -> None:
    entries = (
        _reference("artifacts/fold-a/Predictions.xlsx", sha256="9" * 64),
        _reference("summary/fold-a/Predictions.xlsx", sha256="9" * 64),
        _reference("ChildA/metrics/fold-a/Score.json", sha256="a" * 64),
        _reference("ChildA/summary/fold-a/Score.json", sha256="a" * 64),
    )

    views = partition_artifact_views(entries)

    assert views.summary_mirror_count == 0
    assert views.canonical_entries == entries


def test_r10_shaped_compact_result_stays_below_terminal_payload_budget() -> None:
    canonical = tuple(
        ArtifactReference(
            artifact_id=f"artifact-{index:016x}",
            category="artifacts",
            relative_path=f"artifacts/Native Scientific Output {index:02d}.xlsx",
            local_path=("D:/gpx_token_pilot_r10/P01MCP/native/mcp_state/runs/" f"run-ce2b7a347159403f/workspace/geopi_output/Experiment/Run/artifacts/Native Scientific Output {index:02d}.xlsx"),
            size_bytes=10_000 + index,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            sha256=f"{index + 1:064x}",
            requirement_id=(f"required.output.{index}" if index < 17 else None),
            requirement_ids=((f"required.output.{index}",) if index < 17 else ()),
            scientific_type="native_scientific_output",
            metadata={
                "producer": "geochemistrypi_cli",
                "hash_algorithm": "sha256",
                "output_role": "scientific_output",
                "output_roles": ["scientific_output"],
                "adapter_mapping_ids": [f"mapping-{index}"],
            },
        )
        for index in range(63)
    )
    mirrors = tuple(
        artifact.model_copy(
            update={
                "artifact_id": f"artifact-{index + 1000:016x}",
                "category": "summary",
                "relative_path": artifact.relative_path.replace("artifacts/", "summary/"),
                "local_path": artifact.local_path.replace("/artifacts/", "/summary/"),
                "requirement_id": None,
                "requirement_ids": (),
            }
        )
        for index, artifact in enumerate(canonical)
    )

    def result(artifacts: tuple[ArtifactReference, ...], artifact_view: str) -> RunResultResponse:
        return RunResultResponse(
            **_scientific_result_identity(),
            run_id="run-0123456789abcdef",
            result_record_path="D:/managed/wrapper/result.json",
            result_record_sha256="a" * 64,
            state="succeeded",
            task="classification",
            model="xgboost",
            output_directory="D:/managed/output",
            interaction_trace="D:/managed/wrapper/interaction-trace.json",
            cli_stdout_log="D:/managed/wrapper/stdout.log",
            cli_stderr_log="D:/managed/wrapper/stderr.log",
            cli_exit_code=0,
            cli_started_at="2026-08-30T01:02:03+00:00",
            cli_finished_at="2026-08-30T01:02:07+00:00",
            cli_execution_duration_seconds=4.125,
            cli_version="0.8.1",
            input_sha256="b" * 64,
            input_hash_verified=True,
            reported_metrics={"bounded_metric_payload": "m" * 5000},
            artifact_count=126,
            canonical_artifact_count=63,
            summary_mirror_count=63,
            artifact_index_path="D:/managed/wrapper/artifact-index.json",
            artifact_index_sha256="c" * 64,
            artifact_view=artifact_view,
            artifact_view_count=len(artifacts),
            returned_artifact_count=len(artifacts),
            artifacts=artifacts,
            artifacts_truncated=False,
            limitations=(),
        )

    full = result((*canonical, *mirrors), "all")
    compact = CompactRunResultResponse.from_full(result(canonical, "canonical"))
    full_bytes = len(json.dumps(full.model_dump(mode="json"), separators=(",", ":")).encode("utf-8"))
    compact_bytes = len(json.dumps(compact.model_dump(mode="json"), separators=(",", ":")).encode("utf-8"))

    assert compact_bytes < 28 * 1024
    assert compact_bytes < full_bytes * 0.35
    assert compact.returned_artifact_count == 32
    assert compact.next_artifact_offset == 32
    assert compact.artifacts_truncated is True
    assert compact.cli_started_at == full.cli_started_at
    assert compact.cli_finished_at == full.cli_finished_at
    assert compact.cli_execution_duration_seconds == full.cli_execution_duration_seconds
    first_compact_receipt = compact.model_dump(mode="json")["artifacts"][0]
    assert "relative_path_sha256" not in first_compact_receipt
    assert "requirement_ids_sha256" not in first_compact_receipt
    assert first_compact_receipt["sha256"] == canonical[0].sha256
    invalid_timing = full.model_dump(mode="json")
    invalid_timing["cli_execution_duration_seconds"] = None
    with pytest.raises(ValidationError, match="must provide start, finish, and duration together"):
        RunResultResponse.model_validate(invalid_timing)

    missing_timing = full.model_dump(mode="json")
    missing_timing.update(
        {
            "cli_started_at": None,
            "cli_finished_at": None,
            "cli_execution_duration_seconds": None,
        }
    )
    with pytest.raises(
        ValidationError,
        match="successful or partial scientific result requires a CLI child-process interval",
    ):
        RunResultResponse.model_validate(missing_timing)

    compact_missing_timing = compact.model_dump(mode="json")
    compact_missing_timing.update(
        {
            "cli_started_at": None,
            "cli_finished_at": None,
            "cli_execution_duration_seconds": None,
        }
    )
    with pytest.raises(
        ValidationError,
        match="successful or partial scientific result requires a CLI child-process interval",
    ):
        CompactRunResultResponse.model_validate(compact_missing_timing)

    oversized_metrics = {
        "oversized_native_metric_group": {"values": "x" * 50_000},
        "small_native_metric_group": {"accuracy": 0.75},
    }
    oversized = result(canonical, "canonical").model_copy(update={"reported_metrics": oversized_metrics})
    bounded = CompactRunResultResponse.from_full(oversized)
    assert bounded.reported_metrics_truncated is True
    assert bounded.reported_metric_groups_omitted == 1
    assert bounded.reported_metrics_original_bytes > 16 * 1024
    assert bounded.reported_metrics == {"small_native_metric_group": {"accuracy": 0.75}}
    assert len(json.dumps(bounded.reported_metrics, separators=(",", ":")).encode("utf-8")) <= 8 * 1024
    assert oversized.reported_metrics == oversized_metrics


def test_compact_result_enforces_global_budget_for_extreme_valid_full_result() -> None:
    def requirement_id(artifact_index: int, requirement_index: int) -> str:
        prefix = f"required.artifact.{artifact_index:02d}.{requirement_index:03d}."
        return prefix + "x" * (120 - len(prefix))

    artifacts = []
    for artifact_index in range(32):
        requirement_ids = tuple(requirement_id(artifact_index, index) for index in range(128))
        long_relative_path = f"artifacts/{artifact_index:02d}-" + "p" * 1400
        artifacts.append(
            ArtifactReference(
                artifact_id=f"artifact-{artifact_index:016x}",
                category="artifacts",
                relative_path=long_relative_path,
                local_path=f"D:/managed/output/{long_relative_path}",
                size_bytes=10_000 + artifact_index,
                media_type="application/octet-stream",
                sha256=f"{artifact_index + 1:064x}",
                requirement_id=requirement_ids[0],
                requirement_ids=requirement_ids,
                scientific_type="native_scientific_output",
                metadata={"producer": "geochemistrypi_cli"},
            )
        )

    missing_ids = tuple(f"required.missing.{index:03d}" for index in range(256))
    children = tuple(
        {
            "model": f"failed-model-{index:02d}-" + "m" * 50,
            "state": "failed",
            "output_relative_path": f"children/{index:02d}/" + "o" * 230,
            "artifact_count": 0,
            "error": f"failure-{index:02d}-" + "e" * 989,
        }
        for index in range(20)
    )
    limitations = tuple(f"limitation-{index:02d}-" + "l" * 985 for index in range(64))
    full = RunResultResponse(
        **_scientific_result_identity(),
        run_id="run-0123456789abcdef",
        result_record_path="D:/managed/wrapper/result.json",
        result_record_sha256="a" * 64,
        state="partial_failure",
        contract_status="incomplete",
        missing_artifact_requirement_ids=missing_ids,
        task="classification",
        model="all_models",
        output_directory="D:/managed/output",
        interaction_trace="D:/managed/wrapper/interaction-trace.json",
        cli_stdout_log="D:/managed/wrapper/stdout.log",
        cli_stderr_log="D:/managed/wrapper/stderr.log",
        cli_exit_code=0,
        cli_started_at="2026-08-30T00:00:00+00:00",
        cli_finished_at="2026-08-30T00:00:01+00:00",
        cli_execution_duration_seconds=1.0,
        cli_version="0.8.1",
        input_sha256="b" * 64,
        input_hash_verified=True,
        dataset_preparation={"contract_hash": "c" * 64},
        reported_metrics={f"metric_group_{index}": "v" * 1000 for index in range(8)},
        artifact_count=len(artifacts),
        canonical_artifact_count=len(artifacts),
        artifact_index_path="D:/managed/wrapper/artifact-index.json",
        artifact_index_sha256="d" * 64,
        artifact_view="canonical",
        artifact_view_count=len(artifacts),
        returned_artifact_count=len(artifacts),
        artifacts=tuple(artifacts),
        artifacts_truncated=False,
        aggregate_state="partial_failure",
        aggregate_summary={
            "expected_model_count": len(children),
            "succeeded_count": 0,
            "failed_count": len(children),
        },
        children=children,
        limitations=limitations,
    )

    full_bytes = len(json.dumps(full.model_dump(mode="json"), separators=(",", ":")).encode("utf-8"))
    compact = CompactRunResultResponse.from_full(full)
    compact_bytes = len(json.dumps(compact.model_dump(mode="json"), separators=(",", ":")).encode("utf-8"))

    assert full_bytes > 600_000
    assert compact_bytes <= 64 * 1024
    assert compact.state == "partial_failure"
    assert compact.contract_status == "incomplete"
    assert compact.missing_artifact_requirement_ids_total_count == 256
    assert compact.missing_artifact_requirement_ids_truncated is True
    assert compact.missing_artifact_requirement_ids_sha256 == _canonical_sha256(list(missing_ids))
    assert 0 < compact.returned_artifact_count < len(artifacts)
    assert compact.next_artifact_offset == compact.returned_artifact_count
    assert compact.artifacts_truncated is True

    first_full_artifact = artifacts[0]
    first_compact_artifact = compact.artifacts[0]
    assert len(first_full_artifact.requirement_ids) == 128
    assert len(first_compact_artifact.requirement_ids) == 4
    assert first_compact_artifact.requirement_ids_total_count == 128
    assert first_compact_artifact.requirement_ids_truncated is True
    assert first_compact_artifact.requirement_ids_sha256 == _canonical_sha256(list(first_full_artifact.requirement_ids))
    assert first_compact_artifact.relative_path_truncated is True
    assert first_compact_artifact.relative_path_sha256 == _canonical_sha256(first_full_artifact.relative_path)

    # Validate the per-artifact rule directly using the already bounded first
    # extreme receipt.
    artifact_payload = first_compact_artifact.model_dump(mode="json")
    artifact_payload["relative_path_truncated"] = False
    with pytest.raises(
        ValidationError,
        match="separate SHA-256 only when truncated",
    ):
        type(first_compact_artifact).model_validate(artifact_payload)
    assert compact.children_total_count == len(children)
    assert compact.children_truncated is True
    assert compact.children_sha256 == _canonical_sha256([child.model_dump(mode="json") for child in full.children])
    assert compact.limitations_total_count == len(limitations)
    assert compact.limitations_truncated is True
    assert compact.limitations_sha256 == _canonical_sha256(list(limitations))
    assert len(full.missing_artifact_requirement_ids) == 256
    assert len(full.artifacts[0].requirement_ids) == 128


def test_all_models_result_preserves_146_missing_requirement_id_identity() -> None:
    missing_ids = tuple(f"required.all_models.output.{index:03d}" for index in range(146))
    children = tuple(
        {
            "model": f"model-{index:02d}",
            "state": "succeeded",
            "output_relative_path": f"children/model-{index:02d}",
            "artifact_count": 0,
        }
        for index in range(13)
    )
    full = RunResultResponse(
        **_scientific_result_identity(),
        run_id="run-fedcba9876543210",
        result_record_path="D:/managed/wrapper/all-models-result.json",
        result_record_sha256="1" * 64,
        state="partial_failure",
        contract_status="incomplete",
        missing_artifact_requirement_ids=missing_ids,
        task="classification",
        model="all_models",
        output_directory="D:/managed/output",
        interaction_trace="D:/managed/wrapper/interaction-trace.json",
        cli_stdout_log="D:/managed/wrapper/stdout.log",
        cli_stderr_log="D:/managed/wrapper/stderr.log",
        cli_exit_code=0,
        cli_started_at="2026-08-30T00:00:00+00:00",
        cli_finished_at="2026-08-30T00:00:01+00:00",
        cli_execution_duration_seconds=1.0,
        cli_version="0.8.1",
        input_sha256="2" * 64,
        input_hash_verified=True,
        reported_metrics={},
        artifact_count=0,
        canonical_artifact_count=0,
        artifact_index_path="D:/managed/wrapper/artifact-index.json",
        artifact_index_sha256="3" * 64,
        artifact_view="canonical",
        artifact_view_count=0,
        artifacts=(),
        artifacts_truncated=False,
        aggregate_state="complete",
        aggregate_summary={
            "expected_model_count": len(children),
            "succeeded_count": len(children),
            "failed_count": 0,
        },
        children=children,
        limitations=("Every missing requirement remains listed in the immutable full result.",),
    )

    compact = CompactRunResultResponse.from_full(full)

    assert full.missing_artifact_requirement_ids == missing_ids
    assert compact.state == "partial_failure"
    assert compact.contract_status == "incomplete"
    assert compact.missing_artifact_requirement_ids == missing_ids[:16]
    assert compact.missing_artifact_requirement_ids_total_count == 146
    assert compact.missing_artifact_requirement_ids_truncated is True
    assert compact.missing_artifact_requirement_ids_sha256 == _canonical_sha256(list(missing_ids))


def test_compact_result_hashes_large_preparation_record_without_replaying_it() -> None:
    huge_marker = "preparation-bulk-should-not-be-returned-" + "z" * 120_000
    full = RunResultResponse(
        **_scientific_result_identity(),
        run_id="run-0123456789abcdef",
        result_record_path="D:/managed/wrapper/result.json",
        result_record_sha256="a" * 64,
        state="succeeded",
        task="classification",
        model="xgboost",
        output_directory="D:/managed/output",
        interaction_trace="D:/managed/wrapper/interaction-trace.json",
        cli_stdout_log="D:/managed/wrapper/stdout.log",
        cli_stderr_log="D:/managed/wrapper/stderr.log",
        cli_exit_code=0,
        cli_started_at="2026-08-30T00:00:00+00:00",
        cli_finished_at="2026-08-30T00:00:01+00:00",
        cli_execution_duration_seconds=1.0,
        cli_version="0.8.1",
        input_sha256="c" * 64,
        input_hash_verified=True,
        source_input_sha256="d" * 64,
        dataset_preparation={
            "contract_hash": "e" * 64,
            "contract": {
                "worksheet": "Training",
                "selected_columns": ["Sample", "SiO2", "Label"],
                "filters": [{"column": "SiO2", "operator": "not_null"}],
                "irrelevant_bulk": huge_marker,
            },
            "source_file": {"sha256": "d" * 64},
            "prepared_input": {"sha256": "c" * 64},
            "table": {
                "input_row_count": 2011,
                "source_row_count": 2000,
                "filtered_row_count": 11,
                "row_identity": {"strategy": "source_row", "ordered_sha256": "f" * 64},
            },
            "declared_operations": ["filtering"],
            "executed_view_operations": ["select_worksheet", "select_columns", "filter_rows"],
            "source_mapping": {"unbounded_external_record": huge_marker},
        },
        source_row_count=2000,
        reported_metrics={"accuracy": 0.8},
        artifact_count=0,
        canonical_artifact_count=0,
        artifact_index_path="D:/managed/wrapper/artifact-index.json",
        artifact_index_sha256="1" * 64,
        artifact_view="canonical",
        artifact_view_count=0,
        artifacts=(),
        artifacts_truncated=False,
        limitations=(),
    )

    compact = CompactRunResultResponse.from_full(full)
    payload = json.dumps(compact.model_dump(mode="json"), separators=(",", ":"))

    assert huge_marker not in payload
    assert len(payload.encode("utf-8")) < 8 * 1024
    assert compact.dataset_preparation.preparation_contract_sha256 == "e" * 64
    assert compact.dataset_preparation.input_row_count == 2011
    assert compact.dataset_preparation.prepared_row_count == 2000
    assert compact.dataset_preparation.filtered_row_count == 11
    assert compact.dataset_preparation.projection_mode == "selected"
    assert compact.dataset_preparation.projected_column_count == 3
    assert compact.dataset_preparation.filter_count == 1
    assert compact.dataset_preparation.row_identity_sha256 == "f" * 64


@pytest.mark.parametrize(
    "conflict",
    (
        {"detail": "full"},
        {"artifact_view": "all"},
        {"artifact_offset": 1},
        {"artifact_limit": 1},
    ),
)
def test_conditional_result_identity_rejects_full_all_and_pagination(conflict: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="if_result_sha256"):
        RunResultRequest(
            run_id="run-0123456789abcdef",
            if_result_sha256="a" * 64,
            **conflict,
        )

    request = RunResultRequest(
        run_id="run-0123456789abcdef",
        if_result_sha256="a" * 64,
        wait_seconds=300,
    )
    assert request.wait_seconds == 300


def test_result_request_defaults_to_one_terminal_wait_and_allows_explicit_short_wait() -> None:
    default_wait = RunResultRequest(run_id="run-0123456789abcdef")
    short_wait = RunResultRequest(
        run_id="run-0123456789abcdef",
        wait_seconds=5,
    )

    assert default_wait.wait_seconds == 300
    assert short_wait.wait_seconds == 5


def test_terminal_conditional_helper_suppresses_replayed_diagnostics() -> None:
    error, error_truncated, error_sha256, error_total_utf8_bytes = terminal_error_projection("bounded failure")
    terminal = TerminalRunReceipt(
        run_id="run-0123456789abcdef",
        result_record_path="D:/managed/wrapper/terminal-result.json",
        result_record_sha256="a" * 64,
        scientific_contract_id="scientific-contract-v4/supervised_learning/classification/xgboost",
        scientific_execution_contract_bound=True,
        state="failed",
        stage="failed",
        created_at="2026-08-29T00:00:00+00:00",
        started_at="2026-08-29T00:00:01+00:00",
        finished_at="2026-08-29T00:00:02+00:00",
        progress_message="The managed run failed.",
        error=error,
        error_truncated=error_truncated,
        error_sha256=error_sha256,
        error_total_utf8_bytes=error_total_utf8_bytes,
        result_type="cli_execution_failed",
        retryable=False,
        analysis_process_started=True,
        cli_exit_code=1,
        cli_started_at="2026-08-29T00:00:01+00:00",
        cli_finished_at="2026-08-29T00:00:02+00:00",
        cli_execution_duration_seconds=1.0,
    )

    unchanged = terminal_result_response_view(terminal, "a" * 64)
    changed = terminal_result_response_view(terminal, "b" * 64)

    assert isinstance(unchanged, TerminalRunNotModifiedResponse)
    assert unchanged.state == "failed"
    assert unchanged.result_type == "cli_execution_failed"
    assert unchanged.retryable is False
    assert unchanged.scientific_contract_id == terminal.scientific_contract_id
    assert unchanged.scientific_execution_contract_bound is True
    assert unchanged.scientific_validity == "not_established"
    assert "error" not in unchanged.model_dump(mode="json")
    assert changed is terminal
    assert changed.cli_execution_duration_seconds == 1.0


def test_compact_result_preserves_scientific_contract_transparency() -> None:
    full = _valid_partial_aggregate_result().model_copy(
        update={
            "scientific_contract_id": "scientific-contract-v2/supervised_learning/classification/all_models",
            "scientific_execution_contract_bound": False,
        }
    )
    compact = CompactRunResultResponse.from_full(full)

    assert compact.scientific_contract_id == full.scientific_contract_id
    assert compact.scientific_execution_contract_bound is False
