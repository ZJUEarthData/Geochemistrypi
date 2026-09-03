import hashlib
import json
from pathlib import Path

import pytest
from geochemistrypi_mcp.api.schemas import (
    AnalysisValidationResponse,
    ArtifactReference,
    CompactRunResultResponse,
    PendingRunResultResponse,
    RunResultNotModifiedResponse,
    RunResultResponse,
    RunStatusResponse,
)
from geochemistrypi_mcp.api.terminal_receipts import TerminalRunNotModifiedResponse, TerminalRunReceipt
from geochemistrypi_mcp.api.tools import full_output_contract_schema
from geochemistrypi_mcp.config.constants import CLI_VERSION
from geochemistrypi_mcp.config.settings import McpSettings
from geochemistrypi_mcp.server import create_server
from jsonschema import Draft202012Validator
from mcp import Client


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _retained_output_contract(tool) -> dict:
    advertised = tool.output_schema
    assert advertised is not None
    assert advertised["type"] == "object"
    assert advertised["additionalProperties"] is True
    assert advertised["x-geochemistrypi-output-contract-delivery"] == "hash-addressed-server-enforced"
    assert len(json.dumps(advertised, separators=(",", ":")).encode("utf-8")) < 700
    Draft202012Validator.check_schema(advertised)

    contract_sha256 = advertised["x-geochemistrypi-full-output-schema-sha256"]
    contract = full_output_contract_schema(contract_sha256)
    encoded = json.dumps(
        contract,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == contract_sha256
    assert len(encoded) == advertised["x-geochemistrypi-full-output-schema-utf8-bytes"]
    Draft202012Validator.check_schema(contract)
    return contract


def _scientific_result_identity() -> dict[str, object]:
    return {
        "request_hash": "d" * 64,
        "validation_id": "val-" + "e" * 32,
        "canonical_contract_hash": "f" * 64,
        "compiled_plan_hash": "0" * 64,
        "scientific_contract_id": "scientific-contract-v4/supervised_learning/classification/xgboost",
        "scientific_execution_contract_bound": True,
        "provenance_manifest_path": "C:/managed/wrapper/provenance-manifest.json",
        "provenance_manifest_sha256": "1" * 64,
    }


def test_public_scientific_identity_fields_are_required() -> None:
    assert "scientific_contract_id" in AnalysisValidationResponse.model_json_schema()["required"]
    for model in (
        RunResultResponse,
        CompactRunResultResponse,
        RunResultNotModifiedResponse,
        TerminalRunReceipt,
        TerminalRunNotModifiedResponse,
    ):
        required = set(model.model_json_schema()["required"])
        assert {
            "scientific_contract_id",
            "scientific_execution_contract_bound",
        } <= required


class ResultDeliveryManager:
    def __init__(self, *, pending: bool = False) -> None:
        self.pending = pending
        self.result_calls: list[tuple[str, float, int, int | None, str]] = []
        self.closed = False

    def get_result(
        self,
        run_id: str,
        *,
        wait_seconds: float = 0,
        artifact_offset: int = 0,
        artifact_limit: int | None = None,
        artifact_view: str = "canonical",
    ) -> RunResultResponse | PendingRunResultResponse:
        self.result_calls.append((run_id, wait_seconds, artifact_offset, artifact_limit, artifact_view))
        if self.pending:
            return PendingRunResultResponse.from_status(
                RunStatusResponse(
                    run_id=run_id,
                    state="running",
                    stage="indexing_outputs",
                    created_at="2026-08-30T00:00:00+00:00",
                    started_at="2026-08-30T00:00:01+00:00",
                    progress_message="Indexing native GeochemistryPi outputs.",
                ),
                wait_seconds=wait_seconds,
            )

        artifacts = tuple(
            ArtifactReference(
                artifact_id=f"artifact-{index:016x}",
                category="artifacts",
                relative_path=f"artifacts/Native Output {index}.xlsx",
                local_path=f"C:/managed/output/artifacts/Native Output {index}.xlsx",
                size_bytes=100 + index,
                media_type=("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
                sha256=f"{index + 1:064x}",
                requirement_id=f"required.output-{index}",
                requirement_ids=(f"required.output-{index}",),
                scientific_type="native_scientific_output",
            )
            for index in range(3)
        )
        limit = artifact_limit or len(artifacts)
        page = artifacts[artifact_offset : artifact_offset + limit]
        page_end = artifact_offset + len(page)
        next_offset = page_end if page_end < len(artifacts) else None
        return RunResultResponse(
            **_scientific_result_identity(),
            run_id=run_id,
            result_record_path="C:/managed/wrapper/result.json",
            result_record_sha256="a" * 64,
            state="succeeded",
            task="classification",
            model="xgboost",
            output_directory="C:/managed/output",
            interaction_trace="C:/managed/wrapper/interaction-trace.json",
            cli_stdout_log="C:/managed/wrapper/stdout.log",
            cli_stderr_log="C:/managed/wrapper/stderr.log",
            cli_exit_code=0,
            cli_started_at="2026-08-30T00:00:00+00:00",
            cli_finished_at="2026-08-30T00:00:01+00:00",
            cli_execution_duration_seconds=1.0,
            cli_version=CLI_VERSION,
            input_sha256="b" * 64,
            input_hash_verified=True,
            dataset_preparation={"source_file": {"sha256": "b" * 64}},
            reported_metrics={"accuracy": 0.91, "f1": 0.89},
            artifact_count=len(artifacts),
            canonical_artifact_count=len(artifacts),
            summary_mirror_count=0,
            artifact_index_path="C:/managed/wrapper/artifact-index.json",
            artifact_index_sha256="c" * 64,
            artifact_view=artifact_view,
            artifact_view_count=len(artifacts),
            artifact_offset=artifact_offset,
            returned_artifact_count=len(page),
            next_artifact_offset=next_offset,
            artifacts=page,
            artifacts_truncated=next_offset is not None,
            limitations=("Observed values only.",),
        )

    def close(self) -> None:
        self.closed = True


@pytest.mark.anyio
async def test_compact_continuation_is_additive_artifact_only_and_schema_aligned(
    tmp_path: Path,
) -> None:
    manager = ResultDeliveryManager()
    server = create_server(
        McpSettings(runs_root=tmp_path / "runs", cli_executable=None),
        manager,
    )

    async with Client(server) as client:
        tool = {item.name: item for item in (await client.list_tools()).tools}["get_run_result"]
        output_contract = _retained_output_contract(tool)
        assert {
            "CompactRunResultResponse",
            "CompactRunArtifactPageResponse",
            "PendingRunResultResponse",
            "RequiredTabularObservation",
            "RequiredTabularObservationSummary",
        } <= set(output_contract["$defs"])

        first = await client.call_tool(
            "get_run_result",
            {
                "run_id": "run-0123456789abcdef",
                "artifact_limit": 1,
            },
        )
        assert first.is_error is False
        Draft202012Validator(output_contract).validate(first.structured_content)
        assert first.structured_content["response_detail"] == "compact"
        assert first.structured_content["next_artifact_offset"] == 1
        assert "dataset_preparation" in first.structured_content
        assert "reported_metrics" in first.structured_content
        assert "required_tabular_observations" in first.structured_content
        assert first.structured_content["required_tabular_observations"] == {
            "artifact_index_sha256": None,
            "observations": [],
            "total_count": 0,
            "returned_count": 0,
            "truncated": False,
            "observations_sha256": hashlib.sha256(b"[]").hexdigest(),
            "returned_cell_count": 0,
            "returned_utf8_bytes": 2,
            "omitted_artifact_count": 0,
            "omission_reason_counts": {},
            "omissions_sha256": hashlib.sha256(b"[]").hexdigest(),
        }
        assert "contract_status" in first.structured_content

        continuation = await client.call_tool(
            "get_run_result",
            {
                "run_id": "run-0123456789abcdef",
                "artifact_offset": 1,
                "artifact_limit": 1,
            },
        )
        assert continuation.is_error is False
        Draft202012Validator(output_contract).validate(continuation.structured_content)
        page = continuation.structured_content
        assert page["response_detail"] == "artifact_page"
        assert page["additive"] is True
        assert page["artifact_page_number"] == 2
        assert page["artifact_offset"] == 1
        assert page["artifact_limit"] == 1
        assert page["returned_artifact_count"] == 1
        assert page["next_artifact_offset"] == 2
        assert page["artifacts"][0]["sha256"] == "2".zfill(64)
        assert page["artifact_page_sha256"] == _canonical_sha256(page["artifacts"])
        for repeated_core_field in (
            "dataset_preparation",
            "reported_metrics",
            "contract_status",
            "missing_artifact_requirement_ids",
            "limitations",
            "task",
            "model",
            "output_directory",
        ):
            assert repeated_core_field not in page
        assert len(json.dumps(page, separators=(",", ":"))) < len(json.dumps(first.structured_content, separators=(",", ":")))
        assert "Scientific core fields were not replayed" in continuation.content[0].text

        full_replay = await client.call_tool(
            "get_run_result",
            {
                "run_id": "run-0123456789abcdef",
                "artifact_offset": 1,
                "artifact_limit": 1,
                "detail": "full",
            },
        )
        assert full_replay.is_error is False
        assert full_replay.structured_content["response_detail"] == "full"
        assert "reported_metrics" in full_replay.structured_content

        unchanged = await client.call_tool(
            "get_run_result",
            {
                "run_id": "run-0123456789abcdef",
                "if_result_sha256": "a" * 64,
            },
        )
        assert unchanged.is_error is False
        assert unchanged.structured_content["response_detail"] == "not_modified"
        assert unchanged.structured_content["scientific_contract_id"] == ("scientific-contract-v4/supervised_learning/classification/xgboost")
        assert unchanged.structured_content["scientific_execution_contract_bound"] is True

        invalid_conditional_page = await client.call_tool(
            "get_run_result",
            {
                "run_id": "run-0123456789abcdef",
                "artifact_offset": 1,
                "if_result_sha256": "a" * 64,
            },
        )
        assert invalid_conditional_page.is_error is True
        Draft202012Validator(output_contract).validate(invalid_conditional_page.structured_content)

    assert manager.closed is True


@pytest.mark.anyio
async def test_pending_result_wait_is_a_schema_valid_success_receipt(
    tmp_path: Path,
) -> None:
    manager = ResultDeliveryManager(pending=True)
    server = create_server(
        McpSettings(runs_root=tmp_path / "runs", cli_executable=None),
        manager,
    )

    async with Client(server) as client:
        tool = {item.name: item for item in (await client.list_tools()).tools}["get_run_result"]
        output_contract = _retained_output_contract(tool)
        pending = await client.call_tool(
            "get_run_result",
            {
                "run_id": "run-0123456789abcdef",
                "wait_seconds": 300,
            },
        )

        assert pending.is_error is False
        Draft202012Validator(output_contract).validate(pending.structured_content)
        assert pending.structured_content == {
            "response_detail": "pending",
            "terminal": False,
            "run_id": "run-0123456789abcdef",
            "state": "running",
            "stage": "indexing_outputs",
            "created_at": "2026-08-30T00:00:00+00:00",
            "started_at": "2026-08-30T00:00:01+00:00",
            "cli_pid": None,
            "progress_message": "Indexing native GeochemistryPi outputs.",
            "wait_seconds": 300.0,
            "result_available": False,
            "requery_required": True,
            "recommended_wait_seconds": 5,
            "message": ("The run is still active; no terminal scientific result is available yet."),
        }
        assert "This is not a scientific failure" in pending.content[0].text
        assert "reported_metrics" not in pending.structured_content
        assert "artifacts" not in pending.structured_content

    assert manager.closed is True
