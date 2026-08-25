import json
from pathlib import Path

import pytest
from geochemistrypi_mcp.api.schemas import AnalysisValidationResponse, CancelRunResponse, RunResultResponse, RunStatusResponse, StartAnalysisResponse
from geochemistrypi_mcp.config.constants import CLI_VERSION
from geochemistrypi_mcp.config.settings import McpSettings, SettingsError
from geochemistrypi_mcp.server import create_server
from mcp import Client

SCHEMA_TASK_ENV = "GEOCHEMISTRYPI_MCP_ANALYSIS_SCHEMA_TASK"
VALIDATION_ID = "val-0123456789abcdef0123456789abcdef"
REQUEST_HASH = "1" * 64


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class ProtocolRunManager:
    def __init__(self) -> None:
        self.closed = False
        self.validated = []
        self.legacy_starts = []
        self.reference_starts = []

    def validate(self, request) -> AnalysisValidationResponse:
        self.validated.append(request)
        model = "subaerial_proportion_bootstrap" if request.task == "time_series" else request.model.type
        tuning = "not_applicable" if request.task == "time_series" else request.tuning
        return AnalysisValidationResponse(
            validation_id=VALIDATION_ID,
            request_hash=REQUEST_HASH,
            canonical_contract_hash="2" * 64,
            compiled_plan_hash="3" * 64,
            validation_expires_at="2026-08-22T06:00:00+00:00",
            execution_ready=True,
            scientific_status="valid",
            adapter_status="available",
            artifact_status="planned",
            workflow_family=("time_series" if request.task == "time_series" else "supervised_learning"),
            workflow_mode=(request.mode if request.task == "time_series" else request.task),
            method=model,
            adapter_id="test-adapter",
            adapter_version="1",
            task=request.task,
            models=(model,),
            estimated_model_count=1,
            tuning=tuning,
            training_source="path",
            training_dataset_path=str(request.training_dataset_path),
            training_sha256="0" * 64,
            training_size_bytes=1,
            columns=("SampleID", "SIO2", "Label"),
            identifier_column=request.identifier_column,
            feature_columns=tuple(getattr(request, "feature_columns", ())),
            selected_columns=tuple(getattr(request, "resolved_selected_columns", ())),
            target_column=getattr(request, "target_column", None),
            target_columns=(),
            resolved_model_parameters={},
            experiment_mode="not_applicable" if request.task == "time_series" else "new",
            experiment_name=request.experiment_name,
            interaction_plan="fake-plan",
        )

    def start(self, request) -> StartAnalysisResponse:
        self.legacy_starts.append(request)
        model = "subaerial_proportion_bootstrap" if request.task == "time_series" else request.model.type
        return StartAnalysisResponse(
            run_id="run-0123456789abcdef",
            state="queued",
            models=(model,),
            estimated_model_count=1,
            status_hint="wait once",
            request_hash=None,
            started_from_validation=False,
        )

    def start_validated(self, validation_id: str, request_hash: str, *, expected_task=None) -> StartAnalysisResponse:
        self.reference_starts.append((validation_id, request_hash, expected_task))
        return StartAnalysisResponse(
            run_id="run-0123456789abcdef",
            state="queued",
            models=("subaerial_proportion_bootstrap",),
            estimated_model_count=1,
            status_hint="wait once",
            request_hash=request_hash,
            started_from_validation=True,
        )

    def get_status(self, run_id: str, *, wait_seconds: float = 0) -> RunStatusResponse:
        return RunStatusResponse(
            run_id=run_id,
            state="running",
            stage="running_cli",
            created_at="2026-08-22T05:00:00+00:00",
            started_at="2026-08-22T05:00:01+00:00",
            cli_pid=1234,
            progress_message=f"waited {wait_seconds}",
        )

    def get_result(
        self,
        run_id: str,
        *,
        wait_seconds: float = 0,
        artifact_offset: int = 0,
        artifact_limit: int | None = None,
    ) -> RunResultResponse:
        assert wait_seconds == 300
        assert artifact_offset == 0
        assert artifact_limit == 4
        return RunResultResponse(
            run_id=run_id,
            state="succeeded",
            task="time_series",
            model="subaerial_proportion_bootstrap",
            tuning="not_applicable",
            output_directory="C:/managed/output",
            interaction_trace="C:/managed/wrapper/interaction-trace.json",
            cli_stdout_log="C:/managed/wrapper/stdout.log",
            cli_stderr_log="C:/managed/wrapper/stderr.log",
            cli_exit_code=0,
            cli_version=CLI_VERSION,
            input_sha256="0" * 64,
            input_hash_verified=True,
            reported_metrics={"total_bins": 39},
            artifact_count=0,
            artifact_offset=0,
            returned_artifact_count=0,
            next_artifact_offset=None,
            artifacts=(),
            artifacts_truncated=False,
            limitations=(),
        )

    def cancel(self, run_id: str) -> CancelRunResponse:
        return CancelRunResponse(run_id=run_id, state="cancellation_requested", message="requested")

    def close(self) -> None:
        self.closed = True


def _compact_bytes(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _classification_request(dataset: Path) -> dict:
    return {
        "task": "classification",
        "training_dataset_path": str(dataset),
        "experiment_name": "C5 Protocol",
        "run_name": "Legacy",
        "identifier_column": "SampleID",
        "feature_columns": ["SIO2"],
        "target_column": "Label",
    }


def _annotation_paths(value: object, path: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    found: list[tuple[str, ...]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = (*path, key)
            if key in {"title", "description", "default", "examples"}:
                found.append(child_path)
            found.extend(_annotation_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_annotation_paths(child, (*path, str(index))))
    return found


@pytest.mark.anyio
async def test_advertised_time_series_schemas_remove_only_non_validation_annotations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SCHEMA_TASK_ENV, "time_series")
    server = create_server(
        McpSettings(runs_root=tmp_path / "runs", cli_executable=None),
        ProtocolRunManager(),
    )

    async with Client(server) as client:
        listing = await client.list_tools()
        schemas = {tool.name: tool.input_schema for tool in listing.tools}
        assert all(not _annotation_paths(schema) for schema in schemas.values())
        # The named environment-profile and deterministic filter contracts add
        # validation-bearing fields; keep the scoped schema under a tight revised budget.
        assert sum(_compact_bytes(schema) for schema in schemas.values()) < 12_300
        assert _compact_bytes(listing.model_dump(mode="json", by_alias=True, exclude_none=True)) < 13_700

        time_series = schemas["validate_analysis"]
        assert time_series["additionalProperties"] is False
        assert time_series["properties"]["task"]["const"] == "time_series"
        assert time_series["properties"]["bin_width"]["exclusiveMinimum"] == 0
        assert time_series["properties"]["training_dataset"]["anyOf"][0]["discriminator"]["propertyName"] == "source"
        assert schemas["start_analysis"]["additionalProperties"] is False
        assert set(schemas["start_analysis"]["required"]) == {
            "validation_id",
            "request_hash",
        }


@pytest.mark.anyio
async def test_start_schema_is_a_compact_validation_reference_and_legacy_start_remains_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(SCHEMA_TASK_ENV, raising=False)
    dataset = tmp_path / "rocks.csv"
    dataset.write_text("SampleID,SIO2,Label\nA,50.1,basalt\n", encoding="utf-8")
    manager = ProtocolRunManager()
    server = create_server(McpSettings(runs_root=tmp_path / "runs", cli_executable=None), manager)

    async with Client(server) as client:
        listing = await client.list_tools()
        tools = {tool.name: tool for tool in listing.tools}
        assert len(tools) == 13
        assert all(tool.output_schema is None for tool in tools.values())

        validate_schema = tools["validate_analysis"].input_schema
        start_schema = tools["start_analysis"].input_schema
        assert "title" not in validate_schema
        assert len(validate_schema["oneOf"]) == 6
        assert "title" not in start_schema
        assert start_schema["additionalProperties"] is False
        assert set(start_schema["properties"]) == {"validation_id", "request_hash"}
        assert set(start_schema["required"]) == {"validation_id", "request_hash"}
        assert _compact_bytes(start_schema) < 1000
        assert sum(_compact_bytes(tool.input_schema) for tool in tools.values()) < 65000

        preview = await client.call_tool("validate_analysis", _classification_request(dataset))
        assert preview.is_error is False
        assert preview.structured_content["validation_id"] == VALIDATION_ID
        assert preview.structured_content["request_hash"] == REQUEST_HASH

        preferred = await client.call_tool(
            "start_analysis",
            {"validation_id": VALIDATION_ID, "request_hash": REQUEST_HASH},
        )
        assert preferred.is_error is False
        assert preferred.structured_content["started_from_validation"] is True
        assert manager.reference_starts == [(VALIDATION_ID, REQUEST_HASH, None)]

        legacy = await client.call_tool("start_analysis", _classification_request(dataset))
        assert legacy.is_error is False
        assert legacy.structured_content["started_from_validation"] is False
        assert manager.legacy_starts[-1].task == "classification"

        invalid_alias = _classification_request(dataset)
        invalid_alias["random_seed"] = 2025
        rejected = await client.call_tool("start_analysis", invalid_alias)
        assert rejected.is_error is True
        assert len(manager.legacy_starts) == 1


@pytest.mark.anyio
async def test_time_series_schema_scope_is_strict_small_and_keeps_all_tool_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SCHEMA_TASK_ENV, "time_series")
    manager = ProtocolRunManager()
    server = create_server(McpSettings(runs_root=tmp_path / "runs", cli_executable=None), manager)

    async with Client(server) as client:
        listing = await client.list_tools()
        tools = {tool.name: tool for tool in listing.tools}
        assert len(tools) == 13
        validate_schema = tools["validate_analysis"].input_schema
        assert "title" not in validate_schema
        assert validate_schema["additionalProperties"] is False
        assert validate_schema["properties"]["task"]["const"] == "time_series"
        assert _compact_bytes(validate_schema) < 8000
        assert sum(_compact_bytes(tool.input_schema) for tool in tools.values()) < 15000

        capabilities = await client.call_tool("get_capabilities", {})
        assert capabilities.is_error is False
        assert capabilities.structured_content["analysis_schema_task_scope"] == "time_series"
        assert capabilities.structured_content["supported_tasks"] == [
            "classification",
            "regression",
            "clustering",
            "decomposition",
            "anomaly_detection",
            "time_series",
        ]

        rejected = await client.call_tool(
            "validate_analysis",
            _classification_request(tmp_path / "rocks.csv"),
        )
        assert rejected.is_error is True

        accepted = await client.call_tool(
            "validate_analysis",
            {
                "task": "time_series",
                "training_dataset_path": str(tmp_path / "time-series.csv"),
                "bin_width": 100,
            },
        )
        assert accepted.is_error is False
        assert manager.validated[-1].task == "time_series"

        started = await client.call_tool(
            "start_analysis",
            {"validation_id": VALIDATION_ID, "request_hash": REQUEST_HASH},
        )
        assert started.is_error is False
        assert manager.reference_starts[-1] == (VALIDATION_ID, REQUEST_HASH, "time_series")


def test_invalid_analysis_schema_scope_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SCHEMA_TASK_ENV, "unsupported")
    with pytest.raises(SettingsError, match=SCHEMA_TASK_ENV):
        create_server(
            McpSettings(runs_root=tmp_path / "runs", cli_executable=None),
            ProtocolRunManager(),
        )


@pytest.mark.anyio
async def test_bounded_result_wait_and_artifact_page_are_forwarded_without_polling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(SCHEMA_TASK_ENV, raising=False)
    manager = ProtocolRunManager()
    server = create_server(McpSettings(runs_root=tmp_path / "runs", cli_executable=None), manager)
    async with Client(server) as client:
        result = await client.call_tool(
            "get_run_result",
            {
                "run_id": "run-0123456789abcdef",
                "wait_seconds": 300,
                "artifact_offset": 0,
                "artifact_limit": 4,
            },
        )
        assert result.is_error is False
        assert result.structured_content["returned_artifact_count"] == 0
