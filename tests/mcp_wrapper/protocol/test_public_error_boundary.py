import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from geochemistrypi_mcp.api.output_contracts import PublicToolErrorResponse
from geochemistrypi_mcp.api.schemas import RunResultResponse, RunStatusResponse
from geochemistrypi_mcp.config.settings import McpSettings, SettingsError
from geochemistrypi_mcp.contracts.manifest import CapabilityManifestError
from geochemistrypi_mcp.data.catalog import DatasetCatalogError
from geochemistrypi_mcp.data.inspector import DatasetInspectionError
from geochemistrypi_mcp.data.preparation import DatasetPreparationError
from geochemistrypi_mcp.planning.interaction_plan import PlanCompilationError
from geochemistrypi_mcp.runtime.cli_driver import CliDriverError
from geochemistrypi_mcp.runtime.environment import EnvironmentInspectionError
from geochemistrypi_mcp.runtime.runs import InputIntegrityError, RunManager, RunNotFoundError, RunStateError
from geochemistrypi_mcp.server import create_server
from geochemistrypi_mcp.tracking.experiments import ExperimentStoreError
from geochemistrypi_mcp.tracking.ui import MlflowUiError
from mcp import Client
from pydantic import ValidationError as PydanticValidationError


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _IncompleteScientificResultManager:
    def __init__(self) -> None:
        valid = RunResultResponse(
            run_id="run-0123456789abcdef",
            result_record_path="C:/managed/wrapper/result.json",
            result_record_sha256="a" * 64,
            request_hash="b" * 64,
            validation_id="val-" + "c" * 32,
            canonical_contract_hash="d" * 64,
            compiled_plan_hash="e" * 64,
            scientific_contract_id="scientific-contract-v4/supervised_learning/classification/logistic_regression",
            scientific_execution_contract_bound=True,
            provenance_manifest_path="C:/managed/wrapper/scientific-run-manifest.json",
            provenance_manifest_sha256="f" * 64,
            state="succeeded",
            task="classification",
            model="logistic_regression",
            output_directory="C:/managed/output",
            interaction_trace="C:/managed/wrapper/interaction-trace.json",
            cli_stdout_log="C:/managed/wrapper/stdout.log",
            cli_stderr_log="C:/managed/wrapper/stderr.log",
            cli_exit_code=0,
            cli_started_at="2026-08-30T00:00:00+00:00",
            cli_finished_at="2026-08-30T00:00:01+00:00",
            cli_execution_duration_seconds=1.0,
            cli_version="0.8.1",
            input_sha256="0" * 64,
            input_hash_verified=True,
            reported_metrics={},
            artifact_count=0,
            canonical_artifact_count=0,
            artifact_index_path="C:/managed/wrapper/artifact-index.json",
            artifact_index_sha256="1" * 64,
            artifact_view="canonical",
            artifact_view_count=0,
            artifacts=(),
            artifacts_truncated=False,
            limitations=(),
        )
        # Simulate an object loaded from a pre-contract or corrupt durable
        # record. model_copy deliberately does not revalidate updates.
        self.result = valid.model_copy(update={"validation_id": None})
        self.closed = False

    def get_result(self, *_args, **_kwargs):
        return self.result

    def close(self) -> None:
        self.closed = True


@pytest.mark.anyio
@pytest.mark.parametrize(
    "arguments",
    (
        {"run_id": "run-0123456789abcdef", "detail": "full"},
        {"run_id": "run-0123456789abcdef"},
        {
            "run_id": "run-0123456789abcdef",
            "if_result_sha256": "a" * 64,
        },
    ),
    ids=("full", "compact", "conditional"),
)
async def test_incomplete_success_identity_is_the_same_typed_public_error_for_every_view(
    tmp_path: Path,
    arguments: dict[str, object],
) -> None:
    settings = McpSettings(runs_root=tmp_path / "runs", cli_executable=None)
    runs = _IncompleteScientificResultManager()
    server = create_server(settings, runs)

    async with Client(server) as client:
        result = await client.call_tool("get_run_result", arguments)

    assert runs.closed is True
    assert result.is_error is True
    structured = PublicToolErrorResponse.model_validate(result.structured_content)
    assert structured.result_type == "run_state_invalid"
    assert structured.retryable is False
    assert "immutable successful result identity is incomplete" in result.content[0].text


@pytest.mark.anyio
async def test_long_public_settings_error_is_bounded_and_hashes_the_complete_cause(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_prefix = "A" * 6000
    complete_messages = (
        f"{shared_prefix}-first-complete-tail",
        f"{shared_prefix}-second-complete-tail",
    )
    boundary_message = "B" * 257
    messages = iter((*complete_messages, boundary_message))
    settings = McpSettings(runs_root=tmp_path / "runs", cli_executable=None)
    runs = RunManager(settings)

    def fail_with_long_public_error(*_args, **_kwargs):
        raise SettingsError(next(messages))

    monkeypatch.setattr(runs, "get_status", fail_with_long_public_error)
    server = create_server(settings, runs)

    async with Client(server) as client:
        first = await client.call_tool(
            "get_run_status",
            {"run_id": "run-0123456789abcdef"},
        )
        second = await client.call_tool(
            "get_run_status",
            {"run_id": "run-0123456789abcdef"},
        )
        boundary = await client.call_tool(
            "get_run_status",
            {"run_id": "run-0123456789abcdef"},
        )

    for result, complete_message, tail in (
        (first, complete_messages[0], "first-complete-tail"),
        (second, complete_messages[1], "second-complete-tail"),
    ):
        assert result.is_error is True
        assert "internal wrapper error" not in result.content[0].text
        assert tail not in result.content[0].text
        assert "Valid alternatives" in result.content[0].text
        assert "Next action" in result.content[0].text
        assert len(result.content[0].text) <= 4000
        assert len(result.content[0].text.encode("utf-8")) <= 4000

        structured = PublicToolErrorResponse.model_validate(result.structured_content)
        assert structured.error_type == "request_error"
        assert structured.tool_name == "get_run_status"
        cause = structured.root_causes[0]
        assert len(cause.problem) <= 256
        assert cause.problem.endswith("…")
        assert cause.problem_truncated is True
        assert cause.problem_total_utf8_bytes == len(complete_message.encode("utf-8"))
        assert cause.problem_sha256 == hashlib.sha256(complete_message.encode("utf-8")).hexdigest()
        assert len(cause.problem.encode("utf-8")) < cause.problem_total_utf8_bytes
        assert (
            len(
                json.dumps(
                    result.structured_content,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            <= 64 * 1024
        )

    assert first.structured_content["root_causes_sha256"] != second.structured_content["root_causes_sha256"]
    assert first.structured_content["root_causes"][0]["problem"] == second.structured_content["root_causes"][0]["problem"]
    assert first.structured_content["root_causes"][0]["problem_sha256"] != second.structured_content["root_causes"][0]["problem_sha256"]

    boundary_structured = PublicToolErrorResponse.model_validate(boundary.structured_content)
    boundary_cause = boundary_structured.root_causes[0]
    assert boundary.is_error is True
    assert boundary_cause.problem_truncated is True
    assert boundary_cause.problem.endswith("…")
    assert len(boundary_cause.problem) <= 256
    assert len(boundary_cause.problem.encode("utf-8")) == 256
    assert boundary_cause.problem_total_utf8_bytes == 257
    assert boundary_cause.problem_sha256 == hashlib.sha256(boundary_message.encode("utf-8")).hexdigest()

    false_flag = deepcopy(first.structured_content)
    false_flag["root_causes"][0]["problem_truncated"] = False
    with pytest.raises(PydanticValidationError, match="untruncated problem"):
        PublicToolErrorResponse.model_validate(false_flag)

    false_length = deepcopy(first.structured_content)
    false_length["root_causes"][0]["problem_total_utf8_bytes"] = len(false_length["root_causes"][0]["problem"].encode("utf-8"))
    with pytest.raises(PydanticValidationError, match="truncated problem"):
        PublicToolErrorResponse.model_validate(false_length)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("error", "result_type", "guidance"),
    (
        (
            InputIntegrityError("selected dataset changed"),
            "input_integrity_changed",
            "Do not reuse this validation receipt",
        ),
        (
            EnvironmentInspectionError("dependency inventory is incomplete"),
            "environment_inspection_failed",
            "geochemistrypi-mcp-doctor",
        ),
        (
            CapabilityManifestError("manifest parity identity is invalid"),
            "capability_manifest_invalid",
            "Do not infer capabilities",
        ),
        (
            RunNotFoundError("run does not exist"),
            "run_not_found",
            "Do not guess run IDs",
        ),
        (
            RunStateError("run state is inconsistent"),
            "run_state_invalid",
            "Do not repeat the failed state-changing call",
        ),
        (
            DatasetCatalogError("dataset id is unavailable"),
            "dataset_catalog_failed",
            "Do not guess a dataset ID",
        ),
        (
            DatasetInspectionError("worksheet is unavailable"),
            "dataset_inspection_failed",
            "Do not repeat the same inspection",
        ),
        (
            DatasetPreparationError("selected column is absent"),
            "dataset_preparation_failed",
            "Do not insert defaults",
        ),
        (
            PlanCompilationError("workflow combination is unsupported"),
            "plan_compilation_failed",
            "Do not substitute a model",
        ),
        (
            CliDriverError(
                "CLI exited before terminal output",
                Path("isolated-workspace"),
            ),
            "cli_execution_failed",
            "Preserve the run directory",
        ),
        (
            ExperimentStoreError("tracking store is unreadable"),
            "experiment_store_failed",
            "Do not guess an experiment ID",
        ),
        (
            MlflowUiError("managed port is unavailable"),
            "mlflow_ui_failed",
            "Do not loop start/stop calls",
        ),
        (
            SettingsError("service state root is absent"),
            "settings_invalid",
            "geochemistrypi-mcp-doctor",
        ),
    ),
)
async def test_runtime_public_errors_are_typed_actionable_and_side_effect_free(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
    result_type: str,
    guidance: str,
) -> None:
    settings = McpSettings(runs_root=tmp_path / "runs", cli_executable=None)
    runs = RunManager(settings)

    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(runs, "get_status", fail)
    server = create_server(settings, runs)
    before = tuple(tmp_path.rglob("*"))

    async with Client(server) as client:
        result = await client.call_tool(
            "get_run_status",
            {"run_id": "run-0123456789abcdef"},
        )

    assert tuple(tmp_path.rglob("*")) == before
    assert result.is_error is True
    structured = PublicToolErrorResponse.model_validate(result.structured_content)
    assert structured.error_schema_version == 2
    assert structured.error_type == "request_error"
    assert structured.result_type == result_type
    assert structured.retryable is False
    assert guidance in structured.next_action
    assert "Resolve safe defaults automatically" not in structured.next_action
    assert "internal wrapper error" not in result.content[0].text


@pytest.mark.anyio
@pytest.mark.parametrize("failure_kind", ("backend_validation", "assertion"))
async def test_unknown_backend_failures_remain_internal(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    settings = McpSettings(runs_root=tmp_path / "runs", cli_executable=None)
    runs = RunManager(settings)

    def fail(*_args, **_kwargs):
        if failure_kind == "backend_validation":
            RunStatusResponse.model_validate({"run_id": "invalid"})
        raise AssertionError("unknown implementation bug")

    monkeypatch.setattr(runs, "get_status", fail)
    server = create_server(settings, runs)

    async with Client(server) as client:
        result = await client.call_tool(
            "get_run_status",
            {"run_id": "run-0123456789abcdef"},
        )

    assert result.is_error is True
    assert "internal wrapper error" in result.content[0].text
    assert not result.structured_content
