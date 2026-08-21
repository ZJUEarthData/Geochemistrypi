import sys
from pathlib import Path

import geochemistrypi_mcp
import pytest
from geochemistrypi_mcp.api.schemas import (
    AnalysisValidationResponse,
    AnomalyDetectionRequest,
    CancelRunResponse,
    ClassificationRequest,
    ClusteringRequest,
    DecompositionRequest,
    RegressionRequest,
    RunResultResponse,
    RunStatusResponse,
    StartAnalysisResponse,
)
from geochemistrypi_mcp.config.constants import CLI_VERSION, SERVER_VERSION
from geochemistrypi_mcp.config.settings import McpSettings
from geochemistrypi_mcp.server import SERVER_INSTRUCTIONS, create_server
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _assert_strict_object_schemas(schema: dict) -> None:
    if schema.get("type") == "object" and "properties" in schema:
        assert schema.get("additionalProperties") is False
    for value in schema.values():
        if isinstance(value, dict):
            _assert_strict_object_schemas(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _assert_strict_object_schemas(item)


def test_server_keeps_the_user_conversation_non_technical() -> None:
    instructions = " ".join(SERVER_INSTRUCTIONS.lower().split())

    assert "short" in instructions
    assert "ordinary-language" in instructions
    assert "one brief question at a time" in instructions
    assert "plain language" in instructions
    assert "wait for confirmation" in instructions
    assert "does not need to know mcp tool names, json" in instructions
    assert "treat every explicit scientific choice as authoritative" in instructions
    assert "infer or default only choices the user omitted" in instructions
    assert "rather than silently substituting another choice" in instructions
    assert "never expose implementation details" in instructions


class FakeRunManager:
    def __init__(self) -> None:
        self.closed = False
        self.requests: list[ClassificationRequest | RegressionRequest | ClusteringRequest | DecompositionRequest | AnomalyDetectionRequest] = []

    def start(
        self,
        request: ClassificationRequest | RegressionRequest | ClusteringRequest | DecompositionRequest | AnomalyDetectionRequest,
    ) -> StartAnalysisResponse:
        self.requests.append(request)
        return StartAnalysisResponse(
            run_id="run-0123456789abcdef",
            state="queued",
            models=(request.model.type,),
            estimated_model_count=1,
            status_hint="poll",
        )

    def validate(self, request) -> AnalysisValidationResponse:
        return AnalysisValidationResponse(
            task=request.task,
            models=(request.model.type,),
            estimated_model_count=1,
            tuning=request.tuning,
            training_source="path",
            training_dataset_path=str(request.training_dataset_path),
            training_sha256="0" * 64,
            training_size_bytes=1,
            columns=("SampleID", "SIO2", "Label"),
            identifier_column=request.identifier_column,
            feature_columns=request.feature_columns,
            target_column=request.target_column,
            target_columns=(request.resolved_target_columns if isinstance(request, RegressionRequest) else ((request.target_column,) if isinstance(request, ClassificationRequest) else ())),
            resolved_model_parameters=request.model.model_dump(mode="python", exclude={"type"}),
            experiment_mode="new",
            experiment_name=request.experiment_name,
            interaction_plan="fake-plan",
        )

    def get_status(self, run_id: str) -> RunStatusResponse:
        return RunStatusResponse(
            run_id=run_id,
            state="running",
            stage="running_cli",
            created_at="2026-08-02T12:00:00+00:00",
            started_at="2026-08-02T12:00:01+00:00",
            cli_pid=1234,
            progress_message="running",
        )

    def get_result(self, run_id: str) -> RunResultResponse:
        return RunResultResponse(
            run_id=run_id,
            state="succeeded",
            task="classification",
            model="logistic_regression",
            output_directory="C:/managed/output",
            interaction_trace="C:/managed/wrapper/interaction-trace.json",
            cli_stdout_log="C:/managed/wrapper/stdout.log",
            cli_stderr_log="C:/managed/wrapper/stderr.log",
            cli_exit_code=0,
            cli_version=CLI_VERSION,
            input_sha256="0" * 64,
            input_hash_verified=True,
            reported_metrics={"bounded_text_guard": "x" * 5000},
            artifact_count=0,
            artifacts=(),
            artifacts_truncated=False,
            limitations=(),
        )

    def cancel(self, run_id: str) -> CancelRunResponse:
        return CancelRunResponse(run_id=run_id, state="cancellation_requested", message="requested")

    def close(self) -> None:
        self.closed = True


@pytest.mark.anyio
async def test_tool_discovery_strict_validation_and_structured_results(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "rocks.csv"
    dataset.write_text("SampleID,SIO2,Label\nA,50.1,basalt\n", encoding="utf-8")
    settings = McpSettings(runs_root=tmp_path / "runs", cli_executable=None)
    fake_runs = FakeRunManager()
    server = create_server(settings, fake_runs)

    async with Client(server) as client:
        listing = await client.list_tools()
        tools = {tool.name: tool for tool in listing.tools}
        assert set(tools) == {
            "cancel_run",
            "get_capabilities",
            "get_run_result",
            "get_run_status",
            "inspect_dataset",
            "list_datasets",
            "list_experiments",
            "get_experiment",
            "start_mlflow_ui",
            "mlflow_ui_status",
            "stop_mlflow_ui",
            "validate_analysis",
            "start_analysis",
        }
        request_schema = tools["start_analysis"].input_schema
        assert request_schema["title"] == "AnalysisRequest"
        assert request_schema["type"] == "object"
        assert request_schema["discriminator"]["propertyName"] == "task"
        assert len(request_schema["oneOf"]) == 6
        assert request_schema["$defs"]["ClassificationRequest"]["additionalProperties"] is False
        assert request_schema["$defs"]["RegressionRequest"]["additionalProperties"] is False
        regression_schema = request_schema["$defs"]["RegressionRequest"]
        assert "target_columns" in regression_schema["properties"]
        assert "target_column" not in regression_schema["required"]
        assert "target_columns" not in regression_schema["required"]
        assert "Do not combine" in regression_schema["properties"]["target_columns"]["description"]
        assert request_schema["$defs"]["ClusteringRequest"]["additionalProperties"] is False
        assert request_schema["$defs"]["DecompositionRequest"]["additionalProperties"] is False
        assert request_schema["$defs"]["AnomalyDetectionRequest"]["additionalProperties"] is False
        for analysis_tool in ("validate_analysis", "start_analysis"):
            analysis_schema = tools[analysis_tool].input_schema
            assert analysis_schema == request_schema
            _assert_strict_object_schemas(analysis_schema)
            time_series_schema = analysis_schema["$defs"]["TimeSeriesRequest"]
            assert time_series_schema["additionalProperties"] is False
            assert "top-level" in time_series_schema["description"]
            properties = time_series_schema["properties"]
            assert {
                "task",
                "training_dataset",
                "bin_width",
                "iterations",
                "seed",
                "age_column",
                "maximum_age_column",
                "probability_column",
                "latitude_column",
                "longitude_column",
                "identifier_column",
                "selected_columns",
                "missing_values",
                "feature_engineering",
                "age_unit",
                "fit_curve",
                "experiment_name",
                "run_name",
            } <= set(properties)
            assert {
                "dataset",
                "model",
                "model_parameters",
                "bin_width_ma",
                "bootstrap_iterations",
                "random_seed",
            }.isdisjoint(properties)
            assert "top-level" in properties["training_dataset"]["description"]
            assert "top-level" in properties["bin_width"]["description"]
            assert "iterations" in properties["iterations"]["description"]
            assert "seed" in properties["seed"]["description"]
            assert "Time Series" in tools[analysis_tool].description
            assert "training_dataset" in tools[analysis_tool].description
            assert "bin_width" in tools[analysis_tool].description
        builtin_dataset = request_schema["$defs"]["BuiltInDatasetReference"]
        assert "builtin:" in builtin_dataset["properties"]["dataset_id"]["description"]

        capabilities = await client.call_tool("get_capabilities", {})
        assert capabilities.is_error is False
        assert capabilities.structured_content["supported_tasks"] == [
            "classification",
            "regression",
            "clustering",
            "decomposition",
            "anomaly_detection",
            "time_series",
        ]
        assert capabilities.structured_content["capability_manifest_schema_version"] == 1
        assert capabilities.structured_content["cli_automation_contract_version"] == 1
        assert capabilities.structured_content["supported_data_sources"] == [
            "path",
            "builtin",
            "desktop",
        ]
        assert len(capabilities.structured_content["supported_clients"]) == 14
        assert "cursor" in capabilities.structured_content["supported_clients"]
        assert "task.time_series" not in capabilities.structured_content["known_gaps"]
        assert "branch.world_map" not in capabilities.structured_content["known_gaps"]
        assert "branch.all_models" not in capabilities.structured_content["known_gaps"]
        assert all(item["status"] == "verified" and item["evidence"] for item in capabilities.structured_content["capabilities"] if item["mcp_supported"])
        assert len(capabilities.structured_content["supported_models_by_task"]["classification"]) == 11
        assert len(capabilities.structured_content["supported_models_by_task"]["regression"]) == 15
        assert capabilities.structured_content["supported_models_by_task"]["clustering"] == [
            "kmeans",
            "dbscan",
            "agglomerative",
            "affinity_propagation",
            "mean_shift",
        ]
        assert capabilities.structured_content["supported_models_by_task"]["decomposition"] == [
            "pca",
            "tsne",
            "mds",
        ]
        assert capabilities.structured_content["supported_models_by_task"]["anomaly_detection"] == [
            "isolation_forest",
            "local_outlier_factor",
        ]
        assert capabilities.structured_content["classification_options"]["sample_balancing"] == ["none"]
        assert capabilities.structured_content["regression_options"]["target_columns"] == [
            "single_numeric",
            "multiple_numeric",
        ]
        assert capabilities.structured_content["clustering_options"]["target_columns"] == ["not_applicable"]
        assert capabilities.structured_content["decomposition_options"]["transformed_data"] == ["enabled"]
        assert capabilities.structured_content["anomaly_detection_options"]["detection_labels"] == [
            "1_inlier",
            "-1_outlier",
        ]
        assert capabilities.structured_content["compatibility"] == {
            "schema_version": 2,
            "release_channel": "stable",
            "public_release_ready": True,
            "mcp_python_requires": ">=3.10,<4",
            "cli_python_requires": ">=3.9,<3.10",
            "mcp_sdk_requires": "==2.0.0",
            "supported_cli_versions": [CLI_VERSION],
            "interaction_plan_version": 1,
            "cli_automation_contract_version": 1,
            "artifact_index_schema_version": 1,
            "target_operating_systems": ["windows", "linux", "macos"],
            "pending_release_gates": [],
        }
        assert capabilities.structured_content["resource_limits"] == {
            "maximum_dataset_bytes": 536870912,
            "maximum_columns": 256,
            "maximum_artifact_references": 200,
            "maximum_concurrent_runs": 1,
            "maximum_pending_runs": 8,
            "maximum_process_seconds": 900,
        }
        assert any("sample_balancing" in item for item in capabilities.structured_content["unsupported_interactions"])
        assert any("clustering.optics" in item for item in capabilities.structured_content["unsupported_interactions"])
        assert any("decomposition.automl" in item for item in capabilities.structured_content["unsupported_interactions"])
        assert any("anomaly_detection.automl" in item for item in capabilities.structured_content["unsupported_interactions"])

        inspection = await client.call_tool(
            "inspect_dataset",
            {"dataset_path": str(dataset), "sample_rows": 1},
        )
        assert inspection.is_error is False
        assert inspection.structured_content["row_count"] == 1
        assert inspection.structured_content["sample_rows"][0]["SampleID"] == "A"

        invalid = await client.call_tool(
            "inspect_dataset",
            {"dataset_path": str(dataset), "sample_rows": 1, "unexpected": True},
        )
        assert invalid.is_error is True
        assert "Invalid field 'unexpected'" in invalid.content[0].text
        assert "Actual value" in invalid.content[0].text
        assert "Valid alternatives" in invalid.content[0].text
        assert "Next action" in invalid.content[0].text
        assert "tools/list" not in invalid.content[0].text
        assert "do not ask the user to edit JSON" in invalid.content[0].text

        missing = await client.call_tool("get_run_status", {})
        assert missing.is_error is True
        assert "Resolve it from the conversation or earlier tool results" in missing.content[0].text
        assert "never ask the user for a schema field name or JSON" in missing.content[0].text

        malformed_time_series = {
            "task": "time_series",
            "dataset": {
                "source": "builtin",
                "dataset_id": "builtin:time_series",
            },
            "model": "subaerial_proportion_bootstrap",
            "model_parameters": {"bin_width_ma": 100},
            "bootstrap_iterations": 100,
            "random_seed": 2025,
        }
        invalid_time_series = await client.call_tool(
            "validate_analysis",
            malformed_time_series,
        )
        assert invalid_time_series.is_error is True
        validation_text = invalid_time_series.content[0].text
        assert "time_series.bin_width" in validation_text
        assert "time_series.dataset" in validation_text
        assert "training_dataset" in validation_text
        assert "time_series.model" in validation_text
        assert "time_series.model_parameters" in validation_text
        assert "bin_width" in validation_text
        assert "time_series.bootstrap_iterations" in validation_text
        assert "iterations" in validation_text
        assert "time_series.random_seed" in validation_text
        assert "seed" in validation_text
        assert validation_text.count("Next action:") == 1
        assert len(validation_text) <= 3035

        requests_before_invalid_start = len(fake_runs.requests)
        invalid_start = await client.call_tool(
            "start_analysis",
            malformed_time_series,
        )
        assert invalid_start.is_error is True
        assert len(fake_runs.requests) == requests_before_invalid_start

        bounded_invalid = await client.call_tool(
            "validate_analysis",
            {
                "task": "time_series",
                "training_dataset": {
                    "source": "builtin",
                    "dataset_id": "time_series",
                },
                "bin_width": 0,
                "iterations": 0,
                "age_unit": "years",
                **{f"unsupported_{index}": "sensitive-value" for index in range(20)},
            },
        )
        assert bounded_invalid.is_error is True
        bounded_text = bounded_invalid.content[0].text
        assert "pattern" in bounded_text
        assert "range" in bounded_text
        assert "literal" in bounded_text
        assert "Additional issues omitted" in bounded_text
        assert "sensitive-value" not in bounded_text
        assert bounded_text.count("Next action:") == 1
        assert len(bounded_text) <= 3035

        preview = await client.call_tool(
            "validate_analysis",
            {
                "training_dataset_path": str(dataset),
                "experiment_name": "Protocol",
                "run_name": "Reference",
                "identifier_column": "SampleID",
                "feature_columns": ["SIO2"],
                "target_column": "Label",
            },
        )
        assert preview.is_error is False
        assert preview.structured_content["analysis_process_started"] is False
        assert preview.structured_content["estimated_model_count"] == 1
        assert preview.structured_content["identifier_column"] == "SampleID"
        assert preview.structured_content["feature_columns"] == ["SIO2"]
        assert preview.structured_content["target_column"] == "Label"
        assert preview.structured_content["resolved_model_parameters"]["solver"] == "lbfgs"

        started = await client.call_tool(
            "start_analysis",
            {
                "training_dataset_path": str(dataset),
                "experiment_name": "Protocol",
                "run_name": "Reference",
                "identifier_column": "SampleID",
                "feature_columns": ["SIO2"],
                "target_column": "Label",
            },
        )
        assert started.is_error is False
        assert started.structured_content["state"] == "queued"
        assert fake_runs.requests[-1].task == "classification"

        regression_started = await client.call_tool(
            "start_analysis",
            {
                "task": "regression",
                "training_dataset_path": str(dataset),
                "experiment_name": "Protocol",
                "run_name": "Regression",
                "identifier_column": "SampleID",
                "feature_columns": ["SIO2"],
                "target_column": "Label",
                "model": {"type": "ridge_regression"},
            },
        )
        assert regression_started.is_error is False
        assert fake_runs.requests[-1].task == "regression"

        multi_regression_started = await client.call_tool(
            "start_analysis",
            {
                "task": "regression",
                "training_dataset_path": str(dataset),
                "experiment_name": "Protocol",
                "run_name": "Multi Regression",
                "identifier_column": "SampleID",
                "feature_columns": ["SIO2"],
                "target_columns": ["Label", "SIO2_Second_Target"],
                "model": {"type": "ridge_regression"},
            },
        )
        assert multi_regression_started.is_error is False
        assert fake_runs.requests[-1].resolved_target_columns == ("Label", "SIO2_Second_Target")

        clustering_started = await client.call_tool(
            "start_analysis",
            {
                "task": "clustering",
                "training_dataset_path": str(dataset),
                "experiment_name": "Protocol",
                "run_name": "Clustering",
                "identifier_column": "SampleID",
                "feature_columns": ["SIO2"],
                "model": {"type": "dbscan"},
            },
        )
        assert clustering_started.is_error is False
        assert fake_runs.requests[-1].task == "clustering"

        decomposition_started = await client.call_tool(
            "start_analysis",
            {
                "task": "decomposition",
                "training_dataset_path": str(dataset),
                "experiment_name": "Protocol",
                "run_name": "Decomposition",
                "identifier_column": "SampleID",
                "feature_columns": ["SIO2"],
                "model": {"type": "pca"},
            },
        )
        assert decomposition_started.is_error is False
        assert fake_runs.requests[-1].task == "decomposition"

        result = await client.call_tool(
            "get_run_result",
            {"run_id": "run-0123456789abcdef"},
        )
        assert result.is_error is False
        assert len(result.content[0].text) == 4000
        assert result.structured_content["reported_metrics"]["bounded_text_guard"] == "x" * 5000
    assert fake_runs.closed is True


@pytest.mark.anyio
async def test_real_stdio_server_reserves_stdout_for_protocol_after_tool_error(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "rocks.csv"
    dataset.write_text("SampleID,SIO2,Label\nA,50.1,basalt\n", encoding="utf-8")
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "geochemistrypi_mcp"],
        env={
            "GEOCHEMISTRYPI_MCP_RUNS_ROOT": str(tmp_path / "runs"),
            "GEOCHEMISTRYPI_CLI_EXECUTABLE": str(tmp_path / "missing-geochemistrypi"),
            "PYTHONPATH": str(Path(geochemistrypi_mcp.__file__).resolve().parent.parent),
        },
    )

    async with Client(stdio_client(parameters)) as client:
        before = await client.call_tool("get_capabilities", {})
        assert before.is_error is False
        failed = await client.call_tool(
            "start_analysis",
            {
                "training_dataset_path": str(dataset),
                "experiment_name": "Protocol",
                "run_name": "No CLI",
                "identifier_column": "SampleID",
                "feature_columns": ["SIO2"],
                "target_column": "Label",
            },
        )
        assert failed.is_error is True
        assert "does not exist" in failed.content[0].text
        assert str(tmp_path) not in failed.content[0].text
        assert "<local-path>" in failed.content[0].text
        after = await client.call_tool("get_capabilities", {})
        assert after.is_error is False
        assert after.structured_content["server_version"] == SERVER_VERSION
