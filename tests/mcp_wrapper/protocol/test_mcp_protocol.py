import hashlib
import json
import sys
from pathlib import Path

import geochemistrypi_mcp
import pytest
from geochemistrypi_mcp.api.directory_views import (
    GetExperimentViewRequest,
    ListDatasetsViewRequest,
    ListExperimentsViewRequest,
    get_experiment_response_view,
    list_datasets_response_view,
    list_experiments_response_view,
)
from geochemistrypi_mcp.api.response_views import dataset_inspection_response_view
from geochemistrypi_mcp.api.schemas import (
    AnalysisValidationResponse,
    AnomalyDetectionRequest,
    CancelRunResponse,
    CapabilitiesNotModifiedResponse,
    ClassificationRequest,
    ClusteringRequest,
    CompactRunResultResponse,
    DatasetInspectionRequest,
    DatasetInspectionResponse,
    DecompositionRequest,
    ExperimentSummary,
    GetExperimentResponse,
    ListDatasetsResponse,
    ListExperimentsResponse,
    MlflowUiStatusResponse,
    RegressionRequest,
    RunResultResponse,
    RunStatusResponse,
    StartAnalysisResponse,
)
from geochemistrypi_mcp.api.terminal_receipts import TerminalRunReceipt, terminal_error_projection
from geochemistrypi_mcp.api.tools import full_output_contract_schema
from geochemistrypi_mcp.api.validation_views import compact_analysis_validation
from geochemistrypi_mcp.config.constants import CLI_VERSION, SERVER_VERSION
from geochemistrypi_mcp.config.settings import McpSettings
from geochemistrypi_mcp.server import SERVER_INSTRUCTIONS, create_server
from jsonschema import Draft202012Validator
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


def test_server_keeps_the_user_conversation_non_technical() -> None:
    instructions = " ".join(SERVER_INSTRUCTIONS.lower().split())

    assert "short" in instructions
    assert "ordinary-language" in instructions
    assert "one brief question at a time" in instructions
    assert "plain language" in instructions
    assert "wait for confirmation" in instructions
    assert "explicit request to run or execute" in instructions
    assert "must not trigger a second confirmation turn" in instructions
    assert "do not add an inspection call as a ritual before validation" in instructions
    assert "does not need to know mcp tool names, json" in instructions
    assert "treat every explicit scientific choice as authoritative" in instructions
    assert "infer or default only choices the user omitted" in instructions
    assert "rather than silently substituting another choice" in instructions
    assert "never expose implementation details" in instructions
    assert "one bounded result wait" in instructions
    assert "never start a blocked validation receipt" in instructions
    assert "detail=full" in instructions
    assert "instead of repeating the same validation" in instructions
    assert "do not fetch the same terminal page twice" in instructions
    assert "never describe partial_failure" in instructions
    assert "conditional result field" in instructions
    assert "never poll in a tight loop" in instructions
    assert "pending receipt" in instructions
    assert "not as a tool or scientific failure" in instructions


class FakeRunManager:
    def __init__(self) -> None:
        self.closed = False
        self.requests: list[ClassificationRequest | RegressionRequest | ClusteringRequest | DecompositionRequest | AnomalyDetectionRequest] = []
        self.validation_calls = 0
        self.validation_detail_calls = 0
        self.validation_response: AnalysisValidationResponse | None = None

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
        self.validation_calls += 1
        response = AnalysisValidationResponse(
            validation_id="val-0123456789abcdef0123456789abcdef",
            request_hash="1" * 64,
            canonical_contract_hash="2" * 64,
            compiled_plan_hash="3" * 64,
            validation_expires_at="2026-08-22T06:00:00+00:00",
            execution_ready=True,
            scientific_status="valid",
            adapter_status="available",
            artifact_status="planned",
            workflow_family=("supervised_learning" if request.task in {"classification", "regression"} else "dimension_reduction" if request.task == "decomposition" else request.task),
            workflow_mode=request.task,
            method=request.model.type,
            scientific_contract_id="scientific-contract-v2/test",
            adapter_id="test-adapter",
            adapter_version="1",
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
        self.validation_response = response
        return response

    def get_validation_detail(
        self,
        validation_id: str,
        request_hash: str,
        *,
        expected_task: str | None = None,
    ) -> AnalysisValidationResponse:
        self.validation_detail_calls += 1
        assert self.validation_response is not None
        assert validation_id == self.validation_response.validation_id
        assert request_hash == self.validation_response.request_hash
        assert expected_task is None or expected_task == self.validation_response.task
        return self.validation_response

    def get_status(self, run_id: str, *, wait_seconds: float = 0) -> RunStatusResponse:
        return RunStatusResponse(
            run_id=run_id,
            state="running",
            stage="running_cli",
            created_at="2026-08-02T12:00:00+00:00",
            started_at="2026-08-02T12:00:01+00:00",
            cli_pid=1234,
            progress_message="running",
        )

    def get_result(
        self,
        run_id: str,
        *,
        wait_seconds: float = 0,
        artifact_offset: int = 0,
        artifact_limit: int | None = None,
        artifact_view: str = "canonical",
    ) -> RunResultResponse:
        return RunResultResponse(
            request_hash="1" * 64,
            validation_id="val-" + "2" * 32,
            canonical_contract_hash="3" * 64,
            compiled_plan_hash="4" * 64,
            scientific_contract_id="scientific-contract-v4/supervised_learning/classification/logistic_regression",
            scientific_execution_contract_bound=True,
            provenance_manifest_path="C:/managed/wrapper/provenance-manifest.json",
            provenance_manifest_sha256="5" * 64,
            run_id=run_id,
            result_record_path="C:/managed/wrapper/result.json",
            result_record_sha256="4" * 64,
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
            cli_version=CLI_VERSION,
            input_sha256="0" * 64,
            input_hash_verified=True,
            reported_metrics={"bounded_text_guard": "x" * 5000},
            artifact_count=0,
            canonical_artifact_count=0,
            artifact_index_path="C:/managed/wrapper/artifact-index.json",
            artifact_index_sha256="5" * 64,
            artifact_view=artifact_view,
            artifact_view_count=0,
            artifacts=(),
            artifacts_truncated=False,
            limitations=(),
        )

    def cancel(self, run_id: str) -> CancelRunResponse:
        return CancelRunResponse(run_id=run_id, state="cancellation_requested", message="requested")

    def close(self) -> None:
        self.closed = True


class FailedRunManager(FakeRunManager):
    def get_result(
        self,
        run_id: str,
        *,
        wait_seconds: float = 0,
        artifact_offset: int = 0,
        artifact_limit: int | None = None,
        artifact_view: str = "canonical",
    ) -> TerminalRunReceipt:
        error, error_truncated, error_sha256, error_total_utf8_bytes = terminal_error_projection("The source dataset changed during execution.")
        return TerminalRunReceipt(
            run_id=run_id,
            result_record_path="C:/managed/wrapper/result.json",
            result_record_sha256="6" * 64,
            scientific_contract_id="scientific-contract-v4/supervised_learning/classification/logistic_regression",
            scientific_execution_contract_bound=True,
            state="failed",
            stage="failed",
            created_at="2026-08-29T00:00:00+00:00",
            started_at="2026-08-29T00:00:01+00:00",
            finished_at="2026-08-29T00:00:02+00:00",
            progress_message="Execution failed integrity validation.",
            error=error,
            error_truncated=error_truncated,
            error_sha256=error_sha256,
            error_total_utf8_bytes=error_total_utf8_bytes,
            result_type="input_integrity_changed",
            retryable=False,
            analysis_process_started=True,
            cli_exit_code=0,
        )


class BlockedRunManager(FakeRunManager):
    def validate(self, request) -> AnalysisValidationResponse:
        return (
            super()
            .validate(request)
            .model_copy(
                update={
                    "execution_ready": False,
                    "scientific_status": "requirements_unmet",
                    "adapter_status": "unavailable",
                    "blocking_issues": (
                        "The configured CLI is missing one required public capability.",
                        "The planned artifact contract is not executable.",
                    ),
                }
            )
        )


@pytest.mark.anyio
async def test_all_public_tool_output_schemas_match_serialized_success_and_error_payloads(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "rocks.csv"
    dataset.write_text(
        "SampleID,SIO2,Label\nA,50.1,basalt\n",
        encoding="utf-8",
    )
    manager = FakeRunManager()
    server = create_server(
        McpSettings(runs_root=tmp_path / "runs", cli_executable=None),
        manager,
    )
    request = ClassificationRequest(
        training_dataset_path=dataset,
        experiment_name="Protocol",
        run_name="Output schema",
        identifier_column="SampleID",
        feature_columns=("SIO2",),
        target_column="Label",
    )
    full_inspection = DatasetInspectionResponse(
        source_path=str(dataset),
        resolved_path=str(dataset),
        format="csv",
        size_bytes=dataset.stat().st_size,
        sha256="0" * 64,
        row_count=1,
        row_count_exact=True,
        column_count=3,
        column_names=("SampleID", "SIO2", "Label"),
        sample_rows=(),
        sample_truncated=True,
    )
    compact_inspection = dataset_inspection_response_view(
        full_inspection,
        DatasetInspectionRequest(dataset_path=dataset),
    )
    experiment = ExperimentSummary(
        experiment_id="1",
        name="Protocol",
        lifecycle_stage="active",
        artifact_location="file:///tracking/1",
    )
    ui_status = MlflowUiStatusResponse(
        state="stopped",
        tracking_root=str(tmp_path / "tracking"),
        message="not running",
    )
    full_result = manager.get_result("run-0123456789abcdef")
    compact_result = CompactRunResultResponse.from_full(full_result)
    compact_validation = compact_analysis_validation(manager.validate(request))
    dataset_directory = ListDatasetsResponse(
        source_filter="builtin",
        supported_formats=("csv", "xlsx"),
        datasets=(),
    )
    dataset_compact = list_datasets_response_view(
        dataset_directory,
        ListDatasetsViewRequest(),
    )
    dataset_full = list_datasets_response_view(
        dataset_directory,
        ListDatasetsViewRequest(detail="full"),
    )
    dataset_unchanged = list_datasets_response_view(
        dataset_directory,
        ListDatasetsViewRequest(if_view_sha256=dataset_compact.view_sha256),
    )
    experiment_directory = ListExperimentsResponse(
        tracking_root=str(tmp_path / "tracking"),
        experiment_count=0,
        experiments=(),
    )
    experiment_compact = list_experiments_response_view(
        experiment_directory,
        ListExperimentsViewRequest(),
    )
    experiment_full = list_experiments_response_view(
        experiment_directory,
        ListExperimentsViewRequest(detail="full"),
    )
    experiment_unchanged = list_experiments_response_view(
        experiment_directory,
        ListExperimentsViewRequest(if_view_sha256=experiment_compact.view_sha256),
    )
    experiment_history = GetExperimentResponse(
        tracking_root=str(tmp_path / "tracking"),
        experiment=experiment,
        run_count=0,
        runs=(),
    )
    history_compact = get_experiment_response_view(
        experiment_history,
        GetExperimentViewRequest(experiment_id="1"),
    )
    history_full = get_experiment_response_view(
        experiment_history,
        GetExperimentViewRequest(experiment_id="1", detail="full"),
    )
    history_unchanged = get_experiment_response_view(
        experiment_history,
        GetExperimentViewRequest(
            experiment_id="1",
            if_view_sha256=history_compact.view_sha256,
        ),
    )

    success_payloads = {
        "get_capabilities": (
            CapabilitiesNotModifiedResponse(
                capabilities_sha256="1" * 64,
                capability_view_sha256="2" * 64,
                server_version=SERVER_VERSION,
                capability_manifest_id="test-capabilities",
            ),
        ),
        "list_datasets": (dataset_compact, dataset_full, dataset_unchanged),
        "inspect_dataset": (full_inspection, compact_inspection),
        "list_experiments": (experiment_compact, experiment_full, experiment_unchanged),
        "get_experiment": (history_compact, history_full, history_unchanged),
        "start_mlflow_ui": (ui_status,),
        "mlflow_ui_status": (ui_status,),
        "stop_mlflow_ui": (ui_status,),
        "validate_analysis": (compact_validation,),
        "start_analysis": (manager.start(request),),
        "get_run_status": (manager.get_status("run-0123456789abcdef"),),
        "get_run_result": (full_result, compact_result),
        "cancel_run": (manager.cancel("run-0123456789abcdef"),),
    }

    async with Client(server) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        assert set(success_payloads) == set(tools)
        output_contracts = {name: _retained_output_contract(tool) for name, tool in tools.items()}
        full_inspection_schema = output_contracts["inspect_dataset"]["$defs"]["DatasetInspectionResponse"]
        assert {"source_sha256", "prepared_view_sha256"} <= set(full_inspection_schema["properties"])
        assert compact_validation.readiness.execution_ready is True
        assert "execution_ready" not in compact_validation.model_dump(mode="json")

        for tool_name, responses in success_payloads.items():
            validator = Draft202012Validator(output_contracts[tool_name])
            for response in responses:
                validator.validate(response.model_dump(mode="json"))

        for tool_name, tool in tools.items():
            rejected = await client.call_tool(tool_name, {"unexpected": True})
            assert rejected.is_error is True
            assert rejected.structured_content["tool_name"] == tool_name
            Draft202012Validator(output_contracts[tool_name]).validate(rejected.structured_content)
    assert manager.closed is True


@pytest.mark.anyio
async def test_tool_discovery_strict_validation_and_structured_results(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "rocks.csv"
    dataset.write_text("SampleID,SIO2,Label\nA,50.1,basalt\n", encoding="utf-8")
    wide_dataset = tmp_path / "wide.csv"
    wide_columns = ["SampleID", *(f"OXIDE_{index:02d}" for index in range(20))]
    wide_dataset.write_text(
        ",".join(wide_columns) + "\n" + ",".join(["A", *(str(index) for index in range(20))]) + "\n",
        encoding="utf-8",
    )
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
        request_schema = tools["validate_analysis"].input_schema
        assert "title" not in request_schema
        assert request_schema["type"] == "object"
        assert len(request_schema["oneOf"]) == 2
        routing_schema, detail_schema = request_schema["oneOf"]
        assert routing_schema["additionalProperties"] is True
        assert routing_schema["required"] == ["task"]
        assert routing_schema["properties"]["task"]["enum"] == [
            "classification",
            "regression",
            "clustering",
            "decomposition",
            "anomaly_detection",
            "time_series",
        ]
        assert detail_schema["additionalProperties"] is False
        assert detail_schema["properties"]["detail"]["const"] == "full"
        assert len(json.dumps(request_schema, separators=(",", ":")).encode("utf-8")) < 2_000

        regression_capabilities = await client.call_tool("get_capabilities", {"task": "regression"})
        assert regression_capabilities.is_error is False
        regression_schema = regression_capabilities.structured_content["validation_request_contract"]["request_schema"]
        Draft202012Validator.check_schema(regression_schema)
        _assert_strict_object_schemas(regression_schema)
        assert "target_columns" in regression_schema["properties"]
        assert "target_column" not in regression_schema["required"]
        assert "target_columns" not in regression_schema["required"]
        assert regression_schema["properties"]["target_columns"]["description"].startswith("One or more numeric regression targets")

        time_series_capabilities = await client.call_tool("get_capabilities", {"task": "time_series"})
        assert time_series_capabilities.is_error is False
        time_series_schema = time_series_capabilities.structured_content["validation_request_contract"]["request_schema"]
        Draft202012Validator.check_schema(time_series_schema)
        _assert_strict_object_schemas(time_series_schema)
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
            "model_parameters",
            "bin_width_ma",
            "bootstrap_iterations",
            "random_seed",
        }.isdisjoint(properties)
        assert properties["training_dataset"]["description"].startswith("Required top-level input reference")
        assert properties["bin_width"]["description"].startswith("Required for binned modes")
        assert properties["iterations"]["default"] == 100
        assert properties["seed"]["default"] == 2025

        start_schema = tools["start_analysis"].input_schema
        assert "title" not in start_schema
        assert set(start_schema["properties"]) == {"validation_id", "request_hash"}
        assert start_schema["additionalProperties"] is False
        assert all(tool.output_schema is not None for tool in tools.values())
        assert all(tool.output_schema["type"] == "object" for tool in tools.values())
        output_contracts = {name: _retained_output_contract(tool) for name, tool in tools.items()}
        assert all("PublicToolErrorResponse" in contract["$defs"] for contract in output_contracts.values())
        builtin_dataset = time_series_schema["$defs"]["BuiltInDatasetReference"]
        assert "retain the required 'builtin:' prefix" in builtin_dataset["properties"]["dataset_id"]["description"]

        compact_capabilities = await client.call_tool("get_capabilities", {})
        assert compact_capabilities.is_error is False
        assert compact_capabilities.structured_content["response_detail"] == "compact"
        attestation = compact_capabilities.structured_content["scientific_attestation"]
        assert attestation["public_manual_method_count"] == 36
        assert attestation["v4_attested_method_count"] == 27
        assert attestation["legacy_without_v4_attestation_method_count"] == 9
        assert sum(len(methods) for methods in attestation["v4_attested_methods_by_task"].values()) == 27
        assert sum(len(methods) for methods in attestation["legacy_methods_without_v4_attestation_by_task"].values()) == 9
        assert "capabilities" not in compact_capabilities.structured_content
        assert "supported_clients" not in compact_capabilities.structured_content
        assert len(compact_capabilities.structured_content["capabilities_sha256"]) == 64
        assert "if_capability_view_sha256" in compact_capabilities.content[0].text
        assert "if_capabilities_sha256" not in compact_capabilities.content[0].text

        capabilities = await client.call_tool("get_capabilities", {"detail": "full"})
        assert capabilities.is_error is False
        assert capabilities.structured_content["response_detail"] == "full"
        assert capabilities.structured_content["scientific_attestation"] == attestation
        assert "if_capabilities_sha256" in capabilities.content[0].text
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
        scientific_methods = {
            task: capabilities.structured_content[f"{task}_options"]["scientific_methods"]
            for task in (
                "classification",
                "regression",
                "clustering",
                "decomposition",
                "anomaly_detection",
            )
        }
        assert scientific_methods == {
            "classification": [
                "logistic_regression",
                "support_vector_machine",
                "decision_tree",
                "random_forest",
                "extra_trees",
                "xgboost",
                "multi_layer_perceptron",
                "gradient_boosting",
                "k_nearest_neighbors",
                "stochastic_gradient_descent",
                "adaboost",
            ],
            "regression": [
                "decision_tree",
                "random_forest",
                "extra_trees",
                "gradient_boosting",
                "xgboost",
                "multi_layer_perceptron",
                "lasso_regression",
                "elastic_net",
                "stochastic_gradient_descent",
            ],
            "clustering": ["kmeans", "affinity_propagation"],
            "decomposition": ["pca", "tsne", "mds"],
            "anomaly_detection": [
                "isolation_forest",
                "local_outlier_factor",
            ],
        }
        assert sum(map(len, scientific_methods.values())) == 27
        assert all(set(methods) <= set(capabilities.structured_content["supported_models_by_task"][task]) for task, methods in scientific_methods.items())
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

        wide_inspection = await client.call_tool(
            "inspect_dataset",
            {"dataset_path": str(wide_dataset), "sample_rows": 1},
        )
        assert wide_inspection.is_error is False
        assert wide_inspection.structured_content["column_names"] == wide_columns
        wide_text = wide_inspection.content[0].text
        assert "Showing 12 of 21 ordered columns" in wide_text
        assert "complete ordered column list remains in structured_content" in wide_text
        assert "OXIDE_10" in wide_text
        assert "OXIDE_11" not in wide_text
        assert len(wide_text) < 1000

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
        assert invalid.structured_content["error_type"] == "validation_error"
        assert invalid.structured_content["tool_name"] == "inspect_dataset"
        assert invalid.structured_content["root_cause_count"] == 1
        assert invalid.structured_content["root_causes_total_count"] == 1
        assert invalid.structured_content["root_causes_truncated"] is False
        assert len(invalid.structured_content["root_causes_sha256"]) == 64
        assert invalid.structured_content["root_causes"][0]["field"] == "unexpected"

        missing = await client.call_tool("get_run_status", {})
        assert missing.is_error is True
        assert "Resolve it from the conversation or earlier tool results" in missing.content[0].text
        assert "never ask the user for a schema field name or JSON" in missing.content[0].text

        repeated_artifact_error = await client.call_tool(
            "validate_analysis",
            {
                "task": "classification",
                "training_dataset_path": str(dataset),
                "experiment_name": "Protocol",
                "run_name": "Grouped errors",
                "identifier_column": "SampleID",
                "feature_columns": ["SIO2"],
                "target_column": "Label",
                "artifact_requirements": ["first", "second"],
            },
        )
        assert repeated_artifact_error.is_error is True
        repeated_details = repeated_artifact_error.structured_content
        assert repeated_details["root_cause_count"] == 1
        assert repeated_details["root_causes"][0]["field"] == "classification.artifact_requirements"
        assert repeated_details["root_causes"][0]["occurrences"] == 2
        assert len(repeated_details["root_causes"][0]["locations"]) == 2
        assert repeated_details["root_causes"][0]["locations_total_count"] == 2
        assert repeated_details["root_causes"][0]["locations_truncated"] is False

        heavily_repeated_error = await client.call_tool(
            "validate_analysis",
            {
                "task": "classification",
                "training_dataset_path": str(dataset),
                "experiment_name": "Protocol",
                "run_name": "Bounded repeated errors",
                "identifier_column": "SampleID",
                "feature_columns": ["SIO2"],
                "target_column": "Label",
                "artifact_requirements": ["invalid"] * 1000,
            },
        )
        assert heavily_repeated_error.is_error is True
        repeated_cause = heavily_repeated_error.structured_content["root_causes"][0]
        assert repeated_cause["occurrences"] == 1000
        assert repeated_cause["locations_total_count"] == 1000
        assert repeated_cause["locations_truncated"] is True
        assert len(repeated_cause["locations"]) == 4
        assert len(repeated_cause["locations_sha256"]) == 64
        assert len(heavily_repeated_error.content[0].text) <= 4000
        assert (
            len(
                json.dumps(
                    heavily_repeated_error.structured_content,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            <= 64 * 1024
        )

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
        assert "place supported fields such as 'bin_width'" in validation_text
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
                **{f"unsupported_{index}": "sensitive-value" for index in range(1000)},
            },
        )
        assert bounded_invalid.is_error is True
        bounded_text = bounded_invalid.content[0].text
        assert "pattern" in bounded_text
        assert "range" in bounded_text
        assert "literal" in bounded_text
        assert "Additional issues omitted" not in bounded_text
        assert "total-count and SHA-256 metadata" in bounded_text
        assert "sensitive-value" not in bounded_text
        assert bounded_text.count("Next action:") == 1
        assert len(bounded_text) <= 4000
        bounded_details = bounded_invalid.structured_content
        assert bounded_details["error_type"] == "validation_error"
        assert bounded_details["root_cause_count"] == 24
        assert len(bounded_details["root_causes"]) == 24
        assert bounded_details["root_causes_total_count"] == 1004
        assert bounded_details["root_causes_truncated"] is True
        assert len(bounded_details["root_causes_sha256"]) == 64
        assert len(json.dumps(bounded_details, separators=(",", ":")).encode("utf-8")) <= 64 * 1024
        assert {cause["field"] for cause in bounded_details["root_causes"]} >= {
            "time_series.training_dataset.builtin.dataset_id",
            "time_series.bin_width",
            "time_series.iterations",
            "time_series.age_unit",
        }

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
        assert preview.structured_content["column_roles"]["identifier_column"] == "SampleID"
        feature_columns = preview.structured_content["column_roles"]["feature_columns"]
        assert feature_columns["total_count"] == 1
        assert feature_columns["truncated"] is False
        assert [item["text"] for item in feature_columns["prefix"]] == ["SIO2"]
        assert len(feature_columns["sha256"]) == 64
        assert preview.structured_content["column_roles"]["target_column"] == "Label"
        assert preview.structured_content["resolved_model_parameters"]["solver"] == "lbfgs"

        validation_calls = fake_runs.validation_calls
        full_validation = await client.call_tool(
            "validate_analysis",
            preview.structured_content["full_detail_request"],
        )
        assert full_validation.is_error is False
        assert fake_runs.validation_calls == validation_calls
        assert fake_runs.validation_detail_calls == 1
        assert full_validation.structured_content["validation_id"] == preview.structured_content["validation_id"]
        assert full_validation.structured_content["request_hash"] == preview.structured_content["request_hash"]
        assert full_validation.structured_content["response_detail"] == "full"
        assert full_validation.structured_content["blocking_issues"] == []
        assert len(full_validation.structured_content["blocking_issues_sha256"]) == 64
        assert full_validation.structured_content["artifact_requirement_count"] == 0
        assert full_validation.structured_content["artifact_requirements"] == []

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
        assert fake_runs.requests[-1].resolved_target_columns == (
            "Label",
            "SIO2_Second_Target",
        )

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
        assert len(result.content[0].text) < 1000
        assert "Metrics are included once" in result.content[0].text
        assert "x" * 100 not in result.content[0].text
        assert result.structured_content["response_detail"] == "compact"
        assert result.structured_content["reported_metrics"]["bounded_text_guard"] == "x" * 5000
    assert fake_runs.closed is True


@pytest.mark.anyio
async def test_blocked_validation_text_never_instructs_the_client_to_start(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "rocks.csv"
    dataset.write_text(
        "SampleID,SIO2,Label\nA,50.1,basalt\n",
        encoding="utf-8",
    )
    fake_runs = BlockedRunManager()
    server = create_server(
        McpSettings(runs_root=tmp_path / "runs", cli_executable=None),
        fake_runs,
    )

    async with Client(server) as client:
        result = await client.call_tool(
            "validate_analysis",
            {
                "task": "classification",
                "training_dataset_path": str(dataset),
                "experiment_name": "Protocol",
                "run_name": "Blocked",
                "identifier_column": "SampleID",
                "feature_columns": ["SIO2"],
                "target_column": "Label",
            },
        )

        assert result.is_error is False
        assert result.structured_content["readiness"]["execution_ready"] is False
        assert "Do not start" in result.content[0].text
        assert "call validate_analysis once with the corrected request" in result.content[0].text
        assert "Start this exact request" not in result.content[0].text
    assert fake_runs.closed is True


@pytest.mark.anyio
async def test_failed_run_result_is_a_small_non_scientific_terminal_receipt(
    tmp_path: Path,
) -> None:
    settings = McpSettings(runs_root=tmp_path / "runs", cli_executable=None)
    fake_runs = FailedRunManager()
    server = create_server(settings, fake_runs)

    async with Client(server) as client:
        result = await client.call_tool(
            "get_run_result",
            {"run_id": "run-0123456789abcdef"},
        )

        assert result.is_error is False
        assert "Scientific validity was not established" in result.content[0].text
        assert result.structured_content["response_detail"] == "terminal"
        assert result.structured_content["state"] == "failed"
        assert result.structured_content["result_type"] == "input_integrity_changed"
        assert result.structured_content["retryable"] is False
        assert result.structured_content["verified_artifact_count"] == 0
        assert "artifacts" not in result.structured_content

        full = await client.call_tool(
            "get_run_result",
            {
                "run_id": "run-0123456789abcdef",
                "detail": "full",
            },
        )
        assert full.is_error is False
        assert full.structured_content["response_detail"] == "terminal"
        assert full.structured_content["result_type"] == result.structured_content["result_type"]
        assert full.structured_content["retryable"] == result.structured_content["retryable"]

        unchanged = await client.call_tool(
            "get_run_result",
            {
                "run_id": "run-0123456789abcdef",
                "if_result_sha256": "6" * 64,
            },
        )
        assert unchanged.is_error is False
        assert unchanged.structured_content["response_detail"] == "not_modified"
        assert unchanged.structured_content["terminal_receipt"] is True
        assert unchanged.structured_content["result_type"] == "input_integrity_changed"
        assert unchanged.structured_content["retryable"] is False
        assert "error" not in unchanged.structured_content
        assert "not replayed" in unchanged.content[0].text
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
