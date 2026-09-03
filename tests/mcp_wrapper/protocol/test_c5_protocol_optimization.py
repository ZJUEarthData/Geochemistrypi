import copy
import hashlib
import json
from pathlib import Path

import pytest
from geochemistrypi_mcp.api.schemas import (
    AnalysisRequest,
    AnalysisValidationResponse,
    CancelRunResponse,
    DatasetInspectionRequest,
    RunResultResponse,
    RunStatusResponse,
    StartAnalysisResponse,
    TimeSeriesRequest,
)
from geochemistrypi_mcp.api.tools import _advertised_input_schema, _optimized_advertised_schema, _require_advertised_analysis_task, full_output_contract_schema
from geochemistrypi_mcp.config.constants import CLI_VERSION
from geochemistrypi_mcp.config.settings import McpSettings, SettingsError
from geochemistrypi_mcp.server import create_server
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from mcp import Client

SCHEMA_TASK_ENV = "GEOCHEMISTRYPI_MCP_ANALYSIS_SCHEMA_TASK"
VALIDATION_ID = "val-0123456789abcdef0123456789abcdef"
REQUEST_HASH = "1" * 64
ANALYSIS_TASKS = (
    "classification",
    "regression",
    "clustering",
    "decomposition",
    "anomaly_detection",
    "time_series",
)
EXPECTED_TOOL_NAMES = {
    "get_capabilities",
    "list_datasets",
    "inspect_dataset",
    "list_experiments",
    "get_experiment",
    "start_mlflow_ui",
    "mlflow_ui_status",
    "stop_mlflow_ui",
    "validate_analysis",
    "start_analysis",
    "get_run_status",
    "get_run_result",
    "cancel_run",
}
CONTRACT_METADATA_KEYS = frozenset(
    {
        "description",
        "default",
        "examples",
        "enum",
        "discriminator",
    }
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class ProtocolRunManager:
    def __init__(self) -> None:
        self.closed = False
        self.validated = []
        self.validation_response = None
        self.validation_detail_calls = []
        self.legacy_starts = []
        self.reference_starts = []
        self.result_calls = []

    def validate(self, request) -> AnalysisValidationResponse:
        self.validated.append(request)
        model = "subaerial_proportion_bootstrap" if request.task == "time_series" else request.model.type
        tuning = "not_applicable" if request.task == "time_series" else request.tuning
        response = AnalysisValidationResponse(
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
            scientific_contract_id="scientific-contract-v2/test",
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
        self.validation_response = response
        return response

    def get_validation_detail(
        self,
        validation_id: str,
        request_hash: str,
        *,
        expected_task=None,
    ) -> AnalysisValidationResponse:
        self.validation_detail_calls.append((validation_id, request_hash, expected_task))
        assert self.validation_response is not None
        assert validation_id == self.validation_response.validation_id
        assert request_hash == self.validation_response.request_hash
        return self.validation_response

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
        artifact_view: str = "canonical",
    ) -> RunResultResponse:
        self.result_calls.append((run_id, wait_seconds, artifact_offset, artifact_limit, artifact_view))
        return RunResultResponse(
            request_hash="6" * 64,
            validation_id="val-" + "7" * 32,
            canonical_contract_hash="8" * 64,
            compiled_plan_hash="9" * 64,
            scientific_contract_id="scientific-contract-v2/time_series/test",
            scientific_execution_contract_bound=False,
            provenance_manifest_path="C:/managed/wrapper/provenance-manifest.json",
            provenance_manifest_sha256="a" * 64,
            run_id=run_id,
            result_record_path="C:/managed/wrapper/result.json",
            result_record_sha256="4" * 64,
            state="succeeded",
            task="time_series",
            model="subaerial_proportion_bootstrap",
            tuning="not_applicable",
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
            reported_metrics={"total_bins": 39},
            artifact_count=0,
            canonical_artifact_count=0,
            artifact_index_path="C:/managed/wrapper/artifact-index.json",
            artifact_index_sha256="5" * 64,
            artifact_view=artifact_view,
            artifact_view_count=0,
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


def test_run_result_schema_accepts_the_registered_continuous_method() -> None:
    result = RunResultResponse(
        request_hash="6" * 64,
        validation_id="val-" + "7" * 32,
        canonical_contract_hash="8" * 64,
        compiled_plan_hash="9" * 64,
        scientific_contract_id="scientific-contract-v2/time_series/continuous/test",
        scientific_execution_contract_bound=False,
        provenance_manifest_path="C:/managed/wrapper/provenance-manifest.json",
        provenance_manifest_sha256="a" * 64,
        run_id="run-0000000000000000",
        state="succeeded",
        task="time_series",
        model="spatiotemporal_weighted_continuous_bootstrap",
        tuning="not_applicable",
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
        reported_metrics={"total_bins": 2},
        artifact_count=0,
        artifact_offset=0,
        returned_artifact_count=0,
        next_artifact_offset=None,
        artifacts=(),
        artifacts_truncated=False,
        limitations=(),
    )

    assert result.model == "spatiotemporal_weighted_continuous_bootstrap"


def _compact_bytes(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _retained_output_contract(tool) -> dict:
    advertised = tool.output_schema
    assert advertised is not None
    assert advertised["type"] == "object"
    assert advertised["additionalProperties"] is True
    assert advertised["x-geochemistrypi-output-contract-delivery"] == "hash-addressed-server-enforced"
    assert _compact_bytes(advertised) < 700
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


def _canonical_value(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _contract_metadata(value: object) -> dict[str, set[str]]:
    found = {key: set() for key in CONTRACT_METADATA_KEYS}

    def visit(child: object) -> None:
        if isinstance(child, dict):
            for key, nested in child.items():
                if key in found:
                    if key == "enum" and isinstance(nested, list):
                        found[key].update(_canonical_value(item) for item in nested)
                    else:
                        found[key].add(_canonical_value(nested))
                visit(nested)
        elif isinstance(child, list):
            for nested in child:
                visit(nested)

    visit(value)
    return found


def _raw_and_optimized_schema(model: type, *, require_task: bool = False) -> tuple[dict, dict]:
    raw = _advertised_input_schema(model.model_json_schema())
    if require_task:
        _require_advertised_analysis_task(raw)
    optimized = _optimized_advertised_schema(copy.deepcopy(raw))
    return raw, optimized


def _assert_equivalent_validation(raw: dict, optimized: dict, cases: list[dict]) -> None:
    Draft202012Validator.check_schema(raw)
    Draft202012Validator.check_schema(optimized)
    raw_validator = Draft202012Validator(raw)
    optimized_validator = Draft202012Validator(optimized)
    for case in cases:
        assert optimized_validator.is_valid(case) is raw_validator.is_valid(case), case


def test_advertised_schema_removes_only_generated_titles() -> None:
    source = {
        "title": "Generated model title",
        "description": "Keep this client guidance.",
        "default": {"mode": "safe"},
        "examples": [{"mode": "safe"}],
        "properties": {
            "mode": {
                "title": "Generated field title",
                "description": "Keep this field guidance.",
                "default": "safe",
                "examples": ["safe"],
            }
        },
    }

    advertised = _advertised_input_schema(source)

    assert not _title_paths(advertised)
    assert advertised["description"] == source["description"]
    assert advertised["default"] == source["default"]
    assert advertised["examples"] == source["examples"]
    assert advertised["properties"]["mode"]["description"] == "Keep this field guidance."
    assert advertised["properties"]["mode"]["default"] == "safe"
    assert advertised["properties"]["mode"]["examples"] == ["safe"]


@pytest.mark.parametrize(
    ("model", "require_task"),
    (
        (AnalysisRequest, True),
        (TimeSeriesRequest, True),
        (DatasetInspectionRequest, False),
    ),
)
def test_schema_optimization_keeps_semantic_definitions_and_contract_metadata(model: type, require_task: bool) -> None:
    raw, optimized = _raw_and_optimized_schema(model, require_task=require_task)

    Draft202012Validator.check_schema(optimized)
    assert set(raw.get("$defs", ())) <= set(optimized.get("$defs", ()))
    raw_metadata = _contract_metadata(raw)
    optimized_metadata = _contract_metadata(optimized)
    for key in CONTRACT_METADATA_KEYS:
        assert raw_metadata[key] <= optimized_metadata[key], key


def test_optimized_time_series_schema_preserves_mode_field_ownership() -> None:
    schema = _optimized_advertised_schema(_advertised_input_schema(TimeSeriesRequest.model_json_schema()))
    validator = Draft202012Validator(schema)
    base = {
        "task": "time_series",
        "training_dataset_path": "C:/data/time-series.csv",
        "bin_width": 100,
    }
    validator.validate(base)

    invalid_requests = (
        {**base, "element_columns": ["SIO2"]},
        {
            **base,
            "mode": "continuous",
            "minimum_age_column": "R_MIN_AGE",
            "value_column": "VALUE",
            "probability_column": "SBAP",
        },
        {
            **base,
            "mode": "element_mean",
            "element_columns": ["SIO2"],
            "iterations": 100,
        },
        {
            "task": "time_series",
            "training_dataset_path": "C:/data/time-series.csv",
            "mode": "reference_anomaly_series",
            "time_column": "DATE",
            "signal_columns": ["SIGNAL"],
            "reference_label_column": "REFERENCE",
            "reference_positive_values": ["event"],
            "age_column": "R_AGE",
        },
    )
    for request in invalid_requests:
        with pytest.raises(JsonSchemaValidationError):
            validator.validate(request)


def test_time_series_schema_optimization_preserves_representative_validation() -> None:
    raw, optimized = _raw_and_optimized_schema(TimeSeriesRequest, require_task=True)
    base = {
        "task": "time_series",
        "training_dataset_path": "C:/rocks.csv",
        "bin_width": 100,
    }
    reference = {
        "task": "time_series",
        "training_dataset_path": "C:/rocks.csv",
        "mode": "reference_anomaly_series",
        "time_column": "date",
        "signal_columns": ["signal"],
        "reference_label_column": "kind",
        "reference_positive_values": ["reference"],
    }
    _assert_equivalent_validation(
        raw,
        optimized,
        [
            base,
            {"task": "time_series", "training_dataset_path": "C:/rocks.csv"},
            {
                **base,
                "training_dataset": {
                    "source": "builtin",
                    "dataset_id": "builtin:time_series",
                },
            },
            {
                "task": "time_series",
                "training_dataset": {
                    "source": "builtin",
                    "dataset_id": "builtin:time_series",
                },
                "bin_width": 100,
            },
            {"task": "time_series", "bin_width": 100},
            {
                **base,
                "mode": "continuous",
                "minimum_age_column": "minimum_age",
                "value_column": "value",
            },
            {**base, "mode": "continuous", "minimum_age_column": "minimum_age"},
            {
                **base,
                "mode": "continuous",
                "minimum_age_column": "minimum_age",
                "value_column": "value",
                "filter_minimum": 0,
            },
            {
                **base,
                "mode": "continuous",
                "minimum_age_column": "minimum_age",
                "value_column": "value",
                "filter_minimum": 0,
                "filter_column": "group",
            },
            {**base, "mode": "element_mean", "element_columns": ["SIO2"]},
            {**base, "mode": "element_mean", "element_columns": []},
            reference,
            {**reference, "bin_width": 100},
            {**reference, "missing_values": {"method": "drop_rows"}},
            {**reference, "missing_values": {"method": "error"}},
            {**reference, "comparison_label_column": "kind"},
            {
                **reference,
                "comparison_label_column": "kind",
                "comparison_positive_values": ["comparison"],
            },
            {**reference, "event_dataset_path": "C:/events.csv"},
            {
                **reference,
                "event_dataset_path": "C:/events.csv",
                "event_time_column": "event_date",
            },
            {**reference, "event_time_column": "event_date"},
            {
                **reference,
                "event_dataset_path": "C:/events.csv",
                "event_time_column": "event_date",
                "event_filter_column": "kind",
            },
            {
                **reference,
                "event_dataset_path": "C:/events.csv",
                "event_time_column": "event_date",
                "event_filter_column": "kind",
                "event_filter_values": ["eruption"],
            },
            {
                "task": "time_series",
                "training_dataset": {
                    "source": "path",
                    "path": "C:/rocks.xlsx",
                    "preparation": {"worksheets": [], "union_mode": "rows"},
                },
                "bin_width": 100,
            },
            {
                "task": "time_series",
                "training_dataset": {
                    "source": "path",
                    "path": "C:/rocks.xlsx",
                    "preparation": {
                        "worksheets": ["A", "B"],
                        "union_mode": "rows",
                        "source_sheet_column": "source_sheet",
                        "source_row_column": "source_row",
                        "selected_columns": ["age"],
                    },
                },
                "bin_width": 100,
            },
            {
                "task": "time_series",
                "training_dataset": {
                    "source": "path",
                    "path": "C:/rocks.xlsx",
                    "preparation": {
                        "worksheets": ["A"],
                        "union_mode": "rows",
                        "source_sheet_column": "source_sheet",
                        "source_row_column": "source_row",
                        "selected_columns": ["age"],
                    },
                },
                "bin_width": 100,
            },
        ],
    )


def test_dataset_inspection_schema_optimization_preserves_representative_validation() -> None:
    raw, optimized = _raw_and_optimized_schema(DatasetInspectionRequest)
    _assert_equivalent_validation(
        raw,
        optimized,
        [
            {"dataset_path": "C:/rocks.csv"},
            {"dataset": {"source": "builtin", "dataset_id": "builtin:time_series"}},
            {},
            {
                "dataset_path": "C:/rocks.csv",
                "dataset": {"source": "builtin", "dataset_id": "builtin:time_series"},
            },
            {"dataset_path": "C:/rocks.csv", "dataset": None},
            {
                "dataset": {
                    "source": "path",
                    "path": "C:/rocks.xlsx",
                    "preparation": {"worksheets": []},
                }
            },
            {
                "dataset": {
                    "source": "path",
                    "path": "C:/rocks.xlsx",
                    "preparation": {"worksheets": [], "union_mode": "rows"},
                }
            },
            {
                "dataset": {
                    "source": "path",
                    "path": "C:/rocks.xlsx",
                    "preparation": {
                        "worksheets": ["A", "B"],
                        "union_mode": "rows",
                        "source_sheet_column": "source_sheet",
                        "source_row_column": "source_row",
                        "selected_columns": ["age"],
                    },
                }
            },
        ],
    )


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


def _time_series_request(dataset: Path) -> dict:
    return {
        "task": "time_series",
        "training_dataset_path": str(dataset),
        "bin_width": 100,
    }


def _title_paths(value: object, path: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    found: list[tuple[str, ...]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = (*path, key)
            if key == "title":
                found.append(child_path)
            found.extend(_title_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_title_paths(child, (*path, str(index))))
    return found


@pytest.mark.anyio
async def test_advertised_time_series_schemas_keep_guidance_and_remove_titles(
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
        output_schemas = {tool.name: tool.output_schema for tool in listing.tools}
        assert all(not _title_paths(schema) for schema in schemas.values())
        assert all(schema is not None for schema in output_schemas.values())
        assert all(not _title_paths(schema) for schema in output_schemas.values())
        retained_output_contracts = {tool.name: _retained_output_contract(tool) for tool in listing.tools}
        assert all(not _title_paths(schema) for schema in retained_output_contracts.values())
        # Preserve the complete scientific guidance and meaningful definition
        # names. The budget only guards accidental duplication; it must not be
        # met by deleting descriptions/defaults or rewriting enums as regexes.
        # This conservative ceiling retains descriptions/defaults/examples,
        # semantic $defs names, discriminator mappings, and ordinary local
        # JSON-Pointer refs for broad MCP-client compatibility.
        # Directory discovery now advertises additive compact/full pagination
        # and exact conditional-view selectors. Their descriptions, defaults,
        # enums, and bounds are part of the usable public contract.
        # The four Time Series modes now advertise their complete cross-mode
        # field-ownership matrix instead of relying on runtime-only rejection.
        # Keep a tight ceiling that includes those non-redundant conditions.
        assert sum(_compact_bytes(schema) for schema in schemas.values()) < 33_000
        # tools/list carries only a hash-addressed envelope. Exact success/error
        # contracts remain generated, server-enforced, and retrievable by hash.
        output_schema_bytes = [_compact_bytes(schema) for schema in output_schemas.values()]
        assert max(output_schema_bytes) < 700
        assert sum(output_schema_bytes) < 8_000

        validate_schema = schemas["validate_analysis"]
        time_series = validate_schema["oneOf"][0]
        assert time_series["additionalProperties"] is False
        assert time_series["properties"]["task"]["const"] == "time_series"
        assert time_series["properties"]["training_dataset"]["description"].startswith("Required top-level input reference")
        assert time_series["properties"]["iterations"]["description"].startswith("Top-level bootstrap iterations")
        assert time_series["properties"]["iterations"]["default"] == 100
        assert time_series["properties"]["seed"]["default"] == 2025
        assert any(choice.get("exclusiveMinimum") == 0 for choice in time_series["properties"]["bin_width"]["anyOf"])
        assert time_series["properties"]["training_dataset"]["anyOf"][0]["discriminator"]["propertyName"] == "source"
        assert time_series["properties"]["training_dataset"]["anyOf"][0]["discriminator"]["mapping"]
        assert "BuiltInDatasetReference" in validate_schema["$defs"]
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
        assert set(tools) == EXPECTED_TOOL_NAMES
        assert all(tool.output_schema is not None for tool in tools.values())
        assert all(tool.output_schema["type"] == "object" for tool in tools.values())
        output_contracts = {name: _retained_output_contract(tool) for name, tool in tools.items()}
        assert all("PublicToolErrorResponse" in contract["$defs"] for contract in output_contracts.values())
        assert {
            "CapabilitiesResponse",
            "CompactCapabilitiesResponse",
            "CapabilitiesNotModifiedResponse",
        } <= set(output_contracts["get_capabilities"]["$defs"])
        assert {
            "DatasetInspectionResponse",
            "CompactDatasetInspectionResponse",
        } <= set(output_contracts["inspect_dataset"]["$defs"])
        assert "CompactAnalysisValidationResponse" in output_contracts["validate_analysis"]["$defs"]
        validation_output_definitions = output_contracts["validate_analysis"]["$defs"]
        assert {
            "CompactTextReceipt",
            "CompactSequenceReceipt_CompactTextReceipt_",
            "CompactSequenceReceipt_CompactArtifactRequirementSummary_",
            "CompactMappingReceipt_CompactTextReceipt_",
        } <= set(validation_output_definitions)
        validation_properties = validation_output_definitions["CompactAnalysisValidationResponse"]["properties"]
        for field in ("blocking_issues", "warnings", "artifact_requirements"):
            assert "$ref" in validation_properties[field]
        assert {
            "RunResultResponse",
            "CompactRunResultResponse",
            "CompactRunArtifactPageResponse",
            "RunResultNotModifiedResponse",
            "PendingRunResultResponse",
            "TerminalRunReceipt",
            "TerminalRunNotModifiedResponse",
        } <= set(output_contracts["get_run_result"]["$defs"])

        validate_schema = tools["validate_analysis"].input_schema
        start_schema = tools["start_analysis"].input_schema
        assert "title" not in validate_schema
        assert validate_schema["type"] == "object"
        assert len(validate_schema["oneOf"]) == 2
        routing_schema, detail_schema = validate_schema["oneOf"]
        assert routing_schema["additionalProperties"] is True
        assert routing_schema["required"] == ["task"]
        assert routing_schema["properties"]["task"]["enum"] == list(ANALYSIS_TASKS)
        assert detail_schema["additionalProperties"] is False
        assert detail_schema["properties"]["detail"]["const"] == "full"
        assert _compact_bytes(validate_schema) < 2_000
        assert "title" not in start_schema
        assert start_schema["additionalProperties"] is False
        assert set(start_schema["properties"]) == {"validation_id", "request_hash"}
        assert set(start_schema["required"]) == {"validation_id", "request_hash"}
        assert _compact_bytes(start_schema) < 1000
        input_schema_bytes = sum(_compact_bytes(tool.input_schema) for tool in tools.values())
        output_schema_bytes = [_compact_bytes(tool.output_schema) for tool in tools.values()]
        description_bytes = sum(len((tool.description or "").encode("utf-8")) for tool in tools.values())
        assert input_schema_bytes < 16_000
        assert sum(output_schema_bytes) < 8_000
        assert description_bytes + input_schema_bytes + sum(output_schema_bytes) < 32_000

        task_capabilities = await client.call_tool("get_capabilities", {"task": "regression"})
        assert task_capabilities.is_error is False
        exact_regression_schema = task_capabilities.structured_content["validation_request_contract"]["request_schema"]
        Draft202012Validator.check_schema(exact_regression_schema)
        assert exact_regression_schema["additionalProperties"] is False
        assert exact_regression_schema["properties"]["task"]["const"] == "regression"
        assert "description" in exact_regression_schema["properties"]["target_columns"]

        preview = await client.call_tool("validate_analysis", _classification_request(dataset))
        assert preview.is_error is False
        Draft202012Validator(output_contracts["validate_analysis"]).validate(preview.structured_content)
        assert preview.structured_content["validation_id"] == VALIDATION_ID
        assert preview.structured_content["request_hash"] == REQUEST_HASH

        legacy_without_task = _classification_request(dataset)
        legacy_without_task.pop("task")
        legacy_preview = await client.call_tool("validate_analysis", legacy_without_task)
        assert legacy_preview.is_error is False
        assert manager.validated[-1].task == "classification"

        mixed_task = {
            "task": "time_series",
            "training_dataset_path": str(dataset),
            "bin_width": 100,
            "model": {"type": "logistic_regression"},
        }
        rejected_mixed_task = await client.call_tool("validate_analysis", mixed_task)
        assert rejected_mixed_task.is_error is True
        assert rejected_mixed_task.structured_content is not None
        for tool in tools.values():
            Draft202012Validator(output_contracts[tool.name]).validate(rejected_mixed_task.structured_content)
        assert len(manager.validated) == 2

        preferred = await client.call_tool(
            "start_analysis",
            {"validation_id": VALIDATION_ID, "request_hash": REQUEST_HASH},
        )
        assert preferred.is_error is False
        Draft202012Validator(output_contracts["start_analysis"]).validate(preferred.structured_content)
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
async def test_time_series_advertised_schema_scope_is_strict_small_and_keeps_all_tool_names(
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
        assert len(validate_schema["oneOf"]) == 2
        scientific_schema = validate_schema["oneOf"][0]
        assert scientific_schema["additionalProperties"] is False
        assert scientific_schema["properties"]["task"]["const"] == "time_series"
        assert validate_schema["oneOf"][1]["properties"]["detail"]["const"] == "full"
        assert _compact_bytes(validate_schema) < 20_500
        assert sum(_compact_bytes(tool.input_schema) for tool in tools.values()) < 33_000


@pytest.mark.parametrize("schema_scope", ANALYSIS_TASKS)
@pytest.mark.anyio
async def test_task_schema_scope_only_changes_advertisement_and_keeps_runtime_generic(
    schema_scope: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SCHEMA_TASK_ENV, schema_scope)
    manager = ProtocolRunManager()
    server = create_server(McpSettings(runs_root=tmp_path / "runs", cli_executable=None), manager)
    cross_task_request = _time_series_request(tmp_path / "time-series.csv") if schema_scope == "classification" else _classification_request(tmp_path / "rocks.csv")

    async with Client(server) as client:
        listing = await client.list_tools()
        tools = {tool.name: tool for tool in listing.tools}
        assert set(tools) == EXPECTED_TOOL_NAMES
        assert all(tool.description for tool in tools.values())
        advertised_schema = tools["validate_analysis"].input_schema
        scientific_schema = advertised_schema["oneOf"][0]
        assert scientific_schema["properties"]["task"]["const"] == schema_scope
        assert "task" in scientific_schema["required"]

        capabilities = await client.call_tool("get_capabilities", {})
        assert capabilities.is_error is False
        assert capabilities.structured_content["analysis_schema_task_scope"] == schema_scope
        assert capabilities.structured_content["supported_tasks"] == list(ANALYSIS_TASKS)

        validated = await client.call_tool("validate_analysis", cross_task_request)
        assert validated.is_error is False
        assert manager.validated[-1].task != schema_scope

        validation_count = len(manager.validated)
        full_detail = await client.call_tool(
            "validate_analysis",
            validated.structured_content["full_detail_request"],
        )
        assert full_detail.is_error is False
        assert full_detail.structured_content["task"] != schema_scope
        assert len(manager.validated) == validation_count
        assert manager.validation_detail_calls[-1] == (VALIDATION_ID, REQUEST_HASH, None)

        legacy_start = await client.call_tool("start_analysis", cross_task_request)
        assert legacy_start.is_error is False
        assert manager.legacy_starts[-1].task != schema_scope

        validated_start = await client.call_tool(
            "start_analysis",
            {"validation_id": VALIDATION_ID, "request_hash": REQUEST_HASH},
        )
        assert validated_start.is_error is False
        assert manager.reference_starts[-1] == (VALIDATION_ID, REQUEST_HASH, None)


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
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        result_contract = _retained_output_contract(tools["get_run_result"])
        result_properties = tools["get_run_result"].input_schema["properties"]
        assert result_properties["artifact_view"]["enum"] == ["canonical", "all"]
        assert result_properties["detail"]["enum"] == ["compact", "full"]
        assert result_properties["if_result_sha256"]["anyOf"][0]["pattern"] == "^[0-9a-f]{64}$"
        result = await client.call_tool(
            "get_run_result",
            {
                "run_id": "run-0123456789abcdef",
                "wait_seconds": 300,
                "artifact_offset": 0,
                "artifact_limit": 200,
            },
        )
        assert result.is_error is False
        Draft202012Validator(result_contract).validate(result.structured_content)
        assert result.structured_content["response_detail"] == "compact"
        assert result.structured_content["returned_artifact_count"] == 0
        assert manager.result_calls == [("run-0123456789abcdef", 300, 0, 32, "canonical")]

        unchanged = await client.call_tool(
            "get_run_result",
            {
                "run_id": "run-0123456789abcdef",
                "wait_seconds": 300,
                "if_result_sha256": "4" * 64,
            },
        )
        assert unchanged.is_error is False
        Draft202012Validator(result_contract).validate(unchanged.structured_content)
        assert unchanged.structured_content["response_detail"] == "not_modified"
        assert unchanged.structured_content["not_modified"] is True
        assert "reported_metrics" not in unchanged.structured_content
        assert "artifacts" not in unchanged.structured_content
        assert len(json.dumps(unchanged.structured_content)) < 2000

        full = await client.call_tool(
            "get_run_result",
            {
                "run_id": "run-0123456789abcdef",
                "wait_seconds": 300,
                "artifact_limit": 200,
                "artifact_view": "all",
                "detail": "full",
            },
        )
        assert full.is_error is False
        Draft202012Validator(result_contract).validate(full.structured_content)
        assert full.structured_content["response_detail"] == "full"
        assert manager.result_calls[-1] == ("run-0123456789abcdef", 300, 0, 200, "all")
