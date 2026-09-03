import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from geochemistrypi_mcp.api.schemas import START_READY_METHODS_BY_TASK, AnomalyDetectionRequest, ClassificationRequest, ClusteringRequest, DecompositionRequest, RegressionRequest, TimeSeriesRequest
from geochemistrypi_mcp.api.tools import full_output_contract_schema
from geochemistrypi_mcp.config.settings import McpSettings
from geochemistrypi_mcp.server import create_server
from jsonschema import Draft202012Validator
from mcp import Client

TASK_MODELS = {
    "classification": ClassificationRequest,
    "regression": RegressionRequest,
    "clustering": ClusteringRequest,
    "decomposition": DecompositionRequest,
    "anomaly_detection": AnomalyDetectionRequest,
    "time_series": TimeSeriesRequest,
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _retained_output_contract(tool) -> dict:
    advertised = tool.output_schema
    assert advertised is not None
    assert advertised["type"] == "object"
    assert advertised["additionalProperties"] is True
    assert advertised["x-geochemistrypi-output-contract-delivery"] == "hash-addressed-server-enforced"
    assert advertised["x-geochemistrypi-output-contract-resolver"] == {
        "tool": "get_capabilities",
        "argument": "output_contract_sha256",
        "response_field": "output_contract_schema",
    }
    assert len(_canonical_bytes(advertised)) < 700
    Draft202012Validator.check_schema(advertised)

    contract_sha256 = advertised["x-geochemistrypi-full-output-schema-sha256"]
    contract = full_output_contract_schema(contract_sha256)
    encoded = _canonical_bytes(contract)
    assert hashlib.sha256(encoded).hexdigest() == contract_sha256
    assert len(encoded) == advertised["x-geochemistrypi-full-output-schema-utf8-bytes"]
    Draft202012Validator.check_schema(contract)
    return contract


def _contract_metadata(value: object) -> dict[str, set[bytes]]:
    keys = {
        "description",
        "default",
        "examples",
        "enum",
        "discriminator",
    }
    found = {key: set() for key in keys}

    def visit(child: object) -> None:
        if isinstance(child, dict):
            for key, nested in child.items():
                if key in found:
                    if key == "enum" and isinstance(nested, list):
                        found[key].update(_canonical_bytes(item) for item in nested)
                    else:
                        found[key].add(_canonical_bytes(nested))
                visit(nested)
        elif isinstance(child, list):
            for nested in child:
                visit(nested)

    visit(value)
    return found


def _representative_request(task: str) -> dict[str, object]:
    base: dict[str, object] = {
        "task": task,
        "training_dataset": {
            "source": "path",
            "path": "input/major_oxides.xlsx",
        },
    }
    if task == "time_series":
        return {
            **base,
            "bin_width": 100,
            "seed": 2025,
        }
    base.update(
        {
            "experiment_name": f"{task}-contract",
            "run_name": "one-valid-request",
            "identifier_column": "SAMPLE NAME",
            "feature_columns": ["SIO2(WT%)", "TIO2(WT%)", "AL2O3(WT%)"],
            "reproducibility": {"model_seed": 42},
        }
    )
    if task == "classification":
        base.update(
            {
                "target_column": "Label",
                "model": {"type": "xgboost", "number_of_estimators": 100},
            }
        )
    elif task == "regression":
        base.update(
            {
                "target_column": "Target",
                "model": {"type": "ridge_regression", "alpha": 1.0},
            }
        )
    elif task == "clustering":
        base["model"] = {"type": "kmeans", "number_of_clusters": 3}
    elif task == "decomposition":
        base["model"] = {"type": "pca", "number_of_components": 2}
    else:
        base["model"] = {
            "type": "isolation_forest",
            "number_of_estimators": 100,
            "contamination": 0.1,
            "maximum_features": 3,
            "bootstrap": False,
        }
    return base


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_each_task_capability_view_returns_one_exact_executable_validation_contract(
    tmp_path: Path,
) -> None:
    server = create_server(McpSettings(runs_root=tmp_path / "runs", cli_executable=None))

    async with Client(server) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        assert len(tools) == 13
        publicly_resolved_output_contracts = {}
        for tool_name, tool in tools.items():
            retained_contract = _retained_output_contract(tool)
            advertised = tool.output_schema
            contract_sha256 = advertised["x-geochemistrypi-full-output-schema-sha256"]
            resolved = await client.call_tool(
                "get_capabilities",
                {"output_contract_sha256": contract_sha256},
            )
            assert resolved.is_error is False
            resolved_payload = resolved.structured_content
            assert resolved_payload["response_detail"] == "output_contract"
            assert resolved_payload["output_contract_sha256"] == contract_sha256
            assert resolved_payload["output_contract_utf8_bytes"] == advertised["x-geochemistrypi-full-output-schema-utf8-bytes"]
            assert resolved_payload["output_contract_schema"] == retained_contract
            publicly_resolved_output_contracts[tool_name] = resolved_payload["output_contract_schema"]

        capabilities_output_contract = publicly_resolved_output_contracts["get_capabilities"]
        Draft202012Validator(capabilities_output_contract).validate(resolved.structured_content)
        unknown_contract = await client.call_tool(
            "get_capabilities",
            {"output_contract_sha256": "0" * 64},
        )
        assert unknown_contract.is_error is True
        Draft202012Validator(capabilities_output_contract).validate(unknown_contract.structured_content)
        assert {
            "TaskValidationRequestContract",
            "ValidationRequestNavigation",
        } <= set(capabilities_output_contract["$defs"])
        validate_schema = tools["validate_analysis"].input_schema
        assert len(validate_schema["oneOf"]) == 2
        routing_schema, detail_schema = validate_schema["oneOf"]
        assert routing_schema["additionalProperties"] is True
        assert routing_schema["required"] == ["task"]
        assert set(routing_schema["properties"]["task"]["enum"]) == set(TASK_MODELS)
        assert detail_schema["additionalProperties"] is False
        assert detail_schema["properties"]["detail"]["const"] == "full"
        assert len(_canonical_bytes(validate_schema)) < 2_000
        inspection_path = tools["inspect_dataset"].input_schema["properties"]["dataset_path"]
        assert "startup working directory" in inspection_path["description"]
        assert inspection_path["x-path-resolution-base"] == "mcp_startup_working_directory"
        assert inspection_path["x-relative-path-must-remain-within-base"] is True
        routing_schema_size = len(_canonical_bytes(tools["validate_analysis"].input_schema))

        for task, model in TASK_MODELS.items():
            result = await client.call_tool("get_capabilities", {"task": task})

            assert result.is_error is False
            response = result.structured_content
            Draft202012Validator(capabilities_output_contract).validate(response)
            assert response["response_detail"] == "compact"
            assert response["task_filter"] == task
            assert response["supported_tasks"] == list(TASK_MODELS)
            contract = response["validation_request_contract"]
            schema = contract["request_schema"]
            schema_bytes = _canonical_bytes(schema)

            Draft202012Validator.check_schema(schema)
            raw_metadata = _contract_metadata(model.model_json_schema())
            delivered_metadata = _contract_metadata(schema)
            for metadata_key, raw_values in raw_metadata.items():
                assert raw_values <= delivered_metadata[metadata_key], metadata_key
            assert contract["task"] == task
            assert contract["validation_tool"] == "validate_analysis"
            assert contract["strict_top_level_object"] is True
            assert contract["request_schema_utf8_bytes"] == len(schema_bytes)
            assert contract["request_schema_sha256"] == hashlib.sha256(schema_bytes).hexdigest()
            assert contract["request_schema_utf8_bytes"] > routing_schema_size
            assert schema["additionalProperties"] is False
            assert schema["properties"]["task"]["const"] == task
            assert "task" in schema["required"]
            assert contract["navigation"]["training_dataset_one_of"] == [
                "training_dataset",
                "training_dataset_path",
            ]
            assert contract["navigation"]["dataset_reference_discriminator_path"] == "training_dataset.source"
            assert contract["navigation"]["dataset_reference_sources"] == [
                "path",
                "builtin",
                "desktop",
            ]
            assert contract["navigation"]["path_resolution_policy"] == "absolute_or_mcp_startup_working_directory"
            assert "startup working directory" in json.dumps(
                schema,
                ensure_ascii=False,
            )

            request = _representative_request(task)
            assert Draft202012Validator(schema).is_valid(request)
            parsed = model.model_validate(request)
            assert parsed.task == task
            raw_schema = model.model_json_schema()
            raw_schema["required"] = [
                "task",
                *(field for field in raw_schema.get("required", ()) if field != "task"),
            ]
            raw_validator = Draft202012Validator(raw_schema)
            delivered_validator = Draft202012Validator(schema)
            invalid_extra = {**request, "unexpected_contract_probe": True}
            invalid_missing_task = deepcopy(request)
            invalid_missing_task.pop("task")
            for candidate in (request, invalid_extra, invalid_missing_task):
                assert delivered_validator.is_valid(candidate) is raw_validator.is_valid(candidate)

            if task == "time_series":
                assert contract["navigation"]["model_settings_discriminator_path"] is None
                assert contract["navigation"]["workflow_seed_path"] == "seed"
            else:
                assert contract["navigation"]["model_selection_discriminator_path"] == "model_selection.mode"
                assert contract["navigation"]["model_settings_discriminator_path"] == "model.type"
                assert contract["navigation"]["model_seed_path"] == "reproducibility.model_seed"

            if task == "regression":
                assert contract["navigation"]["regression_target_exactly_one_of"] == ["target_column", "target_columns"]
                minimal = contract["minimal_legal_request_example"]
                assert minimal is not None
                assert Draft202012Validator(schema).is_valid(minimal)
                assert model.model_validate(minimal).task == "regression"
                assert "target_column" in minimal
                assert "target_columns" not in minimal

                multi_target = deepcopy(minimal)
                multi_target["target_columns"] = [multi_target.pop("target_column")]
                assert Draft202012Validator(schema).is_valid(multi_target)
                assert model.model_validate(multi_target).resolved_target_columns == ("target",)

                both_targets = {**minimal, "target_columns": ["other_target"]}
                no_targets = deepcopy(minimal)
                no_targets.pop("target_column")
                assert not Draft202012Validator(schema).is_valid(both_targets)
                assert not Draft202012Validator(schema).is_valid(no_targets)
            else:
                assert contract["navigation"]["regression_target_exactly_one_of"] == []
                assert contract["minimal_legal_request_example"] is None

            assert len(_canonical_bytes(response)) <= 64 * 1024
            if task == "classification":
                unchanged = await client.call_tool(
                    "get_capabilities",
                    {
                        "task": task,
                        "if_capability_view_sha256": response["capability_view_sha256"],
                    },
                )
                assert unchanged.is_error is False
                assert unchanged.structured_content["response_detail"] == "not_modified"


@pytest.mark.anyio
async def test_unfiltered_capability_view_does_not_replay_six_task_schemas(
    tmp_path: Path,
) -> None:
    server = create_server(McpSettings(runs_root=tmp_path / "runs", cli_executable=None))

    async with Client(server) as client:
        result = await client.call_tool("get_capabilities", {})

    assert result.is_error is False
    assert result.structured_content["validation_request_contract"] is None
    assert "request_schema" not in json.dumps(result.structured_content)


@pytest.mark.anyio
async def test_start_ready_templates_cover_every_public_method_and_resolve_exact_schema(
    tmp_path: Path,
) -> None:
    server = create_server(McpSettings(runs_root=tmp_path / "runs", cli_executable=None))

    async with Client(server) as client:
        for task, methods in START_READY_METHODS_BY_TASK.items():
            resolved_schema = None
            for method in methods:
                result = await client.call_tool(
                    "get_capabilities",
                    {"detail": "start_ready", "task": task, "method": method},
                )
                assert result.is_error is False
                response = result.structured_content
                assert response["response_detail"] == "start_ready"
                assert response["task_filter"] == task
                assert response["method_filter"] == method
                assert response["available_methods"] == list(methods)
                assert response["next_action"] == {
                    "next_tool": "validate_analysis",
                    "arguments_source": ("request_template_after_replacing_placeholders_and_overlaying_user_values"),
                    "full_capabilities_required": False,
                    "dataset_inspection_required": False,
                }
                assert response["placeholder_paths"]
                assert response["template_is_structural_only"] is True
                assert response["template_runtime_model_validated"] is True
                assert len(_canonical_bytes(response)) <= 12_288

                template = response["request_template"]
                TASK_MODELS[task].model_validate(template)
                if resolved_schema is None:
                    resolver = response["request_schema_resolver"]
                    assert resolver["tool"] == "get_capabilities"
                    resolved = await client.call_tool(
                        resolver["tool"],
                        resolver["arguments"],
                    )
                    assert resolved.is_error is False
                    resolved_schema = resolved.structured_content
                    assert resolved_schema["response_detail"] == "request_schema"
                    assert resolved_schema["task"] == task
                    assert resolved_schema["request_schema_sha256"] == response["request_schema_sha256"]
                    assert resolved_schema["request_schema_utf8_bytes"] == response["request_schema_utf8_bytes"]
                    Draft202012Validator.check_schema(resolved_schema["request_schema"])
                assert Draft202012Validator(resolved_schema["request_schema"]).is_valid(template)


@pytest.mark.anyio
async def test_start_ready_is_stateless_and_conditional_only_suppresses_retained_payload(
    tmp_path: Path,
) -> None:
    server = create_server(McpSettings(runs_root=tmp_path / "runs", cli_executable=None))
    arguments = {
        "detail": "start_ready",
        "task": "classification",
        "method": "xgboost",
    }

    async with Client(server) as client:
        first = await client.call_tool("get_capabilities", arguments)
        repeated = await client.call_tool("get_capabilities", arguments)
        conditional = await client.call_tool(
            "get_capabilities",
            {
                **arguments,
                "if_capability_view_sha256": first.structured_content["capability_view_sha256"],
            },
        )

    assert first.is_error is False
    assert repeated.is_error is False
    assert first.structured_content == repeated.structured_content
    assert conditional.is_error is False
    assert conditional.structured_content["response_detail"] == "start_ready_not_modified"
    assert conditional.structured_content["requery_required"] is False
    assert "original structured payload" in conditional.structured_content["message"]


@pytest.mark.anyio
async def test_start_ready_rejects_incomplete_or_unknown_selection_and_unknown_schema_hash(
    tmp_path: Path,
) -> None:
    server = create_server(McpSettings(runs_root=tmp_path / "runs", cli_executable=None))

    async with Client(server) as client:
        missing_method = await client.call_tool(
            "get_capabilities",
            {"detail": "start_ready", "task": "classification"},
        )
        unknown_method = await client.call_tool(
            "get_capabilities",
            {
                "detail": "start_ready",
                "task": "classification",
                "method": "not-a-public-method",
            },
        )
        unknown_schema = await client.call_tool(
            "get_capabilities",
            {"request_schema_sha256": "0" * 64},
        )

    assert missing_method.is_error is True
    assert missing_method.structured_content["result_type"] == "invalid_arguments"
    assert unknown_method.is_error is True
    assert unknown_method.structured_content["result_type"] == "invalid_arguments"
    assert unknown_schema.is_error is True
    assert unknown_schema.structured_content["result_type"] == "contract_not_found"
    assert "Do not guess hashes" in unknown_schema.structured_content["next_action"]


@pytest.mark.anyio
async def test_capability_tool_guides_direct_validation_without_hiding_tools(
    tmp_path: Path,
) -> None:
    server = create_server(McpSettings(runs_root=tmp_path / "runs", cli_executable=None))

    async with Client(server) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    assert len(tools) == 13
    capability_description = tools["get_capabilities"].description
    inspection_description = tools["inspect_dataset"].description
    validation_description = tools["validate_analysis"].input_schema["oneOf"][0]["description"]
    assert "detail='start_ready'" in capability_description
    assert "detail='full' is audit-only" in capability_description
    assert "retaining the original payload" in capability_description
    assert "Skip inspection" in inspection_description
    assert "Known dataset paths and columns do not require inspection" in validation_description


@pytest.mark.anyio
async def test_validation_errors_point_to_dataset_model_and_seed_locations(
    tmp_path: Path,
) -> None:
    server = create_server(McpSettings(runs_root=tmp_path / "runs", cli_executable=None))
    invalid = {
        "task": "anomaly_detection",
        "dataset": {"source": "path", "path": "input/major_oxides.xlsx"},
        "experiment_name": "error-guidance",
        "run_name": "one-correction",
        "identifier_column": "SAMPLE NAME",
        "feature_columns": ["SIO2(WT%)", "TIO2(WT%)", "AL2O3(WT%)"],
        "model": {
            "type": "isolation_forest",
            "model_parameters": {"number_of_estimators": 100},
        },
        "reproducibility": {"random_seed": 42},
    }

    async with Client(server) as client:
        result = await client.call_tool("validate_analysis", invalid)

    assert result.is_error is True
    assert result.structured_content["error_type"] == "validation_error"
    alternatives = " ".join(cause["valid_alternative"] for cause in result.structured_content["root_causes"])
    assert "top-level training_dataset reference" in alternatives
    assert "model.type" in alternatives
    assert "reproducibility" in alternatives
    assert "detail='start_ready'" in result.structured_content["next_action"]
    assert "do not inspect them again" in result.structured_content["next_action"]
    assert "repeated guessed requests" in result.structured_content["next_action"]
