import json
from pathlib import Path

from geochemistrypi_mcp.config.clients import SUPPORTED_CLIENTS


def test_client_acceptance_matrix_covers_every_supported_target_and_required_step() -> None:
    repository = Path(__file__).resolve().parents[3]
    matrix_path = Path(__file__).parent / "fixtures" / "client_acceptance_matrix_v1.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert matrix["schema_version"] == 1
    prompt = matrix["canonical_user_prompt"]
    assert prompt == "帮我分析桌面上的 rocks.xlsx，看看哪些元素最能区分不同岩性。"
    assert len(prompt) <= 40
    assert all(
        technical_term not in prompt.lower()
        for technical_term in (
            "mcp",
            "cli",
            "json",
            "target_column",
            "feature_columns",
            "validate_analysis",
            "logistic_regression",
        )
    )
    assert set(matrix["expected_client_behavior"]) == {
        "discover_and_inspect_before_execution",
        "ask_only_plain_language_scientific_questions",
        "preserve_explicit_scientific_choices",
        "never_silently_replace_invalid_choices",
        "preview_in_plain_language_and_wait_for_confirmation",
        "poll_and_summarize_original_outputs",
    }
    assert tuple(target["id"] for target in matrix["targets"]) == SUPPORTED_CLIENTS
    assert len(matrix["targets"]) == 14
    assert set(matrix["required_steps"]) == {
        "discovery",
        "atomic_registration",
        "backup",
        "tools_list",
        "capability_discovery",
        "natural_language_dataset_workflow",
        "status_polling",
        "result_retrieval",
        "owned_unregistration",
        "unrelated_settings_preserved",
    }
    for target in matrix["targets"]:
        assert (repository / target["registration_test"].split("::", 1)[0]).is_file()
        assert (repository / target["protocol_test"].split("::", 1)[0]).is_file()
        assert target["real_client_status"] in {
            "controlled_protocol_verified",
            "pending_external_client_run",
        }
    assert matrix["targets"][0]["real_client_status"] == "controlled_protocol_verified"
    assert all(target["real_client_status"] == "pending_external_client_run" for target in matrix["targets"][1:])
