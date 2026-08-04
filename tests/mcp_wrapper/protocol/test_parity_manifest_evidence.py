import json
from pathlib import Path

from geochemistrypi_mcp.capability_manifest import load_capability_manifest



def _matrix() -> dict:
    path = Path(__file__).resolve().parents[1] / "parity" / "fixtures" / "full_parity_matrix_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _cases() -> list[tuple[str, str, str, str]]:
    matrix = _matrix()
    cases = []
    for task, models in matrix["manual_single_models"].items():
        cases.extend((f"manual.{task}.{model}", task, model, "manual") for model in models)
    for task, models in matrix["automl_models"].items():
        cases.extend((f"automl.{task}.{model}", task, model, "automl") for model in models)
    cases.extend((f"aggregate.{task}.all", task, "all", "aggregate") for task in matrix["aggregate_tasks"])
    return cases


def test_every_supported_capability_has_resolvable_direct_and_mcp_evidence() -> None:
    repository = Path(__file__).resolve().parents[3]
    manifest = load_capability_manifest()
    scenario_ids = {case[0] for case in _cases()}
    groups = {
        "automl.classification": {case[0] for case in _cases() if case[1] == "classification" and case[3] == "automl"},
        "automl.regression": {case[0] for case in _cases() if case[1] == "regression" and case[3] == "automl"},
    }
    for capability in manifest["capabilities"]:
        if not capability["mcp_supported"]:
            continue
        assert capability["status"] == "verified"
        assert capability["evidence"]
        for evidence in capability["evidence"]:
            if evidence.startswith("parity-scenario:"):
                assert evidence.removeprefix("parity-scenario:") in scenario_ids
            elif evidence.startswith("parity-group:"):
                assert groups[evidence.removeprefix("parity-group:")]
            else:
                evidence_path = repository / evidence.split("::", 1)[0]
                assert evidence_path.is_file(), evidence
        if capability["category"] == "model":
            expected = capability["id"].removeprefix("model.")
            assert f"parity-scenario:manual.{expected}" in capability["evidence"]


def test_matrix_branch_inventory_covers_every_pr9i_dimension() -> None:
    matrix = _matrix()
    coverage = {
        value
        for scenario in matrix["branch_scenarios"]
        for value in scenario["coverage"]
    }
    assert {
        "classification_inference",
        "regression_inference",
        "encode_original",
        "map",
        "interval",
        "quantile",
        "error",
        "keep",
        "drop_rows",
        "impute",
        "none",
        "min_max",
        "standardization",
        "mean_normalization",
        "variance_threshold",
        "k_best",
        "disabled",
        "single",
        "multiple",
        "new",
        "existing_id",
        "path",
        "builtin",
        "desktop",
        "seeded_bootstrap",
    } <= coverage
