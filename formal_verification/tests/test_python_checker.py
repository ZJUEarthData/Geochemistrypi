from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

FORMAL = Path(__file__).resolve().parents[1]
PYTHON = FORMAL / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from check_trace import PUBLIC_CHECK_IDS, TraceSpecError, check_trace, load_bundle  # noqa: E402
from generate_counterexamples import baseline_case, build_bundle  # noqa: E402
from run_bridge import portable_log  # noqa: E402


def result_map(report):
    return {item["checkId"]: item["passed"] for item in report["cases"][0]["checks"]}


def test_portable_log_normalizes_line_endings_and_trailing_whitespace():
    assert portable_log("alpha  \r\nbeta\t\n") == "alpha\nbeta\n"


def test_public_surface_has_twenty_stable_unique_checks():
    assert len(PUBLIC_CHECK_IDS) == 20
    assert len(set(PUBLIC_CHECK_IDS)) == 20
    assert PUBLIC_CHECK_IDS[0] == "D01.input_rows_identified"
    assert PUBLIC_CHECK_IDS[-1] == "E02.model_registry_immutable_during_run"


def test_positive_baseline_accepts_noncontiguous_but_injective_codes():
    bundle = {
        "schemaVersion": 2,
        "sourceCommit": "test-commit",
        "generatedAt": "2026-08-11T00:00:00Z",
        "cases": [baseline_case()],
    }
    report = check_trace(bundle)
    assert report["allCasesAccepted"] is True
    assert report["cases"][0]["checks"] == [{"checkId": check_id, "passed": True} for check_id in PUBLIC_CHECK_IDS]
    assert [item["code"] for item in bundle["cases"][0]["labels"]["runtimeMappings"]] == [4, 9]


def test_twenty_counterexamples_are_complete_and_isolated():
    bundle, observations = build_bundle()
    report = check_trace(bundle)
    assert report["caseCount"] == 21
    assert report["counterexampleCount"] == 20
    assert report["coveredCheckCount"] == 20
    assert report["counterexampleCoverageComplete"] is True
    assert report["allCounterexamplesIsolated"] is True
    assert report["allExpectationsMatched"] is True
    assert observations["baselineAccepted"] is True
    for case in report["cases"][1:]:
        assert case["failedCheckIds"] == [case["targetCheckId"]]


def test_production_case_has_no_oracle_bypass():
    case = baseline_case()
    case["caseKind"] = "production"
    case["caseId"] = "production-with-role-gap"
    case["targetCheckId"] = ""
    case["expectedConformant"] = True
    case["dataset"]["roleValidationPairs"] = []
    bundle = {
        "schemaVersion": 2,
        "sourceCommit": "test-commit",
        "generatedAt": "2026-08-11T00:00:00Z",
        "cases": [case],
    }
    report = check_trace(bundle)
    assert report["cases"][0]["accepted"] is False
    assert report["cases"][0]["expectationMatched"] is True
    assert result_map(report)["D04.column_roles_guarded_and_disjoint"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("schemaVersion", 1),
        ("sourceCommit", ""),
        ("cases", []),
    ],
)
def test_invalid_envelope_is_rejected(field, value):
    bundle = {
        "schemaVersion": 2,
        "sourceCommit": "test-commit",
        "generatedAt": "2026-08-11T00:00:00Z",
        "cases": [baseline_case()],
    }
    bundle[field] = value
    with pytest.raises(TraceSpecError):
        check_trace(bundle)


def test_unknown_and_missing_fields_are_rejected():
    bundle = {
        "schemaVersion": 2,
        "sourceCommit": "test-commit",
        "generatedAt": "2026-08-11T00:00:00Z",
        "cases": [baseline_case()],
        "unexpected": True,
    }
    with pytest.raises(TraceSpecError, match="unknown fields"):
        check_trace(bundle)
    del bundle["unexpected"]
    del bundle["cases"][0]["pipeline"]["trainRowIds"]
    with pytest.raises(TraceSpecError, match="missing fields"):
        check_trace(bundle)


def test_boolean_is_not_accepted_as_natural_number():
    case = baseline_case()
    case["pipeline"]["stages"][0]["fitCount"] = True
    bundle = {
        "schemaVersion": 2,
        "sourceCommit": "test-commit",
        "generatedAt": "2026-08-11T00:00:00Z",
        "cases": [case],
    }
    with pytest.raises(TraceSpecError, match="natural number"):
        check_trace(bundle)


def test_duplicate_json_keys_are_rejected(tmp_path):
    trace = tmp_path / "duplicate.json"
    trace.write_text('{"schemaVersion":2,"schemaVersion":2}', encoding="utf-8")
    with pytest.raises(TraceSpecError, match="duplicate JSON object key"):
        load_bundle(trace)


def test_counterexample_target_must_be_public():
    case = baseline_case()
    case.update(
        {
            "caseKind": "counterexample",
            "expectedConformant": False,
            "targetCheckId": "unknown.check",
        }
    )
    bundle = {
        "schemaVersion": 2,
        "sourceCommit": "test-commit",
        "generatedAt": "2026-08-11T00:00:00Z",
        "cases": [case],
    }
    with pytest.raises(TraceSpecError, match="target one public check"):
        check_trace(bundle)


def test_extra_inference_columns_are_allowed_when_effective_order_matches():
    case = baseline_case()
    case["pipeline"]["inferenceInputFeatureSchema"] += ["more-metadata"]
    bundle = {
        "schemaVersion": 2,
        "sourceCommit": "test-commit",
        "generatedAt": "2026-08-11T00:00:00Z",
        "cases": [case],
    }
    assert result_map(check_trace(bundle))["P01.effective_schema_matches_training"] is True


def test_training_subset_fit_is_allowed_but_test_row_fit_is_rejected():
    case = baseline_case()
    case["pipeline"]["stages"][0]["fitRowIds"] = [0, 1]
    bundle = {
        "schemaVersion": 2,
        "sourceCommit": "test-commit",
        "generatedAt": "2026-08-11T00:00:00Z",
        "cases": [case],
    }
    assert result_map(check_trace(bundle))["P02.stateful_fit_uses_training_rows_only"] is True
    case = copy.deepcopy(case)
    case["pipeline"]["stages"][0]["fitRowIds"].append(4)
    assert result_map(check_trace({**bundle, "cases": [case]}))["P02.stateful_fit_uses_training_rows_only"] is False


def test_disabled_feature_engineering_and_preprocessing_are_not_vacuous_misreports():
    case = baseline_case()
    case["dataset"]["featureEngineeringEnabled"] = False
    case["dataset"]["derivedFeatures"] = []
    case["pipeline"]["preprocessingEnabled"] = False
    case["pipeline"]["declaredStageIds"] = []
    case["pipeline"]["materializedStageIds"] = []
    case["pipeline"]["stages"] = []
    bundle = {
        "schemaVersion": 2,
        "sourceCommit": "test-commit",
        "generatedAt": "2026-08-11T00:00:00Z",
        "cases": [case],
    }
    results = result_map(check_trace(bundle))
    assert results["D05.derived_feature_lineage_safe"] is True
    assert results["P02.stateful_fit_uses_training_rows_only"] is True
    assert results["P05.declared_and_materialized_stage_order_equal"] is True
