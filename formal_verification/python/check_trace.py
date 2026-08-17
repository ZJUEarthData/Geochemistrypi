#!/usr/bin/env python3
"""Strict Python mirror of the twenty Lean audit propositions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

JsonObject = Dict[str, Any]


class TraceSpecError(ValueError):
    """The input is not a valid GeoPi trace schema-v2 bundle."""


_STRING = object()
_NAT = object()
_BOOL = object()


def _list_of(item_schema: Any) -> Tuple[str, Any]:
    return ("list", item_schema)


_DERIVED_SCHEMA = {
    "name": _STRING,
    "sourceColumns": _list_of(_STRING),
    "aggregateFitRowIds": _list_of(_NAT),
}

_DATASET_SCHEMA = {
    "rowIds": _list_of(_NAT),
    "rowIdentityNonemptyMask": _list_of(_BOOL),
    "filterInputRowIds": _list_of(_NAT),
    "filterOutputRowIds": _list_of(_NAT),
    "filterXRowIds": _list_of(_NAT),
    "filterTargetRowIds": _list_of(_NAT),
    "filterNameRowIds": _list_of(_NAT),
    "trainRowIds": _list_of(_NAT),
    "testRowIds": _list_of(_NAT),
    "xTrainRowIds": _list_of(_NAT),
    "yTrainRowIds": _list_of(_NAT),
    "nameTrainRowIds": _list_of(_NAT),
    "xTestRowIds": _list_of(_NAT),
    "yTestRowIds": _list_of(_NAT),
    "nameTestRowIds": _list_of(_NAT),
    "featureColumns": _list_of(_STRING),
    "targetColumns": _list_of(_STRING),
    "identifierColumns": _list_of(_STRING),
    "roleValidationPairs": _list_of(_STRING),
    "featureEngineeringEnabled": _BOOL,
    "allowedDerivedSourceColumns": _list_of(_STRING),
    "derivedFeatures": _list_of(_DERIVED_SCHEMA),
}

_STAGE_SCHEMA = {
    "stageId": _STRING,
    "name": _STRING,
    "fitRowIds": _list_of(_NAT),
    "fitCount": _NAT,
    "trainingStateDigest": _STRING,
    "inferenceStateDigest": _STRING,
    "outputValueCount": _NAT,
    "outputNonFiniteCount": _NAT,
}

_PIPELINE_SCHEMA = {
    "preprocessingEnabled": _BOOL,
    "trainRowIds": _list_of(_NAT),
    "declaredStageIds": _list_of(_STRING),
    "materializedStageIds": _list_of(_STRING),
    "trainFeatureSchema": _list_of(_STRING),
    "inferenceInputFeatureSchema": _list_of(_STRING),
    "effectiveInferenceFeatureSchema": _list_of(_STRING),
    "pipelineOutputFeatureSchema": _list_of(_STRING),
    "modelTrainFeatureSchema": _list_of(_STRING),
    "pipelineTrainOutputDigest": _STRING,
    "modelTrainInputDigest": _STRING,
    "stages": _list_of(_STAGE_SCHEMA),
}

_MAPPING_SCHEMA = {"label": _STRING, "code": _NAT}

_LABEL_SCHEMA = {
    "codecEnabled": _BOOL,
    "sourceLabels": _list_of(_STRING),
    "runtimeMappings": _list_of(_MAPPING_SCHEMA),
    "fullMappings": _list_of(_MAPPING_SCHEMA),
    "trainMappings": _list_of(_MAPPING_SCHEMA),
    "testMappings": _list_of(_MAPPING_SCHEMA),
    "persistedMappings": _list_of(_MAPPING_SCHEMA),
    "codecFitCount": _NAT,
    "predictedCodes": _list_of(_NAT),
    "decodedPredictions": _list_of(_STRING),
}

_PREDICTION_SCHEMA = {
    "scope": _STRING,
    "sourceRowIds": _list_of(_NAT),
    "predictionValues": _list_of(_STRING),
    "sampleRowIds": _list_of(_NAT),
    "artifactRowIds": _list_of(_NAT),
    "artifactPredictionValues": _list_of(_STRING),
    "artifactMismatchPolicy": _STRING,
    "modelRunId": _STRING,
    "artifactRunId": _STRING,
}

_EXECUTION_SCHEMA = {
    "eligibleModels": _list_of(_STRING),
    "selectedModelIds": _list_of(_STRING),
    "trainedModelIds": _list_of(_STRING),
    "trainedModelCount": _NAT,
    "registryBefore": _list_of(_STRING),
    "registryAfter": _list_of(_STRING),
    "registryMutationOperations": _list_of(_STRING),
    "activeRunId": _STRING,
    "stateOwnerRunId": _STRING,
}

_CASE_SCHEMA = {
    "caseId": _STRING,
    "caseKind": _STRING,
    "description": _STRING,
    "expectedConformant": _BOOL,
    "targetCheckId": _STRING,
    "dataset": _DATASET_SCHEMA,
    "pipeline": _PIPELINE_SCHEMA,
    "labels": _LABEL_SCHEMA,
    "prediction": _PREDICTION_SCHEMA,
    "execution": _EXECUTION_SCHEMA,
}

_BUNDLE_SCHEMA = {
    "schemaVersion": _NAT,
    "sourceCommit": _STRING,
    "generatedAt": _STRING,
    "cases": _list_of(_CASE_SCHEMA),
}

PUBLIC_CHECK_IDS = [
    "D01.input_rows_identified",
    "D02.split_is_disjoint_partition",
    "D03.supervised_views_row_aligned",
    "D04.column_roles_guarded_and_disjoint",
    "D05.derived_feature_lineage_safe",
    "D06.filtered_rows_keep_lineage",
    "P01.effective_schema_matches_training",
    "P02.stateful_fit_uses_training_rows_only",
    "P03.fitted_state_reused_for_model_and_inference",
    "P04.model_input_schema_matches_pipeline_output",
    "P05.declared_and_materialized_stage_order_equal",
    "P06.observed_stage_outputs_finite",
    "L01.codec_total_and_injective",
    "L02.one_codec_fitted_once_for_all_splits",
    "L03.codec_persisted_and_predictions_decodable",
    "A01.predictions_bound_to_source_rows",
    "A02.artifact_pairs_aligned_and_mismatch_rejected",
    "A03.model_artifact_and_state_share_run",
    "E01.selected_models_eligible_and_trained",
    "E02.model_registry_immutable_during_run",
]


def _type_name(value: Any) -> str:
    return type(value).__name__


def _decode(value: Any, schema: Any, path: str) -> Any:
    if schema is _STRING:
        if not isinstance(value, str):
            raise TraceSpecError(f"{path} must be a string, got {_type_name(value)}")
        return value
    if schema is _BOOL:
        if type(value) is not bool:
            raise TraceSpecError(f"{path} must be a boolean, got {_type_name(value)}")
        return value
    if schema is _NAT:
        if type(value) is not int or value < 0:
            raise TraceSpecError(f"{path} must be a natural number, got {value!r}")
        return value
    if isinstance(schema, tuple) and schema[0] == "list":
        if not isinstance(value, list):
            raise TraceSpecError(f"{path} must be a list, got {_type_name(value)}")
        return [_decode(item, schema[1], f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(schema, dict):
        if not isinstance(value, dict):
            raise TraceSpecError(f"{path} must be an object, got {_type_name(value)}")
        expected = set(schema)
        actual = set(value)
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        if missing:
            raise TraceSpecError(f"{path} is missing fields {missing}")
        if unknown:
            raise TraceSpecError(f"{path} has unknown fields {unknown}")
        return {key: _decode(value[key], item_schema, f"{path}.{key}") for key, item_schema in schema.items()}
    raise AssertionError(f"unsupported schema at {path}")


def decode_bundle(document: Any) -> JsonObject:
    bundle = _decode(document, _BUNDLE_SCHEMA, "bundle")
    if bundle["schemaVersion"] != 2:
        raise TraceSpecError("schemaVersion must equal 2")
    if bundle["sourceCommit"] == "":
        raise TraceSpecError("sourceCommit must not be empty")
    if bundle["cases"] == []:
        raise TraceSpecError("cases must not be empty")
    case_ids = [case["caseId"] for case in bundle["cases"]]
    if any(case_id == "" for case_id in case_ids) or not _nodup(case_ids):
        raise TraceSpecError("caseId values must be non-empty and unique")
    for case in bundle["cases"]:
        kind = case["caseKind"]
        if kind not in {"baseline", "counterexample", "production"}:
            raise TraceSpecError(f"case {case['caseId']} has invalid caseKind {kind}")
        if kind == "baseline" and (not case["expectedConformant"] or case["targetCheckId"] != ""):
            raise TraceSpecError("a baseline must expect conformance and carry no target")
        if kind == "counterexample" and (case["expectedConformant"] or case["targetCheckId"] not in PUBLIC_CHECK_IDS):
            raise TraceSpecError("a counterexample must expect rejection and target one public check")
        if kind == "production" and case["targetCheckId"] != "":
            raise TraceSpecError("a production case must carry no counterexample target")
    return bundle


def _reject_non_json_number(token: str) -> None:
    raise ValueError(f"non-JSON numeric constant {token}")


def _strict_object(pairs: Sequence[Tuple[str, Any]]) -> JsonObject:
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def load_bundle(path: Union[str, Path]) -> JsonObject:
    trace_path = Path(path)
    try:
        raw = trace_path.read_text(encoding="utf-8")
    except OSError as error:
        raise TraceSpecError(f"cannot read {trace_path}: {error}") from error
    try:
        document = json.loads(raw, parse_constant=_reject_non_json_number, object_pairs_hook=_strict_object)
    except (json.JSONDecodeError, ValueError) as error:
        raise TraceSpecError(f"invalid JSON: {error}") from error
    return decode_bundle(document)


def _nodup(values: Sequence[Any]) -> bool:
    return len(values) == len(set(values))


def _same_members(left: Sequence[Any], right: Sequence[Any]) -> bool:
    return set(left) == set(right)


def _disjoint(left: Sequence[Any], right: Sequence[Any]) -> bool:
    return set(left).isdisjoint(right)


def _subset(left: Sequence[Any], right: Sequence[Any]) -> bool:
    return set(left) <= set(right)


def _input_rows_identified(d: Mapping[str, Any]) -> bool:
    mask = d["rowIdentityNonemptyMask"]
    return d["rowIds"] != [] and _nodup(d["rowIds"]) and len(mask) == len(d["rowIds"]) and all(mask)


def _split_is_disjoint_partition(d: Mapping[str, Any]) -> bool:
    train, test, rows = d["trainRowIds"], d["testRowIds"], d["rowIds"]
    return train != [] and test != [] and _nodup(train) and _nodup(test) and _disjoint(train, test) and _same_members(train + test, rows)


def _supervised_views_row_aligned(d: Mapping[str, Any]) -> bool:
    return d["xTrainRowIds"] == d["yTrainRowIds"] == d["nameTrainRowIds"] == d["trainRowIds"] and d["xTestRowIds"] == d["yTestRowIds"] == d["nameTestRowIds"] == d["testRowIds"]


def _column_roles_guarded_and_disjoint(d: Mapping[str, Any]) -> bool:
    features, targets, identifiers = d["featureColumns"], d["targetColumns"], d["identifierColumns"]
    required = {"feature_target", "feature_identifier", "target_identifier"}
    return (
        features != []
        and targets != []
        and identifiers != []
        and _nodup(features)
        and _nodup(targets)
        and _nodup(identifiers)
        and _disjoint(features, targets)
        and _disjoint(features, identifiers)
        and _disjoint(targets, identifiers)
        and required <= set(d["roleValidationPairs"])
    )


def _derived_feature_lineage_safe(d: Mapping[str, Any]) -> bool:
    derived = d["derivedFeatures"]
    if not d["featureEngineeringEnabled"]:
        return derived == []
    allowed = d["allowedDerivedSourceColumns"]
    if derived == [] or allowed == [] or not _nodup(allowed) or not _disjoint(allowed, d["targetColumns"]) or not _disjoint(allowed, d["identifierColumns"]):
        return False
    train = d["trainRowIds"]
    return all(
        feature["name"] != ""
        and feature["sourceColumns"] != []
        and _subset(feature["sourceColumns"], allowed)
        and (feature["aggregateFitRowIds"] == [] or (_nodup(feature["aggregateFitRowIds"]) and _subset(feature["aggregateFitRowIds"], train)))
        for feature in derived
    )


def _filtered_rows_keep_lineage(d: Mapping[str, Any]) -> bool:
    source, output = d["filterInputRowIds"], d["filterOutputRowIds"]
    return (
        source != []
        and _nodup(source)
        and output != []
        and _nodup(output)
        and _subset(output, source)
        and d["filterXRowIds"] == output
        and d["filterTargetRowIds"] == output
        and d["filterNameRowIds"] == output
    )


def _effective_schema_matches_training(p: Mapping[str, Any]) -> bool:
    train, raw, effective = p["trainFeatureSchema"], p["inferenceInputFeatureSchema"], p["effectiveInferenceFeatureSchema"]
    return train != [] and _nodup(train) and _nodup(raw) and _nodup(effective) and effective == train and _subset(effective, raw)


def _stateful_fit_uses_training_rows_only(p: Mapping[str, Any]) -> bool:
    if not p["preprocessingEnabled"]:
        return p["stages"] == []
    train, stages = p["trainRowIds"], p["stages"]
    return train != [] and _nodup(train) and stages != [] and all(stage["fitRowIds"] != [] and _nodup(stage["fitRowIds"]) and _subset(stage["fitRowIds"], train) for stage in stages)


def _fitted_state_reused_for_model_and_inference(p: Mapping[str, Any]) -> bool:
    if not p["preprocessingEnabled"]:
        return True
    return (
        all(stage["fitCount"] == 1 and stage["trainingStateDigest"] != "" and stage["trainingStateDigest"] == stage["inferenceStateDigest"] for stage in p["stages"])
        and p["pipelineTrainOutputDigest"] != ""
        and p["pipelineTrainOutputDigest"] == p["modelTrainInputDigest"]
    )


def _model_input_schema_matches_pipeline_output(p: Mapping[str, Any]) -> bool:
    schema = p["pipelineOutputFeatureSchema"]
    return schema != [] and _nodup(schema) and schema == p["modelTrainFeatureSchema"]


def _declared_and_materialized_stage_order_equal(p: Mapping[str, Any]) -> bool:
    declared, materialized, observed = p["declaredStageIds"], p["materializedStageIds"], [stage["stageId"] for stage in p["stages"]]
    if not p["preprocessingEnabled"]:
        return declared == [] and materialized == [] and observed == []
    return declared != [] and _nodup(declared) and _nodup(materialized) and declared == materialized == observed


def _observed_stage_outputs_finite(p: Mapping[str, Any]) -> bool:
    if not p["preprocessingEnabled"]:
        return True
    return all(stage["outputValueCount"] > 0 and stage["outputNonFiniteCount"] == 0 for stage in p["stages"])


def _mapping_labels(mappings: Sequence[Mapping[str, Any]]) -> List[str]:
    return [item["label"] for item in mappings]


def _mapping_codes(mappings: Sequence[Mapping[str, Any]]) -> List[int]:
    return [item["code"] for item in mappings]


def _encode(mappings: Sequence[Mapping[str, Any]], label: str) -> Optional[int]:
    return next((item["code"] for item in mappings if item["label"] == label), None)


def _decode_code(mappings: Sequence[Mapping[str, Any]], code: int) -> Optional[str]:
    return next((item["label"] for item in mappings if item["code"] == code), None)


def _codec_total_and_injective(label_trace: Mapping[str, Any]) -> bool:
    if not label_trace["codecEnabled"]:
        return label_trace["runtimeMappings"] == []
    mappings, source = label_trace["runtimeMappings"], label_trace["sourceLabels"]
    labels, codes = _mapping_labels(mappings), _mapping_codes(mappings)
    return (
        source != []
        and _nodup(source)
        and mappings != []
        and _same_members(labels, source)
        and _nodup(labels)
        and _nodup(codes)
        and all((code := _encode(mappings, label)) is not None and _decode_code(mappings, code) == label for label in source)
    )


def _one_codec_fitted_once_for_all_splits(label_trace: Mapping[str, Any]) -> bool:
    if not label_trace["codecEnabled"]:
        return label_trace["codecFitCount"] == 0
    runtime = label_trace["runtimeMappings"]
    return label_trace["codecFitCount"] == 1 and runtime == label_trace["fullMappings"] == label_trace["trainMappings"] == label_trace["testMappings"]


def _codec_persisted_and_predictions_decodable(label_trace: Mapping[str, Any]) -> bool:
    if not label_trace["codecEnabled"]:
        return label_trace["persistedMappings"] == [] and label_trace["predictedCodes"] == []
    decoded = [_decode_code(label_trace["runtimeMappings"], code) for code in label_trace["predictedCodes"]]
    return (
        label_trace["persistedMappings"] != []
        and label_trace["persistedMappings"] == label_trace["runtimeMappings"]
        and all(label is not None for label in decoded)
        and decoded == label_trace["decodedPredictions"]
        and set(label_trace["decodedPredictions"]) <= set(label_trace["sourceLabels"])
    )


def _predictions_bound_to_source_rows(p: Mapping[str, Any]) -> bool:
    return (
        p["scope"] in {"test", "application"}
        and p["sourceRowIds"] != []
        and p["predictionValues"] != []
        and p["sourceRowIds"] == p["sampleRowIds"]
        and _nodup(p["sampleRowIds"])
        and len(p["predictionValues"]) == len(p["sampleRowIds"])
    )


def _artifact_pairs_aligned_and_mismatch_rejected(p: Mapping[str, Any]) -> bool:
    return p["sampleRowIds"] == p["artifactRowIds"] and p["predictionValues"] == p["artifactPredictionValues"] and p["artifactMismatchPolicy"] == "reject"


def _model_artifact_and_state_share_run(p: Mapping[str, Any], e: Mapping[str, Any]) -> bool:
    return p["modelRunId"] != "" and p["modelRunId"] == p["artifactRunId"] == e["activeRunId"]


def _selected_models_eligible_and_trained(e: Mapping[str, Any]) -> bool:
    eligible, selected, trained = e["eligibleModels"], e["selectedModelIds"], e["trainedModelIds"]
    return (
        eligible != []
        and _nodup(eligible)
        and selected != []
        and _nodup(selected)
        and _nodup(trained)
        and _subset(selected, eligible)
        and _same_members(selected, trained)
        and e["trainedModelCount"] == len(trained)
    )


def _model_registry_immutable_during_run(e: Mapping[str, Any]) -> bool:
    return e["registryBefore"] != [] and _nodup(e["registryBefore"]) and e["registryMutationOperations"] == [] and e["registryBefore"] == e["registryAfter"]


def checks(case: Mapping[str, Any]) -> List[JsonObject]:
    d, p, labels, a, e = case["dataset"], case["pipeline"], case["labels"], case["prediction"], case["execution"]
    outcomes = [
        (PUBLIC_CHECK_IDS[0], _input_rows_identified(d)),
        (PUBLIC_CHECK_IDS[1], _split_is_disjoint_partition(d)),
        (PUBLIC_CHECK_IDS[2], _supervised_views_row_aligned(d)),
        (PUBLIC_CHECK_IDS[3], _column_roles_guarded_and_disjoint(d)),
        (PUBLIC_CHECK_IDS[4], _derived_feature_lineage_safe(d)),
        (PUBLIC_CHECK_IDS[5], _filtered_rows_keep_lineage(d)),
        (PUBLIC_CHECK_IDS[6], _effective_schema_matches_training(p)),
        (PUBLIC_CHECK_IDS[7], _stateful_fit_uses_training_rows_only(p)),
        (PUBLIC_CHECK_IDS[8], _fitted_state_reused_for_model_and_inference(p)),
        (PUBLIC_CHECK_IDS[9], _model_input_schema_matches_pipeline_output(p)),
        (PUBLIC_CHECK_IDS[10], _declared_and_materialized_stage_order_equal(p)),
        (PUBLIC_CHECK_IDS[11], _observed_stage_outputs_finite(p)),
        (PUBLIC_CHECK_IDS[12], _codec_total_and_injective(labels)),
        (PUBLIC_CHECK_IDS[13], _one_codec_fitted_once_for_all_splits(labels)),
        (PUBLIC_CHECK_IDS[14], _codec_persisted_and_predictions_decodable(labels)),
        (PUBLIC_CHECK_IDS[15], _predictions_bound_to_source_rows(a)),
        (PUBLIC_CHECK_IDS[16], _artifact_pairs_aligned_and_mismatch_rejected(a)),
        (PUBLIC_CHECK_IDS[17], _model_artifact_and_state_share_run(a, e)),
        (PUBLIC_CHECK_IDS[18], _selected_models_eligible_and_trained(e)),
        (PUBLIC_CHECK_IDS[19], _model_registry_immutable_during_run(e)),
    ]
    return [{"checkId": check_id, "passed": bool(passed)} for check_id, passed in outcomes]


def accepted(case: Mapping[str, Any]) -> bool:
    return all(result["passed"] for result in checks(case))


def report_case(case: Mapping[str, Any]) -> JsonObject:
    check_results = checks(case)
    failed = [result["checkId"] for result in check_results if not result["passed"]]
    is_accepted = not failed
    kind = case["caseKind"]
    expectation_matched = True if kind == "production" else is_accepted == case["expectedConformant"]
    isolation_matched = True if kind == "production" else (failed == [] if kind == "baseline" else failed == [case["targetCheckId"]])
    return {
        "caseId": case["caseId"],
        "caseKind": kind,
        "description": case["description"],
        "expectedConformant": case["expectedConformant"],
        "targetCheckId": case["targetCheckId"],
        "accepted": is_accepted,
        "expectationMatched": expectation_matched,
        "failedCheckIds": failed,
        "isolationMatched": isolation_matched,
        "checks": check_results,
    }


def report_bundle(bundle: Mapping[str, Any]) -> JsonObject:
    reports = [report_case(case) for case in bundle["cases"]]
    accepted_count = sum(report["accepted"] for report in reports)
    counterexamples = [report for report in reports if report["caseKind"] == "counterexample"]
    targets = {report["targetCheckId"] for report in counterexamples}
    return {
        "schemaVersion": bundle["schemaVersion"],
        "sourceCommit": bundle["sourceCommit"],
        "caseCount": len(reports),
        "acceptedCount": accepted_count,
        "rejectedCount": len(reports) - accepted_count,
        "counterexampleCount": len(counterexamples),
        "coveredCheckCount": len(targets),
        "counterexampleCoverageComplete": len(targets) == len(PUBLIC_CHECK_IDS) and targets == set(PUBLIC_CHECK_IDS),
        "allCounterexamplesIsolated": all(report["isolationMatched"] for report in reports),
        "allCasesAccepted": all(report["accepted"] for report in reports),
        "allExpectationsMatched": all(report["expectationMatched"] for report in reports),
        "cases": reports,
    }


def check_trace(document: Any) -> JsonObject:
    return report_bundle(decode_bundle(document))


def _write_report(path: Union[str, Path], report: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Check a GeoPi schema-v2 trace against the Lean specification")
    parser.add_argument("trace")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        report = report_bundle(load_bundle(args.trace))
        _write_report(args.output, report)
    except (TraceSpecError, OSError) as error:
        print(f"trace checker error: {error}", file=sys.stderr)
        return 2
    return 0 if report["allCasesAccepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
