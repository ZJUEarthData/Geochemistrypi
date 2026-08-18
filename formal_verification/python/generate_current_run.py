#!/usr/bin/env python3
"""Translate counterexample and production JSON facts into closed Lean constants and proofs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable

from check_trace import PUBLIC_CHECK_IDS, accepted, checks, load_bundle

_LIST_REFERENCES: dict[tuple[str, tuple[Any, ...]], str] = {}


def s(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def b(value: bool) -> str:
    return "true" if value else "false"


def primitive_list_key(values: list[Any]) -> tuple[str, tuple[Any, ...]] | None:
    if not values:
        return None
    if all(type(value) is bool for value in values):
        return ("Bool", tuple(values))
    if all(type(value) is int and value >= 0 for value in values):
        return ("Nat", tuple(values))
    if all(isinstance(value, str) for value in values):
        return ("String", tuple(values))
    return None


def ls_literal(values: list[Any], render: Callable[[Any], str] = s) -> str:
    if values and all(type(value) is int and value == index for index, value in enumerate(values)):
        return f"List.range {len(values)}"
    if len(values) >= 32 and all(value == values[0] for value in values):
        return f"List.replicate {len(values)} {render(values[0])}"
    if len(values) > 256:
        chunks = [values[index : index + 256] for index in range(0, len(values), 256)]
        return "(" + " ++ ".join(ls_literal(chunk, render) for chunk in chunks) + ")"
    return "[" + ", ".join(render(value) for value in values) + "]"


def ls(values: list[Any], render: Callable[[Any], str] = s) -> str:
    key = primitive_list_key(values)
    if key is not None and key in _LIST_REFERENCES:
        return _LIST_REFERENCES[key]
    return ls_literal(values, render)


def collect_primitive_lists(value: Any, counts: dict[tuple[str, tuple[Any, ...]], int]) -> None:
    if isinstance(value, dict):
        for child in value.values():
            collect_primitive_lists(child, counts)
    elif isinstance(value, list):
        key = primitive_list_key(value)
        if key is not None:
            counts[key] = counts.get(key, 0) + 1
        for child in value:
            collect_primitive_lists(child, counts)


def interned_lists(case_name: str, case: dict[str, Any]) -> list[tuple[str, tuple[str, tuple[Any, ...]]]]:
    counts: dict[tuple[str, tuple[Any, ...]], int] = {}
    collect_primitive_lists(case, counts)
    selected = [key for key, count in counts.items() if len(key[1]) >= 128 or count > 1 and len(key[1]) >= 16]
    selected.sort(key=lambda key: (key[0], len(key[1]), repr(key[1][:4])))
    return [(f"{case_name}_list_{index:02d}", key) for index, key in enumerate(selected)]


def record(fields: list[tuple[str, str]]) -> str:
    return "{ " + ", ".join(f"{name} := {value}" for name, value in fields) + " }"


def render_derived(value: dict[str, Any]) -> str:
    return record(
        [
            ("name", s(value["name"])),
            ("sourceColumns", ls(value["sourceColumns"])),
            ("aggregateFitRowIds", ls(value["aggregateFitRowIds"])),
        ]
    )


def render_dataset(value: dict[str, Any]) -> str:
    list_fields = [
        "rowIds",
        "rowIdentityNonemptyMask",
        "filterInputRowIds",
        "filterOutputRowIds",
        "filterXRowIds",
        "filterTargetRowIds",
        "filterNameRowIds",
        "trainRowIds",
        "testRowIds",
        "xTrainRowIds",
        "yTrainRowIds",
        "nameTrainRowIds",
        "xTestRowIds",
        "yTestRowIds",
        "nameTestRowIds",
        "featureColumns",
        "targetColumns",
        "identifierColumns",
        "roleValidationPairs",
        "allowedDerivedSourceColumns",
    ]
    fields = [(name, ls(value[name])) for name in list_fields]
    fields.extend(
        [
            ("featureEngineeringEnabled", b(value["featureEngineeringEnabled"])),
            ("derivedFeatures", ls(value["derivedFeatures"], render_derived)),
        ]
    )
    return record(fields)


def render_stage(value: dict[str, Any]) -> str:
    return record(
        [
            ("stageId", s(value["stageId"])),
            ("name", s(value["name"])),
            ("fitRowIds", ls(value["fitRowIds"])),
            ("fitCount", str(value["fitCount"])),
            ("trainingStateDigest", s(value["trainingStateDigest"])),
            ("inferenceStateDigest", s(value["inferenceStateDigest"])),
            ("outputValueCount", str(value["outputValueCount"])),
            ("outputNonFiniteCount", str(value["outputNonFiniteCount"])),
        ]
    )


def render_pipeline(value: dict[str, Any]) -> str:
    list_fields = [
        "trainRowIds",
        "declaredStageIds",
        "materializedStageIds",
        "trainFeatureSchema",
        "inferenceInputFeatureSchema",
        "effectiveInferenceFeatureSchema",
        "pipelineOutputFeatureSchema",
        "modelTrainFeatureSchema",
    ]
    fields = [("preprocessingEnabled", b(value["preprocessingEnabled"]))]
    fields.extend((name, ls(value[name])) for name in list_fields)
    fields.extend(
        [
            ("pipelineTrainOutputDigest", s(value["pipelineTrainOutputDigest"])),
            ("modelTrainInputDigest", s(value["modelTrainInputDigest"])),
            ("stages", ls(value["stages"], render_stage)),
        ]
    )
    return record(fields)


def render_mapping(value: dict[str, Any]) -> str:
    return record([("label", s(value["label"])), ("code", str(value["code"]))])


def render_labels(value: dict[str, Any]) -> str:
    return record(
        [
            ("codecEnabled", b(value["codecEnabled"])),
            ("sourceLabels", ls(value["sourceLabels"])),
            ("runtimeMappings", ls(value["runtimeMappings"], render_mapping)),
            ("fullMappings", ls(value["fullMappings"], render_mapping)),
            ("trainMappings", ls(value["trainMappings"], render_mapping)),
            ("testMappings", ls(value["testMappings"], render_mapping)),
            ("persistedMappings", ls(value["persistedMappings"], render_mapping)),
            ("codecFitCount", str(value["codecFitCount"])),
            ("predictedCodes", ls(value["predictedCodes"], str)),
            ("decodedPredictions", ls(value["decodedPredictions"])),
        ]
    )


def render_prediction(value: dict[str, Any]) -> str:
    return record(
        [
            ("scope", s(value["scope"])),
            ("sourceRowIds", ls(value["sourceRowIds"])),
            ("predictionValues", ls(value["predictionValues"])),
            ("sampleRowIds", ls(value["sampleRowIds"])),
            ("artifactRowIds", ls(value["artifactRowIds"])),
            ("artifactPredictionValues", ls(value["artifactPredictionValues"])),
            ("artifactMismatchPolicy", s(value["artifactMismatchPolicy"])),
            ("modelRunId", s(value["modelRunId"])),
            ("artifactRunId", s(value["artifactRunId"])),
        ]
    )


def render_execution(value: dict[str, Any]) -> str:
    return record(
        [
            ("eligibleModels", ls(value["eligibleModels"])),
            ("selectedModelIds", ls(value["selectedModelIds"])),
            ("trainedModelIds", ls(value["trainedModelIds"])),
            ("trainedModelCount", str(value["trainedModelCount"])),
            ("registryBefore", ls(value["registryBefore"])),
            ("registryAfter", ls(value["registryAfter"])),
            ("registryMutationOperations", ls(value["registryMutationOperations"])),
            ("activeRunId", s(value["activeRunId"])),
            ("stateOwnerRunId", s(value["stateOwnerRunId"])),
        ]
    )


def render_case(value: dict[str, Any]) -> str:
    return record(
        [
            ("caseId", s(value["caseId"])),
            ("caseKind", s(value["caseKind"])),
            ("description", s(value["description"])),
            ("expectedConformant", b(value["expectedConformant"])),
            ("targetCheckId", s(value["targetCheckId"])),
            ("dataset", render_dataset(value["dataset"])),
            ("pipeline", render_pipeline(value["pipeline"])),
            ("labels", render_labels(value["labels"])),
            ("prediction", render_prediction(value["prediction"])),
            ("execution", render_execution(value["execution"])),
        ]
    )


def identifier(prefix: str, case_id: str, index: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", case_id)
    if not cleaned or cleaned[0].isdigit():
        cleaned = "case_" + cleaned
    return f"{prefix}_{index:02d}_{cleaned}"


PROPOSITION_EXPRESSIONS = {
    PUBLIC_CHECK_IDS[0]: lambda n: f"InputRowsIdentified {n}.dataset",
    PUBLIC_CHECK_IDS[1]: lambda n: f"SplitIsDisjointPartition {n}.dataset",
    PUBLIC_CHECK_IDS[2]: lambda n: f"SupervisedViewsRowAligned {n}.dataset",
    PUBLIC_CHECK_IDS[3]: lambda n: f"ColumnRolesGuardedAndDisjoint {n}.dataset",
    PUBLIC_CHECK_IDS[4]: lambda n: f"DerivedFeatureLineageSafe {n}.dataset",
    PUBLIC_CHECK_IDS[5]: lambda n: f"FilteredRowsKeepLineage {n}.dataset",
    PUBLIC_CHECK_IDS[6]: lambda n: f"EffectiveSchemaMatchesTraining {n}.pipeline",
    PUBLIC_CHECK_IDS[7]: lambda n: f"StatefulFitUsesTrainingRowsOnly {n}.pipeline",
    PUBLIC_CHECK_IDS[8]: lambda n: f"FittedStateReusedForModelAndInference {n}.pipeline",
    PUBLIC_CHECK_IDS[9]: lambda n: f"ModelInputSchemaMatchesPipelineOutput {n}.pipeline",
    PUBLIC_CHECK_IDS[10]: lambda n: f"DeclaredAndMaterializedStageOrderEqual {n}.pipeline",
    PUBLIC_CHECK_IDS[11]: lambda n: f"ObservedStageOutputsFinite {n}.pipeline",
    PUBLIC_CHECK_IDS[12]: lambda n: f"CodecTotalAndInjective {n}.labels",
    PUBLIC_CHECK_IDS[13]: lambda n: f"OneCodecFittedOnceForAllSplits {n}.labels",
    PUBLIC_CHECK_IDS[14]: lambda n: f"CodecPersistedAndPredictionsDecodable {n}.labels",
    PUBLIC_CHECK_IDS[15]: lambda n: f"PredictionsBoundToSourceRows {n}.prediction",
    PUBLIC_CHECK_IDS[16]: lambda n: f"ArtifactPairsAlignedAndMismatchRejected {n}.prediction",
    PUBLIC_CHECK_IDS[17]: lambda n: f"ModelArtifactAndStateShareRun {n}.prediction {n}.execution",
    PUBLIC_CHECK_IDS[18]: lambda n: f"SelectedModelsEligibleAndTrained {n}.execution",
    PUBLIC_CHECK_IDS[19]: lambda n: f"ModelRegistryImmutableDuringRun {n}.execution",
}


def theorem_name(base: str, check_id: str) -> str:
    return f"{base}_{check_id.split('.')[0].lower()}_result"


def append_result_theorem(lines: list[str], name: str, case: dict[str, Any], check_id: str, passed: bool) -> None:
    proposition = PROPOSITION_EXPRESSIONS[check_id](name)
    statement = proposition if passed else f"¬ {proposition}"
    theorem = theorem_name(name, check_id)
    lines.extend([f"theorem {theorem} : {statement} := by", "  decide +kernel", "", f"#print axioms {theorem}", ""])


def generate(counterexamples: dict[str, Any], production: dict[str, Any] | None = None) -> str:
    lines = [
        "import GeoPiVerify.Theorems",
        "",
        "namespace GeoPiVerify.Generated",
        "",
        "/- Closed facts originate from the two schema-v2 JSON traces. Every theorem below is reduced by Lean and checked by the kernel. -/",
        "",
    ]
    all_names: list[str] = []
    bundles = [("counterexample", counterexamples)]
    if production is not None:
        bundles.append(("production", production))
    for prefix, bundle in bundles:
        for index, case in enumerate(bundle["cases"]):
            name = identifier(prefix, case["caseId"], index)
            all_names.append(name)
            references = interned_lists(name, case)
            _LIST_REFERENCES.clear()
            _LIST_REFERENCES.update({key: reference for reference, key in references})
            for reference, (item_type, values) in references:
                render = b if item_type == "Bool" else str if item_type == "Nat" else s
                lines.extend([f"def {reference} : List {item_type} :=", f"  {ls_literal(list(values), render)}", ""])
            lines.extend([f"def {name} : CaseTrace :=", f"  {render_case(case)}", ""])
            _LIST_REFERENCES.clear()
            case_passed = accepted(case)
            statement = f"Conforms {name}" if case_passed else f"¬ Conforms {name}"
            overall = f"{name}_overall_result"
            if case["caseKind"] == "production":
                production_results = checks(case)
                for result in production_results:
                    append_result_theorem(lines, name, case, result["checkId"], result["passed"])
                role_result = next(result for result in production_results if result["checkId"] == PUBLIC_CHECK_IDS[3])
                if not case_passed and not role_result["passed"]:
                    role_theorem = theorem_name(name, PUBLIC_CHECK_IDS[3])
                    lines.extend(
                        [
                            f"theorem {overall} : {statement} := by",
                            "  intro h",
                            f"  exact {role_theorem} (accepted_implies_role_boundary {name} h)",
                            "",
                            f"#print axioms {overall}",
                            "",
                        ]
                    )
                else:
                    lines.extend([f"theorem {overall} : {statement} := by", "  decide +kernel", "", f"#print axioms {overall}", ""])
            elif case["caseKind"] == "counterexample":
                lines.extend([f"theorem {overall} : {statement} := by", "  decide +kernel", "", f"#print axioms {overall}", ""])
                target = case["targetCheckId"]
                target_result = next(result for result in checks(case) if result["checkId"] == target)
                append_result_theorem(lines, name, case, target, target_result["passed"])
            else:
                lines.extend([f"theorem {overall} : {statement} := by", "  decide +kernel", "", f"#print axioms {overall}", ""])
    lines.extend([f"def currentCases : List CaseTrace := [{', '.join(all_names)}]", "", "end GeoPiVerify.Generated", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counterexamples", type=Path, required=True)
    parser.add_argument("--production", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counterexamples = load_bundle(args.counterexamples)
    production = load_bundle(args.production) if args.production else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generate(counterexamples, production), encoding="utf-8")
    print(f"wrote closed Lean facts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
