#!/usr/bin/env python3
"""Generate one positive baseline and twenty manually designed single-node counterexamples."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from check_trace import PUBLIC_CHECK_IDS, check_trace

ROOT = Path(__file__).resolve().parents[2]
AUDITED_GIT_PATHS = [
    "geochemistrypi/data_mining",
    "formal_verification",
    ":(exclude)formal_verification/results",
    ":(exclude)formal_verification/results/**",
    ":(exclude)formal_verification/GeoPiVerify/Generated/CurrentRun.lean",
]


def source_commit() -> str:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no", "--", *AUDITED_GIT_PATHS],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    return commit + ("-dirty" if dirty else "")


def baseline_case() -> dict[str, Any]:
    rows = [0, 1, 2, 3, 4, 5]
    train = [0, 1, 2, 3]
    test = [4, 5]
    mapping = [{"label": "basalt", "code": 4}, {"label": "granite", "code": 9}]
    return {
        "caseId": "counterexample_baseline",
        "caseKind": "baseline",
        "description": "人工反例套件的独立正向基线",
        "expectedConformant": True,
        "targetCheckId": "",
        "dataset": {
            "rowIds": list(rows),
            "rowIdentityNonemptyMask": [True] * len(rows),
            "filterInputRowIds": list(rows),
            "filterOutputRowIds": list(rows),
            "filterXRowIds": list(rows),
            "filterTargetRowIds": list(rows),
            "filterNameRowIds": list(rows),
            "trainRowIds": list(train),
            "testRowIds": list(test),
            "xTrainRowIds": list(train),
            "yTrainRowIds": list(train),
            "nameTrainRowIds": list(train),
            "xTestRowIds": list(test),
            "yTestRowIds": list(test),
            "nameTestRowIds": list(test),
            "featureColumns": ["SiO2", "MgO", "ratio"],
            "targetColumns": ["rock_type"],
            "identifierColumns": ["sample_id"],
            "roleValidationPairs": ["feature_target", "feature_identifier", "target_identifier"],
            "featureEngineeringEnabled": True,
            "allowedDerivedSourceColumns": ["SiO2", "MgO"],
            "derivedFeatures": [{"name": "ratio", "sourceColumns": ["SiO2", "MgO"], "aggregateFitRowIds": []}],
        },
        "pipeline": {
            "preprocessingEnabled": True,
            "trainRowIds": list(train),
            "declaredStageIds": ["scale-1"],
            "materializedStageIds": ["scale-1"],
            "trainFeatureSchema": ["SiO2", "MgO", "ratio"],
            "inferenceInputFeatureSchema": ["sample_id", "SiO2", "MgO", "ratio", "unused"],
            "effectiveInferenceFeatureSchema": ["SiO2", "MgO", "ratio"],
            "pipelineOutputFeatureSchema": ["SiO2", "MgO", "ratio"],
            "modelTrainFeatureSchema": ["SiO2", "MgO", "ratio"],
            "pipelineTrainOutputDigest": "model-input-1",
            "modelTrainInputDigest": "model-input-1",
            "stages": [
                {
                    "stageId": "scale-1",
                    "name": "StandardScaler",
                    "fitRowIds": list(train),
                    "fitCount": 1,
                    "trainingStateDigest": "state-1",
                    "inferenceStateDigest": "state-1",
                    "outputValueCount": 12,
                    "outputNonFiniteCount": 0,
                }
            ],
        },
        "labels": {
            "codecEnabled": True,
            "sourceLabels": ["basalt", "granite"],
            "runtimeMappings": mapping,
            "fullMappings": copy.deepcopy(mapping),
            "trainMappings": copy.deepcopy(mapping),
            "testMappings": copy.deepcopy(mapping),
            "persistedMappings": copy.deepcopy(mapping),
            "codecFitCount": 1,
            "predictedCodes": [4, 9],
            "decodedPredictions": ["basalt", "granite"],
        },
        "prediction": {
            "scope": "test",
            "sourceRowIds": list(test),
            "predictionValues": ["basalt", "granite"],
            "sampleRowIds": list(test),
            "artifactRowIds": list(test),
            "artifactPredictionValues": ["basalt", "granite"],
            "artifactMismatchPolicy": "reject",
            "modelRunId": "run-1",
            "artifactRunId": "run-1",
        },
        "execution": {
            "eligibleModels": ["Logistic Regression", "Decision Tree"],
            "selectedModelIds": ["Logistic Regression"],
            "trainedModelIds": ["Logistic Regression"],
            "trainedModelCount": 1,
            "registryBefore": ["Logistic Regression", "Decision Tree"],
            "registryAfter": ["Logistic Regression", "Decision Tree"],
            "registryMutationOperations": [],
            "activeRunId": "run-1",
            "stateOwnerRunId": "run-1",
        },
    }


def mutant(base: dict[str, Any], index: int, mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    case = copy.deepcopy(base)
    check_id = PUBLIC_CHECK_IDS[index]
    case.update(
        {
            "caseId": f"counterexample_{check_id.split('.')[0].lower()}",
            "caseKind": "counterexample",
            "description": f"仅破坏 {check_id} 的人工反例",
            "expectedConformant": False,
            "targetCheckId": check_id,
        }
    )
    mutate(case)
    return case


def build_bundle() -> tuple[dict[str, Any], dict[str, Any]]:
    base = baseline_case()
    mutations: list[Callable[[dict[str, Any]], None]] = []

    def d01(c: dict[str, Any]) -> None:
        c["dataset"]["rowIdentityNonemptyMask"][0] = False

    mutations.append(d01)

    def d02(c: dict[str, Any]) -> None:
        c["dataset"]["trainRowIds"] = [0, 1, 2, 3, 4]
        c["dataset"]["testRowIds"] = [4, 5]
        for key in ("xTrainRowIds", "yTrainRowIds", "nameTrainRowIds"):
            c["dataset"][key] = list(c["dataset"]["trainRowIds"])

    mutations.append(d02)
    mutations.append(lambda c: c["dataset"].__setitem__("xTestRowIds", [5, 4]))
    mutations.append(lambda c: c["dataset"].__setitem__("roleValidationPairs", ["feature_target", "feature_identifier"]))
    mutations.append(lambda c: c["dataset"]["derivedFeatures"][0].__setitem__("sourceColumns", ["SiO2", "rock_type"]))
    mutations.append(lambda c: c["dataset"].__setitem__("filterNameRowIds", [1, 0, 2, 3, 4, 5]))
    mutations.append(lambda c: c["pipeline"].__setitem__("effectiveInferenceFeatureSchema", ["MgO", "SiO2", "ratio"]))
    mutations.append(lambda c: c["pipeline"]["stages"][0].__setitem__("fitRowIds", [0, 1, 2, 3, 4]))
    mutations.append(lambda c: c["pipeline"]["stages"][0].__setitem__("fitCount", 2))
    mutations.append(lambda c: c["pipeline"].__setitem__("modelTrainFeatureSchema", ["SiO2", "MgO"]))
    mutations.append(lambda c: c["pipeline"].__setitem__("declaredStageIds", ["other-stage"]))
    mutations.append(lambda c: c["pipeline"]["stages"][0].__setitem__("outputNonFiniteCount", 1))

    def l01(c: dict[str, Any]) -> None:
        bad = [{"label": "basalt", "code": 4}, {"label": "granite", "code": 4}]
        for key in ("runtimeMappings", "fullMappings", "trainMappings", "testMappings", "persistedMappings"):
            c["labels"][key] = copy.deepcopy(bad)
        c["labels"]["predictedCodes"] = [4, 4]
        c["labels"]["decodedPredictions"] = ["basalt", "basalt"]
        c["prediction"]["predictionValues"] = ["basalt", "basalt"]
        c["prediction"]["artifactPredictionValues"] = ["basalt", "basalt"]

    mutations.append(l01)
    mutations.append(lambda c: c["labels"].__setitem__("codecFitCount", 2))
    mutations.append(lambda c: c["labels"].__setitem__("persistedMappings", []))

    def a01(c: dict[str, Any]) -> None:
        c["prediction"]["sampleRowIds"] = [5, 4]
        c["prediction"]["artifactRowIds"] = [5, 4]

    mutations.append(a01)
    mutations.append(lambda c: c["prediction"].__setitem__("artifactMismatchPolicy", "positional_fallback"))
    mutations.append(lambda c: c["prediction"].__setitem__("artifactRunId", "run-2"))

    def e01(c: dict[str, Any]) -> None:
        c["execution"]["selectedModelIds"] = ["Unknown Model"]
        c["execution"]["trainedModelIds"] = ["Unknown Model"]

    mutations.append(e01)

    def e02(c: dict[str, Any]) -> None:
        c["execution"]["registryAfter"] = c["execution"]["registryBefore"] + ["all_models"]
        c["execution"]["registryMutationOperations"] = ["append_all_models"]

    mutations.append(e02)

    if len(mutations) != len(PUBLIC_CHECK_IDS):
        raise RuntimeError("counterexample mutation count does not match public check count")
    cases = [base] + [mutant(base, index, mutation) for index, mutation in enumerate(mutations)]
    bundle = {
        "schemaVersion": 2,
        "sourceCommit": source_commit(),
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "cases": cases,
    }
    report = check_trace(bundle)
    if not report["counterexampleCoverageComplete"] or not report["allCounterexamplesIsolated"] or not report["allExpectationsMatched"]:
        failures = {case["caseId"]: case["failedCheckIds"] for case in report["cases"] if not case["isolationMatched"]}
        raise RuntimeError(f"counterexample suite is not isolated: {failures}")
    observations = {
        "designRule": "每个反例仅改变正向基线中的一个语义事实",
        "baselineAccepted": report["cases"][0]["accepted"],
        "counterexampleCount": report["counterexampleCount"],
        "coveredCheckCount": report["coveredCheckCount"],
        "allCounterexamplesIsolated": report["allCounterexamplesIsolated"],
        "cases": [{"caseId": case["caseId"], "targetCheckId": case["targetCheckId"], "failedCheckIds": case["failedCheckIds"]} for case in report["cases"][1:]],
    }
    return bundle, observations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    args = parser.parse_args()
    bundle, observations = build_bundle()
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    args.observations.parent.mkdir(parents=True, exist_ok=True)
    args.trace.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.observations.write_text(json.dumps(observations, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(bundle['cases'])} counterexample cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
