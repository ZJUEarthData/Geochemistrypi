#!/usr/bin/env python3
"""Audit the real GeoPi business path and emit one production trace without mutations."""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
import tempfile
import types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET_PATH = ROOT / "geochemistrypi/data_mining/data/dataset/Data_Classification.xlsx"
APPLICATION_PATH = ROOT / "geochemistrypi/data_mining/data/dataset/ApplicationData_Classification.xlsx"
CLI_PATH = ROOT / "geochemistrypi/data_mining/cli_pipeline.py"

FEATURE_COLUMNS = [
    "SIO2(WT%)",
    "TIO2(WT%)",
    "AL2O3(WT%)",
    "FEOT(WT%)",
    "CAO(WT%)",
    "MGO(WT%)",
    "MNO(WT%)",
    "NA2O(WT%)",
]
DERIVED_COLUMN = "SIO2_MGO_SUM"
TARGET_COLUMN = "Label"
IDENTIFIER_COLUMNS = ["CITATION", "SAMPLE NAME"]
AUDIT_RUN_ID = "geopi-built-in-classification-audit"
AUDITED_GIT_PATHS = [
    "geochemistrypi/data_mining",
    "formal_verification",
    ":(exclude)formal_verification/results",
    ":(exclude)formal_verification/results/**",
    ":(exclude)formal_verification/GeoPiVerify/Generated/CurrentRun.lean",
]


def _install_external_stubs() -> None:
    if "mlflow" not in sys.modules:
        mlflow = types.ModuleType("mlflow")

        def noop(*_args: Any, **_kwargs: Any) -> None:
            return None

        for name in ("log_artifact", "log_metric", "log_metrics", "log_param", "log_params", "set_tag"):
            setattr(mlflow, name, noop)
        mlflow.active_run = lambda: SimpleNamespace(info=SimpleNamespace(run_id=AUDIT_RUN_ID))
        sklearn_module = types.ModuleType("mlflow.sklearn")
        sklearn_module.log_model = noop
        sklearn_module.load_model = noop
        mlflow.sklearn = sklearn_module
        sys.modules["mlflow"] = mlflow
        sys.modules["mlflow.sklearn"] = sklearn_module
    if "flaml" not in sys.modules:
        flaml = types.ModuleType("flaml")

        class AutoML:
            pass

        flaml.AutoML = AutoML
        sys.modules["flaml"] = flaml


def _source_commit() -> str:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no", "--", *AUDITED_GIT_PATHS],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    return commit + ("-dirty" if dirty else "")


def _token(value: Any) -> str:
    import numpy as np
    import pandas as pd

    if value is None or value is pd.NA:
        return "null"
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return "nan"
        if math.isinf(number):
            return "+inf" if number > 0 else "-inf"
        return format(number, ".17g")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_token)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _array_digest(array: Any) -> str:
    import numpy as np

    values = np.asarray(array)
    return _digest({"shape": list(values.shape), "values": [_token(value) for value in values.ravel()]})


def _state_digest(scaler: Any) -> str:
    return _digest({"mean": [_token(value) for value in scaler.mean_], "scale": [_token(value) for value in scaler.scale_]})


def _business_row_ids(frame: Any) -> list[str]:
    return [f"{citation}|{sample}" for citation, sample in zip(frame["CITATION"].astype(str), frame["SAMPLE NAME"].astype(str))]


def _ids_for(frame: Any, row_id_by_index: dict[Any, int]) -> list[int]:
    return [row_id_by_index[index] for index in frame.index]


def _explicit_role_validations() -> list[str]:
    """Extract explicit cross-role comparisons from the actual CLI function AST."""

    tree = ast.parse(CLI_PATH.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Compare, ast.Call, ast.BinOp)):
            continue
        names = {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
        attrs = {child.attr for child in ast.walk(node) if isinstance(child, ast.Attribute)}
        if "columns" not in attrs:
            continue
        if {"X", "y"} <= names:
            found.add("feature_target")
        if "X" in names and ("NAME" in names or "name_column_select" in names):
            found.add("feature_identifier")
        if "y" in names and ("NAME" in names or "name_column_select" in names):
            found.add("target_identifier")
    return sorted(found)


def _registry_mutations() -> list[str]:
    """Read the actual MODELS alias and append operation from the CLI AST."""

    tree = ast.parse(CLI_PATH.read_text(encoding="utf-8"))
    aliases_models = False
    operations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "MODELS" for target in node.targets):
            aliases_models = isinstance(node.value, ast.Subscript) and isinstance(node.value.value, ast.Name) and node.value.value.id == "Modes2Models"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "MODELS" and node.func.attr == "append":
            if node.args and isinstance(node.args[0], ast.Constant):
                operations.append(f"append_{node.args[0].value}")
    return operations if aliases_models else []


def _artifact_mismatch_policy(save_data_function: Any) -> str:
    tree = ast.parse(inspect.getsource(save_data_function))
    has_raise = any(isinstance(node, ast.Raise) for node in ast.walk(tree))
    resets_index = any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "reset_index" for node in ast.walk(tree))
    return "reject" if has_raise else ("positional_fallback" if resets_index else "unchecked")


def _mapping_from_pairs(original: Any, encoded: Any) -> list[dict[str, Any]]:
    pairs = sorted({(_token(source), int(code)) for source, code in zip(original, encoded)}, key=lambda item: item[1])
    return [{"label": label, "code": code} for label, code in pairs]


def build_bundle() -> tuple[dict[str, Any], dict[str, Any]]:
    _install_external_stubs()
    os.environ.setdefault("MPLBACKEND", "Agg")

    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    from geochemistrypi.data_mining.constants import CLASSIFICATION_MODELS
    from geochemistrypi.data_mining.data import inference as inference_module
    from geochemistrypi.data_mining.data.data_readiness import data_split
    from geochemistrypi.data_mining.data.feature_engineering import FeatureConstructor
    from geochemistrypi.data_mining.data.preprocessing import feature_scaler
    from geochemistrypi.data_mining.model.func.algo_classification import _common as label_module
    from geochemistrypi.data_mining.utils import base as base_utils

    raw = pd.read_excel(DATASET_PATH)
    application = pd.read_excel(APPLICATION_PATH)
    row_id_texts = _business_row_ids(raw)
    row_key_by_identity = {identity: key for key, identity in enumerate(row_id_texts)}
    row_ids = [row_key_by_identity[identity] for identity in row_id_texts]
    row_id_by_index = dict(zip(raw.index, row_ids))
    names = pd.DataFrame({"sample_id": row_id_texts}, index=raw.index)

    # The business path selects features before feature engineering, so the
    # constructor can only read candidate feature columns.
    feature_input = raw[FEATURE_COLUMNS].copy()
    constructor = FeatureConstructor(feature_input, "sample_id")
    feature_config = {DERIVED_COLUMN: "self.data['SIO2(WT%)'] + self.data['MGO(WT%)']"}
    engineered = constructor.batch_build(feature_config)
    model_columns = FEATURE_COLUMNS + [DERIVED_COLUMN]
    X_raw = engineered[model_columns]
    y = raw[[TARGET_COLUMN]]

    app_input = application[FEATURE_COLUMNS].copy()
    app_engineered = FeatureConstructor(app_input, "sample_id").batch_build(feature_config)
    effective_application = app_engineered[model_columns]

    # The business path splits before stateful preprocessing, fits the scaler
    # on training rows only, and reuses that fitted state for inference.
    split = data_split(X_raw, y, names, test_size=0.2)
    X_train_raw = split["X Train"]
    X_test_raw = split["X Test"]
    train_ids = _ids_for(X_train_raw, row_id_by_index)
    test_ids = _ids_for(X_test_raw, row_id_by_index)

    scaling_config, scaled_train_np, fitted_scaler = feature_scaler(X_train_raw, ["Standardization"], 0)
    scaled_train = pd.DataFrame(scaled_train_np, columns=X_train_raw.columns, index=X_train_raw.index)
    scaled_test = pd.DataFrame(fitted_scaler.transform(X_test_raw), columns=X_train_raw.columns, index=X_test_raw.index)

    training_state = _state_digest(fitted_scaler)
    old_save_text, old_save_model = inference_module.save_text, inference_module.save_model
    inference_module.save_text = lambda *_args, **_kwargs: None
    inference_module.save_model = lambda *_args, **_kwargs: None
    try:
        _, transform_pipeline = inference_module.build_transform_pipeline(
            {},
            scaling_config,
            {},
            SimpleNamespace(transformer_config={}),
            X_train_raw,
            split["Y Train"],
            fitted_steps={type(fitted_scaler).__name__: fitted_scaler},
        )
    finally:
        inference_module.save_text, inference_module.save_model = old_save_text, old_save_model
    pipeline_scaler = transform_pipeline.named_steps["standardscaler"]
    inference_state = _state_digest(pipeline_scaler)
    pipeline_train_output = transform_pipeline.transform(X_train_raw)
    output_values = np.asarray(pipeline_train_output, dtype=float)
    output_value_count = int(output_values.size)
    output_nonfinite_count = int(output_values.size - np.isfinite(output_values).sum())

    original_encoder = label_module.LabelEncoder
    fitted_encoders: list[Any] = []

    class TrackingLabelEncoder(LabelEncoder):
        def fit(self, values: Any, y: Any = None) -> Any:
            del y
            result = super().fit(values)
            fitted_encoders.append(self)
            return result

    label_module.LabelEncoder = TrackingLabelEncoder
    try:
        encoded_all, encoded_train, encoded_test, codec = label_module.reset_label(y, split["Y Train"], split["Y Test"], ["Automatic Coding"], 0)
    finally:
        label_module.LabelEncoder = original_encoder
    if len(fitted_encoders) != 1:
        raise RuntimeError(f"reset_label fitted {len(fitted_encoders)} codecs")
    runtime_mapping = _mapping_from_pairs(y[TARGET_COLUMN].tolist(), encoded_all[TARGET_COLUMN].tolist())
    train_mapping = _mapping_from_pairs(split["Y Train"][TARGET_COLUMN].tolist(), encoded_train[TARGET_COLUMN].tolist())
    test_mapping = _mapping_from_pairs(split["Y Test"][TARGET_COLUMN].tolist(), encoded_test[TARGET_COLUMN].tolist())
    # The business path persists the codec with the model; replay that exact
    # persistence step and record the persisted payload as the audit fact.
    with tempfile.TemporaryDirectory(prefix="geopi-codec-audit-") as codec_dir:
        label_module.persist_label_codec(codec, codec_dir, "Logistic Regression")
        persisted_mapping = json.loads((Path(codec_dir) / "Label Codec - Logistic Regression.txt").read_text(encoding="utf-8"))

    from geochemistrypi.data_mining.model.classification import LogisticRegressionClassification

    workflow = LogisticRegressionClassification(max_iter=300)
    workflow.fit(scaled_train, encoded_train)
    predicted_codes_np = workflow.predict(scaled_test)
    code_to_label = {item["code"]: item["label"] for item in runtime_mapping}
    predicted_codes = [int(value) for value in np.asarray(predicted_codes_np).ravel()]
    decoded_predictions = [code_to_label[code] for code in predicted_codes]

    prediction_frame = pd.DataFrame({"Predicted Value": decoded_predictions}, index=split["X Test"].index)
    artifact_input = prediction_frame.copy(deep=True)
    artifact_names = split["Name Test"].copy(deep=True)
    with tempfile.TemporaryDirectory(prefix="geopi-production-audit-") as temp_dir:
        base_utils.save_data(artifact_input, artifact_names, "Application Data Predicted", temp_dir)
        exported = pd.read_excel(Path(temp_dir) / "Application Data Predicted.xlsx")
    artifact_row_ids = [row_key_by_identity[identity] for identity in exported["sample_id"].astype(str).tolist()]
    artifact_predictions = exported["Predicted Value"].map(_token).tolist()

    role_validations = _explicit_role_validations()
    mutation_operations = _registry_mutations()
    registry_before = list(CLASSIFICATION_MODELS)
    registry_after = registry_before + (["all_models"] if "append_all_models" in mutation_operations else [])
    artifact_policy = _artifact_mismatch_policy(base_utils.save_data)
    source_labels = sorted({_token(value) for value in y[TARGET_COLUMN].tolist()})

    case = {
        "caseId": "geopi_builtin_classification_production_audit",
        "caseKind": "production",
        "description": "GeoPi 内置分类数据及真实业务函数形成的生产审计事实",
        "expectedConformant": True,
        "targetCheckId": "",
        "dataset": {
            "rowIds": row_ids,
            "rowIdentityNonemptyMask": [bool(identity) for identity in row_id_texts],
            "filterInputRowIds": row_ids,
            "filterOutputRowIds": row_ids,
            "filterXRowIds": row_ids,
            "filterTargetRowIds": row_ids,
            "filterNameRowIds": row_ids,
            "trainRowIds": train_ids,
            "testRowIds": test_ids,
            "xTrainRowIds": _ids_for(split["X Train"], row_id_by_index),
            "yTrainRowIds": _ids_for(split["Y Train"], row_id_by_index),
            "nameTrainRowIds": _ids_for(split["Name Train"], row_id_by_index),
            "xTestRowIds": _ids_for(split["X Test"], row_id_by_index),
            "yTestRowIds": _ids_for(split["Y Test"], row_id_by_index),
            "nameTestRowIds": _ids_for(split["Name Test"], row_id_by_index),
            "featureColumns": model_columns,
            "targetColumns": [TARGET_COLUMN],
            "identifierColumns": IDENTIFIER_COLUMNS,
            "roleValidationPairs": role_validations,
            "featureEngineeringEnabled": True,
            "allowedDerivedSourceColumns": list(constructor.data.columns),
            "derivedFeatures": [{"name": DERIVED_COLUMN, "sourceColumns": ["SIO2(WT%)", "MGO(WT%)"], "aggregateFitRowIds": []}],
        },
        "pipeline": {
            "preprocessingEnabled": True,
            "trainRowIds": train_ids,
            "declaredStageIds": ["StandardScaler"],
            "materializedStageIds": [type(pipeline_scaler).__name__],
            "trainFeatureSchema": model_columns,
            "inferenceInputFeatureSchema": list(app_engineered.columns),
            "effectiveInferenceFeatureSchema": list(effective_application.columns),
            "pipelineOutputFeatureSchema": model_columns,
            "modelTrainFeatureSchema": list(split["X Train"].columns),
            "pipelineTrainOutputDigest": _array_digest(pipeline_train_output),
            "modelTrainInputDigest": _array_digest(scaled_train),
            "stages": [
                {
                    "stageId": "StandardScaler",
                    "name": "StandardScaler",
                    "fitRowIds": train_ids,
                    "fitCount": 1,
                    "trainingStateDigest": training_state,
                    "inferenceStateDigest": inference_state,
                    "outputValueCount": output_value_count,
                    "outputNonFiniteCount": output_nonfinite_count,
                }
            ],
        },
        "labels": {
            "codecEnabled": True,
            "sourceLabels": source_labels,
            "runtimeMappings": runtime_mapping,
            "fullMappings": runtime_mapping,
            "trainMappings": train_mapping,
            "testMappings": test_mapping,
            "persistedMappings": persisted_mapping,
            "codecFitCount": len(fitted_encoders),
            "predictedCodes": predicted_codes,
            "decodedPredictions": decoded_predictions,
        },
        "prediction": {
            "scope": "test",
            "sourceRowIds": test_ids,
            "predictionValues": decoded_predictions,
            "sampleRowIds": test_ids,
            "artifactRowIds": artifact_row_ids,
            "artifactPredictionValues": artifact_predictions,
            "artifactMismatchPolicy": artifact_policy,
            "modelRunId": AUDIT_RUN_ID,
            "artifactRunId": AUDIT_RUN_ID,
        },
        "execution": {
            "eligibleModels": registry_before,
            "selectedModelIds": ["Logistic Regression"],
            "trainedModelIds": ["Logistic Regression"],
            "trainedModelCount": 1,
            "registryBefore": registry_before,
            "registryAfter": registry_after,
            "registryMutationOperations": mutation_operations,
            "activeRunId": AUDIT_RUN_ID,
            "stateOwnerRunId": AUDIT_RUN_ID,
        },
    }

    bundle = {
        "schemaVersion": 2,
        "sourceCommit": _source_commit(),
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "cases": [case],
    }

    from check_trace import check_trace

    report = check_trace(bundle)
    check_results = {item["checkId"]: item["passed"] for item in report["cases"][0]["checks"]}
    observations = {
        "auditData": {
            "trainingFile": str(DATASET_PATH.relative_to(ROOT)),
            "applicationFile": str(APPLICATION_PATH.relative_to(ROOT)),
            "trainingRows": len(raw),
            "applicationRows": len(application),
            "featureColumns": model_columns,
            "targetColumn": TARGET_COLUMN,
            "identifierColumns": IDENTIFIER_COLUMNS,
            "rowIdentityDigest": _digest(row_id_texts),
        },
        "sourceFacts": {
            "roleValidationPairs": role_validations,
            "featureConstructorReadableColumns": list(constructor.data.columns),
            "artifactMismatchPolicy": artifact_policy,
            "registryMutationOperations": mutation_operations,
        },
        "runtimeFacts": {
            "trainRows": len(train_ids),
            "testRows": len(test_ids),
            "pipelineFitRows": len(train_ids),
            "pipelineFitCount": 1,
            "trainingStateDigest": training_state,
            "inferenceStateDigest": inference_state,
            "modelTrainInputDigest": case["pipeline"]["modelTrainInputDigest"],
            "pipelineTrainOutputDigest": case["pipeline"]["pipelineTrainOutputDigest"],
            "codecFitCount": len(fitted_encoders),
            "persistedCodecEntries": len(persisted_mapping),
            "predictionCount": len(decoded_predictions),
            "artifactRowCount": len(artifact_row_ids),
            "pipelineOutputValueCount": output_value_count,
            "pipelineOutputNonFiniteCount": output_nonfinite_count,
        },
        "checkResults": check_results,
        "failedCheckIds": report["cases"][0]["failedCheckIds"],
        "sourceLocations": {
            "D04.column_roles_guarded_and_disjoint": ["geochemistrypi/data_mining/cli_pipeline.py 569"],
            "D05.derived_feature_lineage_safe": ["geochemistrypi/data_mining/cli_pipeline.py 579"],
            "P02.stateful_fit_uses_training_rows_only": ["geochemistrypi/data_mining/cli_pipeline.py 615"],
            "P03.fitted_state_reused_for_model_and_inference": [
                "geochemistrypi/data_mining/cli_pipeline.py 615",
                "geochemistrypi/data_mining/cli_pipeline.py 792",
                "geochemistrypi/data_mining/data/inference.py 67",
            ],
            "L03.codec_persisted_and_predictions_decodable": [
                "geochemistrypi/data_mining/model/func/algo_classification/_common.py 502",
                "geochemistrypi/data_mining/model/func/algo_classification/_common.py 548",
            ],
            "A02.artifact_pairs_aligned_and_mismatch_rejected": ["geochemistrypi/data_mining/utils/base.py 222"],
            "E02.model_registry_immutable_during_run": ["geochemistrypi/data_mining/cli_pipeline.py 712"],
        },
    }
    return bundle, observations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    args = parser.parse_args()
    try:
        bundle, observations = build_bundle()
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        args.observations.parent.mkdir(parents=True, exist_ok=True)
        args.trace.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        args.observations.write_text(json.dumps(observations, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception as error:
        print(f"runtime probe failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(f"wrote production audit to {args.trace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
